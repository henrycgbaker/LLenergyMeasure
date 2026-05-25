"""H2 + H3 + H9 runner for Phase 3b cheap-patterns batch.

Per ``research/mining-substrate-trial/findings/phase3b_hybrid_catalogue.md`` H2, H3, H9. The batch
plays to the LLM's diagnosis-over-synthesis strength established by H4
(LLM excellent at WHERE+WHY gaps exist; weak at single-shot code
synthesis). All three patterns are single-shot reads with NO
output-mutation patches.

Patterns:

H2 (validate): (a)'s output is the input. LLM reads (a) + engine source
  and classifies each (a) entry as CONFIRM / SUSPECT-SPURIOUS /
  UNCERTAIN. Output = (a) with DROPPED entries removed.

H3 (propose + verify): LLM proposes invariants (pure (b)). Verification
  gate: for transformers, runtime_validate_invariants (Phase 2.5
  harness); for vllm + tensorrt, lightweight schema-existence fallback
  (check field in Model.__fields__). Output = (b) entries that pass the
  gate.

H9 (diagnose-only): LLM reads (a)'s output + engine source + gap
  inventory; produces STRUCTURED diagnosis of gap categories. No output
  mutation.

Cells: 3 engines x 3 patterns = 9 LLM calls total.

For H3 vllm + tensorrt schema-existence gate: instead of runtime
import (transitive dep wall), parse the canonical Pydantic / dataclass
class bodies and collect the declared `__fields__` set via AST. A
proposed invariant whose target field is NOT in `__fields__` is
schema-rejected. This is a weaker gate than the transformers runtime
gate; we document the asymmetry honestly.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Project root sits four parents above research/mining-substrate-trial/scripts/strategies/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_TRIAL_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_TRIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_TRIAL_SCRIPTS_DIR))

# Local imports.
from strategies.llm_extractor import (  # noqa: E402
    OllamaBackend,
    extract_with_retry,
)
from trial_scoring import (  # noqa: E402
    ScoringConfig,
    score_invariants,
)


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSpec:
    engine: str
    version_slug: str
    venv_source_root: Path  # e.g. /tmp/trial_<engine>_<version>_venv/src/<engine>
    chunker_module: str  # transformers_chunker | vllm_chunker | tensorrt_chunker
    config_classes: list[str]  # Model class names to introspect for H3 schema gate
    config_module_paths: list[str]  # Files (rel to venv_source_root) hosting the classes


ENGINE_SPECS: dict[str, EngineSpec] = {
    "transformers": EngineSpec(
        engine="transformers",
        version_slug="v4_57_3",
        venv_source_root=Path("/tmp/trial_transformers_v4_57_6_venv/src/transformers"),
        chunker_module="transformers_chunker",
        config_classes=["GenerationConfig", "BitsAndBytesConfig"],
        config_module_paths=[
            "generation/configuration_utils.py",
            "utils/quantization_config.py",
        ],
    ),
    "vllm": EngineSpec(
        engine="vllm",
        version_slug="v0_7_3",
        venv_source_root=Path("/tmp/trial_vllm_v0_7_3_venv/src/vllm"),
        chunker_module="vllm_chunker",
        config_classes=[
            "ModelConfig",
            "CacheConfig",
            "ParallelConfig",
            "SchedulerConfig",
            "EngineArgs",
            "LoRAConfig",
            "DeviceConfig",
            "DecodingConfig",
            "ObservabilityConfig",
            "LoadConfig",
            "PromptAdapterConfig",
            "TokenizerPoolConfig",
            "SamplingParams",
            "BeamSearchParams",
            "GuidedDecodingParams",
        ],
        config_module_paths=[
            "config.py",
            "engine/arg_utils.py",
            "sampling_params.py",
        ],
    ),
    "tensorrt": EngineSpec(
        engine="tensorrt",
        version_slug="v0_21_0",
        venv_source_root=Path("/tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt"),
        chunker_module="tensorrt_chunker",
        config_classes=[
            "BaseLlmArgs",
            "TrtLlmArgs",
            "TorchLlmArgs",
            "CalibConfig",
            "SchedulerConfig",
            "KvCacheConfig",
            "PeftCacheConfig",
            "LookaheadDecodingConfig",
            "DynamicBatchConfig",
            "CacheTransceiverConfig",
            "ExtendedRuntimePerfKnobConfig",
            "DecodingBaseConfig",
            "MedusaDecodingConfig",
            "EagleDecodingConfig",
            "NGramDecodingConfig",
            "MTPDecodingConfig",
            "BuildCacheConfig",
        ],
        config_module_paths=[
            "llmapi/llm_args.py",
            "llmapi/build_cache.py",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Common helpers: load engine source, load invariants, write artefacts
# ---------------------------------------------------------------------------


def _load_invariants(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _summarise_invariants_for_prompt(invariants: list[dict[str, Any]]) -> str:
    """Compact representation of (a) entries for LLM input. Full enough
    for validation/diagnosis without ballooning token count."""
    lines: list[str] = []
    for i, inv in enumerate(invariants):
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", f"<{i}>")
        sev = inv.get("severity", "?")
        native = inv.get("native_type", "?")
        method = (inv.get("miner_source") or {}).get("method", "?")
        match = inv.get("match") or {}
        fields = match.get("fields") or {}
        fields_str = json.dumps(fields, default=str)[:200]
        msg = (inv.get("message_template") or "")[:120]
        lines.append(
            f"- id: {iid}\n"
            f"  severity: {sev}\n"
            f"  native_type: {native}\n"
            f"  miner_source_method: {method}\n"
            f"  match.fields: {fields_str}\n"
            f"  message: {msg}"
        )
    return "\n".join(lines) if lines else "(no invariants)"


def _load_engine_source_for_prompt(spec: EngineSpec, max_chars_per_file: int = 30000) -> str:
    """Load the engine source files most relevant to the (a)/(b)/(d) flow.

    We avoid full-file dumps that blow past 32k context. We cap each file
    at ~30k chars and label which file each block is from. The LLM sees
    enough to validate / diagnose / propose.
    """
    parts: list[str] = []
    for rel_path in spec.config_module_paths:
        full = spec.venv_source_root / rel_path
        if not full.exists():
            parts.append(f"=== {rel_path} ===\n(file not found at {full})\n")
            continue
        text = full.read_text()
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + f"\n... [TRUNCATED {len(text) - max_chars_per_file} chars]"
        parts.append(f"=== {rel_path} ===\n{text}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# H2: LLM validates (a)'s output
# ---------------------------------------------------------------------------

H2_PROMPT_TEMPLATE = """\
You are a precision auditor for a static-analysis invariant catalogue.
You read the static-analysis output (a) below, and the engine source it
was mined from. For each (a) entry, classify whether the static
analysis got it RIGHT.

Three labels per entry:

- CONFIRM: the entry corresponds to a real validation pattern in the
  engine source. The field name, severity, and predicate match what
  the source actually does.

- SUSPECT-SPURIOUS: the entry does NOT correspond to a real validation
  pattern in the source - or - it matches a pattern that has been
  deleted / no longer enforces what the entry says. Give a 1-line
  reason citing the source.

- UNCERTAIN: the entry might be real but you can't confirm from the
  source provided (e.g. validation lives in a method whose body isn't
  in the excerpt; predicate semantics are ambiguous).

Be conservative on DROPPING (only flag SUSPECT-SPURIOUS when you have
clear evidence). False-DROP is worse than false-CONFIRM here - the
trial scores recall against the (a) reference; dropping a real entry
reduces recall directly. When in doubt, prefer UNCERTAIN over
SUSPECT-SPURIOUS.

OUTPUT FORMAT: a single JSON object (no markdown, no commentary, no
code fences, just raw JSON):

{{
  "classifications": [
    {{
      "id": "<entry id from (a) output>",
      "label": "CONFIRM" | "SUSPECT-SPURIOUS" | "UNCERTAIN",
      "reason": "<1-line; cite source line/method when SUSPECT-SPURIOUS>"
    }}
  ]
}}

Include EVERY entry from (a). Order matches (a)'s order. No commentary
outside the JSON.

=== ENGINE: {engine} v{version_slug} ===

=== (a) OUTPUT - {n_entries} entries ===
{output_summary}

=== ENGINE SOURCE EXCERPTS ===
{engine_source}

Now emit the JSON. Start with `{{`.
"""


def run_h2_for_engine(
    spec: EngineSpec,
    *,
    out_root: Path,
    backend: OllamaBackend,
) -> dict[str, Any]:
    """Run H2 (LLM validates (a)) for one engine."""
    engine_out = out_root / "h2_validate" / spec.engine
    engine_out.mkdir(parents=True, exist_ok=True)

    # 1. Load (a)'s output for this engine.
    a_invariants_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / spec.engine
        / spec.version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    a_data = _load_invariants(a_invariants_path)
    a_invs = a_data.get("invariants") or []
    n_entries = len(a_invs)

    # 2. Build prompt.
    engine_source = _load_engine_source_for_prompt(spec)
    output_summary = _summarise_invariants_for_prompt(a_invs)
    prompt = H2_PROMPT_TEMPLATE.format(
        engine=spec.engine,
        version_slug=spec.version_slug,
        n_entries=n_entries,
        output_summary=output_summary,
        engine_source=engine_source,
    )
    (engine_out / "prompt.md").write_text(prompt)

    # 3. Call Ollama.
    print(f"[H2 {spec.engine}] prompt size={len(prompt):,} chars; n_entries={n_entries}")
    t0 = time.time()
    extraction = extract_with_retry(
        backend=backend,
        prompt=prompt,
        chunk_name=f"h2_{spec.engine}",
        output_format="json",
        max_retries=2,
        timeout=1800,
        max_output_tokens=8192,
    )
    elapsed = time.time() - t0

    raw_responses = "\n=== RESPONSE ===\n".join(extraction.raw_responses) or "<no responses>"
    (engine_out / "raw_response.txt").write_text(raw_responses)

    # 4. Parse classifications.
    classifications: list[dict[str, Any]] = []
    if extraction.parsed is not None and isinstance(extraction.parsed, dict):
        for c in extraction.parsed.get("classifications") or []:
            if isinstance(c, dict):
                classifications.append(
                    {
                        "id": str(c.get("id", "")),
                        "label": str(c.get("label", "UNCERTAIN")).strip().upper(),
                        "reason": str(c.get("reason", "")),
                    }
                )

    (engine_out / "classifications.json").write_text(json.dumps(classifications, indent=2))

    # 5. Build filtered output: drop SUSPECT-SPURIOUS entries; keep UNCERTAIN.
    dropped_ids = {c["id"] for c in classifications if c["label"] == "SUSPECT-SPURIOUS"}
    confirm_count = sum(1 for c in classifications if c["label"] == "CONFIRM")
    suspect_count = len(dropped_ids)
    uncertain_count = sum(1 for c in classifications if c["label"] == "UNCERTAIN")

    filtered_invs = [inv for inv in a_invs if (inv.get("id") if isinstance(inv, dict) else None) not in dropped_ids]
    filtered_data = {
        "schema_version": a_data.get("schema_version", "1.0.0"),
        "engine": a_data.get("engine"),
        "engine_version": a_data.get("engine_version"),
        "mined_at": _dt.datetime.now().isoformat() + "+00:00",
        "filtered_by": "h2_validate",
        "dropped_ids": sorted(dropped_ids),
        "invariants": filtered_invs,
    }
    _write_yaml(engine_out / "filtered_proposed.yaml", filtered_data)

    # 6. Score filtered output against the active reference (= (a) itself).
    # Reference is engine_versions/<e>/<active>/outputs/invariants.proposed.yaml,
    # which IS (a). So the recall measures how aggressive LLM was at
    # dropping (a)'s own entries. Precision should stay 1.0 (we only
    # drop entries; we never add).
    cfg = ScoringConfig()
    score = score_invariants(
        reference_invariants=a_data,
        cell_invariants=filtered_data,
        config=cfg,
    )
    (recall, precision, sev_acc, fmode, ref_count, cell_count, int_count,
     recall_misses, precision_spurious, sev_mismatches) = score
    score_json = {
        "pattern": "h2_validate",
        "engine": spec.engine,
        "version_slug": spec.version_slug,
        "wall_sec": elapsed,
        "a_entry_count": n_entries,
        "classifications_returned": len(classifications),
        "confirm_count": confirm_count,
        "suspect_spurious_count": suspect_count,
        "uncertain_count": uncertain_count,
        "drop_rate": suspect_count / n_entries if n_entries else 0.0,
        "filtered_entry_count": len(filtered_invs),
        "recall_vs_active_ref": recall,
        "precision_vs_active_ref": precision,
        "severity_accuracy": sev_acc,
        "failure_mode": str(fmode),
        "intersection_count": int_count,
        "raw_responses_failure_modes": extraction.failure_modes,
    }
    (engine_out / "score.json").write_text(json.dumps(score_json, indent=2))
    print(
        f"[H2 {spec.engine}] DONE: drop={suspect_count}/{n_entries}={score_json['drop_rate']:.1%}, "
        f"recall_vs_ref={recall:.3f}, precision={precision:.3f}, wall={elapsed:.1f}s"
    )
    return score_json


# ---------------------------------------------------------------------------
# H3: LLM proposes -> (a) runtime / schema verifies
# ---------------------------------------------------------------------------


def _collect_fields_via_ast(spec: EngineSpec) -> dict[str, set[str]]:
    """For vllm + tensorrt: build a {class_name: {field_names}} index by
    AST-parsing the source files. This is the schema-existence gate.

    Includes:
    - Pydantic / dataclass class-body annotations (``ast.AnnAssign``).
    - Plain assignments (dataclass defaults without annotation).
    - ``__init__`` parameter names (for hand-written __init__ classes
      like BuildCacheConfig where fields are kwargs not annotations).
    """
    out: dict[str, set[str]] = {}
    for rel_path in spec.config_module_paths:
        full = spec.venv_source_root / rel_path
        if not full.exists():
            continue
        try:
            tree = ast.parse(full.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if spec.config_classes and node.name not in spec.config_classes:
                continue
            fields = set()
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    fields.add(member.target.id)
                elif isinstance(member, ast.Assign):
                    for tgt in member.targets:
                        if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                            # Heuristic: dataclass plain assignments (= default)
                            # look like fields. Skip caps-only (class constants).
                            if not tgt.id.isupper():
                                fields.add(tgt.id)
                elif isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == "__init__":
                    # __init__ kwargs are fields for hand-written __init__ classes.
                    for arg in member.args.args:
                        if arg.arg != "self":
                            fields.add(arg.arg)
                    for arg in getattr(member.args, "kwonlyargs", []) or []:
                        fields.add(arg.arg)
            out.setdefault(node.name, set()).update(fields)
    return out


def _extract_field_from_invariant(inv: dict[str, Any]) -> str | None:
    """Return the candidate field name for the schema-existence gate.

    Match.fields is keyed by `<namespace>.<field>` or sometimes a nested
    structure. We pick the LAST dotted segment of the first field key,
    which is the actual field name in the engine's class.
    """
    match = inv.get("match") or {}
    fields = match.get("fields") or {}
    if not isinstance(fields, dict):
        return None
    key = next(iter(fields.keys()), None)
    if not isinstance(key, str):
        return None
    return key.split(".")[-1]


def _verify_invariants_transformers(b_invs: list[dict[str, Any]], b_data: dict[str, Any], b_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use runtime_validate_invariants on transformers; return (verified, dropped)."""
    from trial_scoring import runtime_validate_invariants

    # Subset to those whose native_type the validator supports.
    supported_substrings = ("GenerationConfig", "BitsAndBytesConfig")
    eligible_ids: set[str] = set()
    skipped_unsupported: list[dict[str, Any]] = []
    for inv in b_invs:
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", "")
        nt = str(inv.get("native_type", ""))
        if any(s in nt for s in supported_substrings):
            eligible_ids.add(iid)
        else:
            skipped_unsupported.append(
                {
                    "id": iid,
                    "drop_reason": f"native_type {nt!r} not supported by transformers runtime gate",
                }
            )

    # Run runtime validation on eligible IDs.
    try:
        rvs = runtime_validate_invariants(b_path, engine="transformers", only_ids=eligible_ids)
    except Exception as exc:
        rvs = []
        for iid in eligible_ids:
            rvs.append(type("RV", (), {"invariant_id": iid, "positive_confirmed": False, "negative_confirmed": False, "observed_outcome": None, "error": f"runtime_validate raised: {exc}"})())

    rv_by_id = {rv.invariant_id: rv for rv in rvs}

    verified: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for inv in b_invs:
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", "")
        if iid not in eligible_ids:
            continue
        rv = rv_by_id.get(iid)
        if rv is None:
            dropped.append({"id": iid, "drop_reason": "no runtime validation result"})
            continue
        if rv.error and not rv.positive_confirmed:
            dropped.append(
                {
                    "id": iid,
                    "drop_reason": f"runtime validation error: {rv.error[:200]}",
                }
            )
            continue
        # Gate criterion: positive case must trigger SOMETHING (gate-pass).
        # We don't require negative_confirmed because some invariants are
        # asymmetric / hard to construct clean negative cases.
        if rv.positive_confirmed:
            verified.append(inv)
        else:
            dropped.append(
                {
                    "id": iid,
                    "drop_reason": f"positive case did not trigger ({rv.observed_outcome or 'no_op'})",
                }
            )

    # Also retain the unsupported-skipped entries as dropped (gate
    # effectively rejects them since we can't check).
    dropped.extend(skipped_unsupported)
    return verified, dropped


def _verify_invariants_schema(
    b_invs: list[dict[str, Any]],
    spec: EngineSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Schema-existence gate: field-in-AST-collected-fields.

    Used for vllm + tensorrt where runtime validation is unavailable in
    this trial environment. Weaker than runtime; documents asymmetry.
    """
    class_fields = _collect_fields_via_ast(spec)
    all_fields: set[str] = set()
    for fs in class_fields.values():
        all_fields.update(fs)

    verified: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for inv in b_invs:
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", "")
        fname = _extract_field_from_invariant(inv)
        if fname is None:
            dropped.append({"id": iid, "drop_reason": "no field key in match.fields"})
            continue
        if fname in all_fields:
            verified.append(inv)
        else:
            dropped.append(
                {
                    "id": iid,
                    "drop_reason": f"field {fname!r} not in declared __fields__ of {sorted(class_fields.keys())[:5]}...",
                    "field": fname,
                }
            )

    return verified, dropped


def run_h3_for_engine(
    spec: EngineSpec,
    *,
    out_root: Path,
) -> dict[str, Any]:
    """Run H3 (LLM proposes via existing (b); verify with runtime/schema gate).

    H3 does NOT make a new LLM call - it REUSES the existing (b)
    extraction at trial_runs/b/<engine>/<version>/invariants.proposed.yaml.
    The novel work is the verification gate.
    """
    engine_out = out_root / "h3_propose_verify" / spec.engine
    engine_out.mkdir(parents=True, exist_ok=True)

    # 1. Load (b) output.
    b_path = (
        _PROJECT_ROOT / "research" / "mining-substrate-trial"
        / "findings"
        / "trial_runs"
        / "b"
        / spec.engine
        / spec.version_slug
        / "invariants.proposed.yaml"
    )
    if not b_path.exists():
        print(f"[H3 {spec.engine}] SKIP: (b) output missing at {b_path}")
        return {"pattern": "h3_propose_verify", "engine": spec.engine, "skipped": True}
    b_data = _load_invariants(b_path)
    b_invs = b_data.get("invariants") or []
    b_count = len(b_invs)

    t0 = time.time()
    if spec.engine == "transformers":
        verified, dropped = _verify_invariants_transformers(b_invs, b_data, b_path)
        gate_kind = "runtime"
    else:
        verified, dropped = _verify_invariants_schema(b_invs, spec)
        gate_kind = "schema_existence"
    elapsed = time.time() - t0

    verified_data = {
        "schema_version": b_data.get("schema_version", "1.0.0"),
        "engine": b_data.get("engine"),
        "engine_version": b_data.get("engine_version"),
        "verified_at": _dt.datetime.now().isoformat() + "+00:00",
        "verified_by": f"h3_propose_verify_{gate_kind}",
        "invariants": verified,
    }
    _write_yaml(engine_out / "verified_invariants.yaml", verified_data)
    _write_yaml(engine_out / "dropped_entries.yaml", {"dropped": dropped})

    # 2. Score verified output vs active reference + b baseline.
    ref_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / spec.engine
        / spec.version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    ref_data = _load_invariants(ref_path)
    cfg = ScoringConfig()
    score = score_invariants(reference_invariants=ref_data, cell_invariants=verified_data, config=cfg)
    (recall_v, precision_v, sev_acc_v, fmode_v, ref_count, cell_count_v, int_count_v,
     _rm, _ps, _sm) = score
    # Also score raw (b) for delta baseline.
    score_b = score_invariants(reference_invariants=ref_data, cell_invariants=b_data, config=cfg)
    (recall_b, precision_b, sev_acc_b, fmode_b, _rc, cell_count_b, int_count_b, _, _, _) = score_b

    score_json = {
        "pattern": "h3_propose_verify",
        "engine": spec.engine,
        "version_slug": spec.version_slug,
        "gate_kind": gate_kind,
        "wall_sec": elapsed,
        "b_emitted_count": b_count,
        "verified_count": len(verified),
        "dropped_count": len(dropped),
        "ref_count": ref_count,
        "intersection_verified_count": int_count_v,
        "intersection_b_count": int_count_b,
        "recall_verified": recall_v,
        "precision_verified": precision_v,
        "recall_b_baseline": recall_b,
        "precision_b_baseline": precision_b,
        "precision_lift": precision_v - precision_b,
        "recall_change": recall_v - recall_b,
    }
    (engine_out / "score.json").write_text(json.dumps(score_json, indent=2))

    # 3. Observations.
    obs = []
    obs.append(f"# H3 observations: {spec.engine}\n")
    obs.append(f"- Gate kind: {gate_kind}\n")
    obs.append(f"- (b) emitted: {b_count}\n")
    obs.append(f"- Verified: {len(verified)}\n")
    obs.append(f"- Dropped: {len(dropped)}\n")
    obs.append(
        f"- recall: (b) baseline {recall_b:.3f} -> verified {recall_v:.3f} (delta {recall_v - recall_b:+.3f})\n"
    )
    obs.append(
        f"- precision: (b) baseline {precision_b:.3f} -> verified {precision_v:.3f} (delta {precision_v - precision_b:+.3f})\n"
    )

    # Identify hallucination patterns dropped (e.g. tensorrt v0_21_0 b had
    # HF GenerationConfig hallucinations).
    hallucination_evidence: list[str] = []
    for d in dropped:
        if "GenerationConfig" in str(d.get("field", "")) and spec.engine == "tensorrt":
            hallucination_evidence.append(d["id"])
    if hallucination_evidence:
        obs.append(
            f"\n- Hallucination pattern caught: {len(hallucination_evidence)} HF GenerationConfig "
            f"entries dropped (tensorrt should not declare HF fields):\n"
        )
        for hid in hallucination_evidence[:10]:
            obs.append(f"  - {hid}\n")

    (engine_out / "observations.md").write_text("".join(obs))
    print(
        f"[H3 {spec.engine}] DONE: {gate_kind} gate, b={b_count}->verified={len(verified)} "
        f"(dropped {len(dropped)}), recall {recall_b:.3f}->{recall_v:.3f}, "
        f"precision {precision_b:.3f}->{precision_v:.3f}, wall={elapsed:.1f}s"
    )
    return score_json


# ---------------------------------------------------------------------------
# H9: LLM diagnoses gaps; structured YAML output
# ---------------------------------------------------------------------------


H9_PROMPT_TEMPLATE = """\
You are a structural-gap analyst for an AST-based static-analysis
catalogue. The output (a) below is what a deterministic AST walker
emitted from the engine source. You are looking for STRUCTURAL gap
PATTERNS that explain why certain validations the source contains do
NOT appear in (a). Each gap pattern is a categorical reason - e.g. the
walker doesn't descend into orelse branches, or doesn't track local-
variable aliases.

For each gap pattern you can identify, emit ONE entry in the JSON
output. Cite a SPECIFIC example field name from this engine that
exhibits the gap. Be honest if you can't find an example - emit no
entries rather than fabricate.

Gap categories you should consider:
- branch-descent: walker handles if/raise but not if/elif/else, not
  else-raise, not nested branches.
- nested-config: walker doesn't recurse into Pydantic / dataclass
  fields typed as another config class.
- local-var-alias: walker compares self.X but source has x = self.X
  and compares the local x.
- normalisation-only: source uses normalisation pattern (no raise);
  walker only emits from if/raise patterns; these validators dormant
  for the walker.
- type-blindness: walker synthesises probe values without consulting
  the declared type; runtime fails for int-typed fields probed with
  string.
- defensive-import: walker fails at import time on bumped versions
  because some symbol was renamed in a dep.
- other: any structural gap that doesn't fit the above.

OUTPUT FORMAT: a single JSON object (no markdown, no commentary, no
code fences, just raw JSON). Use these EXACT keys:

{{
  "gaps": [
    {{
      "category": "branch-descent | nested-config | local-var-alias | normalisation-only | type-blindness | defensive-import | other",
      "severity": "blocks-correctness | reduces-recall | minor",
      "example_field": "<field name in {engine} that exhibits the gap>",
      "structural_reason": "<1-2 sentences; use plain prose without colons or backticks>",
      "cross_engine_pattern": "yes-already-known | yes-new-here | engine-specific | unknown",
      "fix_estimate_loc": <integer>,
      "mergeable_into_spike_refactor": "yes | no | needs-broader-design"
    }}
  ]
}}

IMPORTANT: in structural_reason, avoid colons and backticks (the parser
is strict). Use plain English. If you must reference code, describe it
in words (e.g. "the if/raise pattern" not "`if X: raise`").

If you find no gaps, output: {{"gaps": []}}

=== ENGINE: {engine} v{version_slug} ===

=== (a) OUTPUT ({n_entries} entries) ===
{output_summary}

=== ENGINE SOURCE EXCERPTS ===
{engine_source}

=== KNOWN GAPS FROM PRIOR REVIEW (post_trial_a_gap_closure.md) ===
{known_gaps}

You may RE-CONFIRM known gaps (mark cross_engine_pattern as
yes-already-known) AND surface NEW ones (mark yes-new-here). Be
explicit which is which. Now emit the JSON. Start with `{{`.
"""


# Gap inventory text - small enough to inline, sourced from
# post_trial_a_gap_closure.md.
ENGINE_KNOWN_GAPS: dict[str, str] = {
    "transformers": """\
G-trf-1: defensive imports - the transformers producer's imports crash
on bumped versions (v4_55_4 / v5_9_0). Defensive try/except + AST
fallback would let the walker emit zero invariants gracefully instead
of crashing.
""",
    "vllm": """\
G-vllm-1: EngineArgs.__post_init__ - ZERO raises, all normalisation
(if x is None: x = default). Walker only emits from `if X: raise`.
G-vllm-2: ModelConfig._verify_quantization / _verify_tokenizer_mode /
_verify_cuda_graph compare LOCAL variables (e.g. `quantization` not
`self.quantization`). Walker can't tie local-var-compare to a field
invariant without call-graph analysis.
G-vllm-3: CacheConfig._verify_cache_dtype - if/elif/else chain. Walker
only handles top-level if/raise pattern; doesn't descend into orelse
branches.
""",
    "tensorrt": """\
G-trt-1: type-blind probe synthesis - `_value_satisfying("present",
True)` returns `"x"` even for int-typed fields. Probe construction
needs declared-type information.
G-trt-2: DeprecationWarning poisoning - NOT a walker gap; lives in
validation-emission capture. Out of scope for diagnosis here.
G-trt-3: nested-config dispatch - SchedulerConfig, QuantConfig,
KvCacheConfig are Pydantic-validator-bearing nested classes the walker
doesn't recurse into via class-level type references.
""",
}


def run_h9_for_engine(
    spec: EngineSpec,
    *,
    out_root: Path,
    backend: OllamaBackend,
) -> dict[str, Any]:
    """Run H9 (LLM diagnoses (a)'s structural gaps) for one engine."""
    engine_out = out_root / "h9_diagnose" / spec.engine
    engine_out.mkdir(parents=True, exist_ok=True)

    # 1. Load (a)'s output.
    a_invariants_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / spec.engine
        / spec.version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    a_data = _load_invariants(a_invariants_path)
    a_invs = a_data.get("invariants") or []
    n_entries = len(a_invs)

    # 2. Build prompt.
    engine_source = _load_engine_source_for_prompt(spec)
    output_summary = _summarise_invariants_for_prompt(a_invs)
    known_gaps = ENGINE_KNOWN_GAPS.get(spec.engine, "(no known gaps documented)")
    prompt = H9_PROMPT_TEMPLATE.format(
        engine=spec.engine,
        version_slug=spec.version_slug,
        n_entries=n_entries,
        output_summary=output_summary,
        engine_source=engine_source,
        known_gaps=known_gaps,
    )
    (engine_out / "prompt.md").write_text(prompt)

    # 3. Call Ollama.
    print(f"[H9 {spec.engine}] prompt size={len(prompt):,} chars")
    t0 = time.time()
    extraction = extract_with_retry(
        backend=backend,
        prompt=prompt,
        chunk_name=f"h9_{spec.engine}",
        output_format="json",
        max_retries=2,
        timeout=1800,
        max_output_tokens=4096,
    )
    elapsed = time.time() - t0

    raw_responses = "\n=== RESPONSE ===\n".join(extraction.raw_responses) or "<no responses>"
    (engine_out / "raw_response.txt").write_text(raw_responses)

    # 4. Parse YAML; expect gaps: [...].
    diagnoses: list[dict[str, Any]] = []
    if extraction.parsed is not None and isinstance(extraction.parsed, dict):
        gaps = extraction.parsed.get("gaps") or []
        for g in gaps:
            if isinstance(g, dict):
                diagnoses.append(g)

    # Write structured diagnoses YAML.
    _write_yaml(engine_out / "diagnoses.yaml", {"gaps": diagnoses, "diagnosed_at": _dt.datetime.now().isoformat()})

    # 5. Observations.
    obs = []
    obs.append(f"# H9 observations: {spec.engine}\n\n")
    obs.append(f"- Diagnoses emitted: {len(diagnoses)}\n")
    obs.append(f"- LLM wall: {elapsed:.1f}s\n")
    obs.append(f"- Failure modes: {extraction.failure_modes}\n\n")
    if diagnoses:
        category_counts: dict[str, int] = {}
        cross_engine_counts: dict[str, int] = {}
        for d in diagnoses:
            cat = str(d.get("category", "?"))
            category_counts[cat] = category_counts.get(cat, 0) + 1
            xe = str(d.get("cross_engine_pattern", "?"))
            cross_engine_counts[xe] = cross_engine_counts.get(xe, 0) + 1
        obs.append("## Category breakdown\n")
        for cat, n in sorted(category_counts.items(), key=lambda x: -x[1]):
            obs.append(f"- {cat}: {n}\n")
        obs.append("\n## Cross-engine pattern breakdown\n")
        for xe, n in sorted(cross_engine_counts.items(), key=lambda x: -x[1]):
            obs.append(f"- {xe}: {n}\n")
    (engine_out / "observations.md").write_text("".join(obs))

    score_json = {
        "pattern": "h9_diagnose",
        "engine": spec.engine,
        "version_slug": spec.version_slug,
        "wall_sec": elapsed,
        "diagnoses_count": len(diagnoses),
        "raw_failure_modes": extraction.failure_modes,
    }
    print(f"[H9 {spec.engine}] DONE: {len(diagnoses)} diagnoses, wall={elapsed:.1f}s")
    return score_json


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run H2 + H3 + H9 cells for Phase 3b.")
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["h2", "h3", "h9", "all"],
        default=["all"],
        help="Patterns to run (default: all)",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["transformers", "vllm", "tensorrt", "all"],
        default=["all"],
        help="Engines to run (default: all)",
    )
    parser.add_argument(
        "--out-root",
        default=str(_PROJECT_ROOT / "research" / "mining-substrate-trial" / "findings" / "hybrid_experiments"),
        help="Output root directory",
    )
    args = parser.parse_args(argv)

    patterns = args.patterns
    if "all" in patterns:
        patterns = ["h2", "h3", "h9"]

    engines = args.engines
    if "all" in engines:
        engines = ["transformers", "vllm", "tensorrt"]

    out_root = Path(args.out_root)
    backend = OllamaBackend()

    results: list[dict[str, Any]] = []

    # H2 first (cheapest LLM calls), then H9, then H3 (no LLM call).
    for pattern in patterns:
        for eng in engines:
            spec = ENGINE_SPECS[eng]
            try:
                if pattern == "h2":
                    r = run_h2_for_engine(spec, out_root=out_root, backend=backend)
                elif pattern == "h9":
                    r = run_h9_for_engine(spec, out_root=out_root, backend=backend)
                elif pattern == "h3":
                    r = run_h3_for_engine(spec, out_root=out_root)
                else:
                    continue
                results.append(r)
            except Exception as exc:
                import traceback

                err = {
                    "pattern": pattern,
                    "engine": eng,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                results.append(err)
                print(f"[{pattern} {eng}] CRASH: {exc}")
                # Still continue.

    # Aggregate summary.
    aggregate_path = out_root / "h2_h3_h9_aggregate.json"
    aggregate_path.write_text(json.dumps(results, indent=2))
    print(f"\n[main] aggregate written to {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
