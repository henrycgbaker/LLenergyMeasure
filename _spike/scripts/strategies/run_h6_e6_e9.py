"""H6 + E6 + E9 runner for Phase 3b substrate-decomposition batch.

Three substrate-side variants probing whether the (b) ceiling at 70B-q4
is driven by the chunking decomposition strategy:

H6 (no-chunking)
    Single whole-source LLM call (one for schema, one for invariants).
    Cell: transformers v4.57.3 only (32k context).

E6 (field-anchored)
    For each chunk, prepend the engine's ``Model.__fields__`` declaration.
    Prompt instructs: only propose invariants on fields in this list.
    Cells: transformers v4.57.3 + vllm v0.7.3.

E9 (sequential methodical sweep with cumulative context)
    File-by-file extraction with running notes from prior chunks.
    Each step's prompt embeds prior-extraction summary so the LLM can
    dedup inline + catch cross-class invariants.
    Cells: transformers v4.57.3 + vllm v0.7.3.

Each pattern reuses ``llm_extractor`` (OllamaBackend + extract_with_retry).
Output layout per cell:

    _spike/findings/hybrid_experiments/<pattern>/<engine>/<version>/
      schema.json
      invariants.proposed.yaml
      observations.md
      score.json
      raw_llm_transcripts/*.md
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts import trial_scoring  # noqa: E402
from _spike.scripts.strategies.llm_b_oss import (  # noqa: E402
    _add_invariant_if_unique,
    _pull_invariants_list,
    _write_transcript,
)
from _spike.scripts.strategies.llm_extractor import (  # noqa: E402
    DEFAULT_OLLAMA_URL,
    OllamaBackend,
    extract_with_retry,
    filter_internal_plumbing,
    is_internal_plumbing_field,
)
from _spike.scripts.strategies.prompts import (  # noqa: E402
    INVARIANTS_PROMPT_TEMPLATE,
    SCHEMA_JSON_SCHEMA,
    SCHEMA_PROMPT_TEMPLATE,
)
from _spike.scripts.strategies.transformers_chunker import (  # noqa: E402
    ChunkSpec as TfChunkSpec,
)
from _spike.scripts.strategies.transformers_chunker import (
    get_active_sources as tf_get_active_sources,
)
from _spike.scripts.strategies.transformers_chunker import (
    get_sources_from_path as tf_get_sources_from_path,
)
from _spike.scripts.strategies.transformers_chunker import (
    invariants_chunks as tf_invariants_chunks,
)
from _spike.scripts.strategies.transformers_chunker import (
    schema_chunks as tf_schema_chunks,
)
from _spike.scripts.strategies.vllm_chunker import (
    invariants_chunks as vllm_invariants_chunks,
)
from _spike.scripts.strategies.vllm_chunker import (
    schema_chunks as vllm_schema_chunks,
)

# ---------------------------------------------------------------------------
# Engine spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSpec:
    engine: str
    version_slug: str
    engine_version: str
    config_classes: list[str]
    config_module_paths: list[str]
    venv_source_root: Path
    namespace_for_invariants: str


# Engine specs match the H2/H3/H9 batch for path consistency.
TRANSFORMERS_SPEC = EngineSpec(
    engine="transformers",
    version_slug="v4_57_3",
    engine_version="4.57.3",
    config_classes=["GenerationConfig", "BitsAndBytesConfig", "CompileConfig"],
    config_module_paths=[
        "generation/configuration_utils.py",
        "utils/quantization_config.py",
    ],
    venv_source_root=Path("/tmp/trial_transformers_v4_57_6_venv/src/transformers"),
    namespace_for_invariants="transformers.sampling",
)

VLLM_SPEC = EngineSpec(
    engine="vllm",
    version_slug="v0_7_3",
    engine_version="0.7.3",
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
    venv_source_root=Path("/tmp/vllm-unpacked/vllm"),
    namespace_for_invariants="vllm",
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _collect_fields_via_ast(spec: EngineSpec) -> dict[str, set[str]]:
    """Build {class_name: {field_names}} by AST-parsing the engine source.

    Same approach as the H3 schema-existence gate. Includes:
    - Pydantic / dataclass / msgspec class-body annotations (AnnAssign).
    - Plain assignments (dataclass defaults without annotation).
    - __init__ parameter names.
    - ``self.X = kwargs.pop("X", ...)`` patterns inside __init__ (the
      transformers GenerationConfig pattern; AnnAssign alone misses these).
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
            fields: set[str] = set()
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    fields.add(member.target.id)
                elif isinstance(member, ast.Assign):
                    for tgt in member.targets:
                        if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                            if not tgt.id.isupper():
                                fields.add(tgt.id)
                elif (
                    isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and member.name == "__init__"
                ):
                    for arg in member.args.args:
                        if arg.arg != "self":
                            fields.add(arg.arg)
                    for arg in getattr(member.args, "kwonlyargs", []) or []:
                        fields.add(arg.arg)
                    # Walk the __init__ body for `self.X = kwargs.pop("Y", ...)`
                    # and `X = kwargs.pop("Y", ...)` patterns.
                    for sub in ast.walk(member):
                        if isinstance(sub, ast.Call):
                            func = sub.func
                            is_kwargs_pop = False
                            if (
                                isinstance(func, ast.Attribute)
                                and func.attr == "pop"
                                and isinstance(func.value, ast.Name)
                                and func.value.id == "kwargs"
                            ):
                                is_kwargs_pop = True
                            if is_kwargs_pop and sub.args:
                                first = sub.args[0]
                                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                                    fields.add(first.value)
            # Drop names that look like internal plumbing
            fields = {f for f in fields if not is_internal_plumbing_field(f)}
            out.setdefault(node.name, set()).update(fields)
    return out


def _retry_with_529_backoff(fn, retries: int = 3, base_backoff: int = 60):
    """Wrap a callable; retry on 529 / overload / connection errors.

    Resilience per handoff: 529 retry x3 with 60s backoff.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "529" in msg or "overload" in msg or "connection" in msg:
                wait = base_backoff * (attempt + 1)
                print(
                    f"  [retry] caught {type(exc).__name__}: {exc}; sleeping {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_with_529_backoff: no exception captured but retries exhausted")


# ---------------------------------------------------------------------------
# H6: whole-source no-chunking (transformers only)
# ---------------------------------------------------------------------------


def _tf_sources_for_h6(spec: EngineSpec) -> dict[str, str]:
    """Pull transformers source: prefer file-based AST extraction from the
    pre-staged venv (works regardless of project-venv state); fall back to
    inspect-based extraction if files unavailable.
    """
    if spec.venv_source_root.exists():
        return tf_get_sources_from_path(spec.venv_source_root)
    return tf_get_active_sources()


def _build_h6_invariants_source(spec: EngineSpec) -> str:
    """Concatenate every invariant-relevant source piece into a SINGLE
    block, with section headers preserved so the LLM can navigate.

    For transformers: GenerationConfig.__init__ + GenerationConfig.validate
    (full) + BitsAndBytesConfig (full) + CompileConfig + WatermarkingConfig.
    """
    if spec.engine != "transformers":
        raise ValueError(f"H6 only configured for transformers; got {spec.engine!r}")

    sources = _tf_sources_for_h6(spec)
    if "_extraction_error" in sources:
        raise RuntimeError(f"source extraction failed: {sources['_extraction_error']}")

    parts: list[str] = []
    parts.append(
        "=== SOURCE: transformers.GenerationConfig.__init__ "
        "(sampling_params via kwargs.pop) ===\n" + sources["GenerationConfig.__init__"] + "\n"
    )
    parts.append(
        "=== SOURCE: transformers.GenerationConfig.validate "
        "(the main invariant-emitting method) ===\n" + sources["GenerationConfig.validate"] + "\n"
    )
    parts.append(
        "=== SOURCE: transformers.utils.quantization_config.BitsAndBytesConfig "
        "(type-check invariants on engine_params) ===\n" + sources["BitsAndBytesConfig"] + "\n"
    )
    parts.append(
        "=== SOURCE: transformers.generation.configuration_utils.CompileConfig "
        "(companion of GenerationConfig.compile_config) ===\n" + sources["CompileConfig"] + "\n"
    )
    parts.append(
        "=== SOURCE: transformers.generation.configuration_utils.WatermarkingConfig "
        "(companion of GenerationConfig.watermarking_config) ===\n"
        + sources["WatermarkingConfig"]
        + "\n"
    )
    return "\n".join(parts)


def _build_h6_schema_source(spec: EngineSpec) -> str:
    """Concatenate schema-relevant source: from_pretrained signature + GenerationConfig
    init + docstring + BnB + CompileConfig etc., bounded to fit the 32k context.

    We trim PreTrainedModel.from_pretrained heavily (the full thing is 43k chars; we
    take just signature + docstring kwargs section).
    """
    if spec.engine != "transformers":
        raise ValueError(f"H6 only configured for transformers; got {spec.engine!r}")

    sources = _tf_sources_for_h6(spec)
    if "_extraction_error" in sources:
        raise RuntimeError(f"source extraction failed: {sources['_extraction_error']}")

    # Use the same per-chunk truncation policy as schema_chunks() to keep within budget.
    from _spike.scripts.strategies.transformers_chunker import _truncate_signature_source

    fp_src = _truncate_signature_source(sources["PreTrainedModel.from_pretrained"], max_chars=6000)

    # Pull the latter half of from_pretrained docstring (kwargs section).
    import re as _re

    full_fp = sources["PreTrainedModel.from_pretrained"]
    m = _re.search(r'r?"""(.*?)"""', full_fp, _re.DOTALL)
    docstring_body = m.group(1) if m else ""
    half_point = len(docstring_body) // 2
    docstring_kwargs = docstring_body[half_point:][:12000]

    parts: list[str] = []
    parts.append(
        "=== SOURCE: PreTrainedModel.from_pretrained "
        "(signature + leading docstring) -> engine_params ===\n" + fp_src + "\n"
    )
    parts.append(
        "=== SOURCE: PreTrainedModel.from_pretrained docstring kwargs section "
        "(Sphinx kwargs documenting engine_params via **kwargs) ===\n" + docstring_kwargs + "\n"
    )
    parts.append(
        "=== SOURCE: BitsAndBytesConfig (engine_params quantisation fields) ===\n"
        + sources["BitsAndBytesConfig"][:6000]
        + "\n"
    )
    parts.append(
        "=== SOURCE: CompileConfig ($defs.CompileConfig) ===\n"
        + sources["CompileConfig"][:3000]
        + "\n"
    )
    parts.append(
        "=== SOURCE: WatermarkingConfig ($defs.WatermarkingConfig) ===\n"
        + sources["WatermarkingConfig"][:3000]
        + "\n"
    )
    parts.append(
        "=== SOURCE: GenerationConfig.__init__ -> sampling_params ===\n"
        + sources["GenerationConfig.__init__"]
        + "\n"
    )
    parts.append(
        "=== SOURCE: GenerationConfig FULL docstring "
        "(documents every sampling_params kwarg via Sphinx) ===\n"
        + sources["GenerationConfig.__doc__"][:14000]
        + "\n"
    )
    return "\n".join(parts)


def _merge_schema_payload(
    parsed: dict[str, Any],
    *,
    engine_params: dict[str, Any],
    sampling_params: dict[str, Any],
    defs: dict[str, dict[str, Any]],
    observations: list[str],
    chunk_label: str = "",
) -> None:
    """Merge a parsed chunk_fields payload into the running schema buckets.

    Same merge semantics as ``llm_b_oss.extract_schema`` - factored out so
    both single-call (H6) and multi-call (E6/E9) paths use it.
    """
    chunk_fields = parsed.get("chunk_fields") if isinstance(parsed, dict) else None
    if not isinstance(chunk_fields, dict):
        observations.append(
            f"schema parse {chunk_label!r}: chunk_fields not a dict (got "
            f"{type(chunk_fields).__name__}); skipping"
        )
        return

    for field_name, field_def in chunk_fields.items():
        if not isinstance(field_def, dict):
            continue
        if is_internal_plumbing_field(field_name):
            continue
        namespace = field_def.get("namespace")
        if not isinstance(namespace, str):
            continue
        field_def_clean = {k: v for k, v in field_def.items() if k != "namespace"}
        field_def_clean = {
            k: v for k, v in field_def_clean.items() if v is not None or k == "default"
        }
        if namespace == "engine_params":
            engine_params[field_name] = field_def_clean
        elif namespace == "sampling_params":
            sampling_params[field_name] = field_def_clean
        elif namespace.startswith("$defs."):
            class_name = namespace.removeprefix("$defs.")
            defs.setdefault(class_name, {"type": "object", "properties": {}})
            defs[class_name]["properties"][field_name] = field_def_clean
        else:
            observations.append(
                f"schema parse {chunk_label!r}: unrecognised namespace {namespace!r} "
                f"for field {field_name!r}"
            )


def run_h6_transformers(
    *,
    backend: OllamaBackend,
    out_dir: Path,
    spec: EngineSpec = TRANSFORMERS_SPEC,
    max_retries: int = 2,
    max_output_tokens: int = 8192,
) -> tuple[Path, Path, list[str], float, float]:
    """Single-call schema + single-call invariants on whole transformers source.

    Returns ``(schema_path, invariants_path, observations, schema_wall, invariants_wall)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    observations: list[str] = []

    # --- Schema call ---
    schema_source = _build_h6_schema_source(spec)
    print(f"[H6 transformers schema] source size = {len(schema_source):,} chars")
    observations.append(f"h6_schema_source_chars={len(schema_source)}")

    schema_prompt = SCHEMA_PROMPT_TEMPLATE.format(
        engine=spec.engine,
        engine_version=spec.engine_version,
        chunk_name="h6_whole_file",
        namespaces="engine_params, sampling_params, $defs.CompileConfig, $defs.WatermarkingConfig",
        source=schema_source,
    )

    # Use a synthetic ChunkSpec for transcript writing.
    schema_chunk_marker = TfChunkSpec(
        name="h6_whole_file_schema",
        description="H6 single-shot whole-source schema",
        source=schema_source[:200] + "...<truncated for transcript header>",
        expected_namespaces=["engine_params", "sampling_params"],
        expected_kind="schema",
    )
    schema_extraction = _retry_with_529_backoff(
        lambda: extract_with_retry(
            backend=backend,
            prompt=schema_prompt,
            chunk_name="h6_whole_file_schema",
            output_format="json",
            json_schema=SCHEMA_JSON_SCHEMA,
            max_retries=max_retries,
            max_output_tokens=max_output_tokens,
        )
    )
    schema_wall = schema_extraction.elapsed_sec
    _write_transcript(transcripts_dir, "schema", schema_chunk_marker, schema_extraction)

    engine_params: dict[str, Any] = {}
    sampling_params: dict[str, Any] = {}
    defs: dict[str, dict[str, Any]] = {}
    if schema_extraction.parsed is None:
        observations.append(f"h6 schema extraction FAILED; modes={schema_extraction.failure_modes}")
    else:
        _merge_schema_payload(
            schema_extraction.parsed,
            engine_params=engine_params,
            sampling_params=sampling_params,
            defs=defs,
            observations=observations,
            chunk_label="h6_whole_file_schema",
        )

    engine_params = filter_internal_plumbing(engine_params)
    sampling_params = filter_internal_plumbing(sampling_params)
    schema_envelope: dict[str, Any] = {
        "schema_version": "2.0.0",
        "engine": spec.engine,
        "engine_version": spec.engine_version,
        "discovered_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "discovery_method": "h6_whole_file_no_chunking",
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if defs:
        schema_envelope["$defs"] = defs

    schema_path = out_dir / "schema.json"
    _write_json(schema_path, schema_envelope)

    # --- Invariants call ---
    inv_source = _build_h6_invariants_source(spec)
    print(f"[H6 transformers invariants] source size = {len(inv_source):,} chars")
    observations.append(f"h6_invariants_source_chars={len(inv_source)}")

    inv_prompt = INVARIANTS_PROMPT_TEMPLATE.format(
        engine=spec.engine,
        engine_version=spec.engine_version,
        field_namespace=spec.namespace_for_invariants,
        source=inv_source,
    )

    inv_chunk_marker = TfChunkSpec(
        name="h6_whole_file_invariants",
        description="H6 single-shot whole-source invariants",
        source=inv_source[:200] + "...<truncated for transcript header>",
        expected_namespaces=[spec.namespace_for_invariants],
        expected_kind="invariants",
    )
    inv_extraction = _retry_with_529_backoff(
        lambda: extract_with_retry(
            backend=backend,
            prompt=inv_prompt,
            chunk_name="h6_whole_file_invariants",
            output_format="yaml",
            json_schema=None,
            max_retries=max_retries,
            max_output_tokens=max_output_tokens,
        )
    )
    inv_wall = inv_extraction.elapsed_sec
    _write_transcript(transcripts_dir, "invariants", inv_chunk_marker, inv_extraction)

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str, str]] = set()
    if inv_extraction.parsed is None:
        observations.append(
            f"h6 invariants extraction FAILED; modes={inv_extraction.failure_modes}"
        )
    else:
        inv_list = _pull_invariants_list(inv_extraction.parsed)
        if inv_list is None:
            observations.append("h6 invariants: parsed but no list found")
        else:
            for inv in inv_list:
                if isinstance(inv, dict):
                    _add_invariant_if_unique(
                        inv,
                        merged=merged,
                        seen_ids=seen_ids,
                        seen_identities=seen_identities,
                        engine=spec.engine,
                        added_by_default="h6_no_chunk",
                    )
            observations.append(f"h6 invariants: emitted {len(merged)} unique entries")

    inv_envelope: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": spec.engine,
        "engine_version": spec.engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": "h6_no_chunk",
        "invariants": merged,
    }
    inv_path = out_dir / "invariants.proposed.yaml"
    _write_yaml(inv_path, inv_envelope)

    return schema_path, inv_path, observations, schema_wall, inv_wall


# ---------------------------------------------------------------------------
# E6: field-anchored extension of existing per-class chunks
# ---------------------------------------------------------------------------


def _build_field_anchoring_preamble(
    fields_by_class: dict[str, set[str]], chunk_classes: list[str]
) -> str:
    """Build the field-anchoring preamble for a chunk.

    Lists the `Model.__fields__` for each class the chunk targets, then
    constrains the LLM: propose invariants only on fields in this list.
    """
    lines = [
        "=== E6 FIELD ANCHOR ===",
        "The engine's declared model fields are listed below. For EACH field,",
        "look in the source for a validation pattern (`if X.field <pred>: raise`",
        "or `minor_issues[...] = ...`) and emit an invariant if you find one.",
        "",
        "DOMAIN CONSTRAINT: ONLY propose invariants on fields that appear in",
        "this list. Do NOT propose invariants on fields you cannot find in the",
        "declared __fields__ below - those are out of scope (extra=allow",
        "fields are not in scope for this prompt).",
        "",
    ]
    if not chunk_classes:
        chunk_classes = list(fields_by_class.keys())
    for class_name in chunk_classes:
        fields = fields_by_class.get(class_name)
        if not fields:
            continue
        lines.append(f"--- {class_name}.__fields__ ---")
        # Show fields in deterministic order, comma-separated.
        sorted_fields = sorted(fields)
        # Wrap at 80-char-ish lines.
        joined = ", ".join(sorted_fields)
        lines.append(joined)
        lines.append("")
    return "\n".join(lines)


def _chunk_classes_for_e6(chunk_name: str, all_classes: list[str]) -> list[str]:
    """Heuristic: pick which Pydantic / dataclass models a chunk targets,
    based on the chunk name. Falls back to all classes if no match.
    """
    name_lower = chunk_name.lower()
    matches: list[str] = []
    for cls in all_classes:
        if cls.lower() in name_lower:
            matches.append(cls)
    # Special-case: transformers chunks all key on GenerationConfig / BitsAndBytesConfig.
    if not matches:
        for cls in all_classes:
            # short-form check for transformers
            if (cls == "GenerationConfig" and (
                "generation" in name_lower or "validate" in name_lower
            )) or (cls == "BitsAndBytesConfig" and "bitsandbytes" in name_lower) or (cls == "CompileConfig" and "compile" in name_lower):
                matches.append(cls)
    return matches or all_classes


def run_e6_for_engine(
    *,
    spec: EngineSpec,
    backend: OllamaBackend,
    out_dir: Path,
    max_retries: int = 2,
    max_output_tokens: int = 4096,
) -> tuple[Path, Path, list[str], float, float]:
    """Run E6 (field-anchored) on a single engine.

    Reuses the existing chunker for source decomposition; prepends a
    field-anchor preamble (declared __fields__) to each chunk prompt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    observations: list[str] = []

    # 1. AST-collect declared fields per class.
    fields_by_class = _collect_fields_via_ast(spec)
    total_fields = sum(len(v) for v in fields_by_class.values())
    observations.append(
        f"e6_field_anchor: {len(fields_by_class)} classes, {total_fields} declared fields"
    )
    print(
        f"[E6 {spec.engine}] collected {len(fields_by_class)} classes; "
        f"{total_fields} declared fields"
    )

    # 2. Build chunks via engine's chunker (use the SAME chunks (b) uses).
    if spec.engine == "transformers":
        tf_src_root = spec.venv_source_root if spec.venv_source_root.exists() else None
        schema_chunks = tf_schema_chunks(source_root=tf_src_root)
        invariant_chunks = tf_invariants_chunks(source_root=tf_src_root)
    elif spec.engine == "vllm":
        vllm_src_root = spec.venv_source_root if spec.venv_source_root.exists() else None
        schema_chunks = vllm_schema_chunks(source_root=vllm_src_root)
        invariant_chunks = vllm_invariants_chunks(source_root=vllm_src_root)
    else:
        raise ValueError(f"runner not configured for engine {spec.engine!r}")

    # 3. Schema extraction (unchanged from (b)).
    engine_params: dict[str, Any] = {}
    sampling_params: dict[str, Any] = {}
    defs: dict[str, dict[str, Any]] = {}
    schema_wall = 0.0
    print(f"[E6 {spec.engine}] running {len(schema_chunks)} SCHEMA chunks (no field-anchor)...")
    for i, chunk in enumerate(schema_chunks):
        if not chunk.source.strip():
            observations.append(f"schema chunk {chunk.name!r}: empty source; skipping")
            continue
        prompt = SCHEMA_PROMPT_TEMPLATE.format(
            engine=spec.engine,
            engine_version=spec.engine_version,
            chunk_name=chunk.name,
            namespaces=", ".join(chunk.expected_namespaces),
            source=chunk.source,
        )
        extraction = _retry_with_529_backoff(
            lambda p=prompt, n=chunk.name: extract_with_retry(
                backend=backend,
                prompt=p,
                chunk_name=n,
                output_format="json",
                json_schema=SCHEMA_JSON_SCHEMA,
                max_retries=max_retries,
                max_output_tokens=max_output_tokens,
            )
        )
        schema_wall += extraction.elapsed_sec
        _write_transcript(transcripts_dir, "schema", chunk, extraction)
        if extraction.parsed is None:
            observations.append(
                f"schema chunk {chunk.name!r}: failed; modes={extraction.failure_modes}"
            )
            continue
        _merge_schema_payload(
            extraction.parsed,
            engine_params=engine_params,
            sampling_params=sampling_params,
            defs=defs,
            observations=observations,
            chunk_label=chunk.name,
        )
        print(
            f"  [E6 {spec.engine} schema] chunk {i + 1}/{len(schema_chunks)} "
            f"{chunk.name} elapsed={extraction.elapsed_sec:.1f}s"
        )

    engine_params = filter_internal_plumbing(engine_params)
    sampling_params = filter_internal_plumbing(sampling_params)
    schema_envelope: dict[str, Any] = {
        "schema_version": "2.0.0",
        "engine": spec.engine,
        "engine_version": spec.engine_version,
        "discovered_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "discovery_method": "e6_field_anchored",
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if defs:
        schema_envelope["$defs"] = defs
    schema_path = out_dir / "schema.json"
    _write_json(schema_path, schema_envelope)

    # 4. Invariants extraction with field-anchor preamble.
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str, str]] = set()
    inv_wall = 0.0
    print(
        f"[E6 {spec.engine}] running {len(invariant_chunks)} INVARIANT chunks with field-anchor..."
    )
    for i, chunk in enumerate(invariant_chunks):
        if not chunk.source.strip():
            observations.append(f"invariants chunk {chunk.name!r}: empty source; skipping")
            continue

        # Determine which classes this chunk targets, restrict field-anchor.
        chunk_classes = _chunk_classes_for_e6(chunk.name, spec.config_classes)
        anchor_preamble = _build_field_anchoring_preamble(fields_by_class, chunk_classes)

        ns_list = chunk.expected_namespaces or [spec.namespace_for_invariants]
        field_namespace = ns_list[0]

        # Prefix the source with the field-anchor preamble.
        anchored_source = anchor_preamble + "\n\n" + chunk.source

        prompt = INVARIANTS_PROMPT_TEMPLATE.format(
            engine=spec.engine,
            engine_version=spec.engine_version,
            field_namespace=field_namespace,
            source=anchored_source,
        )

        extraction = _retry_with_529_backoff(
            lambda p=prompt, n=chunk.name: extract_with_retry(
                backend=backend,
                prompt=p,
                chunk_name=n,
                output_format="yaml",
                json_schema=None,
                max_retries=max_retries,
                max_output_tokens=max_output_tokens,
            )
        )
        inv_wall += extraction.elapsed_sec
        _write_transcript(transcripts_dir, "invariants", chunk, extraction)

        if extraction.parsed is None:
            observations.append(
                f"invariants chunk {chunk.name!r}: failed; modes={extraction.failure_modes}"
            )
            continue
        inv_list = _pull_invariants_list(extraction.parsed)
        if inv_list is None:
            observations.append(f"invariants chunk {chunk.name!r}: no list found")
            continue
        chunk_count = 0
        for inv in inv_list:
            if isinstance(inv, dict):
                if _add_invariant_if_unique(
                    inv,
                    merged=merged,
                    seen_ids=seen_ids,
                    seen_identities=seen_identities,
                    engine=spec.engine,
                    added_by_default="e6_field_anchored",
                ):
                    chunk_count += 1
        observations.append(
            f"invariants chunk {chunk.name!r}: emitted {chunk_count} unique; "
            f"anchor_classes={chunk_classes}"
        )
        print(
            f"  [E6 {spec.engine} invariants] chunk {i + 1}/{len(invariant_chunks)} "
            f"{chunk.name} emitted={chunk_count} elapsed={extraction.elapsed_sec:.1f}s"
        )

    inv_envelope: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": spec.engine,
        "engine_version": spec.engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": "e6_field_anchored",
        "invariants": merged,
    }
    inv_path = out_dir / "invariants.proposed.yaml"
    _write_yaml(inv_path, inv_envelope)
    return schema_path, inv_path, observations, schema_wall, inv_wall


# ---------------------------------------------------------------------------
# E9: sequential methodical sweep with cumulative context
# ---------------------------------------------------------------------------


def _running_notes_summary(
    invariants_so_far: list[dict[str, Any]], max_entries: int = 80, max_chars: int = 6000
) -> str:
    """Compact summary of invariants extracted in PRIOR chunks. Fed back
    to the LLM so it can:
    - Dedup inline (won't re-emit an invariant it just produced).
    - Surface cross-class invariants (recognise patterns across chunks).
    """
    if not invariants_so_far:
        return "(no invariants extracted yet; this is the first chunk)"
    lines: list[str] = [
        f"PRIOR EXTRACTIONS ({len(invariants_so_far)} total invariants from earlier chunks):"
    ]
    # Sample the most recent entries first (cumulative pressure).
    recent = invariants_so_far[-max_entries:]
    for inv in recent:
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", "<?>")
        sev = inv.get("severity", "?")
        match = inv.get("match", {}) or {}
        fields = match.get("fields", {}) or {}
        first_key = next(iter(fields.keys()), "?") if isinstance(fields, dict) else "?"
        lines.append(f"  - id={iid} sev={sev} field={first_key}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n  ... [truncated for context]"
    return text


E9_PROMPT_PREFIX = """\
=== E9 SEQUENTIAL EXTRACTION ===
You are doing a METHODICAL FILE-BY-FILE sweep of {engine} v{engine_version}.
This is chunk {chunk_index}/{total_chunks}. You have already extracted
invariants from prior chunks (running notes below). For THIS chunk:

1. EXTRACT new invariants from the source below.
2. DO NOT re-emit invariants already in the running notes (dedup by
   field name + predicate kind).
3. WATCH FOR CROSS-CLASS INVARIANTS: if this chunk's source references
   a field that also appears in an earlier chunk's invariants, note
   whether it adds a cross-class constraint (e.g.
   `max_num_batched_tokens >= max_num_seqs` couples two SchedulerConfig
   fields).
4. If THIS chunk adds NO new invariants beyond the prior list, emit
   `invariants: []` (do not fabricate).

=== RUNNING NOTES (prior extractions) ===
{running_notes}

=== END RUNNING NOTES ===

Now apply the standard invariant extraction prompt below to THIS chunk's
source.

"""


def run_e9_for_engine(
    *,
    spec: EngineSpec,
    backend: OllamaBackend,
    out_dir: Path,
    max_retries: int = 2,
    max_output_tokens: int = 4096,
) -> tuple[Path, Path, list[str], float, float]:
    """Run E9 (sequential methodical sweep with cumulative context) on a single engine.

    Each chunk's prompt is prefixed with a running-notes summary so the
    LLM can dedup inline + surface cross-class patterns. The chunks
    themselves are the same as (b) - the chunker doesn't change.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    observations: list[str] = []

    # 1. Build chunks.
    if spec.engine == "transformers":
        tf_src_root = spec.venv_source_root if spec.venv_source_root.exists() else None
        schema_chunks = tf_schema_chunks(source_root=tf_src_root)
        invariant_chunks = tf_invariants_chunks(source_root=tf_src_root)
    elif spec.engine == "vllm":
        vllm_src_root = spec.venv_source_root if spec.venv_source_root.exists() else None
        schema_chunks = vllm_schema_chunks(source_root=vllm_src_root)
        invariant_chunks = vllm_invariants_chunks(source_root=vllm_src_root)
    else:
        raise ValueError(f"E9 not configured for engine {spec.engine!r}")

    # 2. Schema extraction (unchanged from (b)).
    engine_params: dict[str, Any] = {}
    sampling_params: dict[str, Any] = {}
    defs: dict[str, dict[str, Any]] = {}
    schema_wall = 0.0
    print(f"[E9 {spec.engine}] running {len(schema_chunks)} SCHEMA chunks (no cumulative)...")
    for i, chunk in enumerate(schema_chunks):
        if not chunk.source.strip():
            observations.append(f"schema chunk {chunk.name!r}: empty source; skipping")
            continue
        prompt = SCHEMA_PROMPT_TEMPLATE.format(
            engine=spec.engine,
            engine_version=spec.engine_version,
            chunk_name=chunk.name,
            namespaces=", ".join(chunk.expected_namespaces),
            source=chunk.source,
        )
        extraction = _retry_with_529_backoff(
            lambda p=prompt, n=chunk.name: extract_with_retry(
                backend=backend,
                prompt=p,
                chunk_name=n,
                output_format="json",
                json_schema=SCHEMA_JSON_SCHEMA,
                max_retries=max_retries,
                max_output_tokens=max_output_tokens,
            )
        )
        schema_wall += extraction.elapsed_sec
        _write_transcript(transcripts_dir, "schema", chunk, extraction)
        if extraction.parsed is None:
            observations.append(
                f"schema chunk {chunk.name!r}: failed; modes={extraction.failure_modes}"
            )
            continue
        _merge_schema_payload(
            extraction.parsed,
            engine_params=engine_params,
            sampling_params=sampling_params,
            defs=defs,
            observations=observations,
            chunk_label=chunk.name,
        )
        print(
            f"  [E9 {spec.engine} schema] chunk {i + 1}/{len(schema_chunks)} "
            f"{chunk.name} elapsed={extraction.elapsed_sec:.1f}s"
        )

    engine_params = filter_internal_plumbing(engine_params)
    sampling_params = filter_internal_plumbing(sampling_params)
    schema_envelope: dict[str, Any] = {
        "schema_version": "2.0.0",
        "engine": spec.engine,
        "engine_version": spec.engine_version,
        "discovered_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "discovery_method": "e9_sequential_methodical_sweep",
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if defs:
        schema_envelope["$defs"] = defs
    schema_path = out_dir / "schema.json"
    _write_json(schema_path, schema_envelope)

    # 3. Invariants extraction with CUMULATIVE context.
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str, str]] = set()
    inv_wall = 0.0
    print(
        f"[E9 {spec.engine}] running {len(invariant_chunks)} INVARIANT chunks "
        f"with cumulative context..."
    )
    for i, chunk in enumerate(invariant_chunks):
        if not chunk.source.strip():
            observations.append(f"invariants chunk {chunk.name!r}: empty source; skipping")
            continue

        ns_list = chunk.expected_namespaces or [spec.namespace_for_invariants]
        field_namespace = ns_list[0]

        # Build cumulative-context preamble.
        running_notes = _running_notes_summary(merged, max_entries=80, max_chars=6000)
        preamble = E9_PROMPT_PREFIX.format(
            engine=spec.engine,
            engine_version=spec.engine_version,
            chunk_index=i + 1,
            total_chunks=len(invariant_chunks),
            running_notes=running_notes,
        )

        # Prepend the preamble to the source (so the LLM sees it BEFORE the
        # standard invariants prompt body). We do this by wrapping the
        # source argument; this keeps the locked prompt text intact.
        wrapped_source = preamble + "\n" + chunk.source

        prompt = INVARIANTS_PROMPT_TEMPLATE.format(
            engine=spec.engine,
            engine_version=spec.engine_version,
            field_namespace=field_namespace,
            source=wrapped_source,
        )

        extraction = _retry_with_529_backoff(
            lambda p=prompt, n=chunk.name: extract_with_retry(
                backend=backend,
                prompt=p,
                chunk_name=n,
                output_format="yaml",
                json_schema=None,
                max_retries=max_retries,
                max_output_tokens=max_output_tokens,
            )
        )
        inv_wall += extraction.elapsed_sec
        _write_transcript(transcripts_dir, "invariants", chunk, extraction)

        if extraction.parsed is None:
            observations.append(
                f"invariants chunk {chunk.name!r}: failed; modes={extraction.failure_modes}"
            )
            continue
        inv_list = _pull_invariants_list(extraction.parsed)
        if inv_list is None:
            observations.append(f"invariants chunk {chunk.name!r}: no list found")
            continue
        chunk_count = 0
        for inv in inv_list:
            if isinstance(inv, dict):
                if _add_invariant_if_unique(
                    inv,
                    merged=merged,
                    seen_ids=seen_ids,
                    seen_identities=seen_identities,
                    engine=spec.engine,
                    added_by_default="e9_sequential",
                ):
                    chunk_count += 1
        observations.append(
            f"invariants chunk {chunk.name!r}: emitted {chunk_count} unique "
            f"(cumulative total so far: {len(merged)})"
        )
        print(
            f"  [E9 {spec.engine} invariants] chunk {i + 1}/{len(invariant_chunks)} "
            f"{chunk.name} emitted={chunk_count} cumulative={len(merged)} "
            f"elapsed={extraction.elapsed_sec:.1f}s"
        )

    inv_envelope: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": spec.engine,
        "engine_version": spec.engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": "e9_sequential",
        "invariants": merged,
    }
    inv_path = out_dir / "invariants.proposed.yaml"
    _write_yaml(inv_path, inv_envelope)
    return schema_path, inv_path, observations, schema_wall, inv_wall


# ---------------------------------------------------------------------------
# Scoring + observations write
# ---------------------------------------------------------------------------


def score_and_write(
    *,
    pattern: str,
    spec: EngineSpec,
    out_dir: Path,
    schema_path: Path,
    inv_path: Path,
    observations: list[str],
    wall_clock_sec: float,
    energy_wh: float = 0.0,
) -> dict[str, Any]:
    """Score the cell + write score.json + observations.md.

    Returns a dict with the headline numbers for the summary writeup.
    """
    ref_schema = (
        _PROJECT_ROOT
        / "engine_versions"
        / spec.engine
        / spec.version_slug
        / "outputs"
        / "schema.discovered.json"
    )
    ref_inv = (
        _PROJECT_ROOT
        / "engine_versions"
        / spec.engine
        / spec.version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )

    score, schema_diffs, inv_diffs, type_or_sev_diffs = trial_scoring.score_cell(
        strategy=pattern,
        engine=spec.engine,
        version_slug=spec.version_slug,
        bump_distance="active",
        schema_output_path=schema_path,
        invariants_output_path=inv_path,
        reference_schema_path=ref_schema,
        reference_invariants_path=ref_inv,
        wall_clock_sec=wall_clock_sec,
        energy_wh=energy_wh,
        extra_observations=observations,
    )

    # Score JSON
    score_path = out_dir / "score.json"
    score_path.write_text(score.to_json())

    # Diff artefacts
    trial_scoring.write_diff_artefact(
        diffs=[d for d in schema_diffs if d.kind == "recall_miss"],
        out_path=out_dir / "schema_recall_misses.yaml",
        kind_label="recall_misses",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in schema_diffs if d.kind == "precision_spurious"],
        out_path=out_dir / "schema_precision_spurious.yaml",
        kind_label="precision_spurious",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in inv_diffs if d.kind == "recall_miss"],
        out_path=out_dir / "invariants_recall_misses.yaml",
        kind_label="recall_misses",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in inv_diffs if d.kind == "precision_spurious"],
        out_path=out_dir / "invariants_precision_spurious.yaml",
        kind_label="precision_spurious",
    )

    # Observations.md
    obs_lines = [
        f"# {pattern} {spec.engine} {spec.version_slug} observations",
        "",
        f"- pattern: {pattern}",
        f"- engine: {spec.engine}",
        f"- version_slug: {spec.version_slug}",
        f"- wall_clock_sec: {wall_clock_sec:.1f}",
        f"- energy_wh: {energy_wh:.2f}",
        f"- schema_recall: {score.schema_recall:.4f}",
        f"- schema_precision: {score.schema_precision:.4f}",
        f"- invariant_recall: {score.invariant_recall:.4f}",
        f"- invariant_precision: {score.invariant_precision:.4f}",
        f"- schema_ref_count: {score.schema_reference_count}",
        f"- schema_cell_count: {score.schema_cell_count}",
        f"- schema_intersection: {score.schema_intersection_count}",
        f"- invariant_ref_count: {score.invariant_reference_count}",
        f"- invariant_cell_count: {score.invariant_cell_count}",
        f"- invariant_intersection: {score.invariant_intersection_count}",
        f"- schema_failure_mode: {score.schema_failure_mode}",
        f"- invariant_failure_mode: {score.invariant_failure_mode}",
        "",
        "## Run observations",
        "",
    ]
    for o in observations:
        obs_lines.append(f"- {o}")
    (out_dir / "observations.md").write_text("\n".join(obs_lines))

    return {
        "pattern": pattern,
        "engine": spec.engine,
        "version_slug": spec.version_slug,
        "schema_recall": score.schema_recall,
        "schema_precision": score.schema_precision,
        "invariant_recall": score.invariant_recall,
        "invariant_precision": score.invariant_precision,
        "schema_ref_count": score.schema_reference_count,
        "schema_cell_count": score.schema_cell_count,
        "invariant_ref_count": score.invariant_reference_count,
        "invariant_cell_count": score.invariant_cell_count,
        "wall_clock_sec": wall_clock_sec,
        "schema_failure_mode": score.schema_failure_mode,
        "invariant_failure_mode": score.invariant_failure_mode,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pattern",
        choices=["h6", "e6", "e9"],
        required=True,
        help="Which pattern to run.",
    )
    parser.add_argument(
        "--engine",
        choices=["transformers", "vllm"],
        required=True,
        help="Which engine cell to run.",
    )
    parser.add_argument("--model", default="llama3.1:70b")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_PROJECT_ROOT / "_spike" / "findings" / "hybrid_experiments",
        help="Root output dir for the pattern outputs.",
    )
    args = parser.parse_args(argv)

    if args.pattern == "h6" and args.engine != "transformers":
        print("[error] H6 only supports transformers (context-limited).", file=sys.stderr)
        return 2

    spec = TRANSFORMERS_SPEC if args.engine == "transformers" else VLLM_SPEC

    pattern_dir_map = {
        "h6": "h6_no_chunk",
        "e6": "e6_field_anchored",
        "e9": "e9_sequential",
    }
    pattern_dir = pattern_dir_map[args.pattern]
    out_dir = args.out_root / pattern_dir / spec.engine / spec.version_slug

    backend = OllamaBackend(model=args.model, url=args.ollama_url, num_ctx=args.num_ctx)
    max_output_tokens = (
        args.max_output_tokens
        if args.max_output_tokens
        else (8192 if args.pattern == "h6" else 4096)
    )

    t0 = time.perf_counter()
    if args.pattern == "h6":
        schema_path, inv_path, observations, schema_wall, inv_wall = run_h6_transformers(
            backend=backend,
            out_dir=out_dir,
            spec=spec,
            max_retries=args.max_retries,
            max_output_tokens=max_output_tokens,
        )
    elif args.pattern == "e6":
        schema_path, inv_path, observations, schema_wall, inv_wall = run_e6_for_engine(
            spec=spec,
            backend=backend,
            out_dir=out_dir,
            max_retries=args.max_retries,
            max_output_tokens=max_output_tokens,
        )
    elif args.pattern == "e9":
        schema_path, inv_path, observations, schema_wall, inv_wall = run_e9_for_engine(
            spec=spec,
            backend=backend,
            out_dir=out_dir,
            max_retries=args.max_retries,
            max_output_tokens=max_output_tokens,
        )
    else:
        print(f"[error] unknown pattern {args.pattern!r}", file=sys.stderr)
        return 2
    wall = time.perf_counter() - t0

    observations.insert(
        0,
        f"pattern={args.pattern} engine={spec.engine} "
        f"schema_wall={schema_wall:.1f}s invariants_wall={inv_wall:.1f}s "
        f"total_wall={wall:.1f}s",
    )

    headline = score_and_write(
        pattern=args.pattern,
        spec=spec,
        out_dir=out_dir,
        schema_path=schema_path,
        inv_path=inv_path,
        observations=observations,
        wall_clock_sec=wall,
        energy_wh=0.0,
    )

    print(f"[done {args.pattern} {spec.engine}] {json.dumps(headline)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
