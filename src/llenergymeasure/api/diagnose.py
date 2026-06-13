"""Stage-1 LLM bump-diagnose: a gated, maintainer-invokable proposer.

Productionises the diagnose trial (research-grade harness, findings
``diagnose-trial-findings-2026-06-13``) into a clean, testable module. The
diagnose call is a RECALL-maximising proposer, never an adjudicator: a local
code-tuned model reads the bump's source diff and proposes structured rule
entries; the EXISTING runtime gate (``scripts.validate_rules``) is the precision
filter. Nothing ships ungated - that is the structural hallucination catch the
trial validated.

Two modes (engine-update-workflow design section 8):

- **1a carried-failure triage** (alarm-conditional): given the post-Rung-0
  RESIDUAL of the decay re-gate (``reconcile_regate_report``'s ``residual``) plus
  the diff-scoped source the residual rules cite, classify each residual
  (``decayed_announcement`` / ``dormancy_now_silent`` / ``rule_morphed`` /
  ``still_holds`` / ``unknown``) with a reason, a ``file:line`` citation, and
  constructible ``kwargs_positive`` / ``kwargs_negative`` probes.
- **1b gap-diagnose** (every bump): given the new pin's config-surface source
  diff plus the freshly mined envelope summary, flag what mining missed as
  structured cited entries.

Architecture (load-bearing):

- The model call sits behind :class:`DiagnoseModel` (a ``.complete(prompt) ->
  str`` protocol); the default :class:`OllamaDiagnoseModel` talks to the local
  ollama server (GPU-only path, never exercised in tests - unit tests inject a
  stub returning recorded fixtures).
- Proposals carrying kwargs go through the existing gate IN THE ENGINE
  CONTAINER (a docker subprocess driving ``scripts/diagnose_gate_in_container.py``
  which imports ``scripts.validate_rules`` only inside the image). The gate code
  is NOT reimplemented here.
- Only gate-CONFIRMED entries are written to ``rules.proposed.yaml`` with
  provenance ``llm_diagnose``. Un-confirmed entries and construction-
  unconfirmable claims (silent dormancy) are surfaced for review, never
  auto-written.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from llenergymeasure.config.engine_rules import VALID_SEVERITY

__all__ = [
    "CLASSIFICATIONS",
    "RESPONSE_SCHEMA",
    "DiagnoseError",
    "DiagnoseModel",
    "Diagnosis",
    "GateRunner",
    "OllamaDiagnoseModel",
    "build_carried_triage_prompt",
    "build_gap_diagnose_prompt",
    "diagnose_carried_failures",
    "diagnose_gaps",
    "diff_scoped_source",
    "parse_diagnoses",
    "render_proposed_yaml",
]


class DiagnoseError(Exception):
    """A diagnose run could not produce usable output (bad input, parse, or gate)."""


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

CLASSIFICATIONS = (
    "decayed_announcement",
    "dormancy_now_silent",
    "rule_morphed",
    "still_holds",
    "unknown",
)
"""The classifications the proposer may emit for a carried-failure residual.

Meaning (carried-failure triage):
- ``decayed_announcement`` - the constraint still holds but its warning/error
  MESSAGE was removed or reworded; carry forward with updated text.
- ``dormancy_now_silent`` - a dormant rule whose announcement no longer fires at
  construction; the equivalence holds silently; reclassify and carry.
- ``rule_morphed`` - the constraint is still enforced but its SHAPE changed (new
  co-condition, different exception type, moved check).
- ``still_holds`` - the rule fires exactly as before.
- ``unknown`` - the source does not let the model decide.
"""

# JSON schema for ollama structured output: one object with a `diagnoses` array.
# Each entry is a per-rule verdict carrying a constructible probe (the kwargs
# lever, gateable) and a citation. Mirrors the trial contract that yielded zero
# fabrications.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "classification": {"type": "string", "enum": list(CLASSIFICATIONS)},
                    "reason": {"type": "string"},
                    "citation": {"type": "string"},
                    "kwargs_positive": {"type": "object"},
                    "kwargs_negative": {"type": "object"},
                },
                "required": [
                    "rule_id",
                    "classification",
                    "reason",
                    "citation",
                    "kwargs_positive",
                    "kwargs_negative",
                ],
            },
        }
    },
    "required": ["diagnoses"],
}


@dataclass
class Diagnosis:
    """One parsed per-rule diagnosis with a malformed-kwargs flag.

    ``kwargs_malformed`` records, per probe, any keyword whose JSON value is a
    type-malformed stand-in for a native value the gate needs (the 12B failure
    mode: string ``"True"`` / ``"False"`` / ``"None"`` where native bool/null
    were required). A malformed diagnosis is still PARSED (never crashes the
    run) but is flagged so the caller can drop it before gating rather than feed
    the gate an un-constructible probe.
    """

    rule_id: str
    classification: str
    reason: str
    citation: str
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    kwargs_malformed: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_malformed(self) -> bool:
        return bool(self.kwargs_malformed)


# String stand-ins that signal a model failed to emit a native JSON scalar.
_STRINGLY_TYPED_SENTINELS = frozenset({"true", "false", "none", "null"})


def _malformed_keys(kwargs: dict[str, Any]) -> list[str]:
    """Keys whose value is a stringly-typed bool/null (gate-uncontructible)."""
    bad: list[str] = []
    for key, value in kwargs.items():
        if isinstance(value, str) and value.strip().lower() in _STRINGLY_TYPED_SENTINELS:
            bad.append(key)
    return sorted(bad)


def parse_diagnoses(raw: str) -> list[Diagnosis]:
    """Parse the model's structured-JSON response into :class:`Diagnosis` entries.

    Explicit failure: a non-parsing body, a non-object root, a missing
    ``diagnoses`` array, or zero usable entries raises :class:`DiagnoseError`
    (never a silent empty - that was a spike failure mode). Individual entries
    that parse but carry type-malformed kwargs are KEPT and flagged via
    ``kwargs_malformed`` so the gate-wiring stage can drop them without crashing.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DiagnoseError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise DiagnoseError(f"model output root is not an object: {type(obj).__name__}")
    diags = obj.get("diagnoses")
    if not isinstance(diags, list):
        raise DiagnoseError("model output has no 'diagnoses' array")

    out: list[Diagnosis] = []
    for entry in diags:
        if not isinstance(entry, dict):
            continue
        rule_id = entry.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            continue
        classification = str(entry.get("classification", "unknown"))
        if classification not in CLASSIFICATIONS:
            classification = "unknown"
        pos = entry.get("kwargs_positive") or {}
        neg = entry.get("kwargs_negative") or {}
        if not isinstance(pos, dict):
            pos = {}
        if not isinstance(neg, dict):
            neg = {}
        malformed: dict[str, list[str]] = {}
        if bad_pos := _malformed_keys(pos):
            malformed["kwargs_positive"] = bad_pos
        if bad_neg := _malformed_keys(neg):
            malformed["kwargs_negative"] = bad_neg
        out.append(
            Diagnosis(
                rule_id=rule_id,
                classification=classification,
                reason=str(entry.get("reason", "")),
                citation=str(entry.get("citation", "")),
                kwargs_positive=pos,
                kwargs_negative=neg,
                kwargs_malformed=malformed,
            )
        )
    if not out:
        raise DiagnoseError("model output 'diagnoses' array had no usable entries")
    return out


# ---------------------------------------------------------------------------
# Injectable model call
# ---------------------------------------------------------------------------


class DiagnoseModel(Protocol):
    """A single-shot completion model the diagnose prompt is sent to.

    The whole live path hides behind this one method so tests inject a stub
    returning a recorded fixture and never touch a GPU.
    """

    def complete(self, prompt: str) -> str:
        """Return the model's raw response text for ``prompt``."""
        ...


# Default model the trial selected: matched 70B quality at 3.8x lower wall and
# half the VRAM; the 12B/7B controls confirm we cannot go cheaper.
DEFAULT_MODEL = "qwen2.5-coder:32b"
_OLLAMA_GENERATE = "http://localhost:11434/api/generate"


@dataclass
class OllamaDiagnoseModel:
    """Default :class:`DiagnoseModel`: structured-JSON generate via local ollama.

    LIVE-ONLY: this path needs a GPU and the local ollama server; it is never
    exercised in the unit-test path (tests inject a stub). GPU etiquette is the
    caller's concern (devices 1-3, idle check) - this object just issues one
    deterministic structured-output call.
    """

    model: str = DEFAULT_MODEL
    endpoint: str = _OLLAMA_GENERATE
    num_ctx: int = 16384
    num_predict: int = 4096
    timeout_s: int = 1800

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode())
        return str(body.get("response", ""))


# ---------------------------------------------------------------------------
# Diff-scoping
# ---------------------------------------------------------------------------


def diff_scoped_source(
    *,
    source_root: Path,
    files: Sequence[str],
) -> str:
    """Assemble the file-path-scoped source the prompt embeds.

    Per design section 8, the prompt scope is the source DIFF region by FILE
    PATH (not whole tree, not line-pinned): the files the carried residual rules
    cite (1a) or the walkers' config-surface glob (1b). Each file is rendered
    with 1-based line numbers so the model can emit real ``file:line`` citations.

    Raises :class:`DiagnoseError` if no requested file resolves to non-empty
    content - the trial's empty-input guard, hoisted before the model call so a
    silent empty prompt can never reach the proposer.
    """
    parts: list[str] = []
    for rel in files:
        path = source_root / rel
        if not path.is_file():
            continue
        body = path.read_text()
        if not body.strip():
            continue
        numbered = "\n".join(f"{i + 1:5d}: {line}" for i, line in enumerate(body.splitlines()))
        parts.append(f"### FILE: {rel}\n{numbered}")
    if not parts:
        raise DiagnoseError(
            f"diff-scoped source is empty (no non-empty file among {list(files)} "
            f"under {source_root})"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CARRIED_TRIAGE_PROMPT = """You are auditing an inference-engine config library across an UPSTREAM VERSION BUMP: {engine} {old_version} -> {new_version}.

We previously mined a catalogue of config RULES for {old_version}. After bumping to {new_version}, the runtime gate re-ran each rule's construction probe against the NEW library and, after a deterministic reconciliation pass, these {n} rules remain UNRESOLVED. For each, decide what happened by reading the NEW {new_version} source below.

UNRESOLVED CARRIED RULES (what we knew at {old_version}; each has a positive probe that USED to trigger the rule and a negative probe that did not):
{carried}

NEW {new_version} SOURCE (the files these rules cite - read it carefully; cite real line numbers from it):
{source}

For EACH carried rule above, emit one diagnosis object. classification meanings:
  - decayed_announcement: the rule's constraint still holds, but its warning/error MESSAGE was removed or reworded (so the old message_template no longer matches). Carry forward with updated text.
  - dormancy_now_silent: the rule describes a field SILENTLY ignored under a condition (a dormancy rule), with no warning emitted at construction in the new source. The equivalence still holds silently; carry it forward reclassified as silent dormancy.
  - rule_morphed: the rule still enforces its constraint but its SHAPE changed (a new co-condition, a different exception type, a moved check).
  - still_holds: the rule fires exactly as before at construction.
  - unknown: the new source does not let you decide.

For kwargs_positive / kwargs_negative: emit constructor keyword arguments for the rule's native_type such that, when the object is constructed in the NEW library, the positive kwargs make the rule's condition TRUE (the error/dormancy fires) and the negative kwargs do not. You may reuse or correct the carried probes. These will be CONSTRUCTED and checked by a deterministic gate, so they must be real, buildable kwargs with NATIVE JSON types (true/false/null, not the strings "True"/"False"/"None").

In citation, give the file and line number(s) from the NEW source above that justify your classification. Do not invent line numbers; cite lines you actually see.

Emit a JSON object with a diagnoses array containing exactly {n} objects, one per carried rule_id."""


_GAP_DIAGNOSE_PROMPT = """You are auditing an inference-engine config library after an UPSTREAM VERSION BUMP: {engine} {new_version}.

Deterministic walkers just mined the config surface of {new_version}. Below is a SUMMARY of what they captured, followed by the NEW config-surface SOURCE. Your job is the BLINDNESS check: flag config RULES the source enforces that the mining MISSED (e.g. a constraint that moved into a docstring Args block, a validator the walkers cannot read, a type that the deterministic lift could not recover).

WHAT MINING CAPTURED (the fresh envelope summary - do NOT re-flag anything already here):
{envelope_summary}

NEW {new_version} CONFIG-SURFACE SOURCE (cite real line numbers from it):
{source}

For EACH gap you find, emit one diagnosis object describing a rule mining missed:
  - rule_id: a short stable snake_case id for the missed rule (engine-prefixed).
  - classification: use rule_morphed for a constraint whose shape the walker could not read, or unknown if you are unsure it is real (the gate will adjudicate).
  - reason: WHY mining missed it (e.g. "constraint stated only in the docstring Args, not a Field()").
  - citation: file and line number(s) from the NEW source above.
  - kwargs_positive / kwargs_negative: constructor keyword arguments for the rule's native_type such that the positive makes the constraint fire and the negative does not. Use NATIVE JSON types (true/false/null, not strings). These will be CONSTRUCTED and checked by a deterministic gate.

Only flag gaps you can ground in the source above. If mining looks complete, emit an empty diagnoses array. Emit a JSON object with a diagnoses array."""


def _compact_carried(carried: Sequence[dict[str, Any]]) -> str:
    """Render the carried rule entries (the OLD knowledge) for the prompt."""
    lines: list[str] = []
    for inv in carried:
        lines.append(
            "\n".join(
                [
                    f"- rule_id: {inv.get('id', '')}",
                    f"  native_type: {inv.get('native_type', '')}",
                    f"  severity: {inv.get('severity', '')}",
                    f"  invariant_under_test: {inv.get('invariant_under_test', '')}",
                    f"  old_message_template: {inv.get('message_template', '')}",
                    f"  kwargs_positive: {inv.get('kwargs_positive', {})}",
                    f"  kwargs_negative: {inv.get('kwargs_negative', {})}",
                ]
            )
        )
    return "\n".join(lines)


def build_carried_triage_prompt(
    *,
    engine: str,
    old_version: str,
    new_version: str,
    carried: Sequence[dict[str, Any]],
    source: str,
) -> str:
    """Build the mode-1a prompt from the residual carried entries + scoped source."""
    if not carried:
        raise DiagnoseError("carried-failure triage requires at least one residual rule")
    if not source.strip():
        raise DiagnoseError("carried-failure triage requires non-empty diff-scoped source")
    return _CARRIED_TRIAGE_PROMPT.format(
        engine=engine,
        old_version=old_version,
        new_version=new_version,
        n=len(carried),
        carried=_compact_carried(carried),
        source=source,
    )


def _summarise_envelope(schema: dict[str, Any]) -> str:
    """Compact, prompt-sized summary of a mined schema.discovered.json."""
    lines: list[str] = [
        f"engine: {schema.get('engine', '')}",
        f"engine_version: {schema.get('engine_version', '')}",
    ]
    for section in ("engine_params", "sampling_params"):
        params = schema.get(section)
        if isinstance(params, dict):
            lines.append(f"{section} ({len(params)} fields): {', '.join(sorted(params))}")
    return "\n".join(lines)


def build_gap_diagnose_prompt(
    *,
    engine: str,
    new_version: str,
    schema: dict[str, Any],
    source: str,
) -> str:
    """Build the mode-1b prompt from the fresh mined envelope + config-surface source."""
    if not source.strip():
        raise DiagnoseError("gap-diagnose requires non-empty diff-scoped source")
    return _GAP_DIAGNOSE_PROMPT.format(
        engine=engine,
        new_version=new_version,
        envelope_summary=_summarise_envelope(schema),
        source=source,
    )


# ---------------------------------------------------------------------------
# Gate wiring (output -> gate -> corpus)
# ---------------------------------------------------------------------------

# A callable that runs a list of proposal dicts through the gate IN the engine
# container and returns the verdict list. Injectable so tests stub the gate
# without docker; the default :class:`ContainerGateRunner` drives the real
# in-container gate.
GateRunner = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]

_GATE_SCRIPT = "scripts/diagnose_gate_in_container.py"


@dataclass
class ContainerGateRunner:
    """Default :class:`GateRunner`: drives the gate inside the engine image.

    Reuses the production construct+observe path VERBATIM by running
    ``scripts/diagnose_gate_in_container.py`` (which imports
    ``scripts.validate_rules``) inside the engine cache image - the same code
    that adjudicates the live decay alarm. The host CLI never imports the engine
    or the gate; it shells out to the container, exactly as the rules cell does.
    """

    image: str
    repo: Path
    workdir: Path
    docker_flags: Sequence[str] = ()

    def __call__(self, engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Proposals and verdicts both live in the workdir, bind-mounted at
        # /gateout (rw); the repo is mounted read-only-ish at /repo so the
        # in-container gate can import scripts.validate_rules. This keeps the
        # workdir free to sit anywhere on the host, not just under the repo.
        self.workdir.mkdir(parents=True, exist_ok=True)
        in_path = self.workdir / "diagnose_proposals.json"
        out_path = self.workdir / "diagnose_gate_verdicts.json"
        in_path.write_text(json.dumps(proposals, indent=2))
        cmd = [
            "docker",
            "run",
            "--rm",
            *self.docker_flags,
            "-v",
            f"{self.repo}:/repo",
            "-v",
            f"{self.workdir}:/gateout:rw",
            "-w",
            "/repo",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint",
            "python3",
            self.image,
            _GATE_SCRIPT,
            engine,
            "/gateout/diagnose_proposals.json",
            "/gateout/diagnose_gate_verdicts.json",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return list(json.loads(out_path.read_text()))


def _proposal_from_diagnosis(diag: Diagnosis, carried: dict[str, Any] | None) -> dict[str, Any]:
    """Assemble a gate proposal dict from a diagnosis + its carried entry (if any).

    The carried entry supplies the structural shape the model does not emit
    (``native_type``, ``severity``, ``match``, ``expected_outcome``); the
    diagnosis supplies the (possibly corrected) kwargs probes. For a 1b gap with
    no carried counterpart, the model's classification routes severity.
    """
    carried = carried or {}
    severity = str(carried.get("severity", "")).lower()
    if severity not in VALID_SEVERITY:
        severity = "dormant" if diag.classification == "dormancy_now_silent" else "error"
    return {
        "rule_id": diag.rule_id,
        "classification": diag.classification,
        "native_type": carried.get("native_type") or "",
        "severity": severity,
        "match": carried.get("match") or {"fields": {}},
        "expected_outcome": carried.get("expected_outcome"),
        "kwargs_positive": dict(diag.kwargs_positive),
        "kwargs_negative": dict(diag.kwargs_negative),
        "invariant_under_test": carried.get("invariant_under_test") or diag.reason,
        "message_template": carried.get("message_template"),
        "references": [f"llm_diagnose: {diag.reason}", f"citation: {diag.citation}"],
    }


def _validated_entry(proposal: dict[str, Any], engine: str, new_version: str) -> dict[str, Any]:
    """Shape a gate-confirmed proposal into a committed rules.proposed.yaml entry."""
    entry: dict[str, Any] = {
        "id": proposal["rule_id"],
        "engine": engine,
        "library": engine,
        "invariant_under_test": proposal.get("invariant_under_test") or "",
        "severity": proposal["severity"],
        "native_type": proposal["native_type"],
        "match": proposal.get("match") or {"fields": {}},
        "kwargs_positive": proposal.get("kwargs_positive") or {},
        "kwargs_negative": proposal.get("kwargs_negative") or {},
        "expected_outcome": proposal.get("expected_outcome")
        or {"outcome": "error" if proposal["severity"] == "error" else "dormant_announced"},
        "references": proposal.get("references") or [],
        "added_by": "llm_diagnose",
    }
    if proposal.get("message_template"):
        entry["message_template"] = proposal["message_template"]
    return entry


def render_proposed_yaml(
    *,
    engine: str,
    engine_version: str,
    entries: list[dict[str, Any]],
) -> str:
    """Render gate-confirmed entries as a rules.proposed.yaml document.

    A standalone fragment the maintainer reviews and merges into the engine's
    committed corpus. Always ``added_by: llm_diagnose`` (set per entry).
    """
    doc = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": engine_version,
        "invariants": entries,
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Result + top-level entry points
# ---------------------------------------------------------------------------


@dataclass
class DiagnoseResult:
    """Outcome of one diagnose run: what was proposed, gated, and emitted.

    - ``confirmed`` - gate-confirmed entries (shape: rules.proposed.yaml entries,
      ``added_by: llm_diagnose``); the ONLY entries written.
    - ``unconfirmed`` - proposals the gate did not confirm (reported, not written).
    - ``not_construction_confirmable`` - silent-dormancy claims that cannot
      positive-confirm at construction grain; cited proposals for review.
    - ``dropped_malformed`` - diagnoses dropped before gating for type-malformed
      kwargs (the 12B failure mode).
    """

    engine: str
    engine_version: str
    mode: str
    confirmed: list[dict[str, Any]] = field(default_factory=list)
    unconfirmed: list[dict[str, Any]] = field(default_factory=list)
    not_construction_confirmable: list[dict[str, Any]] = field(default_factory=list)
    dropped_malformed: list[dict[str, Any]] = field(default_factory=list)
    proposed_yaml: str | None = None


def _gate_and_split(
    *,
    engine: str,
    engine_version: str,
    mode: str,
    diagnoses: list[Diagnosis],
    carried_by_id: dict[str, dict[str, Any]],
    gate_runner: GateRunner,
) -> DiagnoseResult:
    """Gate the well-formed diagnoses and split into confirmed / not / unconfirmed."""
    result = DiagnoseResult(engine=engine, engine_version=engine_version, mode=mode)

    gateable: list[Diagnosis] = []
    for diag in diagnoses:
        if diag.is_malformed:
            result.dropped_malformed.append(
                {
                    "rule_id": diag.rule_id,
                    "classification": diag.classification,
                    "malformed": diag.kwargs_malformed,
                    "reason": "type-malformed kwargs (not gate-constructible)",
                }
            )
            continue
        gateable.append(diag)

    if not gateable:
        return result

    proposals = [
        _proposal_from_diagnosis(diag, carried_by_id.get(diag.rule_id)) for diag in gateable
    ]
    proposal_by_id = {p["rule_id"]: p for p in proposals}
    verdicts = gate_runner(engine, proposals)

    for verdict in verdicts:
        rule_id = verdict.get("rule_id", "")
        proposal = proposal_by_id.get(rule_id, {})
        outcome = verdict.get("verdict")
        record = {
            "rule_id": rule_id,
            "classification": proposal.get("classification"),
            "verdict": outcome,
            "gate": verdict,
        }
        if outcome == "confirmed":
            result.confirmed.append(_validated_entry(proposal, engine, engine_version))
        elif outcome == "not_construction_confirmable":
            result.not_construction_confirmable.append(record)
        else:
            result.unconfirmed.append(record)

    if result.confirmed:
        result.proposed_yaml = render_proposed_yaml(
            engine=engine, engine_version=engine_version, entries=result.confirmed
        )
    return result


def diagnose_carried_failures(
    *,
    engine: str,
    old_version: str,
    new_version: str,
    residual_ids: Sequence[str],
    carried_corpus: dict[str, Any],
    source_root: Path,
    cited_files: Sequence[str],
    model: DiagnoseModel,
    gate_runner: GateRunner,
) -> DiagnoseResult:
    """Mode 1a: triage the decay alarm's post-Rung-0 residual failures.

    ``carried_corpus`` is the previous pin's proposed-shape corpus (the
    executable carried catalogue with kwargs); ``residual_ids`` are the ids the
    reconciliation left unresolved (``reconcile_regate_report``'s ``residual``).
    Returns a :class:`DiagnoseResult`; only ``confirmed`` is emitted as YAML.
    """
    by_id = {
        str(inv.get("id", "")): inv
        for inv in carried_corpus.get("invariants", [])
        if isinstance(inv, dict)
    }
    residual = [by_id[rid] for rid in residual_ids if rid in by_id]
    missing = [rid for rid in residual_ids if rid not in by_id]
    if missing:
        raise DiagnoseError(f"residual ids absent from carried corpus: {missing}")
    if not residual:
        raise DiagnoseError("no residual rules to diagnose (empty residual)")

    source = diff_scoped_source(source_root=source_root, files=cited_files)
    prompt = build_carried_triage_prompt(
        engine=engine,
        old_version=old_version,
        new_version=new_version,
        carried=residual,
        source=source,
    )
    diagnoses = parse_diagnoses(model.complete(prompt))
    # Triage only the rules we asked about - ignore any fabricated extra ids.
    residual_id_set = set(residual_ids)
    diagnoses = [d for d in diagnoses if d.rule_id in residual_id_set]
    if not diagnoses:
        raise DiagnoseError("model returned no diagnoses for the residual rules")

    return _gate_and_split(
        engine=engine,
        engine_version=new_version,
        mode="carried_failure_triage",
        diagnoses=diagnoses,
        carried_by_id=by_id,
        gate_runner=gate_runner,
    )


def diagnose_gaps(
    *,
    engine: str,
    new_version: str,
    schema: dict[str, Any],
    source_root: Path,
    config_surface_files: Sequence[str],
    model: DiagnoseModel,
    gate_runner: GateRunner,
) -> DiagnoseResult:
    """Mode 1b: flag what deterministic mining missed on the new pin.

    ``schema`` is the fresh ``schema.discovered.json`` (summarised in-prompt);
    ``config_surface_files`` is the walkers' config-surface glob, resolved to
    paths under ``source_root``. The model proposes missed rules; the gate
    confirms which actually reconstruct. Only ``confirmed`` is emitted.
    """
    source = diff_scoped_source(source_root=source_root, files=config_surface_files)
    prompt = build_gap_diagnose_prompt(
        engine=engine, new_version=new_version, schema=schema, source=source
    )
    diagnoses = parse_diagnoses(model.complete(prompt))
    return _gate_and_split(
        engine=engine,
        engine_version=new_version,
        mode="gap_diagnose",
        diagnoses=diagnoses,
        carried_by_id={},
        gate_runner=gate_runner,
    )
