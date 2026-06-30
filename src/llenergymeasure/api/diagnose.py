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

import ast
import json
import re
import subprocess
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from llenergymeasure.api._source_window import (
    DEFAULT_BUDGET_CHARS,
    SymbolRequest,
    WindowReport,
    _assign_targets,
    _iter_config_classes,
    windowed_source,
)
from llenergymeasure.config.engine_rules import VALID_SEVERITY

__all__ = [
    "CLASSIFICATIONS",
    "DEFAULT_BUDGET_CHARS",
    "RESPONSE_SCHEMA",
    "TIER_D_CLASSIFICATIONS",
    "TIER_D_RESPONSE_SCHEMA",
    "DiagnoseError",
    "DiagnoseModel",
    "Diagnosis",
    "GateRunner",
    "OllamaDiagnoseModel",
    "build_carried_triage_prompt",
    "build_gap_diagnose_prompt",
    "build_tier_d_prompt",
    "carried_symbol_requests",
    "config_surface_symbol_requests",
    "diagnose_carried_failures",
    "diagnose_gaps",
    "diagnose_tier_d",
    "parse_diagnoses",
    "render_proposed_yaml",
    "tier_d_native_types",
    "tier_d_rule_id",
    "tier_d_symbol_requests",
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


TIER_D_CLASSIFICATIONS = (
    "bounded",
    "unconstrained",
    "unknown",
)
"""The classifications the Tier-D (pure-inference) proposer may emit per field.

- ``bounded`` - the field carries a real numeric/semantic constraint stated in
  the source or docstring; the proposer supplies ``fires_when`` (the illegal
  region), an illegal ``kwargs_positive``, and a legal ``kwargs_negative``. Only
  ``bounded`` diagnoses are gated.
- ``unconstrained`` - no real bound (a free counter / port / rank, or a sentinel
  like ``-1``/``0`` the library explicitly accepts as unlimited/disabled). These
  are NOT gated and never emitted - the deterministic noise filter could not
  reject them, but the model can.
- ``unknown`` - the source does not let the model decide; not gated.
"""

# JSON schema for the Tier-D structured-output call: the generic diagnosis shape
# plus the per-field bound (``fires_when`` predicate + human-readable
# ``constraint``). Mirrors the carried/gap RESPONSE_SCHEMA so the same ollama
# structured-output path drives all three modes.
TIER_D_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": list(TIER_D_CLASSIFICATIONS),
                    },
                    "reason": {"type": "string"},
                    "citation": {"type": "string"},
                    "constraint": {"type": "string"},
                    "fires_when": {"type": "object"},
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

    ``fires_when`` / ``constraint`` are Tier-D-only (empty for modes 1a/1b):
    ``fires_when`` is the one-key predicate over the candidate field that
    describes the ILLEGAL region (so the advisory warns only out-of-bound, not on
    every set), and ``constraint`` is the human-readable bound for the message.
    A ``fires_when`` carrying an unsupported operator is recorded in
    ``kwargs_malformed`` under ``"fires_when"`` so the diagnosis is dropped before
    gating rather than emitting an un-evaluable match predicate.
    """

    rule_id: str
    classification: str
    reason: str
    citation: str
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    kwargs_malformed: dict[str, list[str]] = field(default_factory=dict)
    fires_when: dict[str, Any] | None = None
    constraint: str = ""

    @property
    def is_malformed(self) -> bool:
        return bool(self.kwargs_malformed)


# String stand-ins that signal a model failed to emit a native JSON scalar.
_STRINGLY_TYPED_SENTINELS = frozenset({"true", "false", "none", "null"})

# Match operators a Tier-D ``fires_when`` predicate may use: single-field
# numeric bounds, membership, and presence. These are the subset of the loader's
# operator vocabulary (``loader._OPERATOR_HANDLERS``) that a per-field advisory
# can express without a cross-field ``@ref``. A ``fires_when`` outside this set
# is recorded malformed and the diagnosis dropped before gating.
_TIER_D_OPERATORS = frozenset(
    {"<", "<=", ">", ">=", "==", "!=", "in", "not_in", "present", "absent"}
)


def _malformed_keys(kwargs: dict[str, Any]) -> list[str]:
    """Keys whose value is a stringly-typed bool/null (gate-uncontructible)."""
    bad: list[str] = []
    for key, value in kwargs.items():
        if isinstance(value, str) and value.strip().lower() in _STRINGLY_TYPED_SENTINELS:
            bad.append(key)
    return sorted(bad)


def _fires_when_malformed(fires_when: Any) -> bool:
    """True if a Tier-D ``fires_when`` is not a one-key supported-operator dict."""
    if not isinstance(fires_when, dict) or len(fires_when) != 1:
        return True
    (op,) = fires_when.keys()
    return op not in _TIER_D_OPERATORS


def parse_diagnoses(
    raw: str, *, classifications: Sequence[str] = CLASSIFICATIONS
) -> list[Diagnosis]:
    """Parse the model's structured-JSON response into :class:`Diagnosis` entries.

    Explicit failure: a non-parsing body, a non-object root, a missing
    ``diagnoses`` array, or zero usable entries raises :class:`DiagnoseError`
    (never a silent empty - that was a spike failure mode). Individual entries
    that parse but carry type-malformed kwargs are KEPT and flagged via
    ``kwargs_malformed`` so the gate-wiring stage can drop them without crashing.

    ``classifications`` is the allowed classification vocabulary (mode 1a/1b use
    :data:`CLASSIFICATIONS`; Tier-D passes :data:`TIER_D_CLASSIFICATIONS`); an
    entry outside it is coerced to ``unknown``. The optional ``fires_when`` /
    ``constraint`` keys (Tier-D) are read when present; a malformed ``fires_when``
    is flagged in ``kwargs_malformed`` so it never reaches the match predicate.
    """
    allowed = set(classifications)
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
        if classification not in allowed:
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
        fires_when = entry.get("fires_when")
        if fires_when is not None and _fires_when_malformed(fires_when):
            malformed["fires_when"] = [str(fires_when)]
            fires_when = None
        out.append(
            Diagnosis(
                rule_id=rule_id,
                classification=classification,
                reason=str(entry.get("reason", "")),
                citation=str(entry.get("citation", "")),
                kwargs_positive=pos,
                kwargs_negative=neg,
                kwargs_malformed=malformed,
                fires_when=fires_when if isinstance(fires_when, dict) else None,
                constraint=str(entry.get("constraint", "")),
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
    response_schema: dict[str, Any] = field(default_factory=lambda: RESPONSE_SCHEMA)

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": self.response_schema,
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
# Source windowing (per-symbol, within the diff-scoped files)
# ---------------------------------------------------------------------------
#
# The diff-scoping FILE selector (the files a residual rule cites, or the
# config-surface glob) is the caller's; this layer narrows WITHIN those files to
# the cited class + field/validator spans. Feeding the files whole overflows the
# 16K model context and drives invented rule_ids (R7 trial); per-symbol windows
# fixed that. See :mod:`llenergymeasure.api._source_window`.


def _class_name(native_type: str) -> str:
    """The bare class name (last dotted component of a rule's native_type)."""
    return native_type.rsplit(".", 1)[-1] if native_type else ""


def _cited_fields(rule: dict[str, Any]) -> tuple[str, ...]:
    """Field names a carried rule cites: its kwargs probe keys + match-field tails.

    The kwargs_positive / kwargs_negative keys are the real constructor field
    names (``cache_implementation``, ``load_in_4bit``); the ``match.fields`` keys
    are dotted paths whose tail is the same field name. De-duped, order-preserving.
    """
    names: list[str] = []
    seen: set[str] = set()
    for probe in ("kwargs_positive", "kwargs_negative"):
        for key in rule.get(probe) or {}:
            if key not in seen:
                seen.add(key)
                names.append(key)
    match = rule.get("match") or {}
    for dotted in match.get("fields") or {}:
        tail = str(dotted).rsplit(".", 1)[-1]
        if tail and tail not in seen:
            seen.add(tail)
            names.append(tail)
    return tuple(names)


def carried_symbol_requests(
    *,
    carried: Sequence[dict[str, Any]],
    cited_files: Sequence[str],
) -> list[SymbolRequest]:
    """Per-residual windowing requests for mode 1a.

    Each residual rule contributes one request per cited file: its class
    (``native_type`` tail) plus the cited field names. The file selector stays
    the caller's diff-scoping; windowing narrows within those files to the class
    + field/validator spans the proposer needs.
    """
    requests: list[SymbolRequest] = []
    for rule in carried:
        class_name = _class_name(str(rule.get("native_type", "")))
        if not class_name:
            continue
        fields = _cited_fields(rule)
        for rel in cited_files:
            requests.append(SymbolRequest(file=rel, class_name=class_name, fields=fields))
    return requests


def config_surface_symbol_requests(
    *,
    source_root: Path,
    config_surface_files: Sequence[str],
) -> list[SymbolRequest]:
    """Per-class windowing requests for mode 1b (gap-diagnose).

    With no carried rules to cite specific fields, window each config-like class
    on the config surface whole (the AST walker enumerates them). This keeps the
    surface narrowed to config classes - not every helper / import in a large
    config module - while staying field-agnostic.
    """
    requests: list[SymbolRequest] = []
    for rel in config_surface_files:
        path = source_root / rel
        if not path.is_file():
            continue
        body = path.read_text()
        if not body.strip():
            continue
        try:
            module = ast.parse(body)
        except SyntaxError:
            continue
        for cls in _iter_config_classes(module):
            requests.append(SymbolRequest(file=rel, class_name=cls.name))
    return requests


def _is_validator_member(name: str) -> bool:
    """True for method names that can carry a CONSTRUCTION raise on a field.

    Restricts Tier-D's cited-field windows to validator-like members so the char
    budget is spent on the bodies that constrain (``_verify_args`` /
    ``__post_init__`` / ``validate`` / ``@field_validator`` helpers) and not on
    ``__repr__`` / ``compute_hash`` / accessors that merely reference the field.
    """
    if name in {"__post_init__", "post_init", "__init__"}:
        return True
    return any(tok in name for tok in ("verify", "validat", "check"))


def tier_d_symbol_requests(
    *,
    source_root: Path,
    config_surface_files: Sequence[str],
    candidate_leaves: Sequence[str],
) -> list[SymbolRequest]:
    """Per-class CITED-FIELD windowing requests for Tier-D.

    The mode-1b surface window (a 40-line leading slice of each class) shows the
    docstring + field declarations but NOT the ``__post_init__`` / ``_verify_args``
    / ``validate`` / ``@field_validator`` body where a CONSTRUCTION bound's raise
    actually lives (typically 100-400 lines below the header). Tier-D needs that
    body, so it windows each config class with the candidate leaves it DECLARES as
    cited fields - the cited-field path (:func:`_windows_for_class`) then admits
    each field's annotation span AND any same-class method body that references it
    (the validator). A class that declares no candidate leaf is skipped.
    """
    leaves = set(candidate_leaves)
    requests: list[SymbolRequest] = []
    for rel in config_surface_files:
        path = source_root / rel
        if not path.is_file():
            continue
        body = path.read_text()
        if not body.strip():
            continue
        try:
            module = ast.parse(body)
        except SyntaxError:
            continue
        for cls in _iter_config_classes(module):
            declared: set[str] = set()
            for stmt in cls.body:
                declared |= _assign_targets(stmt)
            cls_leaves = tuple(sorted(leaves & declared))
            if cls_leaves:
                requests.append(SymbolRequest(file=rel, class_name=cls.name, fields=cls_leaves))
    return requests


def tier_d_native_types(
    *,
    source_root: Path,
    config_surface_files: Sequence[str],
    candidate_leaves: Sequence[str],
) -> dict[str, str]:
    """Map each candidate leaf to the fully-qualified native_type that DECLARES it.

    Tier-D knows each candidate's owning class deterministically (the config class
    on the surface that declares the leaf), so the gate's construction target does
    not depend on the model formatting a ``file:line`` citation correctly (which
    it often does not). Returns ``{leaf: "<module>.<Class>"}``; the first declaring
    class wins on the rare cross-class collision.
    """
    leaves = set(candidate_leaves)
    out: dict[str, str] = {}
    for rel in config_surface_files:
        path = source_root / rel
        if not path.is_file():
            continue
        body = path.read_text()
        if not body.strip():
            continue
        try:
            module = ast.parse(body)
        except SyntaxError:
            continue
        module_path = _module_from_file(rel)
        for cls in _iter_config_classes(module):
            native_type = f"{module_path}.{cls.name}" if module_path else cls.name
            for stmt in cls.body:
                for target in _assign_targets(stmt):
                    if target in leaves and target not in out:
                        out[target] = native_type
    return out


def _module_from_file(rel: str) -> str:
    """Dotted module path for a source-relative file path.

    ``vllm/config/scheduler.py`` -> ``vllm.config.scheduler``;
    ``config.py`` -> ``config``. The leading package segment is the engine root
    module, so the result is a fully-qualified import path. Even when the class
    is re-exported at a shorter path, ``resolve_native_type``'s module-probe
    (strategy 3) falls back to the bare class name across the engine's modules,
    so a best-effort module prefix only ever helps the first resolution attempt.
    """
    stem = rel[:-3] if rel.endswith(".py") else rel
    return stem.replace("/", ".").strip(".")


@dataclass(frozen=True)
class _ClassSpan:
    """A config class's 1-based inclusive line span and fully-qualified name."""

    start: int
    end: int
    native_type: str


def _config_class_index(
    *, source_root: Path, config_surface_files: Sequence[str]
) -> dict[str, list[_ClassSpan]]:
    """Map each config-surface file to its config classes' spans + native types.

    Built from the SAME AST parse the windowing uses, so the line numbers match
    the windowed source the model cites. Lets a 1b citation (``file:line``) be
    mapped to the enclosing class WITHOUT trusting the model to spell the import
    path - the robust route CR3 requires.
    """
    index: dict[str, list[_ClassSpan]] = {}
    for rel in config_surface_files:
        path = source_root / rel
        if not path.is_file():
            continue
        body = path.read_text()
        if not body.strip():
            continue
        try:
            module = ast.parse(body)
        except SyntaxError:
            continue
        module_path = _module_from_file(rel)
        spans = [
            _ClassSpan(
                start=cls.lineno,
                end=cls.end_lineno or cls.lineno,
                native_type=f"{module_path}.{cls.name}" if module_path else cls.name,
            )
            for cls in _iter_config_classes(module)
        ]
        if spans:
            index[rel] = spans
    return index


def _parse_citation(citation: str) -> tuple[str, int] | None:
    """Extract ``(file, line)`` from a ``file:line`` / ``file:lo-hi`` citation.

    Returns the first cited line (the anchor used to find the enclosing class),
    or ``None`` when the citation carries no parseable ``file:line``. Tolerant of
    a trailing line RANGE (``...:274-280`` -> line 274) and surrounding prose.
    """
    match = re.search(r"([\w./-]+\.py):(\d+)", citation)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _native_type_for_citation(citation: str, class_index: dict[str, list[_ClassSpan]]) -> str:
    """Resolve a 1b diagnosis citation to the enclosing class's native_type.

    Maps the cited ``file:line`` to the config class whose span contains that
    line. Returns the fully-qualified ``{module}.{class}`` so the gate can
    construct it; ``""`` when the citation does not resolve (no parseable
    file:line, file not on the surface, or line outside every class span) - the
    gate then reports infra_error honestly rather than silently confirming.
    """
    parsed = _parse_citation(citation)
    if parsed is None:
        return ""
    cited_file, line = parsed
    spans = class_index.get(cited_file)
    if spans is None:
        # The citation file may be given with a different prefix than the
        # surface-relative key; match on basename as a fallback.
        for rel, rel_spans in class_index.items():
            if rel.endswith(cited_file) or cited_file.endswith(rel):
                spans = rel_spans
                break
    if not spans:
        return ""
    enclosing = [s for s in spans if s.start <= line <= s.end]
    if not enclosing:
        return ""
    # Innermost class wins (smallest span) when classes nest.
    return min(enclosing, key=lambda s: s.end - s.start).native_type


def _assemble_source(
    *,
    source_root: Path,
    requests: Sequence[SymbolRequest],
    budget_chars: int,
    files: Sequence[str],
    member_filter: Callable[[str], bool] | None = None,
) -> WindowReport:
    """Window the requests and guard the empty-source failure mode.

    Raises :class:`DiagnoseError` if windowing produced no source at all (no
    cited symbol resolved in any file) - the trial's empty-input guard, hoisted
    before the model call so a silent empty prompt can never reach the proposer.
    """
    report = windowed_source(
        source_root=source_root,
        requests=requests,
        budget_chars=budget_chars,
        member_filter=member_filter,
    )
    if not report.source.strip():
        detail = "; ".join(report.missing) or "no config classes located"
        raise DiagnoseError(
            f"windowed source is empty (no cited symbol resolved among {list(files)} "
            f"under {source_root}): {detail}"
        )
    return report


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


_TIER_D_PROMPT = """You are auditing config-parameter bounds in an inference-engine library: {engine} {new_version}.

Below is a list of NUMERIC config fields with no declarative bound (no pydantic Field(ge/le), no Literal enum, no mined rule). For each, decide whether the library ENFORCES a bound on it. Look hard in the SOURCE below - not just the docstring, but the method bodies shown beneath each field: `__post_init__`, `_verify_args`, `validate`, `@field_validator`, or any `if <field> ...: raise`. The bound usually lives in a validator, not the declaration.

A field is `bounded` if a construction-time raise constrains it. Pick the ILLEGAL value carefully: it must land in the region the library REJECTS. Watch for explicit legal SENTINELS - if the source accepts a special value (commonly `-1` = unlimited / `0` = disabled, e.g. a guard like `x != -1 and x < 0`), that sentinel is LEGAL, so do NOT use it as your illegal value; pick a value clearly outside the legal set (if `-1` is the legal sentinel and `x < -1` is rejected, use `-2`, never `-1`). A field with no validator raise is `unconstrained`.

FIELDS TO AUDIT (use the given rule_id verbatim for each):
{candidates}

NEW {new_version} SOURCE (cite real file:line from it):
{source}

For EACH field emit one diagnosis object:
  - rule_id: the exact rule_id given for that field.
  - classification: `bounded` if a construction-time raise constrains it; `unconstrained` if no validator constrains it (a free counter / port / rank, or a value the library accepts including its sentinels); `unknown` if the source does not let you decide.
  - reason: WHERE the raise is (e.g. "_verify_args: raises if logprobs < -1") or why it is unconstrained.
  - citation: file and line number(s) of the OWNING CLASS / the raise in the source above (so the gate can construct it).
  - constraint: for `bounded` only - a short human statement of the bound (e.g. "must be >= -1").
  - fires_when: for `bounded` only - a ONE-KEY predicate over THIS field describing the ILLEGAL (rejected) region, using one of: "<", "<=", ">", ">=", "==", "!=", "in", "not_in". E.g. a field rejected when below -1 fires_when {{"<": -1}}; a field that must be <= 1.0 fires_when {{">": 1.0}}.
  - kwargs_positive: for `bounded` only - constructor keyword args setting THIS field to an ILLEGAL value (one satisfying fires_when, and NOT a legal sentinel). NATIVE JSON types (numbers, not strings).
  - kwargs_negative: for `bounded` only - constructor keyword args setting THIS field to a LEGAL value (one NOT satisfying fires_when).

These probes are CONSTRUCTED by a deterministic gate: the illegal kwargs must actually make the library raise and the legal kwargs must construct cleanly, or the proposal is dropped (so a wrong bound costs nothing - but a missed real bound is a missed advisory). Emit a JSON object with a diagnoses array containing one object per field above."""


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


def tier_d_rule_id(engine: str, section: str, leaf: str) -> str:
    """Deterministic rule_id for a Tier-D candidate field.

    Stable across runs (no model freedom) so a diagnosis maps back to its
    candidate by id and the emitted advisory id is reproducible / byte-stable.
    """
    return f"{engine}_tier_d_{section}_{leaf}"


def _compact_candidates(engine: str, candidates: Sequence[dict[str, Any]]) -> str:
    """Render the candidate fields (the audit list) for the Tier-D prompt."""
    lines: list[str] = []
    for cand in candidates:
        section = str(cand["section"])
        leaf = str(cand["leaf"])
        lines.append(
            "\n".join(
                [
                    f"- rule_id: {tier_d_rule_id(engine, section, leaf)}",
                    f"  field: {leaf}",
                    f"  section: {section}",
                    f"  type: {cand.get('type', '')}",
                    f"  default: {cand.get('default', '')}",
                ]
            )
        )
    return "\n".join(lines)


def build_tier_d_prompt(
    *,
    engine: str,
    new_version: str,
    candidates: Sequence[dict[str, Any]],
    source: str,
) -> str:
    """Build the Tier-D pure-inference prompt from the candidate list + source."""
    if not candidates:
        raise DiagnoseError("tier-d diagnose requires at least one candidate field")
    if not source.strip():
        raise DiagnoseError("tier-d diagnose requires non-empty config-surface source")
    return _TIER_D_PROMPT.format(
        engine=engine,
        new_version=new_version,
        candidates=_compact_candidates(engine, candidates),
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
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            # check=True + capture_output hides the container's stderr; the CLI
            # only catches DiagnoseError, so surface the failure with the
            # container's diagnostics instead of an opaque traceback.
            raise DiagnoseError(
                f"diagnose gate container exited {exc.returncode}.\n"
                f"stderr:\n{exc.stderr or '<empty>'}\n"
                f"stdout:\n{exc.stdout or '<empty>'}"
            ) from exc
        if not out_path.exists():
            raise DiagnoseError(
                f"diagnose gate produced no verdicts file at {out_path} "
                "(container exited 0 but wrote nothing)"
            )
        return list(json.loads(out_path.read_text()))


def _proposal_from_diagnosis(
    diag: Diagnosis,
    carried: dict[str, Any] | None,
    *,
    fallback_native_type: str = "",
) -> dict[str, Any]:
    """Assemble a gate proposal dict from a diagnosis + its carried entry (if any).

    The carried entry supplies the structural shape the model does not emit
    (``native_type``, ``severity``, ``match``, ``expected_outcome``); the
    diagnosis supplies the (possibly corrected) kwargs probes. For a 1b gap with
    no carried counterpart, the model's classification routes severity and
    ``fallback_native_type`` (the windowed class identity the citation resolves
    to) supplies the native_type the gate needs to construct - without it a 1b
    proposal always gates infra_error (CR3).
    """
    carried = carried or {}
    severity = str(carried.get("severity", "")).lower()
    if severity not in VALID_SEVERITY:
        severity = "dormant" if diag.classification == "dormancy_now_silent" else "error"
    return {
        "rule_id": diag.rule_id,
        "classification": diag.classification,
        "native_type": carried.get("native_type") or fallback_native_type,
        "severity": severity,
        "match": carried.get("match") or {"fields": {}},
        "expected_outcome": carried.get("expected_outcome"),
        "kwargs_positive": dict(diag.kwargs_positive),
        "kwargs_negative": dict(diag.kwargs_negative),
        "invariant_under_test": carried.get("invariant_under_test") or diag.reason,
        "message_template": carried.get("message_template"),
        "references": [f"llm_diagnose: {diag.reason}", f"citation: {diag.citation}"],
    }


def _validated_entry(proposal: dict[str, Any], engine: str) -> dict[str, Any]:
    """Shape a gate-confirmed proposal into a committed rules.proposed.yaml entry.

    The emitted ``expected_outcome`` is normalised to carry both ``outcome`` and
    a valid ``emission_channel`` (the loader rejects a missing/empty channel);
    the conservative default mirrors the committed corpus (``none`` - no
    user-visible emission for an error raise / silent equivalence).
    """
    severity = proposal["severity"]
    expected = dict(proposal.get("expected_outcome") or {})
    expected.setdefault("outcome", "error" if severity == "error" else "dormant_announced")
    expected.setdefault("emission_channel", "none")
    entry: dict[str, Any] = {
        "id": proposal["rule_id"],
        "engine": engine,
        "library": engine,
        "invariant_under_test": proposal.get("invariant_under_test") or "",
        "severity": severity,
        "native_type": proposal["native_type"],
        "match": proposal.get("match") or {"fields": {}},
        "kwargs_positive": proposal.get("kwargs_positive") or {},
        "kwargs_negative": proposal.get("kwargs_negative") or {},
        "expected_outcome": expected,
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
    - ``source_truncated`` - per-symbol windows dropped to stay under the source
      budget (reported, never silently dropped).
    - ``source_missing`` - cited symbols absent from the new pin's source (a
      class / field that no longer exists); flagged for the maintainer.
    """

    engine: str
    engine_version: str
    mode: str
    confirmed: list[dict[str, Any]] = field(default_factory=list)
    unconfirmed: list[dict[str, Any]] = field(default_factory=list)
    not_construction_confirmable: list[dict[str, Any]] = field(default_factory=list)
    dropped_malformed: list[dict[str, Any]] = field(default_factory=list)
    source_truncated: list[str] = field(default_factory=list)
    source_missing: list[str] = field(default_factory=list)
    proposed_yaml: str | None = None
    # Tier-D only: fields the proposer judged to carry NO real bound (a free
    # counter / port / rank, or a sentinel-accepting field). Reported, never
    # gated or emitted - the model is the noise filter the deterministic
    # candidate enumeration could not be.
    tier_d_unconstrained: list[dict[str, Any]] = field(default_factory=list)


def _gate_and_split(
    *,
    engine: str,
    engine_version: str,
    mode: str,
    diagnoses: list[Diagnosis],
    carried_by_id: dict[str, dict[str, Any]],
    gate_runner: GateRunner,
    native_type_for: Callable[[Diagnosis], str] | None = None,
) -> DiagnoseResult:
    """Gate the well-formed diagnoses and split into confirmed / not / unconfirmed.

    ``native_type_for`` supplies a fallback native_type for a diagnosis that has
    no carried counterpart (1b gap-diagnose), resolved from the cited windowed
    class so the gate can construct the probe (CR3).
    """
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
        _proposal_from_diagnosis(
            diag,
            carried_by_id.get(diag.rule_id),
            fallback_native_type=native_type_for(diag) if native_type_for is not None else "",
        )
        for diag in gateable
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
            result.confirmed.append(_validated_entry(proposal, engine))
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
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> DiagnoseResult:
    """Mode 1a: triage the decay alarm's post-Rung-0 residual failures.

    ``carried_corpus`` is the previous pin's proposed-shape corpus (the
    executable carried catalogue with kwargs); ``residual_ids`` are the ids the
    reconciliation left unresolved (``reconcile_regate_report``'s ``residual``).
    The cited source is WINDOWED per-symbol (each residual's class + cited
    field/validator spans) under ``budget_chars`` so a large cited file cannot
    overflow the model context. Returns a :class:`DiagnoseResult`; only
    ``confirmed`` is emitted as YAML.
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

    requests = carried_symbol_requests(carried=residual, cited_files=cited_files)
    window = _assemble_source(
        source_root=source_root,
        requests=requests,
        budget_chars=budget_chars,
        files=cited_files,
    )
    prompt = build_carried_triage_prompt(
        engine=engine,
        old_version=old_version,
        new_version=new_version,
        carried=residual,
        source=window.source,
    )
    diagnoses = parse_diagnoses(model.complete(prompt))
    # Triage only the rules we asked about - ignore any fabricated extra ids.
    residual_id_set = set(residual_ids)
    diagnoses = [d for d in diagnoses if d.rule_id in residual_id_set]
    if not diagnoses:
        raise DiagnoseError("model returned no diagnoses for the residual rules")

    result = _gate_and_split(
        engine=engine,
        engine_version=new_version,
        mode="carried_failure_triage",
        diagnoses=diagnoses,
        carried_by_id=by_id,
        gate_runner=gate_runner,
    )
    result.source_truncated = window.truncated
    result.source_missing = window.missing
    return result


def diagnose_gaps(
    *,
    engine: str,
    new_version: str,
    schema: dict[str, Any],
    source_root: Path,
    config_surface_files: Sequence[str],
    model: DiagnoseModel,
    gate_runner: GateRunner,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> DiagnoseResult:
    """Mode 1b: flag what deterministic mining missed on the new pin.

    ``schema`` is the fresh ``schema.discovered.json`` (summarised in-prompt);
    ``config_surface_files`` is the walkers' config-surface glob, resolved to
    paths under ``source_root``. The source is WINDOWED to the config-like
    classes on that surface under ``budget_chars`` (not the whole files). The
    model proposes missed rules; the gate confirms which actually reconstruct.
    Only ``confirmed`` is emitted.
    """
    requests = config_surface_symbol_requests(
        source_root=source_root, config_surface_files=config_surface_files
    )
    window = _assemble_source(
        source_root=source_root,
        requests=requests,
        budget_chars=budget_chars,
        files=config_surface_files,
    )
    prompt = build_gap_diagnose_prompt(
        engine=engine, new_version=new_version, schema=schema, source=window.source
    )
    diagnoses = parse_diagnoses(model.complete(prompt))
    # Thread the windowed class identity into each gap proposal: 1b has no
    # carried entry, so without this the proposal's native_type is empty and the
    # gate can never construct it (always infra_error). The citation file:line
    # maps to the enclosing config class's fully-qualified name (CR3).
    class_index = _config_class_index(
        source_root=source_root, config_surface_files=config_surface_files
    )
    result = _gate_and_split(
        engine=engine,
        engine_version=new_version,
        mode="gap_diagnose",
        diagnoses=diagnoses,
        carried_by_id={},
        gate_runner=gate_runner,
        native_type_for=lambda diag: _native_type_for_citation(diag.citation, class_index),
    )
    result.source_truncated = window.truncated
    result.source_missing = window.missing
    return result


# ---------------------------------------------------------------------------
# Tier-D: pure-inference advisories for undeclared-semantic (class-3) fields
# ---------------------------------------------------------------------------
#
# Tier-D is the localiser-not-adjudicator contract applied to numeric fields the
# deterministic surface left UNBOUNDED. The proposer reads the source and, per
# field, either supplies a bound (illegal/legal probes + a fires_when predicate)
# or rejects it as unconstrained. The gate then runs the CONSTRUCTION-VIOLATION
# path (the illegal value must actually raise, the legal value must construct) -
# the empirical sentinel guard that kills the `max_logprobs=-1`-is-legal class of
# hallucination. Confirmed advisories land at severity=warn / outcome=warn /
# llm_proposed=true / added_by=llm_advisory: they WARN at pre-flight, never
# reject (the universal safety gate - an LLM assertion can only ever warn).


def _tier_d_match_fields(engine: str, candidate: dict[str, Any], diag: Diagnosis) -> dict[str, Any]:
    """Build the advisory's ``match.fields`` from the candidate path + fires_when.

    The path is ``<engine>.<section>.<leaf>`` (the project-config location); the
    predicate is the model's ``fires_when`` (the illegal region) so the advisory
    warns only out-of-bound, not on every set (over-warning guard, Risk #5).
    """
    path = f"{engine}.{candidate['section']}.{candidate['leaf']}"
    return {path: diag.fires_when}


def _tier_d_proposal(
    engine: str,
    candidate: dict[str, Any],
    diag: Diagnosis,
    *,
    native_type: str,
) -> dict[str, Any]:
    """Assemble the gate proposal for a Tier-D candidate diagnosis.

    Carries ``tier_d: True`` so the in-container gate routes to the
    construction-violation path (illegal must raise) rather than the
    firing/emission grain, and the carried ``match`` so the gate-soundness locus
    check can attribute the raise. Severity is forced ``warn`` (the emitted
    advisory severity); the gate confirms by construction-violation regardless.
    """
    return {
        "rule_id": diag.rule_id,
        "classification": diag.classification,
        "native_type": native_type,
        "severity": "warn",
        "tier_d": True,
        "match": {"engine": engine, "fields": _tier_d_match_fields(engine, candidate, diag)},
        "expected_outcome": {"outcome": "warn", "emission_channel": "none"},
        "kwargs_positive": dict(diag.kwargs_positive),
        "kwargs_negative": dict(diag.kwargs_negative),
        "invariant_under_test": diag.constraint or diag.reason,
        "message_template": diag.constraint or None,
        "references": [f"llm_advisory: {diag.reason}", f"citation: {diag.citation}"],
    }


def _tier_d_entry(proposal: dict[str, Any], engine: str) -> dict[str, Any]:
    """Shape a construction-confirmed Tier-D proposal into a corpus advisory.

    ``added_by=llm_advisory`` (distinct from ``llm_diagnose`` = gate-confirmed
    FIRING) + ``llm_proposed=True`` so the loader's universal safety gate pins it
    to warn. Frozen + carried byte-stably across re-mines via
    ``build_corpus._CARRIED_PROVENANCES``.
    """
    entry: dict[str, Any] = {
        "id": proposal["rule_id"],
        "engine": engine,
        "library": engine,
        "invariant_under_test": proposal.get("invariant_under_test") or "",
        "severity": "warn",
        "native_type": proposal["native_type"],
        "match": proposal.get("match") or {"fields": {}},
        "kwargs_positive": proposal.get("kwargs_positive") or {},
        "kwargs_negative": proposal.get("kwargs_negative") or {},
        "expected_outcome": {"outcome": "warn", "emission_channel": "none"},
        "references": proposal.get("references") or [],
        "added_by": "llm_advisory",
        "llm_proposed": True,
    }
    if proposal.get("message_template"):
        entry["message_template"] = proposal["message_template"]
    return entry


def diagnose_tier_d(
    *,
    engine: str,
    new_version: str,
    candidates: Sequence[dict[str, Any]],
    source_root: Path,
    config_surface_files: Sequence[str],
    model: DiagnoseModel,
    gate_runner: GateRunner,
    path_validator: Callable[[str], None] | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> DiagnoseResult:
    """Tier-D: pure-inference warn advisories for class-3 (undeclared-semantic) fields.

    ``candidates`` is the deterministic class-3 set (each a mapping with
    ``section`` / ``leaf`` / ``type`` / ``default``;
    :func:`scripts.engine_producers.tier_d_candidates.enumerate_class3_candidates`
    is the caller's source). The config-surface source is windowed per config
    class (mode-1b machinery); the model audits each candidate and supplies a
    bound or rejects it. Only ``bounded`` diagnoses are gated, by the
    construction-violation path; confirmed advisories are emitted at warn.

    ``path_validator`` (injected by the scripts-level driver, which can import
    :mod:`scripts.engine_producers._section_classifier`) raises on an emitted
    match path that does not resolve on the generated Config, so a dead advisory
    is dropped rather than silently committed. ``None`` skips the check (tests /
    callers without the scripts surface).
    """
    result = DiagnoseResult(engine=engine, engine_version=new_version, mode="tier_d")
    if not candidates:
        raise DiagnoseError("tier-d diagnose requires at least one candidate field")

    # Cited-field windowing: show the model each candidate's validator body (where
    # a construction bound's raise lives), not just the 40-line class header.
    candidate_leaves = [str(c["leaf"]) for c in candidates]
    requests = tier_d_symbol_requests(
        source_root=source_root,
        config_surface_files=config_surface_files,
        candidate_leaves=candidate_leaves,
    )
    window = _assemble_source(
        source_root=source_root,
        requests=requests,
        budget_chars=budget_chars,
        files=config_surface_files,
        member_filter=_is_validator_member,
    )
    result.source_truncated = window.truncated
    result.source_missing = window.missing

    prompt = build_tier_d_prompt(
        engine=engine, new_version=new_version, candidates=candidates, source=window.source
    )
    diagnoses = parse_diagnoses(model.complete(prompt), classifications=TIER_D_CLASSIFICATIONS)

    # Map each diagnosis back to the candidate it was asked about (by the
    # deterministic rule_id); ignore any fabricated ids the model invents.
    cand_by_id = {tier_d_rule_id(engine, str(c["section"]), str(c["leaf"])): c for c in candidates}
    # Resolve the gate's construction target from the candidate's DECLARING class
    # (deterministic), not the model's citation - the model often emits an
    # unparseable citation, which left native_type empty -> spurious infra_error.
    native_by_leaf = tier_d_native_types(
        source_root=source_root,
        config_surface_files=config_surface_files,
        candidate_leaves=candidate_leaves,
    )

    gateable: list[tuple[dict[str, Any], Diagnosis, str]] = []
    for diag in diagnoses:
        candidate = cand_by_id.get(diag.rule_id)
        if candidate is None:
            continue
        if diag.classification != "bounded":
            result.tier_d_unconstrained.append(
                {
                    "rule_id": diag.rule_id,
                    "classification": diag.classification,
                    "reason": diag.reason,
                }
            )
            continue
        if diag.is_malformed or diag.fires_when is None:
            result.dropped_malformed.append(
                {
                    "rule_id": diag.rule_id,
                    "classification": diag.classification,
                    "malformed": diag.kwargs_malformed or {"fires_when": ["missing"]},
                    "reason": "type-malformed probe or missing fires_when predicate",
                }
            )
            continue
        native_type = native_by_leaf.get(str(candidate["leaf"]), "")
        proposal = _tier_d_proposal(engine, candidate, diag, native_type=native_type)
        if path_validator is not None:
            (path,) = _tier_d_match_fields(engine, candidate, diag)
            try:
                path_validator(path)
            except Exception as exc:  # UnreachableMatchPathError / UnresolvableMatchPathError
                result.unconfirmed.append(
                    {"rule_id": diag.rule_id, "verdict": "path_unresolved", "reason": str(exc)}
                )
                continue
        gateable.append((proposal, diag, diag.rule_id))

    if not gateable:
        return result

    proposals = [proposal for proposal, _, _ in gateable]
    proposal_by_id = {p["rule_id"]: p for p in proposals}
    verdicts = gate_runner(engine, proposals)

    for verdict in verdicts:
        rule_id = verdict.get("rule_id", "")
        proposal = proposal_by_id.get(rule_id, {})
        if verdict.get("verdict") == "confirmed":
            result.confirmed.append(_tier_d_entry(proposal, engine))
        else:
            result.unconfirmed.append(
                {
                    "rule_id": rule_id,
                    "classification": proposal.get("classification"),
                    "verdict": verdict.get("verdict"),
                    "gate": verdict,
                }
            )

    if result.confirmed:
        result.proposed_yaml = render_proposed_yaml(
            engine=engine, engine_version=new_version, entries=result.confirmed
        )
    return result
