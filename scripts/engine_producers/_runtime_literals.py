"""Runtime-literal discovery: a second schema-discovery stage.

Some engine type knowledge lives only in runtime validation code, not in the
static type surface. The canonical instance is transformers ``early_stopping``:
``GenerationConfig`` accepts ``True``/``False``/``"never"`` at runtime, but
signature-based discovery records ``{"type": "bool"}``, so the generated typed
config rejects the upstream-valid ``"never"``. This stage recovers those
literals and folds them into the discovered schema.

Pipeline position::

    static discovery -> candidate generation -> construction probe -> merged schema

The stage runs after static discovery, inside the same engine container, over
the just-discovered envelope:

1. **Candidate generation** unions four sources, all string-valued only:
   - the shipped rules corpus (equality / membership comparands and
     single-quoted message-template tokens);
   - the engine's own validation source text (``self.<field> in {...}`` /
     ``== "..."`` comparisons) and class docstrings;
   - an optional LLM-proposed file (read only; written by a separate analyst);
   - the previous schema's recorded literals (so a still-valid literal survives
     even if its original evidence source moved).
   Every candidate is filtered against the field's static type: a value the
   static type already expresses is not a candidate.

2. **Construction probe** (the engine is the arbiter): for each candidate, a
   two-leg probe - the literal value must BUILD the native config, and a
   sentinel string must RAISE. A field that accepts both is not string-validated
   at construction grain, so recording it would be unsound; it is dropped.

3. **Merge** records each verified literal under an in-schema ``runtime_literals``
   key on the field, with construction provenance and its evidence.

Staleness is handled by **auto-narrow with loud surfacing**: at a bump, a
previously recorded literal that no longer verifies is dropped from the schema
(no human gate) and a ``NARROWED`` line is emitted for the maintainer's diff.

Only STRING-valued literals are candidate material. The observed problem class
is string literals on non-string static types; and Python's bool/int equality
quirks (``True == 1``) make non-string membership evidence unreliable.

Determinism guarantees (the discovery byte-stability contract): candidates,
evidence, and recorded entries are all sorted; there are NO wall-clock
timestamps in stage output (the envelope's ``discovered_at`` is owned elsewhere
and must not be touched here); the merged schema is a pure function of the pin
and the inputs (rules corpus, engine source, proposals file, previous schema).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine_versions import _outputs  # noqa: E402
from scripts import _candidate_pool  # noqa: E402
from scripts import _engine_constructors as ec  # noqa: E402
from scripts.engine_producers import _current  # noqa: E402

# Sections scanned for field specs, in precedence order (later wins on a
# duplicate leaf name - see :func:`_field_specs`).
_SECTIONS: tuple[str, ...] = ("engine_params", "sampling_params")

# Field-spec map: leaf field name -> (section, spec).
FieldSpecs = dict[str, tuple[str, dict[str, Any]]]

# Corpus predicate operators whose comparands are candidate material. Both the
# symbol form (used by the shipped corpora) and the word-form alias (accepted by
# the rules loader) are taken literally; presence / absence / type predicates and
# numeric-comparison operators are deliberately excluded.
_EQ_MEMBERSHIP_OPS: frozenset[str] = frozenset({"==", "equals", "!=", "not_equal", "in", "not_in"})

# Static type tokens mapped to a canonical scalar (both the Python and the
# JSON-native spelling). A value is string-expressible iff "str" is among the
# mapped tokens of the field's type.
_TYPE_TOKENS: dict[str, str] = {
    "str": "str",
    "string": "str",
    "bool": "bool",
    "boolean": "bool",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
}

# Probe sentinel: a string no engine field legitimately accepts. leg2 builds the
# config with this value and expects the engine to RAISE - proof the field is
# string-validated at construction, not a silent kwargs stash.
_SENTINEL = "__llem_runtime_literal_probe__"


@dataclass(frozen=True)
class LiteralCandidate:
    """One (field, string-value) candidate with its evidence trail.

    Evidence strings use a fixed vocabulary: ``rule:<rule_id>``,
    ``src:<pkg-relative-file>:<line>``, ``doc:<ClassName>.<field>``,
    ``llm:<file>:<line>``.
    """

    field: str
    value: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class CensusFinding:
    """A corpus literal the discovered schema's type cannot express."""

    field: str
    value: str
    static_type: str
    evidence: tuple[str, ...]


@dataclass
class StageReport:
    """Human-readable report lines plus the raw counts behind them."""

    lines: list[str]
    verified: int = 0
    rejected: int = 0
    undiscriminating: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Field specs + expressibility
# ---------------------------------------------------------------------------


def _field_specs(envelope: dict[str, Any]) -> FieldSpecs:
    """Map each leaf field to its (section, spec) over the scanned sections.

    A field present in both sections resolves to the later section in
    :data:`_SECTIONS` (a benign duplicate; the sampling / engine split does not
    overlap on the fields this stage touches).
    """
    specs: FieldSpecs = {}
    for section in _SECTIONS:
        for name, spec in (envelope.get(section) or {}).items():
            if isinstance(spec, dict):
                specs[name] = (section, spec)
    return specs


def _existing_literal_values(spec: dict[str, Any]) -> set[str]:
    """String values already recorded under the field's ``runtime_literals``."""
    out: set[str] = set()
    for entry in spec.get("runtime_literals") or []:
        value = entry.get("value") if isinstance(entry, dict) else None
        if isinstance(value, str):
            out.add(value)
    return out


def _expressible(spec: dict[str, Any], value: str) -> bool:
    """True iff the field's static type (incl. recorded literals) already covers ``value``.

    An expressible value is NOT candidate material - the generated type accepts
    it already. Permissive on anything the stage cannot reason about (``$ref``
    blobs, unknown / Any types, unmappable type tokens): the point is to catch
    string literals stranded on a definitively non-string scalar type.
    """
    if "$ref" in spec:
        return True
    existing = _existing_literal_values(spec)
    if "enum" in spec:
        return value in (spec.get("enum") or []) or value in existing
    if value in existing:
        return True
    type_str = spec.get("type")
    if not type_str or type_str == "unknown":
        return True
    mapped: set[str] = set()
    for token in str(type_str).split("|"):
        tok = token.strip()
        if tok in ("", "None", "unknown"):
            continue
        canonical = _TYPE_TOKENS.get(tok)
        if canonical is None:
            return True  # unmappable non-None token -> permissive Any
        mapped.add(canonical)
    if not mapped:
        return True
    return "str" in mapped


def _accept(
    field_name: str, value: str, evidence: str, fields: FieldSpecs
) -> LiteralCandidate | None:
    """Build a candidate iff ``field_name`` is known and ``value`` is inexpressible."""
    entry = fields.get(field_name)
    if entry is None:
        return None
    _section, spec = entry
    if _expressible(spec, value):
        return None
    return LiteralCandidate(field=field_name, value=value, evidence=(evidence,))


# ---------------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------------


def corpus_candidates(rules_doc: dict[str, Any], fields: FieldSpecs) -> list[LiteralCandidate]:
    """Candidates from the shipped rules corpus.

    Two extractions per rule: (1) string comparands of equality / membership
    predicates (skipping ``None`` and ``@field_ref`` strings); (2) single-quoted
    tokens in the ``message_template``, attached to every match-field leaf of the
    rule. The message-template pass is deliberately over-inclusive - the probe
    arbitrates which attachments are real.
    """
    out: list[LiteralCandidate] = []
    for rule in rules_doc.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id", "?"))
        field_specs = ((rule.get("match") or {}).get("fields")) or {}
        leaves: list[str] = []
        for path, spec in field_specs.items():
            leaf = str(path).rsplit(".", 1)[-1]
            leaves.append(leaf)
            if not isinstance(spec, dict):
                continue
            for op, val in spec.items():
                if op not in _EQ_MEMBERSHIP_OPS:
                    continue
                comparands = val if isinstance(val, (list, tuple)) else [val]
                for comparand in comparands:
                    if isinstance(comparand, str) and not comparand.startswith("@"):
                        cand = _accept(leaf, comparand, f"rule:{rule_id}", fields)
                        if cand is not None:
                            out.append(cand)
        msg = rule.get("message_template")
        if isinstance(msg, str):
            for token in re.findall(r"'([^']{1,40})'", msg):
                for leaf in leaves:
                    cand = _accept(leaf, token, f"rule:{rule_id}", fields)
                    if cand is not None:
                        out.append(cand)
    return out


def scan_source_text(text: str, rel_label: str, fields: FieldSpecs) -> list[LiteralCandidate]:
    """Candidates from ``self.<field> in {...}`` / ``== "..."`` comparisons in source text.

    Pure and import-free (testable without an engine): parses ``text`` and, for
    each single-operator ``ast.Compare`` whose left side is
    ``self.<field>`` with ``field`` known, collects the string members of a
    membership set/tuple/list or the string right-hand side of an equality.
    """
    out: list[LiteralCandidate] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        left = node.left
        if not (
            isinstance(left, ast.Attribute)
            and isinstance(left.value, ast.Name)
            and left.value.id == "self"
            and left.attr in fields
        ):
            continue
        op = node.ops[0]
        comparator = node.comparators[0]
        values: list[str] = []
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(
            comparator, (ast.Set, ast.Tuple, ast.List)
        ):
            for elt in comparator.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.append(elt.value)
        elif (
            isinstance(op, (ast.Eq, ast.NotEq))
            and isinstance(comparator, ast.Constant)
            and isinstance(comparator.value, str)
        ):
            values.append(comparator.value)
        for value in values:
            cand = _accept(left.attr, value, f"src:{rel_label}:{node.lineno}", fields)
            if cand is not None:
                out.append(cand)
    return out


def scan_docstring(doc: str, class_name: str, fields: FieldSpecs) -> list[LiteralCandidate]:
    """Candidates from double-quoted tokens inside a class docstring's Args blocks.

    A line matching ``^\\s*(\\w+)\\s*\\(`` opens the block for that argument name;
    while inside a KNOWN field's block, double-quoted tokens are collected. An
    unknown-name block closes collection (its tokens belong to a field the schema
    does not expose).
    """
    out: list[LiteralCandidate] = []
    current: str | None = None
    for line in (doc or "").splitlines():
        match = re.match(r"^\s*(\w+)\s*\(", line)
        if match is not None:
            name = match.group(1)
            current = name if name in fields else None
        if current is not None:
            for token in re.findall(r'"([^"\s]{1,40})"', line):
                cand = _accept(current, token, f"doc:{class_name}.{current}", fields)
                if cand is not None:
                    out.append(cand)
    return out


def _package_relative_label(cls: type, source_file: Path) -> str:
    """Relativise a class's source file against its top-level package's parent dir.

    Yields e.g. ``transformers/generation/configuration_utils.py``. Falls back to
    the bare file name when the top-level package cannot be resolved.
    """
    try:
        top_name = (cls.__module__ or "").split(".")[0]
        top = importlib.import_module(top_name)
        top_file = getattr(top, "__file__", None)
        if top_file:
            package_dir = Path(top_file).resolve().parent
            return str(source_file.resolve().relative_to(package_dir.parent))
    except (ImportError, AttributeError, ValueError):
        pass
    return source_file.name


def source_scan_candidates(engine: str, fields: FieldSpecs) -> list[LiteralCandidate]:
    """Container-only wrapper: resolve the engine's config classes and scan them.

    Resolves the native constructor classes for both sections (deduped), plus
    transformers ``BitsAndBytesConfig``, then runs :func:`scan_source_text` on
    each class's source file and :func:`scan_docstring` on its docstring. On a
    plain host (no engine installed) every resolution fails and this returns an
    empty list.
    """
    classes: dict[int, type] = {}
    for section in _SECTIONS:
        try:
            for cls in ec.candidate_classes(engine, f"{engine}.{section}.x", []):
                classes[id(cls)] = cls
        except ec.ConstructorResolutionError:
            continue
    if engine == "transformers":
        try:
            bnb = importlib.import_module("transformers").BitsAndBytesConfig
            classes[id(bnb)] = bnb
        except Exception:  # pragma: no cover - container-only import guard
            pass

    out: list[LiteralCandidate] = []
    for cls in classes.values():
        source_file = inspect.getsourcefile(cls)
        if source_file is not None:
            label = _package_relative_label(cls, Path(source_file))
            try:
                text = Path(source_file).read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - container-only IO guard
                text = ""
            if text:
                out.extend(scan_source_text(text, label, fields))
        doc = inspect.getdoc(cls)
        if doc:
            out.extend(scan_docstring(doc, cls.__name__, fields))
    return out


def llm_candidates(proposals_doc: dict[str, Any], fields: FieldSpecs) -> list[LiteralCandidate]:
    """Candidates from an LLM-proposed file (read-only reader).

    Each entry is ``{field, value, citation: {file, line, ...}}``; evidence is
    ``llm:<file>:<line>``.
    """
    out: list[LiteralCandidate] = []
    for entry in proposals_doc.get("candidates") or []:
        if not isinstance(entry, dict):
            continue
        field_name = entry.get("field")
        value = entry.get("value")
        if not isinstance(field_name, str) or not isinstance(value, str):
            continue
        citation = entry.get("citation") or {}
        file_ref = citation.get("file", "?")
        line_ref = citation.get("line", "?")
        cand = _accept(field_name, value, f"llm:{file_ref}:{line_ref}", fields)
        if cand is not None:
            out.append(cand)
    return out


def previous_literals(previous_envelope: dict[str, Any] | None) -> list[LiteralCandidate]:
    """Carry the previous schema's recorded literals forward as candidates.

    A still-valid literal survives re-verification even if its original evidence
    source moved; a literal that no longer verifies is dropped (auto-narrow).
    """
    out: list[LiteralCandidate] = []
    if not previous_envelope:
        return out
    for section in _SECTIONS:
        for name, spec in (previous_envelope.get(section) or {}).items():
            if not isinstance(spec, dict):
                continue
            for entry in spec.get("runtime_literals") or []:
                if not isinstance(entry, dict):
                    continue
                value = entry.get("value")
                if not isinstance(value, str):
                    continue
                evidence = tuple(str(e) for e in (entry.get("evidence") or ()))
                out.append(LiteralCandidate(field=name, value=value, evidence=evidence))
    return out


def _pool(candidates: Iterable[LiteralCandidate]) -> list[LiteralCandidate]:
    """Pool candidates by (field, value), unioning + sorting evidence.

    Final ordering is sorted by (field, value); each pooled candidate's evidence
    is sorted and deduplicated - the determinism backbone of the stage.
    """
    pooled: dict[tuple[str, str], set[str]] = {}
    for cand in candidates:
        pooled.setdefault((cand.field, cand.value), set()).update(cand.evidence)
    return [
        LiteralCandidate(field=name, value=value, evidence=tuple(sorted(evidence)))
        for (name, value), evidence in sorted(pooled.items())
    ]


# ---------------------------------------------------------------------------
# Construction probe
# ---------------------------------------------------------------------------


def _construct(engine: str, cls: type, kwargs: dict[str, Any]) -> None:
    """Construct ``cls`` from ``kwargs`` so the engine's own validation fires.

    For transformers, ``GenerationConfig`` separates construction from
    validation: build then call ``.validate()`` WITHOUT ``strict``. Non-strict is
    load-bearing - ``validate(strict=True)`` conflates a type-valid literal (e.g.
    ``early_stopping="never"``) with a beam-mode inertness complaint, which would
    wrongly reject the literal. Other engines validate at construction via
    :func:`scripts._engine_constructors.construct`. Any exception raised here is
    the engine's own verdict; the caller interprets it.
    """
    if engine == "transformers":
        obj = cls(**kwargs)
        validate = getattr(obj, "validate", None)
        if callable(validate):
            validate()
        return
    ec.construct(engine, cls, kwargs, validate=True)


ConstructFn = Callable[[str, type, dict[str, Any]], None]

ProbeResult = tuple[
    list[LiteralCandidate],
    list[tuple[LiteralCandidate, str]],
    list[LiteralCandidate],
    list[tuple[LiteralCandidate, str]],
]


def probe_candidates_fn(
    engine: str,
    candidates: Iterable[LiteralCandidate],
    fields: FieldSpecs,
    construct: ConstructFn = _construct,
) -> ProbeResult:
    """Two-leg construction probe with the engine as arbiter.

    Returns ``(verified, rejected, undiscriminating, errors)``. For each
    candidate the constructor class is resolved (first class accepting the
    field). leg1 builds ``{field: value}`` and must succeed; leg2 builds
    ``{field: sentinel}`` and must raise. Verified iff leg1 builds AND leg2
    raises. leg1+leg2 both build -> ``undiscriminating`` (the field is not
    string-validated at construction; recording would be unsound). leg1 raises ->
    ``rejected`` (a 120-char exception snippet is kept). No class / import failure
    -> ``errors``.
    """
    verified: list[LiteralCandidate] = []
    rejected: list[tuple[LiteralCandidate, str]] = []
    undiscriminating: list[LiteralCandidate] = []
    errors: list[tuple[LiteralCandidate, str]] = []

    for cand in candidates:
        entry = fields.get(cand.field)
        if entry is None:
            errors.append((cand, "field absent from schema"))
            continue
        section, _spec = entry
        try:
            classes = ec.candidate_classes(engine, f"{engine}.{section}.{cand.field}", [cand.field])
        except ec.ConstructorResolutionError as exc:
            errors.append((cand, f"resolution: {exc}"))
            continue
        cls = next((c for c in classes if ec.accepts(c, cand.field)), None)
        if cls is None:
            errors.append((cand, "no constructor class accepts the field"))
            continue
        try:
            construct(engine, cls, {cand.field: cand.value})
        except Exception as exc:
            rejected.append((cand, str(exc)[:120]))
            continue
        try:
            construct(engine, cls, {cand.field: _SENTINEL})
        except Exception:
            verified.append(cand)
        else:
            undiscriminating.append(cand)
    return verified, rejected, undiscriminating, errors


# ---------------------------------------------------------------------------
# Merge + narrowing + census
# ---------------------------------------------------------------------------


def merge_runtime_literals(
    envelope: dict[str, Any], verified: Iterable[LiteralCandidate], pin: str
) -> None:
    """Record verified literals under each field's ``runtime_literals`` key in place.

    The key is appended after the field's existing keys (insertion order, which
    the runner preserves via ``sort_keys=False``); entries are sorted by value and
    carry sorted evidence. A field with no verified literal gets no key; zero
    verified leaves the envelope byte-identical.
    """
    fields = _field_specs(envelope)
    grouped: dict[tuple[str, str], list[LiteralCandidate]] = {}
    for cand in verified:
        entry = fields.get(cand.field)
        if entry is None:
            continue
        section, _spec = entry
        grouped.setdefault((section, cand.field), []).append(cand)

    for (section, name), cands in grouped.items():
        entries = [
            {
                "value": cand.value,
                "verified": "construction",
                "pin": pin,
                "evidence": sorted(cand.evidence),
            }
            for cand in sorted(cands, key=lambda c: c.value)
        ]
        envelope[section][name]["runtime_literals"] = entries


def _literal_pairs(envelope: dict[str, Any]) -> set[tuple[str, str]]:
    """The (field, value) pairs recorded under ``runtime_literals`` in ``envelope``."""
    pairs: set[tuple[str, str]] = set()
    for section in _SECTIONS:
        for name, spec in (envelope.get(section) or {}).items():
            if not isinstance(spec, dict):
                continue
            for entry in spec.get("runtime_literals") or []:
                value = entry.get("value") if isinstance(entry, dict) else None
                if isinstance(value, str):
                    pairs.add((name, value))
    return pairs


def narrowing_lines(
    previous_envelope: dict[str, Any] | None, new_envelope: dict[str, Any]
) -> list[str]:
    """Loud-surfacing lines for literals present before but not re-verified now."""
    if not previous_envelope:
        return []
    pin = str(new_envelope.get("engine_version", "?"))
    dropped = _literal_pairs(previous_envelope) - _literal_pairs(new_envelope)
    return [
        f"NARROWED: {name} literal '{value}' no longer verified at pin {pin}; "
        "dropped from the schema (auto-narrow)"
        for name, value in sorted(dropped)
    ]


def census(schema: dict[str, Any], rules_doc: dict[str, Any]) -> list[CensusFinding]:
    """Corpus literals the discovered schema type (incl. runtime_literals) cannot express.

    The standing consistency check: a non-empty result means the corpus asserts a
    string value the generated type would reject. Reuses corpus extraction (which
    already filters through :func:`_expressible`, so a recorded runtime literal
    resolves the finding).
    """
    fields = _field_specs(schema)
    findings: list[CensusFinding] = []
    for cand in _pool(corpus_candidates(rules_doc, fields)):
        entry = fields.get(cand.field)
        static_type = str(entry[1].get("type", "")) if entry is not None else ""
        findings.append(
            CensusFinding(
                field=cand.field,
                value=cand.value,
                static_type=static_type,
                evidence=tuple(e for e in cand.evidence if e.startswith("rule:")),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping from ``path`` (empty / absent / non-mapping -> {})."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def run_stage(
    engine: str,
    envelope: dict[str, Any],
    repo_root: Path,
    previous: dict[str, Any] | None,
) -> StageReport:
    """Run the full stage over a discovered ``envelope`` (mutated in place).

    Unions the four candidate sources, probes them, merges the verified literals,
    emits narrowing lines against ``previous``, and closes with a census of the
    MERGED envelope (which should report zero inexpressible corpus literals). No
    printing and no timestamps: the caller prints the returned lines.
    """
    fields = _field_specs(envelope)
    engine_version = str(envelope.get("engine_version", "?"))

    rules_doc = _load_yaml(repo_root / "src/llenergymeasure/engines" / engine / "rules.yaml")
    proposals_doc = _load_yaml(
        _candidate_pool.pool_path(
            repo_root / "engine_versions", engine, engine_version, "runtime_literals.proposed.yaml"
        )
    )

    corpus = corpus_candidates(rules_doc, fields)
    source_and_doc = source_scan_candidates(engine, fields)
    source = [c for c in source_and_doc if c.evidence and c.evidence[0].startswith("src:")]
    docs = [c for c in source_and_doc if c.evidence and c.evidence[0].startswith("doc:")]
    llm = llm_candidates(proposals_doc, fields)
    prev = previous_literals(previous)

    pooled = _pool([*corpus, *source_and_doc, *llm, *prev])

    lines: list[str] = [
        f"candidates corpus={len(corpus)} source={len(source)} doc={len(docs)} "
        f"llm={len(llm)} previous={len(prev)} -> {len(pooled)} unique (field,value)"
    ]

    verified, rejected, undiscriminating, errors = probe_candidates_fn(
        engine, pooled, fields, construct=_construct
    )
    lines.append(
        f"verified={len(verified)} rejected={len(rejected)} "
        f"undiscriminating={len(undiscriminating)} errors={len(errors)}"
    )

    merge_runtime_literals(envelope, verified, engine_version)

    for cand in sorted(verified, key=lambda c: (fields[c.field][0], c.field, c.value)):
        section = fields[cand.field][0]
        lines.append(
            f"RECORDED {section}.{cand.field} '{cand.value}' ({','.join(sorted(cand.evidence))})"
        )

    lines.extend(narrowing_lines(previous, envelope))

    remaining = census(envelope, rules_doc)
    lines.append(f"census after merge: {len(remaining)} corpus literal(s) inexpressible")

    return StageReport(
        lines=[f"runtime-literals: {line}" for line in lines],
        verified=len(verified),
        rejected=len(rejected),
        undiscriminating=len(undiscriminating),
        errors=len(errors),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _src_schema_path(repo_root: Path, engine: str) -> Path:
    return repo_root / "src/llenergymeasure/engines" / engine / _outputs.SCHEMA_FILENAME


def _run_census(repo_root: Path, engines: list[str]) -> int:
    """Report corpus literals inexpressible in each engine's SRC-shadow schema."""
    any_finding = False
    for engine in engines:
        schema_path = _src_schema_path(repo_root, engine)
        rules_path = repo_root / "src/llenergymeasure/engines" / engine / "rules.yaml"
        if not schema_path.is_file():
            print(f"{engine}: SKIPPED - no discovered schema at {schema_path}", file=sys.stderr)
            continue
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        rules_doc = _load_yaml(rules_path)
        findings = census(schema, rules_doc)
        if not findings:
            print(f"{engine}: OK - all corpus literals expressible")
            continue
        any_finding = True
        print(f"{engine}: {len(findings)} corpus literal(s) inexpressible:", file=sys.stderr)
        for finding in findings:
            print(
                f"  {finding.field} '{finding.value}' - static type "
                f"{finding.static_type!r}; cited by {', '.join(finding.evidence)}",
                file=sys.stderr,
            )
    return 1 if any_finding else 0


def _run_probe_report(repo_root: Path, engine: str) -> int:
    """Report-only probe against the OUTPUTS snapshot at the current pin (writes nothing)."""
    library = _current.load_current(engine).get("library")
    pin = str(library.get("current_version")) if isinstance(library, dict) else ""
    schema = json.loads(_outputs.schema_path(engine, pin).read_text(encoding="utf-8"))
    report = run_stage(engine, schema, repo_root, None)
    for line in report.lines:
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--census",
        action="store_true",
        help="Report corpus literals inexpressible in the discovered schema types (exit 1 on any).",
    )
    parser.add_argument(
        "--probe-report",
        action="store_true",
        help="In-container report-only probe against the current pin's snapshot (writes nothing).",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=list(_outputs.ENGINES),
        default=None,
        help="Engine(s) to act on. Defaults to all engines for --census.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    if args.probe_report:
        engines = args.engine or []
        if len(engines) != 1:
            parser.error("--probe-report requires exactly one --engine.")
        return _run_probe_report(repo_root, engines[0])

    if args.census:
        return _run_census(repo_root, args.engine or list(_outputs.ENGINES))

    parser.error("Specify --census or --probe-report.")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
