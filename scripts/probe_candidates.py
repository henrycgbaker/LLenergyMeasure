#!/usr/bin/env python3
"""Probe kernel: construction and identity checks for engine-rule candidates.

Tiers 2 and 3 of the verification ladder (tier 1 is ``check_citations.py``).
Given candidate rules that each claim a severity, this proves the claim against
the REAL engine and writes a verdict back:

- A ``severity: error`` claim earns a CONSTRUCTION PROBE. We build the offending
  config inside the engine and require both legs: a violating value must raise,
  and a satisfying value must construct cleanly. Only then is the error real.
- A ``severity: dormant`` claim earns an IDENTITY PROBE. We construct the config
  at two or more distinct declared values for the field and compare the
  engine-resolved state. Identical resolved state means the declared variation
  was inert - dormant confirmed. No inference runs, no energy is measured.

Probe values are derived DETERMINISTICALLY from the predicate spec (straddle a
threshold; a member and a non-member of a set; the declared values an observed
collision recorded). Nothing is guessed: if a satisfying/violating pair or two
distinct declared values cannot be derived, the verdict is ``unprobeable``.

Closed verdict vocabulary: ``confirmed`` (claim proven), ``refuted`` (probe
contradicted the claim), ``unprobeable`` (no deterministic probe exists for this
claim), ``infra_error`` (a probe leg failed for reasons that do not isolate the
claim - e.g. the satisfying value also raised, or the engine is not importable
here). Candidates are never dropped; the verdict plus a short transcript is
written back so a refuted or unprobeable candidate stays visible.

Runs INSIDE the engine container (imports the live engine via
``_engine_constructors``); stdlib + PyYAML only. Exit 0 when the run completed
(whatever the per-candidate verdicts), 2 on a hard error (bad args, unreadable
candidates file).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _engine_constructors as ec

# Deterministic derivation constants. REF_BASE anchors any field referenced by
# an @field_ref (and any unnamed constructor arg a cross-field probe needs);
# the steps straddle a threshold to its firing and satisfying sides.
REF_BASE = 4
INT_STEP = 1
FLOAT_STEP = 0.5

# Derivation sentinels: _OMIT = do not set this field (absent); _DEFAULT = build
# without the field and read its resolved default (the contrast leg of an
# aliasing dormancy check).
_OMIT = object()
_DEFAULT = object()

_FLAG_KEYS = {"present", "absent"}
_ENGINES = ("vllm", "tensorrt", "transformers")
_BUILTIN_VALUES: dict[str, Any] = {"bool": True, "int": 1, "float": 1.0, "str": "x"}


class Unprobeable(Exception):
    """No deterministic probe can be derived for this candidate's claim."""


@dataclass
class Candidate:
    """A parsed candidate: enough to probe, plus its raw dict for writeback."""

    id: str
    engine: str | None
    severity: str | None
    fields: dict[str, Any]
    normalised: list[str]
    source: str | None
    declared_values: list[Any] | None
    raw: dict[str, Any]


@dataclass
class Verdict:
    """The probe outcome for one candidate (status + which probe + transcript)."""

    status: str
    tier: str | None
    evidence: str

    def __post_init__(self) -> None:
        # Engine exception messages are often multi-line; collapse whitespace so
        # the evidence is one line in both the summary table and the YAML writeback.
        self.evidence = " ".join(self.evidence.split())


@dataclass
class RunStats:
    """Per-engine verdict tally surfaced in the summary."""

    counts: dict[str, int] = field(default_factory=dict)

    def record(self, status: str) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1


# ---------------------------------------------------------------------------
# Predicate-spec value derivation
# ---------------------------------------------------------------------------


def _is_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("@")


def _ref_leaf(value: str) -> str:
    return value[1:].rsplit(".", 1)[-1]


def _iter_refs(spec: Any) -> list[str]:
    """Every ``@field_ref`` string anywhere inside a predicate spec."""
    if _is_ref(spec):
        return [spec]
    if isinstance(spec, dict):
        return [r for v in spec.values() for r in _iter_refs(v)]
    if isinstance(spec, (list, tuple)):
        return [r for v in spec for r in _iter_refs(v)]
    return []


def _referenced_leaves(fields: dict[str, Any]) -> set[str]:
    return {_ref_leaf(r) for spec in fields.values() for r in _iter_refs(spec)}


def _step(threshold: Any) -> Any:
    return (
        INT_STEP if isinstance(threshold, int) and not isinstance(threshold, bool) else FLOAT_STEP
    )


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _bump(value: Any, k: int) -> Any:
    """A value ``k`` steps away from ``value`` (for distinct !=/threshold probes)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Unprobeable(f"cannot derive a distinct numeric probe from {value!r}")
    return value + k * _step(value)


def _contrast(value: Any) -> Any:
    """A value that differs from ``value`` (for the satisfying/nofire leg)."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "_alt"
    raise Unprobeable(f"cannot derive a contrasting value for {value!r}")


def _nonmember(members: list[Any]) -> Any:
    """A single value outside ``members`` (numeric sets -> max+1, else a sentinel)."""
    numeric = [m for m in members if isinstance(m, (int, float)) and not isinstance(m, bool)]
    non_null = [m for m in members if m is not None]
    if numeric and len(numeric) == len(non_null):
        return max(numeric) + 1
    return "__llem_probe_nonmember__"


def _two_nonmembers(members: list[Any]) -> list[Any]:
    numeric = [m for m in members if isinstance(m, (int, float)) and not isinstance(m, bool)]
    non_null = [m for m in members if m is not None]
    if numeric and len(numeric) == len(non_null):
        top = max(numeric)
        return [top + 1, top + 2]
    raise Unprobeable("cannot derive two distinct non-members of a non-numeric set")


def _first_member(members: list[Any]) -> Any:
    for m in members:
        if m is not None:
            return m
    raise Unprobeable("membership set has no non-null member to satisfy the claim")


def _value_of_type(names: Any) -> Any:
    for name in _as_list(names):
        if name in _BUILTIN_VALUES:
            return _BUILTIN_VALUES[name]
    raise Unprobeable(f"cannot synthesise an instance of type(s) {names!r}")


def _value_not_of_type(names: Any) -> Any:
    excluded = {str(n) for n in _as_list(names)}
    for candidate in ("x", 1, 1.0):
        if type(candidate).__name__ not in excluded:
            return candidate
    raise Unprobeable(f"cannot synthesise a value outside type(s) {names!r}")


def _substantive_op(spec: dict[str, Any]) -> tuple[str, Any] | None:
    """The one non-present/absent operator in a spec, or None if flag-only."""
    ops = [(k, v) for k, v in spec.items() if k not in _FLAG_KEYS]
    if not ops:
        return None
    return ops[0]


def _resolve(value: Any, ref_of: Any) -> Any:
    return ref_of(_ref_leaf(value)) if _is_ref(value) else value


def _fire_value(spec: Any, leaf: str, ref_of: Any, referenced: set[str]) -> Any:
    """A value that makes ``spec`` TRUE (so the engine's validator fires)."""
    if not isinstance(spec, dict):
        return spec
    if spec.get("absent"):
        return _OMIT
    op = _substantive_op(spec)
    if op is None:
        if spec.get("present"):
            return REF_BASE if leaf in referenced else True
        return _OMIT
    name, raw = op
    if name == "<":
        t = _resolve(raw, ref_of)
        return t - _step(t)
    if name == "<=":
        return _resolve(raw, ref_of)
    if name == ">":
        t = _resolve(raw, ref_of)
        return t + _step(t)
    if name == ">=":
        return _resolve(raw, ref_of)
    if name == "==":
        return raw
    if name == "!=":
        return _contrast(_resolve(raw, ref_of))
    if name == "in":
        return _first_member(_as_list(raw))
    if name == "not_in":
        return _nonmember(_as_list(raw))
    if name == "type_is":
        return _value_of_type(raw)
    if name == "type_is_not":
        return _value_not_of_type(raw)
    if name == "divisible_by":
        return 2 * _resolve(raw, ref_of)
    if name == "not_divisible_by":
        return _resolve(raw, ref_of) + 1
    raise Unprobeable(f"no fire-value rule for operator {name!r}")


def _nofire_value(spec: Any, leaf: str, ref_of: Any, referenced: set[str]) -> Any:
    """A value that makes ``spec`` FALSE (so the engine constructs cleanly)."""
    if not isinstance(spec, dict):
        return _contrast(spec)
    if spec.get("absent"):
        return REF_BASE if leaf in referenced else True
    op = _substantive_op(spec)
    if op is None:
        if spec.get("present"):
            return _OMIT
        return REF_BASE if leaf in referenced else True
    name, raw = op
    if name == "<":
        return _resolve(raw, ref_of)
    if name == "<=":
        t = _resolve(raw, ref_of)
        return t + _step(t)
    if name == ">":
        return _resolve(raw, ref_of)
    if name == ">=":
        t = _resolve(raw, ref_of)
        return t - _step(t)
    if name == "==":
        return _contrast(raw)
    if name == "!=":
        return _resolve(raw, ref_of)
    if name == "in":
        return _nonmember(_as_list(raw))
    if name == "not_in":
        return _first_member(_as_list(raw))
    if name == "type_is":
        return _value_not_of_type(raw)
    if name == "type_is_not":
        return _value_of_type(raw)
    if name == "divisible_by":
        return _resolve(raw, ref_of) + 1
    if name == "not_divisible_by":
        return 2 * _resolve(raw, ref_of)
    raise Unprobeable(f"no nofire-value rule for operator {name!r}")


def _ref_env(fields: dict[str, Any], referenced: set[str]) -> dict[str, Any]:
    """Numeric fire values of ref-free named fields, to resolve @field_ref."""
    env: dict[str, Any] = {}
    for path, spec in fields.items():
        if _iter_refs(spec):
            continue
        try:
            value = _fire_value(spec, ec.leaf_of(path), lambda _leaf: REF_BASE, referenced)
        except Unprobeable:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            env[ec.leaf_of(path)] = value
    return env


def _ref_of(fields: dict[str, Any], referenced: set[str]) -> Any:
    env = _ref_env(fields, referenced)
    return lambda leaf: env.get(leaf, REF_BASE)


# ---------------------------------------------------------------------------
# Construction probe (severity: error)
# ---------------------------------------------------------------------------


def build_construction_kwargs(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Violating and satisfying kwargs for a construction probe.

    Violating sets every constrained field to its firing value; satisfying flips
    the last (subject) field to a value that does not fire, holding the rest.
    Fields referenced by an ``@field_ref`` but not named are scaffolded to
    :data:`REF_BASE` so the cross-field comparison has both operands.
    """
    referenced = _referenced_leaves(fields)
    ref_of = _ref_of(fields, referenced)
    named = {ec.leaf_of(p) for p in fields}
    subject_path = list(fields)[-1]
    subject_leaf = ec.leaf_of(subject_path)

    violating: dict[str, Any] = {}
    for path, spec in fields.items():
        value = _fire_value(spec, ec.leaf_of(path), ref_of, referenced)
        if value is not _OMIT:
            violating[ec.leaf_of(path)] = value
    for rleaf in referenced:
        if rleaf not in named and rleaf not in violating:
            violating[rleaf] = ref_of(rleaf)

    nofire = _nofire_value(fields[subject_path], subject_leaf, ref_of, referenced)
    satisfying = dict(violating)
    if nofire is _OMIT:
        satisfying.pop(subject_leaf, None)
    else:
        satisfying[subject_leaf] = nofire
    return violating, satisfying, subject_leaf


def _pick_class(classes: list[type], leaves: list[str]) -> type | None:
    for cls in classes:
        if all(ec.accepts(cls, leaf) for leaf in leaves):
            return cls
    return None


def _run_leg(engine: str, cls: type, kwargs: dict[str, Any]) -> str | None:
    """Construct one leg; return None if it built cleanly, else the exception."""
    try:
        ec.construct(engine, cls, kwargs, validate=True)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:200]


def probe_construction(cand: Candidate, engine: str) -> Verdict:
    try:
        violating, satisfying, _ = build_construction_kwargs(cand.fields)
    except Unprobeable as exc:
        return Verdict("unprobeable", "construction", str(exc))

    leaves = [ec.leaf_of(p) for p in cand.fields]
    try:
        classes = ec.candidate_classes(engine, next(iter(cand.fields)), leaves)
    except ec.ConstructorResolutionError as exc:
        return Verdict("unprobeable", "construction", str(exc))
    cls = _pick_class(classes, leaves)
    if cls is None:
        return Verdict("unprobeable", "construction", f"no constructor accepts fields {leaves}")

    positive = _run_leg(engine, cls, violating)
    negative = _run_leg(engine, cls, satisfying)
    v_show, s_show = _show_kwargs(violating), _show_kwargs(satisfying)
    if positive and not negative:
        return Verdict(
            "confirmed",
            "construction",
            f"violating {v_show} raised {positive}; satisfying {s_show} constructed",
        )
    if not positive:
        return Verdict(
            "refuted",
            "construction",
            f"violating {v_show} constructed cleanly - offending value accepted",
        )
    return Verdict(
        "infra_error",
        "construction",
        f"satisfying {s_show} also raised ({negative}); cannot isolate the claim (violating raised {positive})",
    )


# ---------------------------------------------------------------------------
# Identity probe (severity: dormant)
# ---------------------------------------------------------------------------


def identity_values(cand: Candidate, subject_path: str) -> list[Any]:
    """Two or more distinct declared values for the dormant field under test."""
    if cand.source == "observed_collision" and cand.declared_values:
        values = _distinct([_DEFAULT if v == "<absent>" else v for v in cand.declared_values])
        if len(values) >= 2:
            return values
        raise Unprobeable(
            "observed-collision candidate has fewer than two distinct declared values"
        )

    spec = cand.fields[subject_path]
    if not isinstance(spec, dict):
        return [spec, _DEFAULT]
    op = _substantive_op(spec)
    if op is None:
        raise Unprobeable(
            f"present/absent-only dormant field {subject_path!r} has no declared values to compare"
        )
    name, raw = op
    ref_of = _ref_of(cand.fields, _referenced_leaves(cand.fields))
    if name == "in":
        members = _distinct(_as_list(raw))
        if len(members) >= 2:
            return members
        raise Unprobeable("membership set has fewer than two members")
    if name == "not_in":
        return _two_nonmembers(_as_list(raw))
    if name == "!=":
        return [_bump(_resolve(raw, ref_of), 1), _bump(_resolve(raw, ref_of), 2)]
    if name == "==":
        return [raw, _DEFAULT]
    if name in ("<", "<=", ">", ">="):
        t = _resolve(raw, ref_of)
        s = _step(t)
        return {
            "<": [t - s, t - 2 * s],
            "<=": [t, t - s],
            ">": [t + s, t + 2 * s],
            ">=": [t, t + s],
        }[name]
    raise Unprobeable(f"cannot derive identity values for operator {name!r}")


def build_identity_base(fields: dict[str, Any], subject_path: str) -> dict[str, Any]:
    """Kwargs for every field except the subject, held at firing values."""
    referenced = _referenced_leaves(fields)
    ref_of = _ref_of(fields, referenced)
    named = {ec.leaf_of(p) for p in fields}
    base: dict[str, Any] = {}
    for path, spec in fields.items():
        if path == subject_path:
            continue
        value = _fire_value(spec, ec.leaf_of(path), ref_of, referenced)
        if value is not _OMIT:
            base[ec.leaf_of(path)] = value
    for rleaf in referenced:
        if rleaf not in named and rleaf not in base:
            base[rleaf] = ref_of(rleaf)
    return base


def probe_identity(cand: Candidate, engine: str) -> Verdict:
    subject_path = list(cand.fields)[-1]
    try:
        values = identity_values(cand, subject_path)
        base = build_identity_base(cand.fields, subject_path)
    except Unprobeable as exc:
        return Verdict("unprobeable", "identity", str(exc))

    subject_leaf = ec.leaf_of(subject_path)
    leaves = [ec.leaf_of(p) for p in cand.fields]
    try:
        classes = ec.candidate_classes(engine, next(iter(cand.fields)), leaves)
    except ec.ConstructorResolutionError as exc:
        return Verdict("unprobeable", "identity", str(exc))
    cls = _pick_class(classes, leaves)
    if cls is None:
        return Verdict("unprobeable", "identity", f"no constructor accepts fields {leaves}")

    snap_leaves = _distinct([subject_leaf, *cand.normalised])
    snapshots: list[tuple[str, ...]] = []
    for value in values:
        kwargs = dict(base)
        if value is _DEFAULT:
            kwargs.pop(subject_leaf, None)
        else:
            kwargs[subject_leaf] = value
        try:
            instance = ec.construct(engine, cls, kwargs, validate=False)
        except Exception as exc:
            return Verdict(
                "infra_error",
                "identity",
                f"constructing {subject_leaf}={_show(value)} raised {type(exc).__name__}: {exc}"[
                    :200
                ],
            )
        snapshots.append(tuple(_stable(ec.resolved_value(instance, leaf)) for leaf in snap_leaves))

    shown = [_show(v) for v in values]
    if len(set(snapshots)) == 1:
        return Verdict(
            "confirmed",
            "identity",
            f"{subject_leaf} over {shown} -> resolved {list(snapshots[0])} identical",
        )
    return Verdict(
        "refuted", "identity", f"{subject_leaf} over {shown} -> resolved states differ {snapshots}"
    )


# ---------------------------------------------------------------------------
# Dispatch + candidate parsing
# ---------------------------------------------------------------------------


def probe_candidate(cand: Candidate, engine: str, engine_ready: bool) -> Verdict:
    tier = (
        "construction"
        if cand.severity == "error"
        else "identity"
        if cand.severity == "dormant"
        else None
    )
    if cand.severity not in ("error", "dormant"):
        return Verdict(
            "unprobeable",
            tier,
            f"severity {cand.severity!r} is not in the closed set {{error, dormant}}",
        )
    if not cand.fields:
        return Verdict("unprobeable", tier, "candidate has no match.fields to probe")
    if not engine_ready:
        return Verdict(
            "infra_error", tier, f"engine {engine!r} is not importable in this container"
        )
    if cand.severity == "error":
        return probe_construction(cand, engine)
    return probe_identity(cand, engine)


def parse_candidate(raw: Any, index: int) -> Candidate:
    """Lenient parse of a candidate/rule dict (probing tolerates thin entries)."""
    if not isinstance(raw, dict):
        raise Unprobeable(f"entry #{index} is not a mapping")
    match = raw.get("match")
    fields = match.get("fields") if isinstance(match, dict) else None
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    return Candidate(
        id=str(raw.get("id") or f"#{index}"),
        engine=raw.get("engine"),
        severity=raw.get("severity"),
        fields=dict(fields) if isinstance(fields, dict) else {},
        normalised=list(raw.get("normalised_fields") or []),
        source=provenance.get("source"),
        declared_values=provenance.get("declared_values"),
        raw=raw,
    )


def load_document(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a candidates/rules YAML; return (document, raw-entry list).

    Accepts either a ``candidates`` (pool) or ``rules`` (shipped corpus) list.
    """
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    entries = document.get("candidates")
    if entries is None:
        entries = document.get("rules")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a 'candidates' or 'rules' list")
    return document, entries


# ---------------------------------------------------------------------------
# Rendering + writeback
# ---------------------------------------------------------------------------


def _distinct(seq: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in seq:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _show(value: Any) -> str:
    if value is _DEFAULT:
        return "<default>"
    if value is _OMIT:
        return "<omit>"
    return repr(value)


def _show_kwargs(kwargs: dict[str, Any]) -> str:
    return "{" + ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) + "}"


def _stable(value: Any) -> str:
    if value is ec.MISSING:
        return "<missing>"
    try:
        return repr(value)
    except Exception:
        return object.__repr__(value)


def write_verdicts(
    document: dict[str, Any],
    entries: list[dict[str, Any]],
    verdicts: list[Verdict],
    run_date: str,
    out_path: Path,
) -> None:
    for raw, verdict in zip(entries, verdicts, strict=True):
        raw["verdict"] = {
            "status": verdict.status,
            "tier": verdict.tier,
            "date": run_date,
            "evidence": verdict.evidence,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False))


def print_summary(rows: list[tuple[str, Verdict]], stats: RunStats) -> None:
    id_width = max((len(cid) for cid, _ in rows), default=2)
    id_width = min(max(id_width, 2), 60)
    print(f"{'id':<{id_width}}  {'tier':<12}  {'verdict':<12}  evidence")
    print(f"{'-' * id_width}  {'-' * 12}  {'-' * 12}  {'-' * 40}")
    for cid, verdict in rows:
        evidence = (
            verdict.evidence if len(verdict.evidence) <= 100 else verdict.evidence[:97] + "..."
        )
        print(f"{cid:<{id_width}}  {verdict.tier or '-':<12}  {verdict.status:<12}  {evidence}")
    tally = ", ".join(f"{status}={count}" for status, count in sorted(stats.counts.items()))
    print(f"\n{len(rows)} candidate(s): {tally}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Construction/identity probe kernel for engine-rule candidates."
    )
    parser.add_argument(
        "--engine", required=True, choices=_ENGINES, help="Engine whose container this runs in."
    )
    parser.add_argument(
        "--candidates", required=True, nargs="+", type=Path, help="Candidate/rule YAML file(s)."
    )
    parser.add_argument(
        "--out", type=Path, help="Write verdicts here (default: in place). One input only."
    )
    parser.add_argument(
        "--date",
        help="Verdict date (default: $LLEM_PROBE_DATE or today, for reproducible writeback).",
    )
    args = parser.parse_args(argv)

    if args.out and len(args.candidates) > 1:
        parser.error("--out accepts a single --candidates file; run per file otherwise")
    run_date = args.date or os.environ.get("LLEM_PROBE_DATE") or date.today().isoformat()

    engine_ready = ec.engine_importable(args.engine)
    all_rows: list[tuple[str, Verdict]] = []
    stats = RunStats()
    for candidates_path in args.candidates:
        try:
            document, entries = load_document(candidates_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        verdicts: list[Verdict] = []
        for index, raw in enumerate(entries):
            try:
                cand = parse_candidate(raw, index)
            except Unprobeable as exc:
                verdicts.append(Verdict("unprobeable", None, str(exc)))
                continue
            verdict = probe_candidate(cand, args.engine, engine_ready)
            verdicts.append(verdict)
            all_rows.append((cand.id, verdict))
        for verdict in verdicts:
            stats.record(verdict.status)
        write_verdicts(document, entries, verdicts, run_date, args.out or candidates_path)

    print_summary(all_rows, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
