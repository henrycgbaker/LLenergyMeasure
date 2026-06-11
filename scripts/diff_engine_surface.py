#!/usr/bin/env python3
"""Surface-trend census over two engine pins' ``outputs/`` directories.

The blindness heuristic (design ``engine-update-workflow.md`` section 6,
signal 2). A pure artefact diff - no engine imports, runs anywhere - that
compares the committed mined artefacts of two pins and reports the coarse
shape of the knowledge surface: how many fields each schema section carries,
how many of them carry an enum or a bound, the ``$defs`` class inventory, and
how many invariants exist per provenance and per severity.

The signal it serves: shrinkage (the surface got smaller across a bump), or
flatness on a major bump (the surface barely moved when the engine churned
~2x), is a walker-regression flag - the deterministic substrate may have
silently stopped reading an idiom. The study's observed cliff class (the
substrate emitting fewer invariants after an idiom shift) fires this directly.

GRAIN DISCIPLINE (audit constraint, hard): metrics stay at encoding-agnostic
grain - field presence + coarse predicate bucket (does this field carry *an*
enum / *a* bound, yes/no). The study retracted its value-grain metric (which
enum members, which bound values) as encoding-variance-confounded: the same
constraint serialises differently across encodings, so value-grain deltas
measure encoding churn, not surface churn. This script never inspects enum
*members* or bound *values*, only their presence.

Inputs are two directories, each holding (any subset of)::

    schema.discovered.json        schema envelope (engine_params, sampling_params, $defs)
    rules.proposed.yaml      mined invariant candidates
    rules.validated.yaml     gate-confirmed corpus

Usage::

    python scripts/diff_engine_surface.py OLD_OUTPUTS_DIR NEW_OUTPUTS_DIR [--out report.json]

Exit codes:
    0 - report produced (this script never fails on "shrinkage"; it reports,
        the human decides - design section 6: no auto-blocking thresholds in v0.10.0)
    2 - error (a directory is missing, or an artefact is unparseable)

stdout : human-readable text summary
--out  : structured JSON report (stable shape for a later CI step)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_FILE = "schema.discovered.json"
_PROPOSED_FILE = "rules.proposed.yaml"
_VALIDATED_FILE = "rules.validated.yaml"

_SCHEMA_SECTIONS = ("engine_params", "sampling_params")

# Coarse-bucket predicate detection (encoding-agnostic). A field "carries an
# enum" if it declares an explicit ``enum`` list OR its ``type`` string encodes
# a Literal; it "carries a bound" if it declares any numeric-bound key. We look
# only at presence, never at the values.
_BOUND_KEYS = frozenset({"ge", "gt", "le", "lt", "minimum", "maximum", "bounds", "constraints"})


class SurfaceDiffError(Exception):
    """A pin directory is missing or an artefact could not be parsed."""


# ---------------------------------------------------------------------------
# Census dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SchemaSectionCensus:
    """Coarse counts for one schema section (engine_params / sampling_params)."""

    fields: int = 0
    enum_fields: int = 0
    bound_fields: int = 0


@dataclass
class PinCensus:
    """The full coarse census of one pin's outputs directory."""

    engine: str | None = None
    engine_version: str | None = None
    schema_sections: dict[str, SchemaSectionCensus] = field(default_factory=dict)
    defs_classes: list[str] = field(default_factory=list)
    proposed_by_severity: dict[str, int] = field(default_factory=dict)
    proposed_by_provenance: dict[str, int] = field(default_factory=dict)
    validated_cases: int = 0
    validated_by_outcome: dict[str, int] = field(default_factory=dict)


@dataclass
class CountDelta:
    old: int
    new: int

    @property
    def delta(self) -> int:
        return self.new - self.old

    def as_dict(self) -> dict[str, int]:
        return {"old": self.old, "new": self.new, "delta": self.delta}


# ---------------------------------------------------------------------------
# Field predicate detection (coarse bucket only)
# ---------------------------------------------------------------------------


def _field_carries_enum(field_schema: dict[str, Any]) -> bool:
    if isinstance(field_schema.get("enum"), list) and field_schema["enum"]:
        return True
    type_str = str(field_schema.get("type", ""))
    return "Literal[" in type_str


def _field_carries_bound(field_schema: dict[str, Any]) -> bool:
    return any(key in field_schema for key in _BOUND_KEYS)


# ---------------------------------------------------------------------------
# Per-artefact census
# ---------------------------------------------------------------------------


def census_schema(schema: dict[str, Any]) -> tuple[dict[str, SchemaSectionCensus], list[str]]:
    """Census the schema envelope: per-section coarse counts + $defs class names."""
    sections: dict[str, SchemaSectionCensus] = {}
    for section in _SCHEMA_SECTIONS:
        params = schema.get(section)
        if not isinstance(params, dict):
            continue
        census = SchemaSectionCensus(fields=len(params))
        for field_schema in params.values():
            if not isinstance(field_schema, dict):
                continue
            if _field_carries_enum(field_schema):
                census.enum_fields += 1
            if _field_carries_bound(field_schema):
                census.bound_fields += 1
        sections[section] = census

    defs = schema.get("$defs")
    defs_classes = sorted(defs.keys()) if isinstance(defs, dict) else []
    return sections, defs_classes


def _invariant_provenance(invariant: dict[str, Any]) -> str:
    """Coarse provenance label for one invariant.

    The corpus records provenance as ``miner_source.method`` (the engine method
    the walker read the rule from: ``validate``, ``__post_init__``,
    ``_verify_args``, ...). That method name is the stable, encoding-agnostic
    provenance bucket. Falls back to ``unknown`` when absent.
    """
    miner_source = invariant.get("miner_source")
    if isinstance(miner_source, dict):
        method = miner_source.get("method")
        if method:
            return str(method)
    return "unknown"


def census_proposed(corpus: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    """Census proposed invariants: counts per severity and per provenance bucket."""
    by_severity: dict[str, int] = {}
    by_provenance: dict[str, int] = {}
    for invariant in corpus.get("invariants", []):
        if not isinstance(invariant, dict):
            continue
        severity = str(invariant.get("severity", "unknown"))
        by_severity[severity] = by_severity.get(severity, 0) + 1
        provenance = _invariant_provenance(invariant)
        by_provenance[provenance] = by_provenance.get(provenance, 0) + 1
    return by_severity, by_provenance


def census_validated(envelope: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Census the validated envelope: case count and counts per observed outcome."""
    cases = envelope.get("cases", [])
    by_outcome: dict[str, int] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        outcome = str(case.get("outcome", "unknown"))
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
    return len(cases) if isinstance(cases, list) else 0, by_outcome


# ---------------------------------------------------------------------------
# Loading + directory census
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SurfaceDiffError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SurfaceDiffError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise SurfaceDiffError(f"could not read YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SurfaceDiffError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def census_pin(outputs_dir: Path) -> PinCensus:
    """Census every artefact present in one pin's outputs directory.

    Missing artefacts are tolerated (a pin may not have a validated corpus yet,
    e.g. vllm pre-#445); only a missing *directory* is an error.
    """
    if not outputs_dir.is_dir():
        raise SurfaceDiffError(f"not a directory: {outputs_dir}")

    census = PinCensus()

    schema_path = outputs_dir / _SCHEMA_FILE
    if schema_path.is_file():
        schema = _load_json(schema_path)
        census.engine = schema.get("engine")
        census.engine_version = schema.get("engine_version")
        census.schema_sections, census.defs_classes = census_schema(schema)

    proposed_path = outputs_dir / _PROPOSED_FILE
    if proposed_path.is_file():
        proposed = _load_yaml(proposed_path)
        census.engine = census.engine or proposed.get("engine")
        census.engine_version = census.engine_version or proposed.get("engine_version")
        census.proposed_by_severity, census.proposed_by_provenance = census_proposed(proposed)

    validated_path = outputs_dir / _VALIDATED_FILE
    if validated_path.is_file():
        validated = _load_yaml(validated_path)
        census.validated_cases, census.validated_by_outcome = census_validated(validated)

    return census


# ---------------------------------------------------------------------------
# Trend diff
# ---------------------------------------------------------------------------


def _bucket_delta(old: dict[str, int], new: dict[str, int]) -> dict[str, dict[str, int]]:
    """Per-bucket old/new/delta over the union of keys in two count maps."""
    return {
        key: CountDelta(old.get(key, 0), new.get(key, 0)).as_dict() for key in sorted(old | new)
    }


def diff_surface(old: PinCensus, new: PinCensus) -> dict[str, Any]:
    """Build the structured surface-trend report from two pin censuses."""
    schema_trend: dict[str, dict[str, dict[str, int]]] = {}
    for section in sorted(set(old.schema_sections) | set(new.schema_sections)):
        old_sec = old.schema_sections.get(section, SchemaSectionCensus())
        new_sec = new.schema_sections.get(section, SchemaSectionCensus())
        schema_trend[section] = {
            "fields": CountDelta(old_sec.fields, new_sec.fields).as_dict(),
            "enum_fields": CountDelta(old_sec.enum_fields, new_sec.enum_fields).as_dict(),
            "bound_fields": CountDelta(old_sec.bound_fields, new_sec.bound_fields).as_dict(),
        }

    old_defs = set(old.defs_classes)
    new_defs = set(new.defs_classes)
    defs_trend = {
        "old_count": len(old_defs),
        "new_count": len(new_defs),
        "added": sorted(new_defs - old_defs),
        "removed": sorted(old_defs - new_defs),
    }

    return {
        "engine": new.engine or old.engine,
        "old_version": old.engine_version,
        "new_version": new.engine_version,
        "schema": schema_trend,
        "defs_classes": defs_trend,
        "invariants_by_severity": _bucket_delta(old.proposed_by_severity, new.proposed_by_severity),
        "invariants_by_provenance": _bucket_delta(
            old.proposed_by_provenance, new.proposed_by_provenance
        ),
        "validated_cases": CountDelta(old.validated_cases, new.validated_cases).as_dict(),
        "validated_by_outcome": _bucket_delta(old.validated_by_outcome, new.validated_by_outcome),
        "shrinkage_flags": _shrinkage_flags(old, new),
    }


def _shrinkage_flags(old: PinCensus, new: PinCensus) -> list[str]:
    """Coarse regression flags: where the surface got smaller across the bump.

    Informational only (no thresholds, no blocking). The report leads with
    these so a reviewer sees the walker-regression candidates first.
    """
    flags: list[str] = []
    for section in sorted(set(old.schema_sections) | set(new.schema_sections)):
        old_sec = old.schema_sections.get(section, SchemaSectionCensus())
        new_sec = new.schema_sections.get(section, SchemaSectionCensus())
        if new_sec.fields < old_sec.fields:
            flags.append(f"schema.{section}.fields shrank {old_sec.fields} -> {new_sec.fields}")
        if new_sec.enum_fields < old_sec.enum_fields:
            flags.append(
                f"schema.{section}.enum_fields shrank "
                f"{old_sec.enum_fields} -> {new_sec.enum_fields}"
            )
        if new_sec.bound_fields < old_sec.bound_fields:
            flags.append(
                f"schema.{section}.bound_fields shrank "
                f"{old_sec.bound_fields} -> {new_sec.bound_fields}"
            )
    old_total = sum(old.proposed_by_severity.values())
    new_total = sum(new.proposed_by_severity.values())
    if new_total < old_total:
        flags.append(f"invariants total shrank {old_total} -> {new_total}")
    if len(new.defs_classes) < len(old.defs_classes):
        flags.append(f"$defs classes shrank {len(old.defs_classes)} -> {len(new.defs_classes)}")
    return flags


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _fmt_delta(d: dict[str, int]) -> str:
    sign = "+" if d["delta"] >= 0 else ""
    return f"{d['old']} -> {d['new']} ({sign}{d['delta']})"


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    engine = report.get("engine") or "<unknown>"
    lines.append(
        f"Surface trend [{engine}]: {report.get('old_version')} -> {report.get('new_version')}"
    )
    lines.append("")

    flags = report.get("shrinkage_flags") or []
    if flags:
        lines.append("Shrinkage flags (possible walker regression):")
        for flag in flags:
            lines.append(f"  ! {flag}")
    else:
        lines.append("Shrinkage flags: none")
    lines.append("")

    lines.append("Schema sections (field / enum-carrying / bound-carrying):")
    for section, trend in report.get("schema", {}).items():
        lines.append(f"  {section}:")
        lines.append(f"    fields:       {_fmt_delta(trend['fields'])}")
        lines.append(f"    enum_fields:  {_fmt_delta(trend['enum_fields'])}")
        lines.append(f"    bound_fields: {_fmt_delta(trend['bound_fields'])}")
    lines.append("")

    defs = report.get("defs_classes", {})
    lines.append(
        f"$defs classes: {defs.get('old_count', 0)} -> {defs.get('new_count', 0)} "
        f"(+{len(defs.get('added', []))} / -{len(defs.get('removed', []))})"
    )
    if defs.get("added"):
        lines.append(f"    added:   {', '.join(defs['added'])}")
    if defs.get("removed"):
        lines.append(f"    removed: {', '.join(defs['removed'])}")
    lines.append("")

    lines.append("Invariants by severity:")
    for bucket, d in report.get("invariants_by_severity", {}).items():
        lines.append(f"    {bucket}: {_fmt_delta(d)}")
    lines.append("Invariants by provenance (miner_source.method):")
    for bucket, d in report.get("invariants_by_provenance", {}).items():
        lines.append(f"    {bucket}: {_fmt_delta(d)}")
    lines.append("")
    lines.append(
        f"Validated cases: {_fmt_delta(report.get('validated_cases', {'old': 0, 'new': 0, 'delta': 0}))}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_report(old_dir: Path, new_dir: Path) -> dict[str, Any]:
    """Census both pins and return the structured surface-trend report."""
    return diff_surface(census_pin(old_dir), census_pin(new_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path, help="Old pin's outputs/ directory")
    parser.add_argument("new", type=Path, help="New pin's outputs/ directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If given, write the structured JSON report to this path.",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.old, args.new)
    except SurfaceDiffError as exc:
        print(f"surface-diff error: {exc}", file=sys.stderr)
        return 2

    print(render_text(report))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote JSON report {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
