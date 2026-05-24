#!/usr/bin/env python3
"""Semantic differ for discovered engine schema JSONs.

Classifies changes between two schema versions as safe or breaking.

Usage:
    python scripts/diff_discovered_schemas.py old.json new.json

Exit codes:
    0 = identical or safe changes only
    1 = breaking changes detected
    2 = error (malformed input, missing file, etc.)

stdout: JSON with structured diff
stderr: human-readable summary
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Param sections to diff (metadata fields like discovered_at are excluded
# implicitly by only diffing these sections).
_PARAM_SECTIONS = ("engine_params", "sampling_params")


@dataclass
class Change:
    section: str
    field: str
    kind: str  # added, removed, type_changed, default_changed, description_changed
    severity: str  # safe or breaking
    detail: str = ""


@dataclass
class DiffResult:
    safe: list[Change] = field(default_factory=list)
    breaking: list[Change] = field(default_factory=list)
    metadata_changes: dict[str, dict[str, str]] = field(default_factory=dict)
    is_breaking: bool = False
    summary: str = ""


def _type_atoms(type_repr: object) -> set[str]:
    """Extract the set of primitive type atoms from any canonical or legacy ``type`` repr.

    Handles four shapes:
    - Legacy v1 compact string: ``"int | None"`` (split on ``|``).
    - Canonical v2 primitive: ``"string"`` (single atom).
    - Canonical v2 type-array: ``["string", "null"]``.
    - Canonical v2 anyOf branches: handled separately via :func:`_anyof_atoms`.
    """
    if isinstance(type_repr, str):
        return {t.strip() for t in type_repr.split("|") if t.strip()}
    if isinstance(type_repr, list):
        return {str(t) for t in type_repr}
    return set()


def _anyof_atoms(spec: dict) -> set[str]:
    """Extract the set of type atoms from a canonical ``anyOf`` branch list."""
    branches = spec.get("anyOf") if isinstance(spec, dict) else None
    if not isinstance(branches, list):
        return set()
    atoms: set[str] = set()
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        t = branch.get("type")
        if isinstance(t, str):
            atoms.add(t)
        elif isinstance(t, list):
            atoms.update(str(x) for x in t)
    return atoms


def _spec_atoms(spec: dict) -> set[str]:
    """Return the canonical type-atom set for a field spec across legacy + v2 shapes."""
    if not isinstance(spec, dict):
        return set()
    type_atoms = _type_atoms(spec.get("type"))
    if type_atoms:
        return type_atoms
    return _anyof_atoms(spec)


def _is_type_narrowed(old_spec: dict, new_spec: dict) -> bool:
    """Return True if new spec's atom set is a strict subset of old."""
    old_atoms = _spec_atoms(old_spec)
    new_atoms = _spec_atoms(new_spec)
    return bool(old_atoms) and bool(new_atoms) and new_atoms < old_atoms


def _is_type_widened(old_spec: dict, new_spec: dict) -> bool:
    """Return True if new spec's atom set is a strict superset of old."""
    old_atoms = _spec_atoms(old_spec)
    new_atoms = _spec_atoms(new_spec)
    return bool(old_atoms) and bool(new_atoms) and old_atoms < new_atoms


def _format_type_repr(spec: dict) -> str:
    """Render a field-spec ``type`` for human-readable diff output."""
    if not isinstance(spec, dict):
        return ""
    if "anyOf" in spec:
        atoms = _anyof_atoms(spec)
        return " | ".join(sorted(atoms)) if atoms else "anyOf"
    t = spec.get("type")
    if isinstance(t, list):
        return " | ".join(str(x) for x in t)
    return str(t) if t is not None else ""


def diff_schemas(old: dict, new: dict) -> DiffResult:
    """Compare two schema dicts and classify changes."""
    result = DiffResult()

    # Check metadata changes.
    for key in ("engine_version", "schema_version"):
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            result.metadata_changes[key] = {"old": str(old_val), "new": str(new_val)}

    # Diff param sections.
    for section in _PARAM_SECTIONS:
        old_params = old.get(section, {})
        new_params = new.get(section, {})

        old_keys = set(old_params.keys())
        new_keys = set(new_params.keys())

        # Added fields.
        for key in sorted(new_keys - old_keys):
            change = Change(section=section, field=key, kind="added", severity="safe")
            result.safe.append(change)

        # Removed fields.
        for key in sorted(old_keys - new_keys):
            change = Change(section=section, field=key, kind="removed", severity="breaking")
            result.breaking.append(change)

        # Changed fields.
        for key in sorted(old_keys & new_keys):
            old_field = old_params[key]
            new_field = new_params[key]

            # Compare on type-atom sets so the differ stays correct across
            # the v1 compact-string shape and the v2 canonical JSON Schema
            # shape (type-array or anyOf branches). The textual ``type``
            # field can change shape without semantic change (e.g. v1
            # ``"int | None"`` -> v2 ``["integer", "null"]``); those are
            # caught by the equality fallback below.
            old_type_repr = _format_type_repr(old_field)
            new_type_repr = _format_type_repr(new_field)
            shape_changed = old_field.get("type") != new_field.get("type") or old_field.get(
                "anyOf"
            ) != new_field.get("anyOf")

            if shape_changed:
                if _is_type_narrowed(old_field, new_field):
                    change = Change(
                        section=section,
                        field=key,
                        kind="type_narrowed",
                        severity="breaking",
                        detail=f"{old_type_repr} -> {new_type_repr}",
                    )
                    result.breaking.append(change)
                elif _is_type_widened(old_field, new_field):
                    change = Change(
                        section=section,
                        field=key,
                        kind="type_widened",
                        severity="safe",
                        detail=f"{old_type_repr} -> {new_type_repr}",
                    )
                    result.safe.append(change)
                elif _spec_atoms(old_field) != _spec_atoms(new_field):
                    change = Change(
                        section=section,
                        field=key,
                        kind="type_changed",
                        severity="breaking",
                        detail=f"{old_type_repr} -> {new_type_repr}",
                    )
                    result.breaking.append(change)

            old_default = old_field.get("default")
            new_default = new_field.get("default")
            if old_default != new_default:
                change = Change(
                    section=section,
                    field=key,
                    kind="default_changed",
                    severity="safe",
                    detail=f"{old_default!r} -> {new_default!r}",
                )
                result.safe.append(change)

            old_desc = old_field.get("description", "")
            new_desc = new_field.get("description", "")
            if old_desc != new_desc and old_desc and new_desc:
                change = Change(
                    section=section,
                    field=key,
                    kind="description_changed",
                    severity="safe",
                )
                result.safe.append(change)

    result.is_breaking = len(result.breaking) > 0

    # Build summary.
    parts = []
    if result.metadata_changes:
        for k, v in result.metadata_changes.items():
            parts.append(f"{k}: {v['old']} -> {v['new']}")
    if result.safe:
        parts.append(f"{len(result.safe)} safe change(s)")
    if result.breaking:
        parts.append(f"{len(result.breaking)} BREAKING change(s)")
    if not parts:
        parts.append("No changes")
    result.summary = "; ".join(parts)

    return result


def _change_to_dict(c: Change) -> dict:
    d = {"section": c.section, "field": c.field, "kind": c.kind}
    if c.detail:
        d["detail"] = c.detail
    return d


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: diff_discovered_schemas.py <old.json> <new.json>", file=sys.stderr)
        return 2

    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])

    try:
        old = json.loads(old_path.read_text())
        new = json.loads(new_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading schemas: {e}", file=sys.stderr)
        return 2

    result = diff_schemas(old, new)

    # Structured output to stdout.
    output = {
        "safe": [_change_to_dict(c) for c in result.safe],
        "breaking": [_change_to_dict(c) for c in result.breaking],
        "metadata_changes": result.metadata_changes,
        "is_breaking": result.is_breaking,
        "summary": result.summary,
    }
    print(json.dumps(output, indent=2))

    # Human-readable to stderr.
    print(f"\n{result.summary}", file=sys.stderr)
    if result.breaking:
        print("\nBreaking changes:", file=sys.stderr)
        for c in result.breaking:
            detail = f" ({c.detail})" if c.detail else ""
            print(f"  - [{c.section}] {c.field}: {c.kind}{detail}", file=sys.stderr)

    return 1 if result.is_breaking else 0


if __name__ == "__main__":
    sys.exit(main())
