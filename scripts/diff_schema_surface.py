#!/usr/bin/env python3
"""Report what changed in an engine's config surface between two versions.

This is the bump-review aid the absorb workflow reaches for after a schema
re-mine: given two version snapshots of one engine, it names every field-level
change in the discovered configuration surface so a human can read the delta
instead of eyeballing two large JSON files.

It reads only ``schema.discovered.json`` (resolved through
``engine_versions/_outputs``); it imports no engine libraries and runs
anywhere. Per section (``engine_params``, ``sampling_params``) and per ``$defs``
class it reports:

- fields added / removed,
- type changes,
- bound changes (minimum / maximum / exclusiveMinimum / exclusiveMaximum),
- enum-membership changes (members added / removed),
- default changes.

Output is deterministic: every listing is sorted by name, and enum-membership
deltas preserve the source declaration order.

Usage::

    python scripts/diff_schema_surface.py --engine vllm --old 0.7.3 --new 0.19.1
    python scripts/diff_schema_surface.py --engine vllm --old 0.7.3 --new 0.19.1 --format json

Exit codes:
    0 - the two surfaces are identical
    1 - surface changes were found (the expected outcome of a real bump)
    2 - error (an unknown engine/version, or an unreadable/malformed snapshot)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine_versions import _outputs  # noqa: E402

#: Config-surface sections diffed field-by-field.
_SECTIONS = ("engine_params", "sampling_params")

#: Numeric-bound keys compared per field (JSON-Schema spelling, as discovered).
_BOUND_KEYS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")

#: Rendered stand-in for a field that declares no default at all (distinct from
#: a field whose default is ``None``).
_ABSENT = "<absent>"


class SurfaceDiffError(Exception):
    """A snapshot could not be resolved, read, or parsed."""


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldShape:
    """The comparable shape of one field, distilled from its raw schema entry."""

    type: str | None
    has_default: bool
    default: Any
    bounds: tuple[tuple[str, Any], ...]
    enum: tuple[Any, ...] | None


def _type_token(schema: dict[str, Any]) -> str | None:
    """A single comparable type string for a field or nested member.

    Prefers the explicit ``type`` (the top-level sections carry Python-style
    type strings); falls back to a ``$ref`` target or a rendered ``anyOf``
    union (the shapes ``$defs`` object properties use).
    """
    if "type" in schema:
        return str(schema["type"])
    if "$ref" in schema:
        return str(schema["$ref"])
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        members = [_type_token(m) or "?" for m in any_of if isinstance(m, dict)]
        if members:
            return " | ".join(members)
    return None


def _enum_members(schema: dict[str, Any]) -> tuple[Any, ...] | None:
    """The declared enum members of a field or class, or ``None`` if not an enum."""
    members = schema.get("enum")
    return tuple(members) if isinstance(members, list) else None


def _normalise(schema: dict[str, Any]) -> FieldShape:
    return FieldShape(
        type=_type_token(schema),
        has_default="default" in schema,
        default=schema.get("default"),
        bounds=tuple((key, schema[key]) for key in _BOUND_KEYS if key in schema),
        enum=_enum_members(schema),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _enum_delta(old: tuple[Any, ...] | None, new: tuple[Any, ...] | None) -> dict[str, list[Any]]:
    """Members added and removed between two enum member tuples.

    Uses value containment (not sets): enum members can be unhashable, e.g.
    vLLM's ``CUDAGraphMode`` carries list-valued members.
    """
    old_members = list(old or ())
    new_members = list(new or ())
    added = [member for member in new_members if member not in old_members]
    removed = [member for member in old_members if member not in new_members]
    return {"added": added, "removed": removed}


def _default_repr(shape: FieldShape) -> Any:
    return shape.default if shape.has_default else _ABSENT


def _diff_shapes(old: FieldShape, new: FieldShape) -> dict[str, Any]:
    """The set of dimensions in which one field changed (empty if unchanged)."""
    changes: dict[str, Any] = {}
    if old.type != new.type:
        changes["type"] = {"old": old.type, "new": new.type}
    if (old.has_default, old.default) != (new.has_default, new.default):
        changes["default"] = {"old": _default_repr(old), "new": _default_repr(new)}
    if old.bounds != new.bounds:
        changes["bounds"] = {"old": dict(old.bounds), "new": dict(new.bounds)}
    enum_delta = _enum_delta(old.enum, new.enum)
    if enum_delta["added"] or enum_delta["removed"]:
        changes["enum"] = enum_delta
    return changes


def _diff_field_map(old_map: dict[str, Any], new_map: dict[str, Any]) -> dict[str, Any]:
    """Diff a name -> field-schema map into added / removed / changed."""
    old_fields = {name: fs for name, fs in old_map.items() if isinstance(fs, dict)}
    new_fields = {name: fs for name, fs in new_map.items() if isinstance(fs, dict)}

    added = {
        name: _type_token(new_fields[name])
        for name in sorted(new_fields.keys() - old_fields.keys())
    }
    removed = {
        name: _type_token(old_fields[name])
        for name in sorted(old_fields.keys() - new_fields.keys())
    }
    changed: dict[str, Any] = {}
    for name in sorted(old_fields.keys() & new_fields.keys()):
        delta = _diff_shapes(_normalise(old_fields[name]), _normalise(new_fields[name]))
        if delta:
            changed[name] = delta
    return {"added": added, "removed": removed, "changed": changed}


def _diff_defs(old_defs: dict[str, Any], new_defs: dict[str, Any]) -> dict[str, Any]:
    """Diff the ``$defs`` class inventory: added / removed / changed classes.

    A common class is reported when its enum membership shifts (enum classes)
    or its ``properties`` map changes (object classes).
    """
    added = sorted(new_defs.keys() - old_defs.keys())
    removed = sorted(old_defs.keys() - new_defs.keys())
    changed: dict[str, Any] = {}
    for name in sorted(old_defs.keys() & new_defs.keys()):
        old_class = old_defs[name] if isinstance(old_defs[name], dict) else {}
        new_class = new_defs[name] if isinstance(new_defs[name], dict) else {}
        entry: dict[str, Any] = {}

        enum_delta = _enum_delta(_enum_members(old_class), _enum_members(new_class))
        if enum_delta["added"] or enum_delta["removed"]:
            entry["enum"] = enum_delta

        old_props = old_class.get("properties")
        new_props = new_class.get("properties")
        if isinstance(old_props, dict) or isinstance(new_props, dict):
            props_delta = _diff_field_map(
                old_props if isinstance(old_props, dict) else {},
                new_props if isinstance(new_props, dict) else {},
            )
            if props_delta["added"] or props_delta["removed"] or props_delta["changed"]:
                entry["properties"] = props_delta

        if entry:
            changed[name] = entry
    return {"added": added, "removed": removed, "changed": changed}


def _map_has_changes(diff: dict[str, Any]) -> bool:
    return bool(diff["added"] or diff["removed"] or diff["changed"])


def build_report(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> dict[str, Any]:
    """Build the structured surface diff of two discovered-schema dicts."""
    sections = {
        section: _diff_field_map(
            old_schema.get(section, {}) or {}, new_schema.get(section, {}) or {}
        )
        for section in _SECTIONS
    }
    defs = _diff_defs(old_schema.get("$defs", {}) or {}, new_schema.get("$defs", {}) or {})
    changed = any(_map_has_changes(diff) for diff in sections.values()) or _map_has_changes(defs)
    return {
        "engine": new_schema.get("engine") or old_schema.get("engine"),
        "old_version": old_schema.get("engine_version"),
        "new_version": new_schema.get("engine_version"),
        "changed": changed,
        "sections": sections,
        "defs": defs,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_field_changes(name: str, changes: dict[str, Any], indent: str) -> list[str]:
    lines = [f"{indent}~ {name}"]
    if "type" in changes:
        lines.append(f"{indent}    type:    {changes['type']['old']} -> {changes['type']['new']}")
    if "default" in changes:
        lines.append(
            f"{indent}    default: {changes['default']['old']!r} -> {changes['default']['new']!r}"
        )
    if "bounds" in changes:
        lines.append(
            f"{indent}    bounds:  {changes['bounds']['old']} -> {changes['bounds']['new']}"
        )
    if "enum" in changes:
        enum = changes["enum"]
        parts = []
        if enum["added"]:
            parts.append(f"+{enum['added']}")
        if enum["removed"]:
            parts.append(f"-{enum['removed']}")
        lines.append(f"{indent}    enum:    {' '.join(parts)}")
    return lines


def _render_field_map(diff: dict[str, Any], indent: str) -> list[str]:
    lines: list[str] = []
    for name, type_str in diff["added"].items():
        lines.append(f"{indent}+ {name}: {type_str}")
    for name in diff["removed"]:
        lines.append(f"{indent}- {name}")
    for name, changes in diff["changed"].items():
        lines.extend(_render_field_changes(name, changes, indent))
    return lines


def _counts(diff: dict[str, Any]) -> str:
    return f"+{len(diff['added'])} -{len(diff['removed'])} ~{len(diff['changed'])}"


def render_text(report: dict[str, Any]) -> str:
    engine = report["engine"] or "<unknown>"
    lines = [
        f"Schema surface diff [{engine}]: {report['old_version']} -> {report['new_version']}",
    ]
    if not report["changed"]:
        lines.append("")
        lines.append("No surface changes.")
        return "\n".join(lines)

    for section in _SECTIONS:
        diff = report["sections"][section]
        lines.append("")
        lines.append(f"{section}: {_counts(diff)}")
        lines.extend(_render_field_map(diff, "  "))

    defs = report["defs"]
    added, removed, changed = defs["added"], defs["removed"], defs["changed"]
    lines.append("")
    lines.append(f"$defs classes: +{len(added)} -{len(removed)} ~{len(changed)}")
    for name in added:
        lines.append(f"  + {name}")
    for name in removed:
        lines.append(f"  - {name}")
    for name, entry in changed.items():
        lines.append(f"  ~ {name}")
        if "enum" in entry:
            enum = entry["enum"]
            parts = []
            if enum["added"]:
                parts.append(f"+{enum['added']}")
            if enum["removed"]:
                parts.append(f"-{enum['removed']}")
            lines.append(f"      enum: {' '.join(parts)}")
        if "properties" in entry:
            lines.append(f"      properties: {_counts(entry['properties'])}")
            lines.extend(_render_field_map(entry["properties"], "        "))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading + CLI
# ---------------------------------------------------------------------------


def _load_schema(engine: str, version: str) -> dict[str, Any]:
    path = _outputs.schema_path(engine, version)
    if not path.is_file():
        available = _outputs.workspace_versions(engine)
        raise SurfaceDiffError(
            f"no discovered schema for {engine} {version} at {path}; "
            f"workspace has: {', '.join(available) or '(none)'}"
        )
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SurfaceDiffError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SurfaceDiffError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=_outputs.ENGINES)
    parser.add_argument("--old", required=True, metavar="VERSION", help="Baseline version.")
    parser.add_argument(
        "--new", required=True, metavar="VERSION", help="Version compared against it."
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    try:
        old_schema = _load_schema(args.engine, args.old)
        new_schema = _load_schema(args.engine, args.new)
    except SurfaceDiffError as exc:
        print(f"schema-surface-diff error: {exc}", file=sys.stderr)
        return 2

    report = build_report(old_schema, new_schema)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))

    return 1 if report["changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
