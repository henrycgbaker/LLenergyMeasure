#!/usr/bin/env python3
"""Generate the discovered-schema doc for one engine.

Produces ``docs/reference/engines/schema-{engine}.md`` from the introspector-mined
JSON at ``src/llenergymeasure/engines/<engine>/schema.discovered.json``. The output
is a per-engine reference of every parameter the engine exposes, with
type, default, description, and deprecation state, plus any limitations
the introspector flagged.

Run::

    python scripts/generate_schema_doc.py --engine transformers \\
        --out docs/reference/engines/schema-transformers.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.config.ssot import Engine  # noqa: E402
from scripts.engine_producers._common import DEFAULT_SCHEMA_FILENAME  # noqa: E402

_ENGINE_DISPLAY_NAMES: dict[str, str] = {
    "transformers": "Transformers",
    "vllm": "vLLM",
    "tensorrt": "TensorRT-LLM",
}


def _load_schema(engine: str) -> dict[str, Any]:
    """Read + parse the discovered-schema JSON for ``engine``.

    Returns ``{}`` for missing/empty schemas to degrade gracefully when
    discovery hasn't run yet, rather than failing the markdown render.
    """
    path = _PROJECT_ROOT / "src" / "llenergymeasure" / "engines" / engine / DEFAULT_SCHEMA_FILENAME
    if not path.is_file():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _escape_pipe(value: str) -> str:
    """Escape pipes for plain-text GFM table cells.

    Code-spans (backticked) handle pipes natively in GFM, so this is only
    needed for non-code cell content (e.g. descriptions).
    """
    return value.replace("|", "\\|")


def _format_default(value: Any) -> str:
    """Render a default value for display in a markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    if isinstance(value, str):
        return f"`{value}`" if value else '`""`'
    return f"`{json.dumps(value)}`"


def _format_field_name(name: str, deprecated: bool) -> str:
    """Render a field name with a deprecation warning prefix."""
    if deprecated:
        return f"⚠️ `{name}`"
    return f"`{name}`"


def _render_type_cell(meta: dict[str, Any]) -> str:
    """Render a canonical JSON Schema type spec as a compact human-readable string."""
    if "anyOf" in meta:
        parts: list[str] = []
        for sub in meta["anyOf"]:
            if not isinstance(sub, dict):
                continue
            parts.append(_render_type_cell(sub))
        return " \\| ".join(parts) if parts else "-"
    type_val = meta.get("type")
    if type_val is None:
        return "-"
    if isinstance(type_val, list):
        return " \\| ".join("null" if t is None else str(t) for t in type_val)
    if "enum" in meta:
        members = ", ".join(repr(v) for v in meta["enum"])
        return f"Literal[{members}]"
    if type_val == "object" and meta.get("description"):
        # Class-typed fields surface the class name on description.
        return str(meta["description"])
    return str(type_val)


def _render_table(params: dict[str, Any]) -> list[str]:
    """Render a parameter dict as a GFM table; returns list of lines."""
    if not params:
        return ["_No parameters discovered._", ""]
    lines = [
        "| Field | Type | Default | Description |",
        "|-------|------|---------|-------------|",
    ]
    for name, meta in params.items():
        if not isinstance(meta, dict):
            continue
        type_str = _render_type_cell(meta)
        default_str = _format_default(meta.get("default"))
        # ``description`` is reserved for the rendered class-name hint when
        # ``type == "object"``; otherwise (e.g. TRT-LLM Pydantic field docs)
        # it's the human description. Avoid double-printing when used as type.
        if meta.get("type") == "object" and meta.get("description") == type_str:
            description = ""
        else:
            description = str(meta.get("description") or "").strip()
        description = _escape_pipe(description.replace("\n", " "))
        deprecated = bool(meta.get("deprecated"))
        field = _format_field_name(name, deprecated)
        lines.append(f"| {field} | `{type_str}` | {default_str} | {description} |")
    lines.append("")
    return lines


def _render_limitations(limitations: list[Any]) -> list[str]:
    """Render the discovery_limitations list as a blockquote callout."""
    if not limitations:
        return []
    lines = [
        "> **Discovery limitations**",
        "> ",
    ]
    for entry in limitations:
        if not isinstance(entry, dict):
            continue
        section = str(entry.get("section", "?"))
        reason = str(entry.get("reason", "")).strip()
        fields = entry.get("fields") or []
        field_strs = [f"`{f}`" for f in fields if isinstance(f, str)]
        lines.append(f"> - **`{section}`** - {reason}")
        if field_strs:
            lines.append(f">   Affected fields: {', '.join(field_strs)}")
    lines.append("")
    return lines


def _render(engine: str) -> str:
    """Build the schema digest Markdown for ``engine``."""
    schema = _load_schema(engine)
    display = _ENGINE_DISPLAY_NAMES.get(engine, engine.title())
    engine_version = str(schema.get("engine_version", "<unknown>"))
    discovered_at = schema.get("discovered_at", "<unknown>")
    schema_version = schema.get("schema_version", "<unknown>")
    engine_params = schema.get("engine_params") or {}
    sampling_params = schema.get("sampling_params") or {}
    limitations = schema.get("discovery_limitations") or []

    lines: list[str] = [
        f"# {display} Engine Schema",
        "",
        "<!-- Auto-generated by scripts/generate_schema_doc.py -- do not edit manually -->",
        "",
        f"Engine version: **{engine_version}**  ",
        f"Discovered at: {discovered_at}  ",
        f"Schema version: {schema_version}",
        "",
        f"**Summary:** {len(engine_params)} engine parameters, "
        f"{len(sampling_params)} sampling parameters.",
        "",
    ]

    lines.extend(_render_limitations(limitations))

    lines.append("## Engine Parameters")
    lines.append("")
    lines.extend(_render_table(engine_params))

    lines.append("## Sampling Parameters")
    lines.append("")
    lines.extend(_render_table(sampling_params))

    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        required=True,
        choices=tuple(e.value for e in Engine),
        help="Engine name.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output Markdown path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    text = _render(args.engine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
