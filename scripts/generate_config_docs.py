#!/usr/bin/env python3
"""Generate configuration reference documentation from Pydantic models.

This script extracts parameter information from the ExperimentConfig Pydantic model
and generates markdown documentation. Run this script to regenerate docs after
changing config models.

Usage:
    python scripts/generate_config_docs.py

Output:
    docs/generated/config-reference.md
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pydantic import BaseModel  # noqa: E402


def get_type_str(annotation: Any) -> str:
    """Convert a type annotation to a readable string.

    Args:
        annotation: Type annotation from model field.

    Returns:
        Human-readable type string.
    """
    if annotation is None:
        return "Any"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional (Union with None)
    if origin is type(None):
        return "None"

    if origin in (list, set, tuple):
        if args:
            inner = ", ".join(get_type_str(a) for a in args)
            return f"{origin.__name__}[{inner}]"
        return origin.__name__

    if origin is dict:
        if args and len(args) >= 2:
            return f"dict[{get_type_str(args[0])}, {get_type_str(args[1])}]"
        return "dict"

    # Handle Union types (including Optional which is Union[T, None])
    if hasattr(origin, "__name__") and origin.__name__ == "Union":
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            # This is Optional[T]
            return f"{get_type_str(non_none[0])} | None"
        return " | ".join(get_type_str(a) for a in args)

    # Handle Literal
    if hasattr(origin, "__name__") and origin.__name__ == "Literal":
        return f"Literal[{', '.join(repr(a) for a in args)}]"

    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def extract_field_info(model: type[BaseModel], prefix: str = "") -> list[dict[str, Any]]:
    """Extract field information from a Pydantic model recursively.

    Args:
        model: Pydantic model class.
        prefix: Path prefix for nested fields.

    Returns:
        List of field info dictionaries.
    """
    fields = []

    for field_name, field_info in model.model_fields.items():
        path = f"{prefix}{field_name}" if prefix else field_name
        annotation = model.__annotations__.get(field_name)

        # Check if this is a nested model
        is_nested = (
            annotation is not None
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        )

        field_data = {
            "path": path,
            "name": field_name,
            "type": get_type_str(annotation),
            "default": _format_default(field_info.default),
            "description": field_info.description or "",
            "required": field_info.is_required(),
            "is_nested": is_nested,
        }

        fields.append(field_data)

        # Recursively extract nested model fields
        if is_nested:
            nested_fields = extract_field_info(annotation, prefix=f"{path}.")
            fields.extend(nested_fields)

    return fields


def _format_default(default: Any) -> str:
    """Format a default value for display."""
    if default is None:
        return "None"
    if callable(default):
        # Handle factory defaults
        try:
            val = default()
            if isinstance(val, list | dict) and not val:
                return f"{type(val).__name__}()"
            return repr(val)
        except Exception:
            return "<factory>"
    return repr(default)


def generate_markdown(fields: list[dict[str, Any]], presets: dict[str, Any]) -> str:
    """Generate markdown documentation from field information.

    Args:
        fields: List of field info dictionaries.
        presets: Dictionary of preset configurations.

    Returns:
        Markdown string.
    """
    lines = [
        "# Configuration Reference",
        "",
        "> This file is auto-generated from Pydantic models. Do not edit manually.",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Table of Contents",
        "",
    ]

    # Group by top-level section
    sections: dict[str, list[dict[str, Any]]] = {}
    for field in fields:
        top_level = field["path"].split(".")[0]
        if top_level not in sections:
            sections[top_level] = []
        sections[top_level].append(field)

    # Generate TOC
    for section in sorted(sections.keys()):
        lines.append(f"- [{section}](#{section.lower().replace('_', '-')})")

    lines.extend(["", "---", ""])

    # Generate sections
    for section in sorted(sections.keys()):
        section_fields = sections[section]
        lines.append(f"## {section}")
        lines.append("")

        # Create table
        lines.append("| Parameter | Type | Default | Description |")
        lines.append("|-----------|------|---------|-------------|")

        for field in section_fields:
            path = field["path"]
            type_str = field["type"].replace("|", "\\|")
            default = field["default"].replace("|", "\\|")
            desc = field["description"].replace("|", "\\|")
            required = " *(required)*" if field["required"] else ""

            lines.append(f"| `{path}` | {type_str} | {default}{required} | {desc} |")

        lines.append("")

    # Add presets section
    lines.extend(
        [
            "---",
            "",
            "## Built-in Presets",
            "",
            "Presets provide convenient defaults for common use cases.",
            "",
        ]
    )

    for preset_name, preset_config in sorted(presets.items()):
        lines.append(f"### {preset_name}")
        lines.append("")
        lines.append("```yaml")

        for key, value in preset_config.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for kk, vv in v.items():
                            lines.append(f"    {kk}: {_yaml_value(vv)}")
                    else:
                        lines.append(f"  {k}: {_yaml_value(v)}")
            else:
                lines.append(f"{key}: {_yaml_value(value)}")

        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _yaml_value(value: Any) -> str:
    """Format a value for YAML display."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return repr(value)
    return str(value)


def main() -> None:
    """Generate configuration documentation."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.constants import PRESETS

    print("Extracting field information from ExperimentConfig...")
    fields = extract_field_info(ExperimentConfig)
    print(f"  Found {len(fields)} parameters")

    print("Generating markdown...")
    markdown = generate_markdown(fields, PRESETS)

    output_dir = project_root / "docs" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "config-reference.md"
    output_path.write_text(markdown)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
