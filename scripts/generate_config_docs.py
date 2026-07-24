#!/usr/bin/env python3
"""Generate configuration reference documentation from ExperimentConfig JSON schema.

Uses Pydantic model_json_schema() to extract the full schema, then renders
a structured Markdown reference grouped by section.

Usage:
    python scripts/generate_config_docs.py
    python scripts/generate_config_docs.py --output docs/reference/study-config.md

Output:
    Markdown to stdout (or --output path). Suitable for inlining into docs/reference/study-config.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from llenergymeasure.config._doc_helpers import default_label, type_label
from scripts._docgen_common import add_output_option, emit_markdown

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _description(prop: dict[str, Any]) -> str:
    return prop.get("description", "")


# Anything that is not a lowercase word char, hyphen, or space. github-slugger
# (which Docusaurus uses to derive heading anchors) removes exactly this
# punctuation from an ASCII heading and keeps [a-z0-9_-]; spaces map to hyphens.
_SLUGGER_DROP = re.compile(r"[^a-z0-9_ -]")


def _heading_anchor(title: str) -> str:
    """Slugify a heading the way Docusaurus (github-slugger) does.

    Docusaurus derives heading anchors with github-slugger: lowercase, drop
    punctuation, then map spaces to hyphens. Punctuation is REMOVED,
    not hyphenated - so a code span like ``transformers.engine_params`` yields
    ``transformersengine_params``, not ``transformers-engine_params``. The table
    of contents must slugify with the same rule or its in-page links resolve to
    anchors that do not exist.
    """
    return _SLUGGER_DROP.sub("", title.lower()).replace(" ", "-")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

_SECTION_ORDER = [
    ("top-level", "Top-Level Fields"),
    ("warmup", "Warmup (`warmup:`)"),
    ("baseline", "Baseline (`baseline:`)"),
    ("transformers", "Transformers Engine (`transformers:`)"),
    ("transformers_engine_params", "Transformers Engine Params (`transformers.engine_params:`)"),
    (
        "transformers_sampling_params",
        "Transformers Sampling Params (`transformers.sampling_params:`)",
    ),
    (
        "transformers_llem_execution",
        "Transformers Execution Knobs (`transformers.llem_execution:`)",
    ),
    ("vllm", "vLLM Engine (`vllm:`)"),
    ("vllm_engine_params", "vLLM Engine Params (`vllm.engine_params:`)"),
    ("vllm_sampling_params", "vLLM Sampling Params (`vllm.sampling_params:`)"),
    ("tensorrt", "TensorRT-LLM Engine (`tensorrt:`)"),
    ("tensorrt_engine_params", "TensorRT-LLM Engine Params (`tensorrt.engine_params:`)"),
    ("tensorrt_sampling_params", "TensorRT-LLM Sampling Params (`tensorrt.sampling_params:`)"),
]

# Map from JSON schema $defs key to our section key. The per-engine Config
# models are code-generated, so Pydantic emits module-qualified $def names
# (``llenergymeasure__config__generated__<engine>__<Model>``).
_DEF_TO_SECTION: dict[str, str] = {
    "WarmupConfig": "warmup",
    "BaselineConfig": "baseline",
    # TransformersSection / TransformersLlemExecution are hand-written (unique names),
    # so pydantic emits them as simple $def keys, not module-qualified ones.
    "TransformersSection": "transformers",
    "llenergymeasure__config__generated__transformers__EngineParams": (
        "transformers_engine_params"
    ),
    "llenergymeasure__config__generated__transformers__SamplingParams": (
        "transformers_sampling_params"
    ),
    "TransformersLlemExecution": "transformers_llem_execution",
    "llenergymeasure__config__generated__vllm__Config": "vllm",
    "llenergymeasure__config__generated__vllm__EngineParams": "vllm_engine_params",
    "llenergymeasure__config__generated__vllm__SamplingParams": "vllm_sampling_params",
    "llenergymeasure__config__generated__tensorrt__Config": "tensorrt",
    "llenergymeasure__config__generated__tensorrt__EngineParams": "tensorrt_engine_params",
    "llenergymeasure__config__generated__tensorrt__SamplingParams": "tensorrt_sampling_params",
}


def _ref_display_name(ref: str) -> str:
    """Human-readable name for a ``$ref`` target.

    Code-generated engine models carry module-qualified ``$def`` names
    (``llenergymeasure__config__generated__vllm__EngineParams``); show only the
    final class segment in the rendered type column.
    """
    return ref.split("/")[-1].split("__")[-1]


def _render_table(props: dict[str, Any], defs: dict[str, Any]) -> list[str]:
    lines = [
        "| Field | Type | Default | Description |",
        "|-------|------|---------|-------------|",
    ]
    for name, prop in props.items():
        # Resolve $ref to get actual property info
        if "$ref" in prop:
            section_name = _ref_display_name(prop["$ref"])
            # Use field-level description (from ExperimentConfig.Field) not class docstring
            desc = _description(prop)
            lines.append(f"| `{name}` | {section_name} | *(see section)* | {desc} |")
            continue

        # anyOf with $ref (Optional[SubModel])
        any_of = prop.get("anyOf") or []
        ref_in_anyof = next((p for p in any_of if "$ref" in p), None)
        if ref_in_anyof:
            section_name = _ref_display_name(ref_in_anyof["$ref"])
            # Use field-level description (from ExperimentConfig.Field) not class docstring
            desc = _description(prop)
            has_null = any(p.get("type") == "null" for p in any_of)
            type_str = f"{section_name} | None" if has_null else section_name
            default = default_label(prop)
            lines.append(f"| `{name}` | {type_str} | {default} | {desc} |")
            continue

        type_str = type_label(prop, defs)
        default = default_label(prop)
        desc = _description(prop)
        lines.append(f"| `{name}` | {type_str} | {default} | {desc} |")
    return lines


def render_markdown(schema: dict[str, Any]) -> str:
    defs = schema.get("$defs", {})
    top_props = schema.get("properties", {})

    # Build section content
    sections: dict[str, list[str]] = {}

    # Top-level fields
    sections["top-level"] = _render_table(top_props, defs)

    # Sub-model sections from $defs
    for def_name, section_key in _DEF_TO_SECTION.items():
        if def_name in defs:
            def_schema = defs[def_name]
            props = def_schema.get("properties", {})
            if props:
                sections[section_key] = _render_table(props, defs)

    # Render output
    lines: list[str] = [
        "<!-- Auto-generated by scripts/generate_config_docs.py -- do not edit manually -->",
        "",
        "## Configuration Reference",
        "",
        "Full reference for all `ExperimentConfig` fields.",
        "All fields except `model` are optional and have sensible defaults.",
        "",
    ]

    # Table of contents
    lines.append("**Sections:**")
    for section_key, section_title in _SECTION_ORDER:
        if section_key in sections:
            anchor = _heading_anchor(section_title)
            lines.append(f"- [{section_title}](#{anchor})")
    lines.append("")

    # Sections
    for section_key, section_title in _SECTION_ORDER:
        if section_key not in sections:
            continue
        lines.append(f"### {section_title}")
        lines.append("")
        lines.extend(sections[section_key])
        lines.append("")

    lines.extend(_sweep_axis_notation_section())

    return "\n".join(lines)


def _sweep_axis_notation_section() -> list[str]:
    """Prose reference for the numeric sweep-axis range shorthands.

    A sweep axis value is normally an explicit YAML list. Three mapping
    shorthands are accepted as compact notation for such a list; they expand at
    load time, so downstream consumers only ever see plain lists. This is
    hand-authored (not schema-derived) because the shorthands are parsed in
    ``config/sweep_idioms.py``, not on any ``ExperimentConfig`` field.
    """
    return [
        "### Sweep-Axis Range Shorthands",
        "",
        "An independent sweep axis (`sweep:` entry mapping to a list of scalars) may be",
        "written as one of three compact range shorthands instead of an explicit list.",
        "Each expands to a plain list at load time, so the two forms are interchangeable.",
        "",
        "| Shorthand | Meaning | Example | Expands to |",
        "|-----------|---------|---------|------------|",
        "| `{min: a, max: b, num: n}` | `n` evenly spaced values, endpoints inclusive | "
        "`{min: 0, max: 8, num: 5}` | `[0, 2, 4, 6, 8]` |",
        "| `{log: {min: a, max: b, num: n}}` | `n` log-spaced values (`min > 0`), endpoints "
        "inclusive | `{log: {min: 1, max: 100, num: 3}}` | `[1, 10, 100]` |",
        "| `{pow2: {min: a, max: b}}` | ascending powers of two within `[a, b]` | "
        "`{pow2: {min: 4, max: 32}}` | `[4, 8, 16, 32]` |",
        "",
        "Values stay integers when all bounds are integers and every produced value is",
        "integral; otherwise they are floats (rounded to kill binary-float noise).",
        "A mapping that matches none of these shapes is rejected at load time. To sweep a",
        "literal mapping value, set it in the base config or use a group entry (list of",
        "dicts).",
        "",
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = add_output_option(
        argparse.ArgumentParser(description="Generate config reference Markdown")
    )
    args = parser.parse_args()

    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    emit_markdown(render_markdown(schema), args.output)


if __name__ == "__main__":
    main()
