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
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llenergymeasure.config._doc_helpers import default_label, type_label

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _description(prop: dict[str, Any]) -> str:
    return prop.get("description", "")


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
    ("vllm", "vLLM Engine (`vllm:`)"),
    ("vllm_engine_params", "vLLM Engine Params (`vllm.engine_params:`)"),
    ("vllm_sampling_params", "vLLM Sampling Params (`vllm.sampling_params:`)"),
    ("tensorrt", "TensorRT-LLM Engine (`tensorrt:`)"),
    ("tensorrt_engine_params", "TensorRT-LLM Engine Params (`tensorrt.engine_params:`)"),
    ("tensorrt_sampling_params", "TensorRT-LLM Sampling Params (`tensorrt.sampling_params:`)"),
    ("harness", "Harness Overrides (`harness:`)"),
    ("harness_transformers", "Transformers Harness (`harness.transformers:`)"),
]

# Map from JSON schema $defs key to our section key. The per-engine Config
# models are code-generated, so Pydantic emits module-qualified $def names
# (``llenergymeasure__engines__<engine>__config__<Model>``).
_DEF_TO_SECTION: dict[str, str] = {
    "WarmupConfig": "warmup",
    "BaselineConfig": "baseline",
    "llenergymeasure__engines__transformers__config__Config": "transformers",
    "llenergymeasure__engines__transformers__config__EngineParams": "transformers_engine_params",
    "llenergymeasure__engines__transformers__config__SamplingParams": (
        "transformers_sampling_params"
    ),
    "llenergymeasure__engines__vllm__config__Config": "vllm",
    "llenergymeasure__engines__vllm__config__EngineParams": "vllm_engine_params",
    "llenergymeasure__engines__vllm__config__SamplingParams": "vllm_sampling_params",
    "llenergymeasure__engines__tensorrt__config__Config": "tensorrt",
    "llenergymeasure__engines__tensorrt__config__EngineParams": "tensorrt_engine_params",
    "llenergymeasure__engines__tensorrt__config__SamplingParams": "tensorrt_sampling_params",
    "HarnessConfig": "harness",
    "TransformersHarness": "harness_transformers",
}


def _ref_display_name(ref: str) -> str:
    """Human-readable name for a ``$ref`` target.

    Code-generated engine models carry module-qualified ``$def`` names
    (``llenergymeasure__engines__vllm__config__EngineParams``); show only the
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
            anchor = section_title.lower()
            for ch in " /`.:()`":
                anchor = anchor.replace(ch, "-")
            while "--" in anchor:
                anchor = anchor.replace("--", "-")
            anchor = anchor.strip("-")
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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate config reference Markdown")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write output to this file path (default: stdout)",
    )
    args = parser.parse_args()

    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    markdown = render_markdown(schema)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
