#!/usr/bin/env python3
"""Generate docs/reference/engines/invalid-combos.md from config introspection SSOT.

Pure renderer: all data comes from llenergymeasure.config.introspection.
No static lists maintained in this script.

Run: python scripts/generate_invalid_combos_doc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llenergymeasure.config.introspection import (
    get_capability_matrix_markdown,
    get_runtime_limitations,
    get_validation_rules,
)


def generate_markdown() -> str:
    """Generate the invalid combinations markdown document."""
    lines = [
        "# Invalid Parameter Combinations",
        "",
        "> Auto-generated from config validators and test results.",
        "",
        "This document lists parameter combinations that will fail validation or runtime.",
        "The tool validates these at config load time and provides clear error messages.",
        "The full verified rule set per engine lives at",
        "`src/llenergymeasure/engines/<engine>/rules.yaml`.",
        "",
        "## Config Validation Errors",
        "",
        "These combinations are rejected at config load time with a clear error message.",
        "",
        "| Engine | Invalid Combination | Reason | Resolution |",
        "|---------|---------------------|--------|------------|",
    ]

    for invariant in get_validation_rules():
        lines.append(
            f"| {invariant['engine']} | `{invariant['combination']}` | "
            f"{invariant['reason']} | {invariant['resolution']} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Limitations",
            "",
            "These combinations pass config validation but may fail at runtime",
            "due to hardware, model, or package requirements.",
            "",
            "| Engine | Parameter | Limitation | Resolution |",
            "|---------|-----------|------------|------------|",
        ]
    )

    for limitation in get_runtime_limitations():
        lines.append(
            f"| {limitation['engine']} | `{limitation['parameter']}` | "
            f"{limitation['limitation']} | {limitation['resolution']} |"
        )

    lines.extend(
        [
            "",
            "## Engine Capability Matrix",
            "",
            get_capability_matrix_markdown(),
            "",
            "## Recommended Configurations by Use Case",
            "",
            "### Memory-Constrained (Consumer GPU)",
            "```yaml",
            "engine: transformers",
            "transformers:",
            "  load_in_4bit: true",
            "  bnb_4bit_quant_type: nf4",
            "```",
            "",
            "### High Throughput (Production)",
            "```yaml",
            "engine: vllm",
            "vllm:",
            "  engine:",
            "    gpu_memory_utilization: 0.9",
            "    enable_prefix_caching: true",
            "```",
            "",
            "### Maximum Performance (Ampere+)",
            "```yaml",
            "engine: tensorrt",
            "tensorrt:",
            "  dtype: float16",
            "  quant_config:",
            "    quant_algo: FP8  # Hopper only",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Generate and write the invalid combos documentation."""
    output_path = (
        Path(__file__).parent.parent / "docs" / "reference" / "engines" / "invalid-combos.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = generate_markdown()
    output_path.write_text(content)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
