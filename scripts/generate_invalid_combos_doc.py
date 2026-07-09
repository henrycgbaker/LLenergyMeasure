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
    get_dormant_rules,
    get_runtime_limitations,
    get_validation_rules,
)


def generate_markdown() -> str:
    """Generate the invalid combinations markdown document."""
    lines = [
        "# Invalid Parameter Combinations",
        "",
        "> Auto-generated. Do not edit by hand: run",
        "> `python scripts/generate_invalid_combos_doc.py` (or `make docs-all`).",
        "",
        "This document lists parameter combinations that fail validation or run",
        "differently than declared. The error rules are enforced at config load",
        "time with a clear error message; the dormant rules are accepted but",
        "silently normalised by the engine. Both are derived from the live rule",
        "corpus (`src/llenergymeasure/engines/<engine>/rules.yaml`) plus the",
        "cross-engine `ExperimentConfig` validators, so this page cannot drift",
        "from what actually fires at runtime.",
        "",
        "## Config Validation Errors",
        "",
        "These combinations are rejected at config load time with a clear error",
        "message. The `all` rows are cross-engine `ExperimentConfig` validators;",
        "the rest come from each engine's shipped rule corpus.",
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
            "## Dormant Parameters",
            "",
            "These combinations pass validation, but the engine silently normalises",
            "or ignores the declared field: the declared value is not the effective",
            "value. The study planner deduplicates configs that differ only in a",
            "dormant field, so the GPU runs such a cell once. `Normalised fields`",
            "names the paths the engine drives back to their default.",
            "",
            "| Engine | Combination | Effect | Normalised fields |",
            "|---------|-------------|--------|-------------------|",
        ]
    )

    for dormant in get_dormant_rules():
        lines.append(
            f"| {dormant['engine']} | `{dormant['combination']}` | "
            f"{dormant['effect']} | {dormant['normalised_fields']} |"
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
