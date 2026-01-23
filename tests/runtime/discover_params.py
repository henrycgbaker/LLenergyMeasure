#!/usr/bin/env python3
"""Auto-discover parameters from Pydantic models.

This script provides a CLI interface to the introspection SSOT module.
The actual parameter discovery logic lives in:
    src/llenergymeasure/config/introspection.py

This is a thin wrapper for backwards compatibility and CLI usage.

Usage:
    # List all discovered params (from project root)
    python -m tests.runtime.discover_params

    # Output as JSON (for CI/tooling)
    python -m tests.runtime.discover_params --format json

    # Generate test values override template
    python -m tests.runtime.discover_params --format overrides

    # Validate manual params match discovered
    python -m tests.runtime.discover_params --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path (tests/runtime/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# SSOT Delegation to config/introspection.py
# =============================================================================


def discover_params_from_model(
    model_class: type,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Extract all parameters from a Pydantic model with metadata.

    Delegates to the SSOT introspection module.
    """
    from llenergymeasure.config.introspection import get_params_from_model

    return get_params_from_model(model_class, prefix=prefix)


def discover_all_backends() -> dict[str, dict[str, dict[str, Any]]]:
    """Discover params for all backends using SSOT introspection."""
    from llenergymeasure.config.introspection import get_all_params

    all_params = get_all_params()
    # Return only backend sections (exclude 'shared')
    return {
        "pytorch": all_params["pytorch"],
        "vllm": all_params["vllm"],
        "tensorrt": all_params["tensorrt"],
    }


def discover_shared_params() -> dict[str, dict[str, Any]]:
    """Discover shared params using SSOT introspection."""
    from llenergymeasure.config.introspection import get_shared_params

    return get_shared_params()


def format_as_text(all_params: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Format discovered params as readable text."""
    lines = ["# Auto-Discovered Parameters", ""]
    lines.append("Source: src/llenergymeasure/config/introspection.py (SSOT)")
    lines.append("")

    for backend, params in sorted(all_params.items()):
        lines.append(f"## {backend.upper()} ({len(params)} params)")
        lines.append("")

        for param, meta in sorted(params.items()):
            # Support both 'type' and 'type_str' keys for compatibility
            type_str = meta.get("type_str") or meta.get("type", "unknown")
            if meta.get("options"):
                type_str = f"Literal{meta['options']}"

            default_str = (
                f"default={meta['default']}" if meta.get("default") is not None else "required"
            )
            lines.append(f"  {param}")
            lines.append(f"    Type: {type_str}, {default_str}")
            if meta.get("description"):
                lines.append(f"    Desc: {meta['description']}")
            if meta.get("test_values"):
                lines.append(f"    Test: {meta['test_values']}")
            lines.append("")

    return "\n".join(lines)


def format_as_json(all_params: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Format as JSON for tooling."""
    return json.dumps(all_params, indent=2, default=str)


def format_as_overrides(all_params: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Generate a test value overrides template."""
    lines = [
        "# Test Value Overrides",
        "# Copy values you want to customize into TEST_VALUE_OVERRIDES in test_all_params.py",
        "",
        "TEST_VALUE_OVERRIDES: dict[str, list[Any]] = {",
    ]

    for backend, params in sorted(all_params.items()):
        lines.append(f"    # {backend.upper()}")
        for param, meta in sorted(params.items()):
            if meta["test_values"]:
                lines.append(f'    # "{param}": {meta["test_values"]},')
        lines.append("")

    lines.append("}")
    return "\n".join(lines)


def validate_manual_params() -> tuple[bool, list[str]]:
    """Check if manual param definitions match discovered params."""
    from .test_all_params import PYTORCH_PARAMS, TENSORRT_PARAMS, VLLM_PARAMS

    all_discovered = discover_all_backends()
    issues = []

    manual_maps = {
        "pytorch": PYTORCH_PARAMS,
        "vllm": VLLM_PARAMS,
        "tensorrt": TENSORRT_PARAMS,
    }

    for backend, manual_params in manual_maps.items():
        discovered = set(all_discovered[backend].keys())
        manual = set(manual_params.keys())

        # Params in Pydantic but not in manual test list
        missing = discovered - manual
        if missing:
            for p in sorted(missing):
                issues.append(f"[{backend}] Missing from manual: {p}")

        # Params in manual but not in Pydantic (might be deprecated)
        extra = manual - discovered
        if extra:
            for p in sorted(extra):
                issues.append(f"[{backend}] Extra in manual (not in Pydantic): {p}")

    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover params from Pydantic models")
    parser.add_argument(
        "--format",
        choices=["text", "json", "overrides"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate manual params match discovered",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "vllm", "tensorrt", "shared"],
        help="Show only specific backend",
    )

    args = parser.parse_args()

    if args.validate:
        valid, issues = validate_manual_params()

        if valid:
            print("✓ Manual param definitions match Pydantic models")
            sys.exit(0)
        else:
            print("✗ Param mismatch detected:")
            for issue in issues:
                print(f"  {issue}")
            print("\nRun with --discover to use auto-discovery, or update manual definitions.")
            sys.exit(1)

    # Discover params
    if args.backend == "shared":
        all_params = {"shared": discover_shared_params()}
    elif args.backend:
        all_discovered = discover_all_backends()
        all_params = {args.backend: all_discovered[args.backend]}
    else:
        all_params = discover_all_backends()
        all_params["shared"] = discover_shared_params()

    # Format output
    if args.format == "json":
        print(format_as_json(all_params))
    elif args.format == "overrides":
        print(format_as_overrides(all_params))
    else:
        print(format_as_text(all_params))


if __name__ == "__main__":
    main()
