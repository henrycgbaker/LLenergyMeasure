#!/usr/bin/env python3
"""Check that engine current.yaml versions match discovered schema engine_versions.

Each engine has a canonical version pinned in
``engine_versions/<engine>/current.yaml`` under ``library.current_version``. The
discovered schema at ``src/llenergymeasure/engines/<engine>/schema.discovered.json``
must agree.

Engines covered:
  - vllm
  - tensorrt
  - transformers

Usage:
    python scripts/check_discovered_schema_versions.py [--engine ENGINE]

When ``--engine`` is omitted, all three engines are checked. The CI
matrix in ``ci.yml`` runs one job per engine with ``--engine`` set.

Exit codes:
    0 = match (or all matches when no --engine)
    1 = mismatch detected
    2 = error (missing file, parse failure)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_versions import _outputs
from scripts.engine_producers._common import DEFAULT_SCHEMA_FILENAME

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENGINES = ("vllm", "tensorrt", "transformers")

# The two on-disk copies of a discovered schema whose parameter surfaces must
# not diverge: the src/ shadow the runtime loader and absorb read, and the
# versioned outputs/ snapshot codegen reads. Only the version envelope is
# compared elsewhere; these are the keys that carry the actual config surface.
_SURFACE_KEYS = ("engine_params", "sampling_params")


def _normalize_version(version: str) -> str:
    """Strip leading 'v' prefix for comparison."""
    return version.lstrip("v")


def _current_version(current_yaml: Path) -> str | None:
    """Return ``library.current_version`` from an engine current.yaml."""
    data = yaml.safe_load(current_yaml.read_text()) or {}
    library = data.get("library") or {}
    value = library.get("current_version")
    return None if value is None else str(value)


def _parse_schema_version(schema_path: Path) -> Any:
    """Extract engine_version from a discovered schema JSON."""
    data = json.loads(schema_path.read_text())
    return data.get("engine_version")


def _surface(schema_path: Path) -> dict[str, Any]:
    """Parameter surface (engine + sampling params) of a discovered schema."""
    data = json.loads(schema_path.read_text())
    return {key: data.get(key) for key in _SURFACE_KEYS}


def main(repo_root: Path | None = None, engines: tuple[str, ...] | None = None) -> int:
    root = repo_root or REPO_ROOT
    engines_dir = root / "src" / "llenergymeasure" / "engines"
    engine_versions_dir = root / "engine_versions"

    errors: list[str] = []
    mismatches: list[str] = []

    for engine in engines or _ENGINES:
        current_yaml = engine_versions_dir / engine / "current.yaml"
        try:
            pinned_version = _current_version(current_yaml)
        except FileNotFoundError:
            errors.append(f"{engine}: current.yaml not found: {current_yaml}")
            continue

        schema_path = engines_dir / engine / DEFAULT_SCHEMA_FILENAME
        try:
            schema_version = _parse_schema_version(schema_path)
        except FileNotFoundError:
            errors.append(f"{engine}: schema not found: {schema_path}")
            continue

        if pinned_version is None:
            errors.append(f"{engine}: library.current_version not found in {current_yaml.name}")
            continue

        if schema_version is None:
            errors.append(f"{engine}: engine_version not found in {schema_path.name}")
            continue

        if _normalize_version(pinned_version) != _normalize_version(str(schema_version)):
            mismatches.append(
                f"MISMATCH: {current_yaml.name} pins library.current_version={pinned_version} "
                f"but schema was discovered against {schema_version}\n"
                f"  Run: ./scripts/refresh_discovered_schemas.sh {engine}"
            )
            continue  # version already diverged; a surface diff would just be noise

        # Guard the two schema copies against a surface divergence the version
        # envelope alone cannot catch: the versioned outputs/ snapshot codegen
        # reads must expose the same parameter surface as the src/ shadow the
        # runtime loader and absorb read.
        outputs_schema = (
            engine_versions_dir / engine / _outputs.safe_version(pinned_version) / "outputs"
        ) / DEFAULT_SCHEMA_FILENAME
        try:
            outputs_surface = _surface(outputs_schema)
        except FileNotFoundError:
            errors.append(f"{engine}: versioned schema snapshot not found: {outputs_schema}")
            continue
        if _surface(schema_path) != outputs_surface:
            mismatches.append(
                f"SURFACE MISMATCH: {schema_path} and {outputs_schema} agree on version "
                f"{pinned_version} but their parameter surfaces differ\n"
                f"  Re-run discovery so both copies carry the same surface."
            )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if mismatches:
        for m in mismatches:
            print(m, file=sys.stderr)
        return 1

    print("All schema versions match pinned engine_versions.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=_ENGINES,
        default=None,
        help="Check only the named engine. Omit to check all three.",
    )
    args = parser.parse_args()
    sys.exit(main(engines=(args.engine,) if args.engine else None))
