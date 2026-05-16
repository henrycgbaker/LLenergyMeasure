#!/usr/bin/env python3
"""Check that engine current.yaml versions match discovered schema engine_versions.

Each engine has a canonical version pinned in
``engine_versions/<engine>/current.yaml`` under ``library.current_version``. The
discovered schema at ``engine_versions/<engine>/v<current>/outputs/schema.discovered.json``
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

from scripts.engine_producers._common import DEFAULT_SCHEMA_FILENAME

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENGINES = ("vllm", "tensorrt", "transformers")


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


def _safe_version(version: str) -> str:
    return "v" + _normalize_version(version).replace(".", "_").replace("-", "_")


def main(repo_root: Path | None = None, engines: tuple[str, ...] | None = None) -> int:
    root = repo_root or REPO_ROOT
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

        if pinned_version is None:
            errors.append(f"{engine}: library.current_version not found in {current_yaml.name}")
            continue

        schema_path = (
            engine_versions_dir
            / engine
            / _safe_version(pinned_version)
            / "outputs"
            / DEFAULT_SCHEMA_FILENAME
        )
        try:
            schema_version = _parse_schema_version(schema_path)
        except FileNotFoundError:
            errors.append(f"{engine}: schema not found: {schema_path}")
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
