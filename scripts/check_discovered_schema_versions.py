#!/usr/bin/env python3
"""Check that engine SSOT versions match vendored schema engine_versions.

Each engine has a canonical version pinned in
``engine_versions/<engine>.yaml`` under ``library.current_version``. The
discovered schema in ``src/llenergymeasure/config/discovered_schemas``
must agree.

Engines covered:
  - vllm
  - tensorrt
  - transformers

Exit codes:
    0 = all versions match
    1 = mismatch detected
    2 = error (missing file, parse failure)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENGINES = ("vllm", "tensorrt", "transformers")


def _normalize_version(version: str) -> str:
    """Strip leading 'v' prefix for comparison."""
    return version.lstrip("v")


def _ssot_current_version(ssot_path: Path) -> str | None:
    """Return ``library.current_version`` from an engine SSOT yaml."""
    data = yaml.safe_load(ssot_path.read_text()) or {}
    library = data.get("library") or {}
    value = library.get("current_version")
    return None if value is None else str(value)


def _parse_schema_version(schema_path: Path) -> Any:
    """Extract engine_version from a vendored schema JSON."""
    data = json.loads(schema_path.read_text())
    return data.get("engine_version")


def main(repo_root: Path | None = None) -> int:
    root = repo_root or REPO_ROOT
    schema_dir = root / "src" / "llenergymeasure" / "config" / "discovered_schemas"
    ssot_dir = root / "engine_versions"

    errors: list[str] = []
    mismatches: list[str] = []

    for engine in _ENGINES:
        ssot_path = ssot_dir / f"{engine}.yaml"
        try:
            ssot_version = _ssot_current_version(ssot_path)
        except FileNotFoundError:
            errors.append(f"{engine}: SSOT not found: {ssot_path}")
            continue

        schema_path = schema_dir / f"{engine}.json"
        try:
            schema_version = _parse_schema_version(schema_path)
        except FileNotFoundError:
            errors.append(f"{engine}: schema not found: {schema_path}")
            continue

        if ssot_version is None:
            errors.append(f"{engine}: library.current_version not found in {ssot_path.name}")
            continue

        if schema_version is None:
            errors.append(f"{engine}: engine_version not found in {schema_path.name}")
            continue

        if _normalize_version(ssot_version) != _normalize_version(str(schema_version)):
            mismatches.append(
                f"MISMATCH: {ssot_path.name} pins library.current_version={ssot_version} "
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

    print("All schema versions match SSOT engine_versions.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
