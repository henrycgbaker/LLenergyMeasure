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

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENGINES = ("vllm", "tensorrt", "transformers")


def _normalize_version(version: str) -> str:
    """Strip leading 'v' prefix for comparison."""
    return version.lstrip("v")


def _parse_ssot_version(ssot_path: Path) -> str | None:
    """Extract library.current_version from an engine SSOT yaml.

    Uses a tiny PyYAML-free parser to keep this script free of dependencies
    that might not be installed in every CI lane.
    """
    text = ssot_path.read_text()
    in_library = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            in_library = line.strip().startswith("library:")
            continue
        if in_library and line.lstrip().startswith("current_version:"):
            value = line.split(":", 1)[1].strip()
            return value.strip("\"'")
    return None


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
        if not ssot_path.exists():
            errors.append(f"{engine}: SSOT not found: {ssot_path}")
            continue

        schema_path = schema_dir / f"{engine}.json"
        if not schema_path.exists():
            errors.append(f"{engine}: schema not found: {schema_path}")
            continue

        ssot_version = _parse_ssot_version(ssot_path)
        if ssot_version is None:
            errors.append(f"{engine}: library.current_version not found in {ssot_path.name}")
            continue

        schema_version = _parse_schema_version(schema_path)
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
