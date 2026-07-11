#!/usr/bin/env python3
"""Promote versioned discovered-schema snapshots into the packaged src copies.

Schema discovery writes one snapshot per engine version to
``engine_versions/<engine>/v<safe>/outputs/schema.discovered.json`` - the single
discovery write target. This script is the ONLY writer of the packaged shadow at
``src/llenergymeasure/engines/<engine>/schema.discovered.json``: for each engine
at its current pin it byte-copies the versioned snapshot into the src tree.

The copy is verbatim - there is no transformation here. If a transformation is
ever needed it belongs in discovery or in codegen, not in promotion, so that the
two copies stay a pure byte-promotion of each other. The CI surface-equality
guard (``check_discovered_schema_versions.py``) is the drift tripwire for exactly
this invariant.

``v<safe>`` is derived from ``library.current_version`` in each engine's
``current.yaml`` via :mod:`engine_versions._outputs`, the one place that
version-to-directory name-mangling lives.

Usage:
    python scripts/promote_schemas.py [--engine ENGINE]

When ``--engine`` is omitted, all three engines are promoted.

Exit codes:
    0 = promoted (or all promoted when no --engine)
    2 = error (missing snapshot, missing/unreadable current.yaml)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_versions import _outputs

REPO_ROOT = Path(__file__).resolve().parent.parent


def _current_version(current_yaml: Path) -> str:
    """Return ``library.current_version`` from an engine current.yaml."""
    data = yaml.safe_load(current_yaml.read_text()) or {}
    library = data.get("library") or {}
    value = library.get("current_version")
    if value is None:
        raise ValueError(f"library.current_version not found in {current_yaml}")
    return str(value)


def promote_engine(engine: str, repo_root: Path) -> bool:
    """Byte-copy one engine's versioned snapshot schema into its src shadow.

    Resolves the pin from ``current.yaml``, locates the versioned snapshot under
    ``engine_versions/`` (via the ``_outputs`` mangling helper), and writes its
    bytes verbatim to the packaged ``src/`` copy. Returns ``True`` when the src
    copy's bytes actually changed.
    """
    current_yaml = repo_root / "engine_versions" / engine / "current.yaml"
    version = _current_version(current_yaml)

    outputs_schema = (
        repo_root
        / "engine_versions"
        / engine
        / _outputs.safe_version(version)
        / "outputs"
        / _outputs.SCHEMA_FILENAME
    )
    src_schema = (
        repo_root / "src" / "llenergymeasure" / "engines" / engine / _outputs.SCHEMA_FILENAME
    )

    outputs_bytes = outputs_schema.read_bytes()
    changed = not src_schema.exists() or src_schema.read_bytes() != outputs_bytes
    src_schema.parent.mkdir(parents=True, exist_ok=True)
    src_schema.write_bytes(outputs_bytes)
    return changed


def main(repo_root: Path | None = None, engines: tuple[str, ...] | None = None) -> int:
    root = repo_root or REPO_ROOT
    errors: list[str] = []

    for engine in engines or _outputs.ENGINES:
        try:
            changed = promote_engine(engine, root)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{engine}: {exc}")
            continue
        status = "updated" if changed else "unchanged"
        print(f"{engine}: promoted outputs -> src ({status})", file=sys.stderr)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=_outputs.ENGINES,
        default=None,
        help="Promote only the named engine. Omit to promote all three.",
    )
    args = parser.parse_args()
    sys.exit(main(engines=(args.engine,) if args.engine else None))
