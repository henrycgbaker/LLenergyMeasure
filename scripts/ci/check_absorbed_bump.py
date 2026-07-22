#!/usr/bin/env python3
"""Guard against a bare engine version-pin bump merging without absorption.

Advancing an engine pin (``engine_versions/<engine>/current.yaml``) is only the
first line of a bump. The pipeline (``make absorb ENGINE=<engine> SRC=...``)
also regenerates the knowledge products the new pin implies:

- the versioned snapshot under ``engine_versions/<engine>/<version>/outputs/``
  (the mined ``schema.discovered.json`` + ``curated.yaml``),
- the generated config model at
  ``src/llenergymeasure/config/generated/<engine>.py``, and
- the packaged engine copies under ``src/llenergymeasure/engines/<engine>/``
  (the ``rules.yaml`` corpus and the promoted ``schema.discovered.json``).

A PR that moves only the pin has skipped ``make absorb``: it would ship a typed
config validated against the OLD engine surface while claiming the new version.
Renovate's regex manager emits exactly this shape (it edits ``current_version``
in place and nothing else), so a bare bump must fail at PR time.

This is a pure path check. It reads the PR's changed-file paths (one per line)
from stdin or ``--changed-files PATH`` and, for each engine whose
``current.yaml`` moved, asserts both the versioned snapshot outputs AND the
packaged src copies also changed. It exits non-zero listing every engine whose
bump is incomplete, naming the absorb command to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

ENGINES: tuple[str, ...] = ("transformers", "vllm", "tensorrt")

# Packaged knowledge a real absorb regenerates beside the engine, under
# src/llenergymeasure/engines/<engine>/. The generated config model lives in the
# config layer instead (config/generated/<engine>.py) and is checked separately.
# plugin.py and __init__.py are hand-written glue, not absorb outputs, so they do
# not count as evidence.
_ENGINE_KNOWLEDGE_FILES: frozenset[str] = frozenset({"rules.yaml", "schema.discovered.json"})


def _bumped_engines(changed: set[str]) -> list[str]:
    """Return engines whose ``current.yaml`` pin moved in this changeset."""
    return [e for e in ENGINES if f"engine_versions/{e}/current.yaml" in changed]


def _has_snapshot_outputs(engine: str, changed: set[str]) -> bool:
    """True when a versioned snapshot output for ``engine`` changed."""
    prefix = f"engine_versions/{engine}/"
    return any(f.startswith(prefix) and "/outputs/" in f for f in changed)


def _has_src_copies(engine: str, changed: set[str]) -> bool:
    """True when a packaged src knowledge file for ``engine`` changed.

    Absorb writes across two homes: the generated config model in the config
    layer, and the rules corpus / promoted schema beside the engine.
    """
    if f"src/llenergymeasure/config/generated/{engine}.py" in changed:
        return True
    prefix = f"src/llenergymeasure/engines/{engine}/"
    return any(
        f.startswith(prefix) and PurePosixPath(f).name in _ENGINE_KNOWLEDGE_FILES for f in changed
    )


def check_absorbed_bump(changed_files: list[str]) -> list[str]:
    """Return one error paragraph per engine whose pin bump is unaccompanied.

    An empty list means every bumped engine shipped its regenerated knowledge
    (or no pin moved at all).
    """
    changed = {f.strip() for f in changed_files if f.strip()}
    errors: list[str] = []
    for engine in _bumped_engines(changed):
        missing: list[str] = []
        if not _has_snapshot_outputs(engine, changed):
            missing.append(
                f"engine_versions/{engine}/<version>/outputs/ "
                "(mined schema.discovered.json + curated.yaml)"
            )
        if not _has_src_copies(engine, changed):
            missing.append(
                f"the generated config (src/llenergymeasure/config/generated/{engine}.py) "
                f"plus the engine knowledge under src/llenergymeasure/engines/{engine}/ "
                "(rules.yaml / schema.discovered.json)"
            )
        if not missing:
            continue
        lines = [f"{engine}: current.yaml bumped the pin but the absorbed knowledge is missing:"]
        lines.extend(f"  - {item}" for item in missing)
        lines.append(f"  Run: make absorb ENGINE={engine} SRC=<{engine} source at the new pin>")
        lines.append("  then commit the regenerated snapshot outputs and src/ copies.")
        errors.append("\n".join(lines))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--changed-files",
        default="-",
        help="Path to a newline-delimited list of changed files, or '-' for stdin (default).",
    )
    args = parser.parse_args(argv)

    if args.changed_files == "-":
        changed_files = sys.stdin.read().splitlines()
    else:
        changed_files = Path(args.changed_files).read_text(encoding="utf-8").splitlines()

    errors = check_absorbed_bump(changed_files)
    if not errors:
        print("Engine pin bumps ship their absorbed knowledge (or no pin moved).")
        return 0

    print("::error::Bare engine version-pin bump detected (absorption skipped).", file=sys.stderr)
    for block in errors:
        print(block, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
