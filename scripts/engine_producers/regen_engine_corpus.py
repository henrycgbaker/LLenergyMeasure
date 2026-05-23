"""Sync per-version engine corpus (SSOT) into the loader's shadow location.

The mined-corpus SSOT lives at
``engine_versions/<engine>/v<safe>/outputs/{invariants.proposed.yaml,
invariants.validated.yaml, schema.discovered.json}``. The loader reads from
``src/llenergymeasure/engines/<engine>/{same-files}`` (the "data shadow" that
ships in the wheel). This script copies SSOT -> shadow.

Two modes:

- ``--write`` (default): ``shutil.copy2`` each source onto its destination.
  Idempotent; preserves mtime.
- ``--check``: read both files, exit 1 with a unified diff per drifted file.
  No writes. CI parity gate.

Single source of truth for path derivation: :func:`_current.current_outputs_dir`
and :func:`_current.safe_version`. The script is intentionally tiny - the
heavy lifting lives in the per-engine miners that produce the SSOT.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
from pathlib import Path

# Project root on sys.path so this script works both as ``python -m
# scripts.engine_producers.regen_engine_corpus`` and as a direct
# ``python scripts/engine_producers/regen_engine_corpus.py``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._current import (  # noqa: E402
    _find_repo_root,
    current_outputs_dir,
)

ENGINES: tuple[str, ...] = ("vllm", "tensorrt", "transformers")
CORPUS_FILES: tuple[str, ...] = (
    "invariants.proposed.yaml",
    "invariants.validated.yaml",
    "schema.discovered.json",
)


def _shadow_dir(engine: str) -> Path:
    """Return ``src/llenergymeasure/engines/<engine>/`` for the given engine."""
    return (
        _find_repo_root(Path(__file__).resolve()) / "src" / "llenergymeasure" / "engines" / engine
    )


def _file_diff(src: Path, dst: Path) -> str:
    """Return a unified diff string (src vs dst); empty if byte-identical."""
    src_text = src.read_text().splitlines(keepends=True)
    dst_text = dst.read_text().splitlines(keepends=True) if dst.exists() else []
    return "".join(
        difflib.unified_diff(
            dst_text,
            src_text,
            fromfile=str(dst),
            tofile=str(src),
        )
    )


def sync_engine(engine: str, *, write: bool) -> list[str]:
    """Sync (or check) one engine. Return list of human-readable drift messages.

    Empty list = no drift detected (or write succeeded). Non-empty list under
    ``write=False`` indicates the caller should exit non-zero. Missing source
    files raise ``FileNotFoundError`` loud - the SSOT path is computed from
    ``current.yaml``; a missing source means the per-version archive hasn't
    been produced yet and the caller must surface that as a real error, not
    silently skip.
    """
    outputs = current_outputs_dir(engine)
    shadow = _shadow_dir(engine)
    drift: list[str] = []
    for filename in CORPUS_FILES:
        src = outputs / filename
        dst = shadow / filename
        if not src.exists():
            raise FileNotFoundError(
                f"Source file missing for {engine}: {src}. "
                f"Check engine_versions/{engine}/current.yaml points at a "
                f"version whose outputs/ directory has been populated by the "
                f"cells workflow."
            )
        if write:
            shadow.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            continue
        # --check mode: compare bytes and accumulate diff per file.
        src_bytes = src.read_bytes()
        dst_bytes = dst.read_bytes() if dst.exists() else b""
        if src_bytes == dst_bytes:
            continue
        diff = _file_diff(src, dst)
        drift.append(f"{engine}/{filename} drift:\n{diff}")
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync per-version engine corpus artefacts from engine_versions/ "
            "(SSOT) to src/llenergymeasure/engines/ (data shadow)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "Don't write; exit 1 with a unified diff per drifted file. Used as the CI parity gate."
        ),
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Sync SSOT -> shadow via shutil.copy2 (default).",
    )
    args = parser.parse_args(argv)

    # Default to --write when neither flag is passed; --check is opt-in.
    write = not args.check

    all_drift: list[str] = []
    for engine in ENGINES:
        all_drift.extend(sync_engine(engine, write=write))

    if not write and all_drift:
        for entry in all_drift:
            print(entry, file=sys.stderr)
        print(
            "\nDrift detected between engine_versions/<engine>/v<current>/outputs/ "
            "and src/llenergymeasure/engines/<engine>/. Run:\n"
            "    python scripts/engine_producers/regen_engine_corpus.py --write\n"
            "to sync.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
