"""Sync the per-pin engine corpus (SSOT) into the loader's data shadow.

The mined-corpus SSOT lives at
``engine_versions/<engine>/v<safe>/outputs/`` (resolved from each engine's
``current.yaml`` pin). The runtime loader reads from
``src/llenergymeasure/engines/<engine>/`` - the data shadow that ships in
the wheel. This script copies SSOT -> shadow.

Files synced (per pin): schema.discovered.json, rules.proposed.yaml,
rules.validated.yaml, curated.yaml, and overlay.yaml when the SSOT
carries one (optional hand-authored narrowings/completions).

Two modes:

- ``--check`` (default): compare SSOT against shadow; exit 1 with a precise
  per-file drift report. No writes. CI parity gate.
- ``--write``: copy each SSOT file onto its shadow and report what changed.

``--engine <name>`` restricts the run to one engine (repeatable); the
default is all engines.

This is deliberately small: file comparison + copy. The heavy lifting lives
in the per-engine miners that produce the SSOT.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._current import current_outputs_dir  # noqa: E402

ENGINES: tuple[str, ...] = ("transformers", "vllm", "tensorrt")

# Files always synced. overlay.yaml is handled separately: it is optional
# and only synced when the SSOT carries one.
CORPUS_FILES: tuple[str, ...] = (
    "schema.discovered.json",
    "rules.proposed.yaml",
    "rules.validated.yaml",
    "curated.yaml",
)
OPTIONAL_FILES: tuple[str, ...] = ("overlay.yaml",)


def _shadow_dir(engine: str) -> Path:
    """Return ``src/llenergymeasure/engines/<engine>/``."""
    return _PROJECT_ROOT / "src" / "llenergymeasure" / "engines" / engine


def _file_diff(src: Path, dst: Path) -> str:
    """Return a unified diff (shadow vs SSOT); empty when byte-identical."""
    src_lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    dst_lines = dst.read_text(encoding="utf-8").splitlines(keepends=True) if dst.exists() else []
    return "".join(difflib.unified_diff(dst_lines, src_lines, fromfile=str(dst), tofile=str(src)))


def sync_engine(engine: str, *, write: bool) -> tuple[list[str], list[str]]:
    """Sync (or check) one engine's corpus files.

    Returns ``(drift, changed)``:

    - ``drift``: per-file drift reports. In ``--check`` mode a non-empty list
      means the caller must exit non-zero.
    - ``changed``: in ``--write`` mode, the files that were (re)written.

    A missing SSOT outputs dir or source file raises ``FileNotFoundError`` -
    every current pin is expected to carry a full corpus, so a gap is a real
    error, not a silently-tolerated state.
    """
    outputs = current_outputs_dir(engine)
    if not outputs.is_dir():
        raise FileNotFoundError(
            f"{engine}: SSOT outputs dir not found ({outputs}). "
            f"Check engine_versions/{engine}/current.yaml and the vendored pin."
        )

    # Optional files are synced only when the SSOT carries one, so their
    # absence is never reported as drift.
    names = [*CORPUS_FILES, *(n for n in OPTIONAL_FILES if (outputs / n).exists())]

    shadow = _shadow_dir(engine)
    drift: list[str] = []
    changed: list[str] = []
    for name in names:
        src = outputs / name
        dst = shadow / name
        if not src.exists():
            raise FileNotFoundError(f"{engine}: SSOT file missing ({src}).")

        src_bytes = src.read_bytes()
        dst_bytes = dst.read_bytes() if dst.exists() else b""
        if src_bytes == dst_bytes:
            continue

        if write:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            changed.append(f"{engine}/{name}")
        else:
            drift.append(f"{engine}/{name} drift:\n{_file_diff(src, dst)}")
    return drift, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the per-pin engine corpus from engine_versions/ (SSOT) into "
            "src/llenergymeasure/engines/ (data shadow)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify SSOT/shadow parity; exit 1 with a per-file drift report. Default mode.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Copy SSOT -> shadow (mutates the working tree) and report what changed.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=ENGINES,
        help="Restrict to one or more engines (repeatable). Default: all engines.",
    )
    args = parser.parse_args(argv)

    engines = tuple(args.engine) if args.engine else ENGINES
    all_drift: list[str] = []
    all_changed: list[str] = []
    for engine in engines:
        drift, changed = sync_engine(engine, write=args.write)
        all_drift.extend(drift)
        all_changed.extend(changed)

    if args.write:
        if all_changed:
            for entry in all_changed:
                print(f"[regen-corpus] wrote: {entry}")
        else:
            print("[regen-corpus] shadow already in sync; nothing written.")
        return 0

    if all_drift:
        for entry in all_drift:
            print(entry, file=sys.stderr)
        print(
            "\nDrift between engine_versions/<engine>/v<pin>/outputs/ and "
            "src/llenergymeasure/engines/<engine>/.\n"
            "Resync the shadow from the SSOT:\n"
            "  uv run python scripts/engine_producers/regen_engine_corpus.py --write",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
