"""Sync per-version engine corpus (SSOT) into the loader's shadow location.

The mined-corpus SSOT lives at
``engine_versions/<engine>/v<safe>/outputs/{invariants.proposed.yaml,
invariants.validated.yaml, schema.discovered.json}``. The loader reads from
``src/llenergymeasure/engines/<engine>/{same-files}`` (the "data shadow" that
ships in the wheel). This script copies SSOT -> shadow.

Two modes:

- ``--check`` (default): read both files, exit 1 with a unified diff per
  drifted file. CI parity gate. **Safe default** (no mutation).
- ``--write``: ``shutil.copy2`` each source onto its destination.
  Idempotent; preserves mtime. **Opt-in** (mutates working tree).

Single source of truth for path derivation: :func:`_current.current_outputs_dir`
and :func:`_current.safe_version`. The script is intentionally tiny - the
heavy lifting lives in the per-engine miners that produce the SSOT.

Findings addressed (PR-B /code-review at PR #665):
- F#5/F#8: missing-source paths are tolerated (skip + log) rather than
  raise. The Renovate-pre-vendor window (current.toml bumped, vendor PR
  not yet landed) is legitimate; crashing CI on it is foot-gunny.
- F#9: ``--write`` is opt-in; bare invocation is ``--check`` (Unix
  formatter/linter convention).
- F#10: ``--engine <name>`` filter (repeatable) so writeback can run
  per-engine matching just the cells that uploaded artefacts.
- F#11: explicit ``encoding='utf-8'`` on text reads; CI runners with
  non-UTF8 locales would otherwise raise ``UnicodeDecodeError``.
- F#14: handle read-only destinations (chmod 444 / artefact perms /
  editor lock) via unlink-then-copy fallback.
- F#13: missing-source notices collect across engines rather than
  short-circuit on the first; better dev iteration.
- F#15: clearer error wording for missing-vs-wrong-type
  ``library.current_version`` (the fix lives in ``_current.py``).
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
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
    # curated.yaml is the Move-2 per-version curation: which mined fields
    # llem exposes via the generated Config class. Mirrors archive -> shadow
    # alongside the other 3 corpus artefacts so the loader sees a consistent
    # snapshot of {what's mined, what's validated, what's discovered, what's
    # exposed} for the active pin.
    "curated.yaml",
)


def _shadow_dir(engine: str) -> Path:
    """Return ``src/llenergymeasure/engines/<engine>/`` for the given engine."""
    return (
        _find_repo_root(Path(__file__).resolve()) / "src" / "llenergymeasure" / "engines" / engine
    )


def _file_diff(src: Path, dst: Path) -> str:
    """Return a unified diff string (src vs dst); empty if byte-identical."""
    src_text = src.read_text(encoding="utf-8").splitlines(keepends=True)
    dst_text = (
        dst.read_text(encoding="utf-8").splitlines(keepends=True) if dst.exists() else []
    )
    return "".join(
        difflib.unified_diff(
            dst_text,
            src_text,
            fromfile=str(dst),
            tofile=str(src),
        )
    )


def _copy_resilient(src: Path, dst: Path) -> None:
    """``shutil.copy2`` with read-only-dst fallback (F#14).

    If the destination exists and is read-only (chmod 444, artefact mode,
    editor lock), ``shutil.copy2`` raises ``PermissionError``. Unlink first
    in that case - the parent directory's write bit is the relevant
    permission for the new file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        if dst.exists():
            dst.unlink()
            shutil.copy2(src, dst)
        else:
            raise


def _path_has_uncommitted_changes(path: Path) -> bool:
    """Return True if ``path`` differs from its git HEAD version.

    Used by the F#3 destructive-write warning: if a dev has local edits
    in ``src/llenergymeasure/engines/<e>/<file>`` that --write is about
    to clobber, surface that loudly so the edit doesn't silently vanish.

    Returns False on any git error (file untracked, not in repo, git not
    on PATH) - those cases don't warrant a warning.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(path)],
            check=False,
            capture_output=True,
        )
    except (FileNotFoundError, OSError):
        return False
    # `git diff --quiet`: 0 = same, 1 = differs, 128 = not-in-repo / other.
    return result.returncode == 1


def sync_engine(engine: str, *, write: bool) -> tuple[list[str], list[str]]:
    """Sync (or check) one engine.

    Returns a pair ``(drift_messages, skip_messages)``.

    - ``drift_messages``: non-empty under ``write=False`` means the caller
      must exit non-zero. Empty otherwise.
    - ``skip_messages``: non-empty when the per-version outputs/ directory
      is missing or any expected file is absent. F#5/F#8 - the
      Renovate-pre-vendor window is legitimate and must not crash CI; the
      caller decides how loud to be (here: print to stderr; do not exit
      non-zero in --check, do not abort --write).
    """
    drift: list[str] = []
    skipped: list[str] = []
    try:
        outputs = current_outputs_dir(engine)
    except FileNotFoundError as exc:
        # current.toml itself is missing - the engine isn't tracked. Skip
        # with a clear note rather than crashing CI.
        skipped.append(f"{engine}: {exc}")
        return drift, skipped

    if not outputs.is_dir():
        skipped.append(
            f"{engine}: outputs directory not present ({outputs}). "
            f"Cells haven't produced this version yet (Renovate-pre-vendor "
            f"window); skipping."
        )
        return drift, skipped

    shadow = _shadow_dir(engine)
    for filename in CORPUS_FILES:
        src = outputs / filename
        dst = shadow / filename
        if not src.exists():
            skipped.append(
                f"{engine}/{filename}: source missing ({src}); skipping. "
                f"Trigger cells to populate."
            )
            continue
        src_bytes = src.read_bytes()
        # F#12: zero-byte SSOT is a corruption signal, not "in parity with
        # an empty shadow". Surface it as a distinct skip rather than
        # silently passing as "no drift".
        if not src_bytes:
            skipped.append(
                f"{engine}/{filename}: SSOT is zero bytes ({src}). "
                f"Likely a corrupted writeback or interrupted mining run. "
                f"Re-trigger cells; do not overwrite the shadow from this."
            )
            continue
        if write:
            # F#3 (option a): warn loudly when --write would clobber
            # uncommitted local edits in src/<e>/. Dev escape valve
            # preserved (no --force gate); the warning just makes the
            # destructiveness visible.
            if (
                dst.exists()
                and dst.read_bytes() != src_bytes
                and _path_has_uncommitted_changes(dst)
            ):
                print(
                    f"[regen-corpus] WARNING: {dst} has uncommitted "
                    f"local edits that --write is about to overwrite "
                    f"with SSOT content from {src}.\n"
                    f"  If those edits are intentional, commit them "
                    f"first OR move the change to the SSOT and re-run.\n"
                    f"  Pipeline is meant to be archive -> shadow only; "
                    f"local src/<e>/ edits are a dev escape valve, not "
                    f"the supported workflow.",
                    file=sys.stderr,
                )
            _copy_resilient(src, dst)
            continue
        # --check mode: compare bytes and accumulate diff per file.
        dst_bytes = dst.read_bytes() if dst.exists() else b""
        if src_bytes == dst_bytes:
            continue
        drift.append(f"{engine}/{filename} drift:\n{_file_diff(src, dst)}")
    return drift, skipped


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
            "Verify parity between SSOT and shadow. Exit 1 with a unified "
            "diff per drifted file. CI parity gate. **Default mode.**"
        ),
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help=(
            "Sync SSOT -> shadow via shutil.copy2 (mutates working tree). "
            "Opt-in; used by the writeback bot after cells produce fresh "
            "archive content."
        ),
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=ENGINES,
        default=None,
        help=(
            "Restrict the run to one or more engines (repeatable). Used by "
            "the writeback bot to run only the engines whose cells uploaded "
            "artefacts on this pipeline run. Default: all engines."
        ),
    )
    args = parser.parse_args(argv)

    # F#9: default to --check (the safe / non-mutating mode). --write is
    # opt-in. Matches Unix formatter/linter convention (ruff, black).
    write = args.write
    engines = tuple(args.engine) if args.engine else ENGINES

    all_drift: list[str] = []
    all_skipped: list[str] = []
    for engine in engines:
        drift, skipped = sync_engine(engine, write=write)
        all_drift.extend(drift)
        all_skipped.extend(skipped)

    # Skipped engines are informational; always surface them on stderr so a
    # CI log shows what didn't run, but never make them cause non-zero exit.
    for entry in all_skipped:
        print(f"[regen-corpus] skipped: {entry}", file=sys.stderr)

    if not write and all_drift:
        for entry in all_drift:
            print(entry, file=sys.stderr)
        print(
            "\nDrift detected between engine_versions/<engine>/v<current>/outputs/ "
            "and src/llenergymeasure/engines/<engine>/.\n"
            "Resolution paths:\n"
            "  - Hand-edit in src/<engine>/ ? Revert it; the SSOT is the archive.\n"
            "  - Local edit in engine_versions/<engine>/v<safe>/outputs/ ?\n"
            "      uv run python scripts/engine_producers/regen_engine_corpus.py --write\n"
            "  - Drift from a stale shadow after Renovate bump ?\n"
            "      Trigger cells (gh workflow run engine-pipeline.yml) and let the\n"
            "      writeback bot land both sides atomically.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
