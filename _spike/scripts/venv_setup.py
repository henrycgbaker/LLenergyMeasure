"""Per-engine-per-version venv scaffolding for trial matrix execution (Phase 2.5).

Builds isolated venvs at ``/tmp/trial_<engine>_<version_slug>_venv/`` for
non-active cells in the trial matrix. Trial_runner calls these lazily
on first hit.

Strategy capability matrix:

| Strategy | Needs source? | Needs runtime import? |
|----------|---------------|------------------------|
| (a)      | yes (AST walk)| yes (run the miner)    |
| (b)      | yes (chunker) | NO (LLM reads source)  |
| (c)      | yes (chunker) | NO (LLM reads source)  |
| (d-ab)   | yes (both)    | NO for the LLM half    |
| (d-ac)   | yes (both)    | NO for the LLM half    |

For Phase 2.5 / Phase 3 matrix execution, source-only venvs are the
common case (strategies b/c/d). Strategy (a) needs the package
installable - typically a CUDA-bearing image, not a venv.

Source-only venv shape:

::

    /tmp/trial_<engine>_<version_slug>_venv/
        bin/python                  # venv stub
        src/<engine>/               # unpacked wheel source (the chunker reads from here)
        src/dist-info               # original metadata
        .source_only                # marker file (so the runner knows not to expect import)

The Phase 1 miner-lift agents pre-populated `/tmp/vllm-unpacked/` and
`/tmp/trt-llm-0.21.0/` from the active wheels. For other versions, this
module downloads via `pip download <pkg>==<ver> --no-deps` and unpacks
into the venv source dir.

Usage::

    from _spike.scripts.venv_setup import ensure_source_only_venv

    venv_path = ensure_source_only_venv(engine="vllm", version_slug="v0_7_3")
    # -> Path("/tmp/trial_vllm_v0_7_3_venv/")
    # source available at venv_path / "src" / "vllm"

Phase 2.5 scope: source-only venvs only. Strategy (a) needs a CUDA-
bearing runtime (out of venv scope; use the canonical container).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

VENV_ROOT_TEMPLATE = "/tmp/trial_{engine}_{version_slug}_venv"
"""Template for the per-engine-per-version venv path."""


# Map from engine + version_slug to (pip name, pip version)
ENGINE_PIP_SPEC: dict[tuple[str, str], tuple[str, str]] = {
    # transformers
    ("transformers", "v4_57_3"): ("transformers", "4.57.3"),
    ("transformers", "v4_56_0"): ("transformers", "4.56.0"),
    # vllm
    ("vllm", "v0_7_3"): ("vllm", "0.7.3"),
    ("vllm", "v0_7_0"): ("vllm", "0.7.0"),
    ("vllm", "v0_8_0"): ("vllm", "0.8.0"),
    # tensorrt
    ("tensorrt", "v0_21_0"): ("tensorrt-llm", "0.21.0"),
    ("tensorrt", "v0_20_0"): ("tensorrt-llm", "0.20.0"),
    ("tensorrt", "v0_22_0"): ("tensorrt-llm", "0.22.0"),
}


# Pre-existing unpacked source trees (from Phase 1 miner-lift work).
# These are SHARED across cells; the venv_setup module checks for them
# before downloading. Saves time + bandwidth.
EXISTING_UNPACKED: dict[tuple[str, str], Path] = {
    ("vllm", "v0_7_3"): Path("/tmp/vllm-unpacked/vllm"),
    ("tensorrt", "v0_21_0"): Path("/tmp/trt-llm-0.21.0/tensorrt_llm"),
}


@dataclass(frozen=True)
class SourceOnlyVenv:
    """Marker for a source-only venv (no runtime install).

    The strategy chunkers (vllm_chunker, tensorrt_chunker) read from
    ``source_dir``; they don't need the venv's python interpreter.

    Strategy (a) miners need a different setup (a CUDA-bearing image
    container); they do not use this scaffolding.
    """

    engine: str
    version_slug: str
    venv_path: Path
    source_dir: Path
    """Directory containing the engine's Python source tree."""


def ensure_source_only_venv(
    *,
    engine: str,
    version_slug: str,
    venv_root_template: str = VENV_ROOT_TEMPLATE,
) -> SourceOnlyVenv:
    """Build (lazily) a source-only venv for the given (engine, version).

    Steps:
    1. Check if the venv dir already exists with a `src/<engine>` subtree.
       If yes, return it.
    2. Check the EXISTING_UNPACKED map for a pre-staged source tree
       (from Phase 1). If hit, symlink it into the venv structure.
    3. Otherwise: `pip download` the wheel, unpack, and stage.

    Returns a :class:`SourceOnlyVenv` describing the result.

    Raises ``LookupError`` if (engine, version_slug) is not in
    ``ENGINE_PIP_SPEC`` and no pre-staged path exists.
    """
    venv_path = Path(venv_root_template.format(engine=engine, version_slug=version_slug))
    source_dir = venv_path / "src" / engine
    marker = venv_path / ".source_only"

    # Fast path: already built
    if marker.exists() and source_dir.exists():
        return SourceOnlyVenv(
            engine=engine,
            version_slug=version_slug,
            venv_path=venv_path,
            source_dir=source_dir,
        )

    venv_path.mkdir(parents=True, exist_ok=True)
    (venv_path / "src").mkdir(parents=True, exist_ok=True)

    # Existing pre-staged source (Phase 1 miner-lift artefact)
    pre_staged = EXISTING_UNPACKED.get((engine, version_slug))
    if pre_staged is not None and pre_staged.exists():
        # Symlink the existing tree into the venv structure.
        if source_dir.exists() or source_dir.is_symlink():
            # If it's a wrong-target symlink or empty dir, clean and re-link.
            if source_dir.is_symlink():
                source_dir.unlink()
            else:
                shutil.rmtree(source_dir)
        source_dir.symlink_to(pre_staged.resolve())
        marker.write_text(
            f"engine={engine}\nversion_slug={version_slug}\nsource_from=pre_staged({pre_staged})\n"
        )
        return SourceOnlyVenv(
            engine=engine,
            version_slug=version_slug,
            venv_path=venv_path,
            source_dir=source_dir,
        )

    # pip download path
    spec = ENGINE_PIP_SPEC.get((engine, version_slug))
    if spec is None:
        raise LookupError(
            f"No pip spec for ({engine}, {version_slug}). "
            f"Add a row to ENGINE_PIP_SPEC or pre-stage source under EXISTING_UNPACKED."
        )
    pip_name, pip_version = spec
    return _pip_download_and_unpack(
        engine=engine,
        version_slug=version_slug,
        pip_name=pip_name,
        pip_version=pip_version,
        venv_path=venv_path,
        source_dir=source_dir,
        marker=marker,
    )


def _pip_download_and_unpack(
    *,
    engine: str,
    version_slug: str,
    pip_name: str,
    pip_version: str,
    venv_path: Path,
    source_dir: Path,
    marker: Path,
) -> SourceOnlyVenv:
    """Download a wheel via pip + unpack into venv/src/<engine>.

    Uses ``pip download --no-deps`` to skip transitive deps (we only
    need the engine's source; the chunker reads files, not Python
    imports).
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # Download wheel + sdist into tmp
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--dest",
            str(tmp),
            f"{pip_name}=={pip_version}",
        ]
        subprocess.run(cmd, check=True)
        wheels = list(tmp.glob("*.whl"))
        if not wheels:
            tarballs = list(tmp.glob("*.tar.gz"))
            if not tarballs:
                raise RuntimeError(
                    f"pip download produced no wheel or tarball for {pip_name}=={pip_version}"
                )
            # Unpack tarball
            import tarfile

            with tarfile.open(tarballs[0]) as tf:
                tf.extractall(venv_path / "src")
        else:
            # Unpack wheel as zip
            import zipfile

            with zipfile.ZipFile(wheels[0]) as zf:
                zf.extractall(venv_path / "src")

    # The unpacked source should be at venv_path/src/<engine>/<python files>.
    # If it's nested differently (e.g. version_slug subdir), normalise.
    candidates = list((venv_path / "src").glob(f"{engine}*"))
    canonical = None
    for c in candidates:
        if c.is_dir() and (c / "__init__.py").exists():
            canonical = c
            break
    if canonical is None:
        # Sometimes the wheel unpacks to engine/ directly without prefix.
        if (venv_path / "src" / engine).is_dir():
            canonical = venv_path / "src" / engine
    if canonical is None:
        raise RuntimeError(f"Could not locate {engine} package directory under {venv_path / 'src'}")

    # Ensure source_dir points to canonical.
    if canonical != source_dir:
        if source_dir.exists() or source_dir.is_symlink():
            if source_dir.is_symlink():
                source_dir.unlink()
            else:
                shutil.rmtree(source_dir)
        source_dir.symlink_to(canonical.resolve())

    marker.write_text(
        f"engine={engine}\n"
        f"version_slug={version_slug}\n"
        f"pip_spec={pip_name}=={pip_version}\n"
        f"source_from=pip_download\n"
    )
    return SourceOnlyVenv(
        engine=engine,
        version_slug=version_slug,
        venv_path=venv_path,
        source_dir=source_dir,
    )


# ---------------------------------------------------------------------------
# CLI: ad-hoc venv prep
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a source-only venv for a (engine, version) cell."
    )
    parser.add_argument("--engine", required=True, choices=["transformers", "vllm", "tensorrt"])
    parser.add_argument("--version-slug", required=True, help="e.g. v0_7_3 or v4_57_3")
    args = parser.parse_args(argv)
    sv = ensure_source_only_venv(engine=args.engine, version_slug=args.version_slug)
    print(f"engine={sv.engine}")
    print(f"version_slug={sv.version_slug}")
    print(f"venv_path={sv.venv_path}")
    print(f"source_dir={sv.source_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
