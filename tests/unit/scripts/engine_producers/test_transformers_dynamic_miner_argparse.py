"""Argparse-layer regression tests for the transformers dynamic miner.

Separate from :mod:`test_transformers_dynamic_miner` because that module
``pytest.importorskip("transformers")`` at module load; the argparse contract
is a pure-Python wrapper around the heavy library-mining body and stays
verifiable without the live library installed.

Background: prior to the fix, the per-version vendored miner accepted an
``argv`` parameter on its ``main(argv)`` signature for stub-compatibility
but never parsed it; the output path was hardcoded to
``src/llenergymeasure/engines/transformers/_staging/transformers_dynamic_miner.yaml``.
The orchestrator's per-version cell wiring passes ``--out <engine_versions/.../outputs/_staging/...>``;
the miner silently ignored it and the merger missed 28 dynamic-mined
invariants. See ``.product/research/transformers-sampling-regression-2026-05-23.md``
for the full diagnostic.

These tests exercise the argparse layer against each vendored miner
producer directly (not the stub) so a regression in any one of the
vendored copies surfaces.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_VENDORED_MINERS: tuple[Path, ...] = (
    _PROJECT_ROOT
    / "engine_versions"
    / "transformers"
    / "v4_57_3"
    / "producers"
    / "dynamic_invariant_miner.py",
    _PROJECT_ROOT
    / "engine_versions"
    / "transformers"
    / "v5_7_0"
    / "producers"
    / "dynamic_invariant_miner.py",
)


_EXPECTED_STAGING_BASENAME = "transformers_dynamic_invariant_miner.yaml"


def _load_miner_module(path: Path) -> Any:
    """Import a vendored miner file as a standalone module under a synthetic name.

    Each per-version producer must be loaded under a distinct module name to
    avoid Python's import cache collapsing them into one - they live in
    sibling directories sharing the same basename.
    """
    safe_version = path.parent.parent.name  # ``v4_57_3`` etc
    module_name = f"_test_dynamic_miner_{safe_version}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Pre-register so any internal ``from ... import`` doesn't loop on partial init.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "miner_path",
    _VENDORED_MINERS,
    ids=lambda p: p.parent.parent.name,
)
def test_default_out_uses_registry_basename(miner_path: Path) -> None:
    """The ``--out`` default basename matches the orchestrator registry entry.

    The orchestrator (``build_corpus._ENGINE_EXTRACTORS``) names the staging
    file ``transformers_dynamic_invariant_miner.yaml``. The miner default
    must match exactly; relying on the merger's ``transformers_*.yaml``
    discovery glob to paper over a basename mismatch is the latent bug the
    fix removes.
    """
    miner = _load_miner_module(miner_path)
    # Argparse is built inside main(); the default is reachable by inspecting
    # the parser without invoking the library-dependent body. We mimic the
    # construction here so we never invoke the heavy walk path.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    # The miner module's main() doesn't expose the parser; reproduce the
    # default-extraction by parsing an empty argv against a faithful clone
    # of the miner's parser - the basename is what the test cares about.
    # Simpler: just inspect the miner's parser default by calling main with a
    # patched parse_args that captures the default. We do that via
    # monkeypatching argparse.ArgumentParser.parse_args.
    captured: dict[str, Any] = {}

    original_parse_args = argparse.ArgumentParser.parse_args

    def _capture(
        self: argparse.ArgumentParser, args: list[str] | None = None
    ) -> argparse.Namespace:
        # Trigger normal parse_args so defaults populate, then capture.
        ns = original_parse_args(self, args)
        captured["out"] = ns.out
        # Raise to short-circuit the rest of main() before any library import.
        raise _ParseCaptured

    argparse.ArgumentParser.parse_args = _capture  # type: ignore[assignment]
    try:
        with pytest.raises(_ParseCaptured):
            miner.main([])
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args  # type: ignore[assignment]

    out_default = captured["out"]
    assert isinstance(out_default, Path), (
        f"miner {miner_path} --out default is not a Path: {out_default!r}"
    )
    assert out_default.name == _EXPECTED_STAGING_BASENAME, (
        f"miner {miner_path} writes basename {out_default.name!r}; "
        f"orchestrator registry expects {_EXPECTED_STAGING_BASENAME!r}. "
        "Without a basename match the merger's `transformers_*.yaml` glob is "
        "the only thing keeping the staging file discoverable - a fragile "
        "contract that previously hid the latent --out bug."
    )


@pytest.mark.parametrize(
    "miner_path",
    _VENDORED_MINERS,
    ids=lambda p: p.parent.parent.name,
)
def test_out_argument_routes_write_path(miner_path: Path, tmp_path: Path) -> None:
    """The ``--out`` flag controls where the staging YAML lands.

    Regression for the hardcoded-path bug: the prior ``main(argv)``
    silently ignored ``argv``, so calling ``main(["--out", "/tmp/x.yaml"])``
    wrote to the hardcoded ``src/.../_staging/`` location regardless.

    We stub out the library-dependent producers
    (``_resolve_source_paths``, ``walk_generation_config_invariants``) so the
    test runs without the ``transformers`` package installed - the bug
    being verified is purely path-routing, independent of mining content.
    """
    miner = _load_miner_module(miner_path)

    target = tmp_path / "regression_test.yaml"
    sentinel_hardcoded = (
        _PROJECT_ROOT
        / "src"
        / "llenergymeasure"
        / "engines"
        / "transformers"
        / "_staging"
        / "transformers_dynamic_miner.yaml"
    )
    # Snapshot the hardcoded-location mtime (if any) so we can assert the
    # miner did NOT touch it. The path may not exist on a clean dev tree;
    # the assertion below tolerates that.
    pre_existed = sentinel_hardcoded.exists()
    pre_mtime = sentinel_hardcoded.stat().st_mtime if pre_existed else None

    # Stub out the library-dependent functions so the miner short-circuits
    # to the write step without importing transformers.
    miner._resolve_source_paths = lambda: (
        "0.0.0-stub",
        "/nonexistent/abs.py",
        "stub/rel.py",
    )
    miner.walk_generation_config_invariants = lambda **_: []

    rc = miner.main(["--out", str(target)])

    assert rc == 0, "miner.main returned non-zero exit"
    assert target.is_file(), (
        f"--out path {target!s} was not written; miner is ignoring --out "
        "and falling back to its hardcoded path again."
    )
    # The hardcoded path must not have been touched by this invocation.
    if pre_existed:
        assert sentinel_hardcoded.stat().st_mtime == pre_mtime, (
            f"miner touched the legacy hardcoded path {sentinel_hardcoded!s} "
            "despite --out being supplied."
        )
    else:
        assert not sentinel_hardcoded.exists(), (
            f"miner created the legacy hardcoded path {sentinel_hardcoded!s} "
            "despite --out being supplied."
        )


def test_orchestrator_registry_matches_miner_default_basename() -> None:
    """Cross-module invariant: registry name == miner --out default basename.

    Prevents the basename from drifting back out of sync. The
    orchestrator's ``_ENGINE_EXTRACTORS`` registry is the source of truth
    for which staging filename the merger looks for; the miner default must
    track it.
    """
    from scripts.engine_producers.build_corpus import _ENGINE_EXTRACTORS

    transformers_extractors = {
        e.module: e.staging_basename for e in _ENGINE_EXTRACTORS["transformers"]
    }
    dynamic_basename = transformers_extractors.get(
        "scripts.engine_producers.transformers_dynamic_invariant_miner"
    )
    assert dynamic_basename == _EXPECTED_STAGING_BASENAME, (
        f"orchestrator registry expects {dynamic_basename!r}; "
        f"this test pins it to {_EXPECTED_STAGING_BASENAME!r}. "
        "If you renamed the staging file, update the miner --out default and "
        "_EXPECTED_STAGING_BASENAME in this test together."
    )


class _ParseCaptured(BaseException):
    """Sentinel to short-circuit main() once parse_args has fired."""
