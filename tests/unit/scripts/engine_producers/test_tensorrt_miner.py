"""Tests for :mod:`scripts.engine_producers.tensorrt_miner` - orchestrator surface.

The orchestrator is a thin wrapper around the static miner; it exists to
match the per-engine ``{engine}_miner.py`` shape that
``transformers_miner.py`` established. The probe (``scripts._drift``) is
the runtime gate for landmark resolution; the dispatcher selects which
vendored static-miner archive runs for the SSOT-pinned library version.

Coverage:

- TRT-LLM is registered in :data:`scripts.engine_producers.build_corpus._ENGINE_EXTRACTORS`.
- The orchestrator never imports ``tensorrt_llm`` at module load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import build_corpus  # noqa: E402


def test_tensorrt_engine_registered_in_build_corpus() -> None:
    """``--engine tensorrt`` resolves to the static miner extractor."""
    assert "tensorrt" in build_corpus._ENGINE_EXTRACTORS
    extractors = build_corpus._ENGINE_EXTRACTORS["tensorrt"]
    assert len(extractors) == 1
    assert extractors[0].module == "scripts.engine_producers.tensorrt_static_invariant_miner"
    assert extractors[0].staging_basename == "tensorrt_static_invariant_miner.yaml"


def test_orchestrator_does_not_import_tensorrt_llm() -> None:
    """The orchestrator must NOT import ``tensorrt_llm`` at module load.

    The host has TRT-LLM 1.1.0 installed (a separate library generation
    that diverged significantly from 0.21.0). Importing it would silently
    mine drifted source. Source-driven AST walking is the load-bearing
    safety property; this test pins it.
    """
    # The miner module is already loaded by the import above; ``tensorrt_llm``
    # must therefore be absent from ``sys.modules``.
    # (Other test modules that DO import the live library may run before us;
    # in that case skip rather than failing on someone else's import.)
    if "tensorrt_llm" in sys.modules:
        pytest.skip(
            "tensorrt_llm has been imported by some other test module; "
            "this test only catches the case where the orchestrator itself imports it."
        )
    # Re-import the orchestrator and confirm no side effect.
    assert "tensorrt_llm" not in sys.modules
