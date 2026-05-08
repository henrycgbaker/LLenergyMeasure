"""End-to-end test of the producer module lazy LANDMARKS pattern.

Each producer module (3 invariants miners + 3 schema introspectors) exposes
``LANDMARKS`` via PEP 562 ``__getattr__`` that delegates to the dispatcher.
This test verifies the full chain ``producer module -> __getattr__ -> dispatcher
-> archive subpackage -> LANDMARKS`` resolves correctly at the current SSOT
``library.current_version`` for every (engine, producer) cell.

The test also asserts that the producer's exposed tuple is byte-identical to
what the archive returns directly. Drift between the two is a refactor-time
trap: if the producer re-export wiring drifts from the archive contents
(e.g. someone updates the archive but forgets to flip the ``producer=`` arg
in the producer module's dispatcher call), this test fails loud.
"""

from __future__ import annotations

import importlib

import pytest

from llenergymeasure._engine_archive._dispatcher import load_machinery
from scripts.engine_miners._ssot import load_ssot

# (producer module path, engine, producer kind) for every probe-driving
# producer module currently shipped. Mirrors ``_PRODUCER_MODULES`` in
# ``scripts/_probe.py`` plus the SSOT-side ``static`` / ``discovery`` keys.
_PRODUCERS: list[tuple[str, str, str]] = [
    ("scripts.engine_miners.transformers_miner", "transformers", "static"),
    ("scripts.engine_miners.vllm_static_miner", "vllm", "static"),
    ("scripts.engine_miners.tensorrt_static_miner", "tensorrt", "static"),
    ("scripts.engine_introspectors.transformers_introspector", "transformers", "discovery"),
    ("scripts.engine_introspectors.vllm_introspector", "vllm", "discovery"),
    ("scripts.engine_introspectors.tensorrt_introspector", "tensorrt", "discovery"),
]


@pytest.mark.parametrize("module_path,engine,producer", _PRODUCERS)
def test_producer_lazy_landmarks_matches_archive(
    module_path: str, engine: str, producer: str
) -> None:
    """Producer's exposed ``LANDMARKS`` must match the archive's tuple."""
    producer_module = importlib.import_module(module_path)
    landmarks_via_producer = producer_module.LANDMARKS

    ssot = load_ssot(engine)
    library = ssot["library"]
    assert isinstance(library, dict)
    version = str(library["current_version"])

    landmarks_via_archive = load_machinery(
        engine=engine,
        version=version,
        producer=producer,  # type: ignore[arg-type]
    ).LANDMARKS

    assert landmarks_via_producer == landmarks_via_archive, (
        f"Producer {module_path} LANDMARKS drift from archive at "
        f"{engine}=={version} ({producer}). Producer wiring may reference "
        f"the wrong producer kind, the wrong engine, or the wrong SSOT field."
    )
    assert isinstance(landmarks_via_producer, tuple)
    assert len(landmarks_via_producer) > 0
    assert all(isinstance(s, str) for s in landmarks_via_producer)
