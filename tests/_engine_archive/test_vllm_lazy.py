"""Per-engine lazy-LANDMARKS tests for the vLLM producers at the current pin.

Asserts the producer shim modules' PEP 562 ``__getattr__`` hooks resolve
``LANDMARKS`` through the dispatcher to the pinned per-version producer, and
that the dispatcher resolves the pin EXACTLY (no ``<= target`` fallback to an
older dir). After the producer dedup each engine vendors a single producer at
its current pin; the prior pin keeps only its ``landmarks.yaml`` + ``outputs/``
corpus, so there is no cross-version producer to drift against.

Mirror of ``tests/_engine_archive/test_dispatcher.py`` style: pytest,
parametrize across producer kinds, no fixtures.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from engine_versions._dispatcher import load_producer
from scripts.engine_producers._current import current_version, safe_version

_ENGINE = "vllm"

# (shim_module_path, producer_kind) pairs. The producer kind is the
# dispatcher's producer argument name.
_PRODUCERS: tuple[tuple[str, str], ...] = (
    ("scripts.engine_producers.vllm_static_invariant_miner", "static_invariant_miner"),
    ("scripts.engine_producers.vllm_schema_introspector", "schema_introspector"),
)


def _import_producer(module_path: str) -> ModuleType:
    """Import (or re-import) the shim module fresh.

    Re-import avoids cross-test contamination of the module-global
    ``LANDMARKS`` cache populated by ``_get_landmarks()``.
    """
    return importlib.import_module(module_path)


@pytest.mark.parametrize("module_path,producer", _PRODUCERS)
def test_producer_landmarks_is_non_empty_tuple_of_dotted_paths(
    module_path: str, producer: str
) -> None:
    """The lazy LANDMARKS export must be a non-empty tuple of dotted paths."""
    module = _import_producer(module_path)
    landmarks = module.LANDMARKS  # triggers PEP 562 __getattr__
    assert isinstance(landmarks, tuple), f"{module_path}.LANDMARKS must be a tuple"
    assert len(landmarks) > 0, f"{module_path}.LANDMARKS must be non-empty"
    for entry in landmarks:
        assert isinstance(entry, str), f"LANDMARKS entry {entry!r} must be a string"
        assert "." in entry, f"LANDMARKS entry {entry!r} must be a dotted path"


@pytest.mark.parametrize("module_path,producer", _PRODUCERS)
def test_shim_resolves_to_exact_pin_producer(module_path: str, producer: str) -> None:
    """The shim must dispatch to the pin's own vendored producer, resolved exactly.

    ``load_producer`` at the current pin must import the pin's producer module
    (not fall back to an older ``<= target`` dir), and the shim's lazy
    ``LANDMARKS`` must equal that producer's. A fallback firing here would mean
    the pinned producer dir went missing or the dispatcher mis-resolved the pin.
    """
    pin = current_version(_ENGINE)
    pinned = load_producer(engine=_ENGINE, version=pin, producer=producer)
    expected = f"engine_versions.{_ENGINE}.{safe_version(pin)}.producers.{producer}"
    assert pinned.__name__ == expected
    shim = _import_producer(module_path)
    assert shim.LANDMARKS == pinned.LANDMARKS
