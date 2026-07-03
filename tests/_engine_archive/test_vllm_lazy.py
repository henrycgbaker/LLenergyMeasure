"""Per-engine lazy-LANDMARKS tests for vLLM at v0.19.1.

Asserts that the producer modules' PEP 562 ``__getattr__`` hooks resolve
``LANDMARKS`` to the same tuple that ``load_producer`` returns directly
from the per-version archive subpackage. This catches drift between the
producer-side wiring (script ``_get_landmarks`` + dispatcher call) and
the archive contents (``_engine_archive/vllm/v0_19_1/producers/*.py``).

Mirror of ``tests/_engine_archive/test_dispatcher.py`` style: pytest,
parametrize across producer kinds, no fixtures.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from engine_versions._dispatcher import load_producer

# (producer_module_path, producer_kind) pairs. The producer kind is the
# dispatcher's producer argument name.
_PRODUCERS: tuple[tuple[str, str], ...] = (
    ("scripts.engine_producers.vllm_static_invariant_miner", "static_invariant_miner"),
    ("scripts.engine_producers.vllm_schema_introspector", "schema_introspector"),
)


def _import_producer(module_path: str) -> ModuleType:
    """Import (or re-import) the producer module fresh.

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
def test_producer_landmarks_matches_dispatcher(module_path: str, producer: str) -> None:
    """Producer-side LANDMARKS must equal what the dispatcher returns directly.

    Drift between these two means the producer module's ``_get_landmarks``
    is resolving a different version (or wiring) than the per-version
    archive at v0.19.1 actually contains - exactly the failure mode the
    archive subpackage is meant to prevent.
    """
    module = _import_producer(module_path)
    direct = load_producer(engine="vllm", version="0.19.1", producer=producer).LANDMARKS
    assert direct == module.LANDMARKS
