"""Per-engine lazy-LANDMARKS tests for vLLM at v0.7.3.

Asserts that the producer modules' PEP 562 ``__getattr__`` hooks resolve
``LANDMARKS`` to the same tuple that ``load_machinery`` returns directly
from the per-version archive subpackage. This catches drift between the
producer-side wiring (script ``_get_landmarks`` + dispatcher call) and
the archive contents (``_engine_archive/vllm/v0_7_3/machinery/*.py``).

Mirror of ``tests/_engine_archive/test_dispatcher.py`` style: pytest,
parametrize across producer kinds, no fixtures.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from llenergymeasure._engine_archive._dispatcher import load_machinery

# (producer_module_path, producer_kind) pairs. The producer kind is the
# SSOT-side name (``static`` / ``discovery``) the dispatcher routes on.
_PRODUCERS: tuple[tuple[str, str], ...] = (
    ("scripts.engine_miners.vllm_static_miner", "static"),
    ("scripts.engine_introspectors.vllm_introspector", "discovery"),
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
    archive at v0.7.3 actually contains - exactly the failure mode the
    archive subpackage is meant to prevent.
    """
    module = _import_producer(module_path)
    direct = load_machinery(engine="vllm", version="0.7.3", producer=producer).LANDMARKS
    assert direct == module.LANDMARKS
