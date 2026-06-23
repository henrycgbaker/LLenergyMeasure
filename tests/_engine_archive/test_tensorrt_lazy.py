"""Per-engine LANDMARKS resolution tests for the tensorrt producers at the pin.

Engine-specific complement to ``tests/_engine_archive/test_dispatcher.py``:
asserts the two tensorrt producer shims expose ``LANDMARKS`` lazily via PEP 562
``__getattr__``, that the dispatcher resolves the current pin EXACTLY (no
``<= target`` fallback), and that the shim's resolved tuple matches the pinned
producer's. After the producer dedup the prior pin keeps only its
``landmarks.yaml`` + ``outputs/`` corpus, so there is no cross-version producer
to drift against.

The miner module name is ``tensorrt_static_invariant_miner`` (not
``tensorrt_miner``) - see ``scripts/_drift.py`` ``_PRODUCER_MODULES`` for the
canonical mapping.
"""

from __future__ import annotations

import importlib

from engine_versions._dispatcher import load_producer
from scripts.engine_producers._current import current_version, safe_version

_ENGINE = "tensorrt"

# (shim_module_path, producer_kind) pairs.
_PRODUCERS: tuple[tuple[str, str], ...] = (
    ("scripts.engine_producers.tensorrt_static_invariant_miner", "static_invariant_miner"),
    ("scripts.engine_producers.tensorrt_schema_introspector", "schema_introspector"),
)


def _assert_landmark_shape(landmarks: object) -> tuple[str, ...]:
    """Shared shape contract: non-empty tuple of dotted ``tensorrt_llm.*`` paths."""
    assert isinstance(landmarks, tuple), f"LANDMARKS must be tuple, got {type(landmarks)!r}"
    assert len(landmarks) > 0, "LANDMARKS must be non-empty"
    for entry in landmarks:
        assert isinstance(entry, str), f"LANDMARKS entry must be str, got {type(entry)!r}"
        assert entry.startswith("tensorrt_llm."), (
            f"LANDMARKS entry {entry!r} must start with 'tensorrt_llm.'"
        )
    return landmarks  # type: ignore[return-value]


def test_tensorrt_producer_landmarks_resolve_to_exact_pin() -> None:
    """Both tensorrt shims must dispatch to the pin's vendored producer, exactly.

    For each producer kind, ``load_producer`` at the current pin imports the
    pin's own module (no fallback), the shim's lazy ``LANDMARKS`` matches it, and
    the tuple satisfies the ``tensorrt_llm.*`` shape contract.
    """
    pin = current_version(_ENGINE)
    for module_path, producer in _PRODUCERS:
        pinned = load_producer(engine=_ENGINE, version=pin, producer=producer)
        expected = f"engine_versions.{_ENGINE}.{safe_version(pin)}.producers.{producer}"
        assert pinned.__name__ == expected, module_path
        shim = importlib.import_module(module_path)
        landmarks = _assert_landmark_shape(shim.LANDMARKS)
        assert landmarks == pinned.LANDMARKS, module_path
