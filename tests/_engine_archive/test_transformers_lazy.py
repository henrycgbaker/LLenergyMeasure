"""Per-engine LANDMARKS resolution tests for transformers v4.57.3.

Engine-specific complement to ``tests/_engine_archive/test_dispatcher.py``:
asserts the two transformers producer modules expose ``LANDMARKS`` lazily
via PEP 562 ``__getattr__`` and that the resolved tuple matches the
machinery loaded directly through the dispatcher.

Mirror style of ``test_dispatcher.py``: keep imports lazy where the assertion
target is the module attribute, fail loud otherwise.
"""

from __future__ import annotations

import importlib

from engine_versions._dispatcher import load_producer

_TRANSFORMERS_VERSION = "4.57.3"


def _assert_landmark_shape(landmarks: object) -> tuple[str, ...]:
    """Shared shape contract: non-empty tuple of dotted ``transformers.*`` paths."""
    assert isinstance(landmarks, tuple), f"LANDMARKS must be tuple, got {type(landmarks)!r}"
    assert len(landmarks) > 0, "LANDMARKS must be non-empty"
    for entry in landmarks:
        assert isinstance(entry, str), f"LANDMARKS entry must be str, got {type(entry)!r}"
        assert entry.startswith("transformers."), (
            f"LANDMARKS entry {entry!r} must start with 'transformers.'"
        )
    return landmarks  # type: ignore[return-value]


def test_transformers_miner_landmarks_resolve_lazily() -> None:
    """``scripts.engine_producers.transformers_miner.LANDMARKS`` resolves via PEP 562."""
    module = importlib.import_module("scripts.engine_producers.transformers_miner")
    landmarks = _assert_landmark_shape(module.LANDMARKS)

    archived = load_producer(
        engine="transformers", version=_TRANSFORMERS_VERSION, producer="static_invariant_miner"
    ).LANDMARKS
    assert landmarks == archived


def test_transformers_schema_introspector_landmarks_resolve_lazily() -> None:
    """``scripts.engine_producers.transformers_schema_introspector.LANDMARKS`` resolves via PEP 562."""
    module = importlib.import_module("scripts.engine_producers.transformers_schema_introspector")
    landmarks = _assert_landmark_shape(module.LANDMARKS)

    archived = load_producer(
        engine="transformers", version=_TRANSFORMERS_VERSION, producer="schema_introspector"
    ).LANDMARKS
    assert landmarks == archived
