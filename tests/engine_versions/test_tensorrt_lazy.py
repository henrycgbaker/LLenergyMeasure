"""Per-engine LANDMARKS resolution tests for tensorrt-llm v1.2.1.

Engine-specific complement to ``tests/engine_versions/test_dispatcher.py``:
asserts the tensorrt schema-introspector producer module exposes
``LANDMARKS`` lazily via PEP 562 ``__getattr__`` and that the resolved tuple
matches the machinery loaded directly through the dispatcher.

Mirror style of ``test_dispatcher.py``: keep imports lazy where the
assertion target is the module attribute, fail loud otherwise.
"""

from __future__ import annotations

import importlib

from engine_versions._dispatcher import load_producer

_TENSORRT_VERSION = "1.2.1"


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


def test_tensorrt_schema_introspector_landmarks_resolve_lazily() -> None:
    """``scripts.engine_producers.tensorrt_schema_introspector.LANDMARKS`` resolves via PEP 562."""
    module = importlib.import_module("scripts.engine_producers.tensorrt_schema_introspector")
    landmarks = _assert_landmark_shape(module.LANDMARKS)

    archived = load_producer(
        engine="tensorrt", version=_TENSORRT_VERSION, producer="schema_introspector"
    ).LANDMARKS
    assert landmarks == archived
