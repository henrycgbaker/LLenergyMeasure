"""Runtime engine availability detection."""

from __future__ import annotations

from llenergymeasure.config.ssot import ENGINES, Engine


def is_engine_available(engine: str) -> bool:
    """Check if an engine is available (installed and importable).

    Imports the engine's ``availability_probe`` package (from the ``ssot.ENGINES``
    descriptor) via the builtin ``__import__`` - kept over ``importlib`` so an
    unexpected error during import still surfaces rather than being masked.

    Args:
        engine: Engine name ("transformers", "vllm", or "tensorrt").

    Returns:
        True if engine is importable, False otherwise.
    """
    try:
        probe = ENGINES[Engine(engine)].availability_probe
    except (ValueError, KeyError):
        # Unknown engine
        return False
    try:
        __import__(probe)
        return True
    except (ImportError, OSError):
        # ImportError: package not installed
        # OSError: library dependency missing (tensorrt_llm on some systems)
        return False
