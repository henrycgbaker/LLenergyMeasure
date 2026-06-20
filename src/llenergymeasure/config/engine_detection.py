"""Runtime engine availability detection."""

from __future__ import annotations

from llenergymeasure.config.ssot import Engine

KNOWN_ENGINES: list[Engine] = list(Engine)


def is_engine_available(engine: str) -> bool:
    """Check if an engine is available (installed and importable).

    Args:
        engine: Engine name ("transformers", "vllm", or "tensorrt").

    Returns:
        True if engine is importable, False otherwise.
    """
    try:
        if engine == Engine.TRANSFORMERS:
            import torch  # noqa: F401
        elif engine == Engine.VLLM:
            import vllm  # noqa: F401
        elif engine == Engine.TENSORRT:
            import tensorrt_llm  # noqa: F401
        else:
            # Unknown engine
            return False
        return True
    except (ImportError, OSError, Exception):
        # ImportError: package not installed
        # OSError: library dependency missing (tensorrt_llm on some systems)
        # Exception: catch-all for any other import-time errors
        return False
