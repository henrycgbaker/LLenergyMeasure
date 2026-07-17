"""OOM and import error helpers for inference engine implementations.

Extracted from the repeated patterns in transformers.py, vllm.py, and
tensorrt.py to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.utils.exceptions import EngineError


def is_oom_error(exc: Exception) -> bool:
    """Check whether an exception is a CUDA out-of-memory error."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def raise_engine_error(exc: Exception, engine_name: str, *, hint: str = "") -> None:
    """Raise a EngineError wrapping *exc* with a user-friendly message.

    If the error is OOM, includes remediation hints. Otherwise wraps
    generically as "<engine> inference failed".

    Args:
        exc: The original exception.
        engine_name: Human-readable engine name (e.g. "vLLM", "TRT-LLM").
        hint: Extra remediation text appended to OOM messages.
    """
    if is_oom_error(exc):
        msg = f"{engine_name} CUDA out of memory."
        if hint:
            msg = f"{msg} Try: {hint}"
        raise EngineError(f"{msg} Original error: {exc}") from exc
    raise EngineError(f"{engine_name} inference failed: {exc}") from exc


def require_import(module: str) -> Any:
    """Import *module* or raise EngineError pointing at the Docker contract.

    Engine libraries (`transformers`, `vllm`, `tensorrt_llm`) only resolve
    inside their respective Docker images. A host import failure is the
    expected state when running `llem` outside a container - surface that
    plainly rather than suggesting a host install that no longer exists.

    Args:
        module: Fully-qualified module name (e.g. "vllm").

    Returns:
        The imported module object.
    """
    try:
        import importlib

        return importlib.import_module(module)
    except ImportError as e:
        raise EngineError(
            f"{module} is not available on host. Engine code runs inside Docker - "
            "see https://henrycgbaker.github.io/llenergymeasure/contributing/development "
            "for the build/run pattern."
        ) from e
