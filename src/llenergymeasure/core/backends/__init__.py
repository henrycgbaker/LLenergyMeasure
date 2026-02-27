"""Inference backends for llenergymeasure."""

import importlib.util

from llenergymeasure.core.backends.protocol import InferenceBackend
from llenergymeasure.exceptions import BackendError

__all__ = ["InferenceBackend", "detect_default_backend", "get_backend"]


def detect_default_backend() -> str:
    """Detect the default available backend.

    Returns 'pytorch' if transformers is installed.

    Raises:
        BackendError: If no supported backend is installed.
    """
    if importlib.util.find_spec("transformers") is not None:
        return "pytorch"
    # Future: check vllm, tensorrt_llm
    raise BackendError(
        "No inference backend installed. Install one with: pip install llenergymeasure[pytorch]"
    )


def get_backend(name: str) -> InferenceBackend:
    """Get an inference backend instance by name.

    Args:
        name: Backend name ('pytorch', 'vllm', 'tensorrt').

    Returns:
        An InferenceBackend instance.

    Raises:
        BackendError: If the backend name is unknown.
    """
    if name == "pytorch":
        from llenergymeasure.core.backends.pytorch import PyTorchBackend

        return PyTorchBackend()
    raise BackendError(f"Unknown backend: {name!r}. Available: pytorch")
