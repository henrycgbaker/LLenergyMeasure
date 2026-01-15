"""Inference backend registry with lazy loading.

This module provides a registry for inference backends (PyTorch, vLLM, TensorRT-LLM)
with lazy importing to avoid dependency errors when optional backends aren't installed.

Usage:
    from llm_energy_measure.core.inference_backends import get_backend, list_backends

    # Get a backend by name
    backend = get_backend("vllm")

    # Check available backends
    available = list_backends()

The registry follows the same pattern as core/energy_backends/ for consistency.
"""

import importlib
from typing import TYPE_CHECKING

from llm_energy_measure.exceptions import ConfigurationError

from .protocols import (
    BackendResult,
    BackendRuntime,
    ConfigWarning,
    CudaManagement,
    InferenceBackend,
    LaunchMode,
    RuntimeCapabilities,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "BackendResult",
    "BackendRuntime",
    "ConfigWarning",
    "CudaManagement",
    "InferenceBackend",
    "LaunchMode",
    "RuntimeCapabilities",
    "get_backend",
    "is_backend_available",
    "list_backends",
    "register_backend",
]

# Registered backend instances/classes
_BACKENDS: dict[str, type[InferenceBackend]] = {}

# Lazy backend definitions: name -> "module.path:ClassName"
_LAZY_BACKENDS: dict[str, str] = {
    "pytorch": "llm_energy_measure.core.inference_backends.pytorch:PyTorchBackend",
    "vllm": "llm_energy_measure.core.inference_backends.vllm:VLLMBackend",
    "tensorrt": "llm_energy_measure.core.inference_backends.tensorrt:TensorRTBackend",
}

# Install hints for optional backends
_INSTALL_HINTS: dict[str, str] = {
    "vllm": "pip install llm-energy-measure[vllm]",
    "tensorrt": "pip install llm-energy-measure[tensorrt]",
}


def register_backend(name: str, backend_cls: type[InferenceBackend]) -> None:
    """Register a backend class.

    Args:
        name: Backend identifier (e.g., 'pytorch', 'vllm').
        backend_cls: Class implementing InferenceBackend protocol.
    """
    _BACKENDS[name] = backend_cls


def get_backend(name: str) -> InferenceBackend:
    """Get a backend instance by name.

    Uses lazy loading for optional backends (vLLM, TensorRT) to avoid
    import errors when dependencies aren't installed.

    Args:
        name: Backend name ('pytorch', 'vllm', 'tensorrt').

    Returns:
        Instantiated backend.

    Raises:
        ConfigurationError: If backend name is unknown.
        BackendNotAvailableError: If backend dependencies aren't installed.
    """
    # Check if already loaded
    if name in _BACKENDS:
        return _BACKENDS[name]()

    # Try lazy loading
    if name in _LAZY_BACKENDS:
        module_path, class_name = _LAZY_BACKENDS[name].rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            backend_cls: type[InferenceBackend] = getattr(module, class_name)
            _BACKENDS[name] = backend_cls
            return backend_cls()
        except ImportError as e:
            install_hint = _INSTALL_HINTS.get(name)
            hint_msg = f" Install with: {install_hint}" if install_hint else ""
            raise ConfigurationError(
                f"Backend '{name}' requires additional dependencies.{hint_msg}"
            ) from e
        except AttributeError as e:
            raise ConfigurationError(
                f"Backend '{name}' class '{class_name}' not found in module."
            ) from e

    # Unknown backend
    available = list_backends()
    raise ConfigurationError(
        f"Unknown backend '{name}'. Available backends: {', '.join(available)}"
    )


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        List of backend names (includes lazy backends that may not be installed).
    """
    # Combine registered and lazy backends
    all_names = set(_BACKENDS.keys()) | set(_LAZY_BACKENDS.keys())
    return sorted(all_names)


def is_backend_available(name: str) -> bool:
    """Check if a backend is available (installed and usable).

    Args:
        name: Backend name.

    Returns:
        True if backend is installed and its is_available() returns True.
    """
    try:
        backend = get_backend(name)
        return backend.is_available()
    except ConfigurationError:
        return False
