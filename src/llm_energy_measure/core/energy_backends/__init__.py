"""Energy measurement backends for LLM Bench.

This module provides a plugin registry for energy measurement backends.
Each backend implements the EnergyBackend protocol.

Usage:
    # Get the default backend
    backend = get_backend("codecarbon")

    # Use the backend
    tracker = backend.start_tracking()
    # ... run inference ...
    metrics = backend.stop_tracking(tracker)

    # List available backends
    backends = list_backends()

    # Register a custom backend
    register_backend("custom", CustomBackend)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_energy_measure.core.energy_backends.base import EnergyBackend
from llm_energy_measure.core.energy_backends.codecarbon import (
    CodeCarbonBackend,
    CodeCarbonData,
    warm_up,
)
from llm_energy_measure.exceptions import ConfigurationError

if TYPE_CHECKING:
    pass

# Registry of available backends
_BACKENDS: dict[str, type[EnergyBackend]] = {}


def register_backend(name: str, backend_cls: type[EnergyBackend]) -> None:
    """Register an energy backend.

    Args:
        name: Unique name for the backend.
        backend_cls: Backend class implementing EnergyBackend protocol.

    Raises:
        ValueError: If name is already registered.
    """
    if name in _BACKENDS:
        raise ValueError(f"Backend '{name}' is already registered")
    _BACKENDS[name] = backend_cls


def get_backend(name: str, **kwargs: object) -> EnergyBackend:
    """Get an instance of a registered backend.

    Args:
        name: Name of the registered backend.
        **kwargs: Arguments to pass to the backend constructor.

    Returns:
        Instance of the requested backend.

    Raises:
        ConfigurationError: If backend name is not registered.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys()) or "(none)"
        raise ConfigurationError(f"Unknown backend: '{name}'. Available: {available}")
    return _BACKENDS[name](**kwargs)  # type: ignore[return-value]


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        List of registered backend names.
    """
    return list(_BACKENDS.keys())


def clear_backends() -> None:
    """Clear all registered backends.

    Primarily for testing purposes.
    """
    _BACKENDS.clear()


def _register_default_backends() -> None:
    """Register built-in backends."""
    register_backend("codecarbon", CodeCarbonBackend)


# Auto-register default backends on import
_register_default_backends()


__all__ = [
    "CodeCarbonBackend",
    "CodeCarbonData",
    "EnergyBackend",
    "clear_backends",
    "get_backend",
    "list_backends",
    "register_backend",
    "warm_up",
]
