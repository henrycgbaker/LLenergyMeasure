"""Runtime backend availability detection."""

from __future__ import annotations

KNOWN_BACKENDS: list[str] = ["pytorch", "vllm", "tensorrt"]


def is_backend_available(backend: str) -> bool:
    """Check if a backend is available (installed and importable).

    Args:
        backend: Backend name ("pytorch", "vllm", or "tensorrt").

    Returns:
        True if backend is importable, False otherwise.
    """
    try:
        if backend == "pytorch":
            import torch  # noqa: F401
        elif backend == "vllm":
            import vllm  # noqa: F401
        elif backend == "tensorrt":
            import tensorrt_llm  # noqa: F401
        else:
            # Unknown backend
            return False
        return True
    except (ImportError, OSError, Exception):
        # ImportError: package not installed
        # OSError: library dependency missing (tensorrt_llm on some systems)
        # Exception: catch-all for any other import-time errors
        return False


def get_available_backends() -> list[str]:
    """Get list of all available backends on the system.

    Returns:
        List of backend names that are installed and importable.
    """
    return [b for b in KNOWN_BACKENDS if is_backend_available(b)]


def get_backend_install_hint(backend: str) -> str:
    """Get installation command for a backend.

    Args:
        backend: Backend name.

    Returns:
        pip install command string for the backend.
    """
    hints = {
        "pytorch": "pip install llenergymeasure",
        "vllm": "Docker recommended — see docs/deployment.md",
        "tensorrt": "Docker recommended — see docs/deployment.md",
    }
    return hints.get(backend, f"pip install llenergymeasure[{backend}]")
