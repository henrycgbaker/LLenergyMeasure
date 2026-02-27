"""Pre-flight validation module.

Runs before any GPU allocation or model loading. Collects all failures into a
single PreFlightError so the user sees all problems at once.

Boundary:
    Pydantic handles schema validation (types, enums, missing fields).
    Pre-flight handles runtime checks: backend installed? model accessible? CUDA available?
"""

import importlib.util
import logging
from pathlib import Path

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.exceptions import PreFlightError

logger = logging.getLogger(__name__)

# Map from backend name to the package that provides it.
_BACKEND_PACKAGES: dict[str, str] = {
    "pytorch": "transformers",
    "vllm": "vllm",
    "tensorrt": "tensorrt_llm",
}


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------


def _check_cuda_available() -> bool:
    """Return True if CUDA is available via torch.

    Uses importlib.util.find_spec() first to avoid importing torch when it is
    not installed (heavy module init).
    """
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


def _check_backend_installed(backend: str) -> bool:
    """Return True if the package that provides *backend* is importable."""
    package = _BACKEND_PACKAGES.get(backend)
    if package is None:
        # Unknown backend — Pydantic already blocked invalid values; treat as missing.
        return False
    return importlib.util.find_spec(package) is not None


def _check_model_accessible(model_id: str) -> str | None:
    """Check whether *model_id* is reachable.

    Returns an error string if a definitive failure is detected, None otherwise
    (including when we cannot determine reachability).
    """
    # Local path — starts with /, ./, or ~
    if model_id.startswith("/") or model_id.startswith("./") or model_id.startswith("~"):
        path = Path(model_id).expanduser()
        if not path.exists():
            return f"{model_id} not found — path does not exist"
        return None

    # Hub model — use huggingface_hub if available
    if importlib.util.find_spec("huggingface_hub") is None:
        return None  # Cannot check — skip rather than block

    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return None  # Accessible
    except Exception as exc:
        exc_str = str(exc)
        if "401" in exc_str or "403" in exc_str or "gated" in exc_str.lower():
            return f"{model_id} gated model — no HF_TOKEN → export HF_TOKEN=<your_token>"
        if "404" in exc_str or "not found" in exc_str.lower():
            return f"{model_id} not found on HuggingFace Hub"
        # Network error, timeout, etc. — don't block
        logger.debug("Model accessibility check skipped (network error): %s", exc)
        return None


def _warn_if_persistence_mode_off() -> None:
    """Log a warning if GPU persistence mode is disabled.

    Never raises — always wrapped in a broad except.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
            if mode == pynvml.NVML_FEATURE_DISABLED:
                logger.warning(
                    "GPU persistence mode is off. First experiment may have higher "
                    "latency. Enable: sudo nvidia-smi -pm 1"
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_preflight(config: ExperimentConfig) -> None:
    """Run all pre-flight checks for *config*.

    Collects every failure into a single ``PreFlightError`` so the user sees
    all problems at once. Raises nothing on success.

    Raises:
        PreFlightError: One or more checks failed.
    """
    failures: list[str] = []

    # Check 1: CUDA available
    if not _check_cuda_available():
        failures.append("CUDA not available — is a GPU present and CUDA installed?")

    # Check 2: Backend installed
    if not _check_backend_installed(config.backend):
        package = _BACKEND_PACKAGES.get(config.backend, config.backend)
        failures.append(
            f"{config.backend} not installed — pip install llenergymeasure[{config.backend}]"
            f" (missing: {package})"
        )

    # Check 3: Model accessible
    model_error = _check_model_accessible(config.model)
    if model_error is not None:
        failures.append(model_error)

    if failures:
        n = len(failures)
        lines = "\n".join(f"  \u2717 {f}" for f in failures)
        raise PreFlightError(f"Pre-flight failed: {n} issue(s) found\n{lines}")

    # Non-blocking warning (after all checks pass)
    _warn_if_persistence_mode_off()
