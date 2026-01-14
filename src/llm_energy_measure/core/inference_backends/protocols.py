"""Protocol definitions for inference backends.

This module defines the unified interface for inference backends (PyTorch, vLLM, TensorRT-LLM),
enabling a modular plug-in architecture where backends can be swapped without changing
the orchestration layer.

Design Philosophy:
- Backends handle BOTH model loading AND inference (vLLM requires this)
- Each backend uses native types internally (torch.Tensor, RequestOutput, etc.)
- Conversion to BackendResult happens at the boundary
- Config validation allows robust handling of backend-specific parameters
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.domain.model_info import ModelInfo


@dataclass
class BackendResult:
    """Backend-agnostic inference result.

    All backends convert their native output types to this common format
    at the inference boundary. This enables unified metrics collection
    regardless of the underlying backend.
    """

    # Token counts
    total_tokens: int
    input_tokens: int
    output_tokens: int

    # Timing
    inference_time_sec: float

    # Per-batch timing (if available)
    batch_latencies_ms: list[float] = field(default_factory=list)

    # Optional: raw outputs for debugging/analysis (not torch tensors)
    output_texts: list[str] | None = None

    # Backend-specific metadata
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Throughput in tokens/second."""
        if self.inference_time_sec > 0:
            return self.total_tokens / self.inference_time_sec
        return 0.0

    @property
    def latency_per_token_ms(self) -> float:
        """Average latency per token in milliseconds."""
        if self.total_tokens > 0:
            return (self.inference_time_sec * 1000) / self.total_tokens
        return 0.0


@dataclass
class BackendRuntime:
    """Runtime context provided to backends during initialization.

    This abstraction decouples backends from HuggingFace Accelerate,
    allowing vLLM and TensorRT-LLM to manage their own distributed execution.

    Attributes:
        device: Target device (None if backend manages devices internally)
        process_index: Index of current process in distributed setup
        num_processes: Total number of processes
        is_main_process: Whether this is the main/rank-0 process
        accelerator: HuggingFace Accelerator (only for PyTorch backend)
    """

    device: Any | None  # torch.device or None
    process_index: int
    num_processes: int
    is_main_process: bool
    accelerator: Any | None = None  # Accelerator, only for PyTorch

    @property
    def manages_distribution(self) -> bool:
        """Whether backend should handle its own multi-GPU distribution."""
        return self.accelerator is None


@dataclass
class ConfigWarning:
    """Warning about config/backend compatibility.

    Used when a config parameter is set but not supported by the selected backend,
    or when parameter semantics differ between backends.
    """

    param: str
    message: str
    severity: Literal["warning", "error"] = "warning"
    suggestion: str | None = None


@runtime_checkable
class InferenceBackend(Protocol):
    """Unified interface for inference backends.

    Backends handle both model loading and inference, allowing each to use
    its native APIs internally. The only shared interface point is the
    BackendResult returned from run_inference().

    Implementations:
    - PyTorchBackend: HuggingFace Transformers + Accelerate
    - VLLMBackend: vLLM with PagedAttention and continuous batching
    - TensorRTBackend: TensorRT-LLM with compiled inference (future)

    Example:
        backend = get_backend("vllm")
        if not backend.is_available():
            raise BackendNotAvailableError("vllm")

        warnings = backend.validate_config(config)
        for w in warnings:
            logger.warning(f"{w.param}: {w.message}")

        backend.initialize(config, runtime)
        result = backend.run_inference(prompts, config)
        backend.cleanup()
    """

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'pytorch', 'vllm', 'tensorrt')."""
        ...

    @property
    def version(self) -> str:
        """Backend version string for result tracking."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is installed and usable.

        Returns:
            True if the backend's dependencies are available.
        """
        ...

    def initialize(self, config: "ExperimentConfig", runtime: BackendRuntime) -> None:
        """Load model and prepare for inference.

        The backend manages its own model state internally. For vLLM/TensorRT,
        this includes setting up tensor parallelism and memory allocation.

        Args:
            config: Experiment configuration with model, precision, sharding settings.
            runtime: Runtime context (devices, process info, optional accelerator).

        Raises:
            BackendInitializationError: If model loading fails.
        """
        ...

    def run_inference(self, prompts: list[str], config: "ExperimentConfig") -> BackendResult:
        """Run inference on prompts.

        Args:
            prompts: List of input prompts.
            config: Configuration for decoder params, batching, etc.

        Returns:
            BackendResult with token counts, timing, and optional outputs.

        Raises:
            BackendInferenceError: If inference fails.
        """
        ...

    def cleanup(self) -> None:
        """Release resources (GPU memory, processes, etc.).

        Called after inference completes, even on error. Should be idempotent.
        """
        ...

    def get_model_info(self) -> "ModelInfo":
        """Return model metadata for metrics and reporting.

        Returns:
            ModelInfo with parameter count, architecture details, etc.
        """
        ...

    def get_supported_params(self) -> set[str]:
        """Return set of config parameter names this backend supports.

        Used for config validation to detect unsupported parameters early.

        Returns:
            Set of parameter names (e.g., {'temperature', 'top_p', 'batch_size'}).
        """
        ...

    def validate_config(self, config: "ExperimentConfig") -> list[ConfigWarning]:
        """Validate config compatibility with this backend.

        Checks for:
        - Unsupported parameters (error or warning based on criticality)
        - Semantic differences (e.g., batch_size meaning differs)
        - Deprecated options

        Args:
            config: Configuration to validate.

        Returns:
            List of warnings/errors. Empty list means config is fully compatible.
        """
        ...
