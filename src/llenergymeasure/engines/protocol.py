"""Engine protocol contracts for inference plugins and the harness interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llenergymeasure.config.models import ExperimentConfig


@dataclass
class InferenceOutput:
    """Minimal output from one engine inference run.

    Engine-specific data (e.g. vLLM RequestOutput objects) goes in extras.
    The harness uses these fields to assemble the full ExperimentResult.

    The extended-metrics fields below are best-effort: an empty list or ``None``
    means the engine could not provide that signal for this run (the harness
    leaves the corresponding result field null). Engine-internal opaque objects
    (model handles, RequestOutput lists) stay in ``extras``.
    """

    elapsed_time_sec: float
    input_tokens: int
    output_tokens: int
    peak_memory_mb: float
    model_memory_mb: float
    batch_times: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    inference_time_sec: float = 0.0  # Set by harness after perf_counter brackets

    # Extended-metrics capture (best-effort; empty/None = engine cannot provide)
    per_request_latencies_ms: list[float] = field(default_factory=list)
    """Per-request end-to-end latency in ms. Empty when the engine cannot
    attribute timing per request (e.g. a single batched call)."""
    ttft_ms: list[float] = field(default_factory=list)
    """Per-request time-to-first-token in ms. Empty for non-streaming engines."""
    itl_ms: list[float] = field(default_factory=list)
    """Inter-token latency samples in ms. Empty for non-streaming engines."""
    num_batches: int | None = None
    """Number of static batches processed. None for continuous batching (vLLM)."""
    padding_tokens: int | None = None
    """Total padding tokens added across batches. None when not measurable
    (continuous batching, or engines that do not pad)."""
    kv_cache_stats: dict[str, Any] | None = None
    """KV-cache stats dict (hit_rate/blocks_used/blocks_total/kv_cache_mb).
    None for engines that do not expose a paged KV cache (Transformers/TRT-LLM)."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class EnginePlugin(Protocol):
    """Contract for thin inference plugins.

    MeasurementHarness owns the full measurement lifecycle (energy tracking,
    CUDA sync, FLOPs estimation, result assembly). Plugins own only inference.
    """

    @property
    def name(self) -> str:
        """Engine identifier (e.g. 'transformers', 'vllm', 'tensorrt')."""
        ...

    @property
    def version(self) -> str:
        """Engine library version string for reproducibility."""
        ...

    def load_model(
        self,
        config: ExperimentConfig,
        on_substep: Callable[[str, float], None] | None = None,
    ) -> Any:
        """Load model into memory. Returns opaque model object passed to warmup/run_inference/cleanup.

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for reporting
                sub-operation progress (e.g. tokenizer loaded, engine compiled).
        """
        ...

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt and return latency in ms.

        Returns 0.0 to signal the harness should skip CV-based convergence
        (e.g. vLLM/TRT-LLM use single-token kernel warmup instead).

        Args:
            config: Experiment configuration.
            model: Opaque model object from load_model().
            prompt: Single warmup prompt text.

        Returns:
            Latency in milliseconds, or 0.0 to opt out of convergence loop.
        """
        ...

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run inference over all prompts.

        Args:
            config: Experiment configuration.
            model: Opaque model object from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.
        """
        ...

    def cleanup(self, model: Any) -> None:
        """Release model from memory and clear CUDA cache."""
        ...

    def check_hardware(self, config: ExperimentConfig) -> list[str]:
        """Return host-GPU compatibility errors (empty list when compatible).

        - Never raises; errors propagate via the returned list.
        - Returns ``[]`` when no GPU is visible (containers without a visible
          device must not block at config time).
        - Pure: no weight loading, no GPU allocation, no engine construction.
        """
        ...

    def capture_observed_params(
        self, config: ExperimentConfig, model: Any, output: InferenceOutput
    ) -> dict[str, Any]:
        """Extract library-observed effective parameters after inference.

        Called by the harness AFTER the NVML measurement window closes
        (post ``t_inference_end`` + ``_cuda_sync``), so the capture overhead
        does not perturb energy or timing measurements.

        Returns a dict with keys ``"engine"``, ``"sampling"``, and
        ``"library_version"`` - same shape as
        :func:`llenergymeasure.engines._helpers.assemble_observed_params`.

        Engines must never raise; any extraction failure should be caught and
        logged at DEBUG, returning empty dicts for the failed sub-field.
        """
        ...
