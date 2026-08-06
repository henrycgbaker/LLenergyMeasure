"""Engine protocol contracts for inference plugins and the harness interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llenergymeasure.config.models import ExperimentConfig

if TYPE_CHECKING:
    # Server-lifecycle value types live at the infra altitude (the docker /
    # process plumbing lives there). They are referenced only in annotations,
    # so a TYPE_CHECKING import keeps this widely-imported core module free of a
    # runtime infra dependency; runtime_checkable conformance keys on method
    # names, not these types.
    from llenergymeasure.infra.server_lifecycle import (
        ProbeRequest,
        ServerHandle,
        ServerPlacement,
    )


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
    latency_measurement_mode: str | None = None
    """Engine-declared capture mode for ttft_ms/itl_ms (a
    ``LatencyMeasurementMode`` value string, e.g. ``"true_streaming"`` or
    ``"proportional"``). Plain string to keep the protocol dependency-free; the
    harness maps it to the enum. None when the engine performed no streaming/ITL
    capture for this run."""
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
    ) -> tuple[Any, Any]:
        """Load model into memory.

        Returns an opaque ``(model, aux)`` tuple passed verbatim as the ``model``
        argument to warmup/run_inference/cleanup; the second element is
        engine-specific (tokenizer for transformers, sampling params for
        vLLM/TRT-LLM).

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
        """Return host/engine compatibility errors (empty list when compatible).

        The single preflight compatibility hook: it carries host-GPU hardware
        gates AND engine-specific config/checkpoint compatibility (e.g. TRT-LLM
        rejecting HF pre-quantised AWQ/GPTQ checkpoints).

        - Never raises; errors propagate via the returned list.
        - GPU-dependent gates return ``[]`` when no GPU is visible (containers
          without a visible device must not block at config time); GPU-independent
          config/checkpoint checks may still return errors in that case.
        - No weight loading, no GPU allocation, no engine construction. May read
          model metadata (e.g. a checkpoint ``config.json``), not weights.
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
        :func:`llenergymeasure.engines._observed.assemble_observed_params`.

        Engines must never raise; any extraction failure should be caught and
        logged at DEBUG, returning empty dicts for the failed sub-field.
        """
        ...


@runtime_checkable
class ServerCapable(Protocol):
    """Additive server-lifecycle extension for engines that can serve online.

    This is a SIBLING of :class:`EnginePlugin`'s single-call ``run_inference``
    contract, never a tightening of it: an engine opts into
    online-serving measurement by ALSO implementing these three methods, and the
    offline ``run_inference`` surface is untouched. An engine "claims server
    support" only by implementing all three - there are no partial
    implementations, and the readiness probe is a required member:
    an engine cannot be server-capable without driving a real request through
    its serving path.

    Lifecycle (a long-lived server, parallel to the run-to-completion batch
    dispatch):

    - :meth:`launch` starts the server (a sibling container under container
      placement, or a host subprocess under process placement) and returns a
      :class:`~llenergymeasure.infra.server_lifecycle.ServerHandle` exposing the
      base URL, the process/container identity, and log access.
    - :meth:`await_ready` polls liveness THEN drives a real inference request
      through the serving path; readiness is satisfied ONLY when that request
      completes (``/health`` alone never suffices). The server-lifecycle layer
      owns the probe MECHANICS; the request SHAPE (``probe_request``) is supplied
      by the caller (the server warmup protocol draws it from the measured traffic
      distribution).
    - :meth:`shutdown` stops the server gracefully with a hard-kill escalation;
      it is idempotent and leaves nothing leaked on any exit path.
    """

    def launch(self, config: ExperimentConfig, placement: ServerPlacement) -> ServerHandle:
        """Launch the engine's server and return a handle to it.

        Allocates a free port, resolves the image (container placement) or
        builds the host command (process placement), starts the server, and
        returns immediately (readiness is awaited separately). A launch that
        fails part-way cleans up its own partial state before raising.
        """
        ...

    def await_ready(
        self,
        handle: ServerHandle,
        probe_request: ProbeRequest,
        *,
        timeout: float,
    ) -> None:
        """Block until the server is ready, or raise.

        Liveness poll THEN a real inference request through the serving path.
        Returns ``None`` on success; raises a
        :class:`~llenergymeasure.infra.server_lifecycle.ServerLifecycleError`
        subclass on launch failure, readiness timeout, or an unreachable
        docker-outside-of-docker topology.
        """
        ...

    def shutdown(self, handle: ServerHandle) -> None:
        """Stop the server (graceful, escalating to a hard kill); idempotent."""
        ...
