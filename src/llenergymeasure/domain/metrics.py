"""Metrics domain models for LLM Bench."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# FLOPs Result
# =============================================================================


class FlopsResult(BaseModel):
    """FLOPs estimation result with provenance tracking.

    Tracks both the estimated value and the method used to obtain it,
    allowing downstream consumers to understand confidence levels.

    Note: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
    computation happens at FP16 after dequantization.
    """

    value: float = Field(..., description="Estimated FLOPs count")
    method: Literal["calflops", "architecture", "parameter_estimate", "palm_formula"] = Field(
        ..., description="Estimation method used"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level of the estimate"
    )
    precision: str = Field(..., description="Compute precision (e.g., fp16, fp32)")
    notes: str | None = Field(default=None, description="Additional context or warnings")

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid (non-zero) estimate."""
        return self.value > 0


# =============================================================================
# Schema v3: Energy Breakdown, Throttle, Warmup Result
# =============================================================================


class MultiGPUMetrics(BaseModel):
    """Per-device energy breakdown for multi-GPU experiments."""

    num_gpus: int = Field(..., description="Number of GPUs used")
    energy_per_gpu_j: list[float] = Field(..., description="Per-device energy in joules")
    energy_total_j: float = Field(..., description="Sum of energy across all devices")
    energy_per_output_token_j: float = Field(
        ..., description="Primary cross-configuration efficiency metric"
    )


class EnergyBreakdown(BaseModel):
    """Detailed energy breakdown with baseline adjustment.

    Separates raw measured energy from baseline-adjusted values to enable
    accurate attribution of energy to inference work (not idle power).
    """

    raw_j: float = Field(..., description="Total measured energy in Joules")
    adjusted_j: float | None = Field(
        default=None,
        description="Baseline-adjusted energy (raw - baseline * duration) in Joules",
    )
    baseline_power_w: float | None = Field(
        default=None,
        description="Measured baseline idle power in Watts",
    )
    baseline_method: str | None = Field(
        default=None,
        description="How baseline was obtained ('cached', 'validated', 'fresh', 'unavailable')",
    )
    baseline_timestamp: datetime | None = Field(
        default=None,
        description="When baseline power was measured",
    )
    baseline_cache_age_sec: float | None = Field(
        default=None,
        description="Age of cached baseline measurement in seconds",
    )


class ThrottleAxis(BaseModel):
    """One throttling axis (thermal or power): the combined flag plus its hw/sw split.

    ``any`` is the axis-level "did this kind of throttling happen at all"
    indicator; ``hw`` / ``sw`` name the hardware- and software-driven slowdown
    causes NVML reports for that axis. ``any`` is the OR of the two causes.
    """

    any: bool = Field(
        default=False,
        description="Either hardware or software slowdown on this axis was seen.",
    )
    hw: bool = Field(
        default=False,
        description="Hardware slowdown on this axis was seen.",
    )
    sw: bool = Field(
        default=False,
        description="Software slowdown on this axis was seen.",
    )


class ThrottleInfo(BaseModel):
    """GPU throttling information, symmetric across the thermal and power axes.

    Each axis (``thermal``, ``power``) carries a combined ``any`` flag plus the
    ``hw`` / ``sw`` split, so ``throttle.thermal.any`` and ``throttle.power.any``
    are the two top-level "did this axis throttle" questions. Any throttling can
    invalidate energy and performance measurements.
    """

    thermal: ThrottleAxis = Field(
        default_factory=ThrottleAxis,
        description="Thermal throttling (hardware/software thermal slowdown).",
    )
    power: ThrottleAxis = Field(
        default_factory=ThrottleAxis,
        description="Power throttling (hardware power brake / software power cap).",
    )
    throttle_duration_sec: float = Field(
        default=0.0,
        description="Estimated duration of throttling in seconds",
    )
    max_temperature_c: float | None = Field(
        default=None,
        description="Peak GPU temperature during experiment in Celsius",
    )
    throttle_timestamps: list[float] = Field(
        default_factory=list,
        description="Timestamps (seconds from start) when throttle was detected",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def detected(self) -> bool:
        """Whether any throttling (thermal or power) occurred during experiment."""
        return self.thermal.any or self.power.any


class WarmupResult(BaseModel):
    """Result of warmup convergence detection.

    Records whether the warmup phase achieved stable latency (low CV)
    before the measurement phase began.
    """

    converged: bool = Field(..., description="Whether convergence was achieved")
    final_cv: float = Field(..., description="Final coefficient of variation")
    iterations_completed: int = Field(..., description="Number of warmup prompts run")
    target_cv: float = Field(..., description="Configured CV threshold")
    max_prompts: int = Field(..., description="Configured maximum warmup iterations")
    latencies_ms: list[float] = Field(
        default_factory=list,
        description="Warmup latencies in ms (for debugging)",
    )
    thermal_floor_wait_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds spent in thermal floor wait after warmup. Set by caller, not by warmup_until_converged().",
    )


# =============================================================================
# Extended Efficiency Metrics - Consistent schema with conditional computation
# =============================================================================


class MemoryEfficiencyMetrics(BaseModel):
    """Memory efficiency metrics.

    All fields always present in schema. Values are null when not computable.
    """

    # Raw memory values (null when not measured, never a silent 0.0)
    total_vram_mb: float = Field(default=0.0, description="Total GPU VRAM in MB")
    model_memory_mb: float | None = Field(
        default=None,
        description=(
            "Model weights memory after load, before inference (MB). "
            "For out-of-process engines (vLLM V1, TRT-LLM) this is an NVML "
            "whole-device reading (includes CUDA context and co-tenants); for "
            "in-process Transformers it is the torch per-process allocation. "
            "null = not measured (no torch/NVML reading available)."
        ),
    )
    peak_memory_mb: float | None = Field(
        default=None,
        description=(
            "Peak GPU memory during inference measurement window (MB). "
            "Reflects KV cache + activations + batch buffers, not model weights. "
            "For out-of-process engines this is an NVML whole-device reading "
            "(see model_memory_mb). null = not measured."
        ),
    )
    inference_memory_mb: float | None = Field(
        default=None,
        description=(
            "Inference-only memory: peak minus model baseline (MB). "
            "Derived: max(0.0, peak_memory_mb - model_memory_mb), computed by the "
            "harness only when both peak and model baseline were measured. When "
            "both come from NVML the shared whole-device context term cancels in "
            "the subtraction, so the delta is meaningful. "
            "null = not measured or not computable."
        ),
    )
    kv_cache_mb: float | None = Field(default=None, description="KV cache memory in MB (vLLM only)")

    # Derived efficiency metrics (null if not computable)
    tokens_per_gb_vram: float | None = Field(
        default=None, description="Output tokens per GB of VRAM used"
    )
    model_memory_utilisation: float | None = Field(
        default=None, description="Model memory / total VRAM (0.0-1.0)"
    )
    kv_cache_memory_ratio: float | None = Field(
        default=None, description="KV cache / peak memory (vLLM only)"
    )


class GPUUtilisationMetrics(BaseModel):
    """GPU utilisation during inference.

    Collected via pynvml background sampling. Null if pynvml unavailable.
    """

    sm_utilisation_mean: float | None = Field(
        default=None, description="Mean SM utilisation (0-100%)"
    )
    sm_utilisation_samples: int = Field(default=0, description="Number of samples collected")
    memory_bandwidth_utilisation: float | None = Field(
        default=None, description="Memory bandwidth utilisation (0-100%)"
    )


class BatchEfficiencyMetrics(BaseModel):
    """Batch processing efficiency.

    Only applicable for static batching (Transformers, TensorRT). Null for vLLM
    continuous batching.
    """

    effective_batch_size: float | None = Field(
        default=None, description="Average actual batch size"
    )
    batch_utilisation: float | None = Field(
        default=None, description="Actual / configured batch size (0.0-1.0)"
    )
    padding_overhead: float | None = Field(
        default=None, description="Padding tokens / total tokens (0.0-1.0)"
    )
    num_batches: int | None = Field(default=None, description="Number of batches processed")


class KVCacheEfficiencyMetrics(BaseModel):
    """KV cache efficiency metrics.

    vLLM-specific. Always null for Transformers/TensorRT engines.
    """

    kv_cache_hit_rate: float | None = Field(
        default=None, description="Prefix cache hit rate (0.0-1.0, vLLM only)"
    )
    kv_cache_blocks_used: int | None = Field(default=None, description="KV cache blocks used")
    kv_cache_blocks_total: int | None = Field(
        default=None, description="Total KV cache blocks available"
    )


class RequestLatencyMetrics(BaseModel):
    """Per-request end-to-end latency statistics.

    E2E latency = total time from request submission to completion.
    """

    e2e_latency_mean_ms: float | None = Field(
        default=None, description="Mean E2E latency per request"
    )
    e2e_latency_median_ms: float | None = Field(default=None, description="Median E2E latency")
    e2e_latency_p95_ms: float | None = Field(
        default=None, description="95th percentile E2E latency"
    )
    e2e_latency_p99_ms: float | None = Field(
        default=None, description="99th percentile E2E latency"
    )
    e2e_latency_samples: int = Field(default=0, description="Number of request samples")


class ExtendedEfficiencyMetrics(BaseModel):
    """Extended efficiency metrics container.

    Consistent schema - all fields always present in results.
    Individual values are null when not computable for the configuration.

    Design principles:
    - Graceful degradation: null values, never errors
    - Engine-agnostic where possible
    - Late aggregation: raw samples stored, stats computed at aggregation
    """

    # Core efficiency metrics
    tpot_ms: float | None = Field(
        default=None,
        description="Time Per Output Token in ms (ITL mean, streaming only)",
    )
    token_efficiency_index: float | None = Field(
        default=None,
        description="Composite: throughput * tokens_per_joule * precision_factor",
    )

    # Grouped metrics (always present as objects, internal fields may be null)
    memory: MemoryEfficiencyMetrics = Field(
        default_factory=MemoryEfficiencyMetrics,
        description="Memory efficiency metrics",
    )
    gpu_utilisation: GPUUtilisationMetrics = Field(
        default_factory=GPUUtilisationMetrics,
        description="GPU utilisation during inference",
    )
    batch: BatchEfficiencyMetrics = Field(
        default_factory=BatchEfficiencyMetrics,
        description="Batch efficiency (static batching only)",
    )
    kv_cache: KVCacheEfficiencyMetrics = Field(
        default_factory=KVCacheEfficiencyMetrics,
        description="KV cache efficiency (vLLM only)",
    )
    request_latency: RequestLatencyMetrics = Field(
        default_factory=RequestLatencyMetrics,
        description="Per-request E2E latency statistics",
    )


# =============================================================================
# Latency Measurement Types - For streaming inference metrics
# =============================================================================


class LatencyMeasurementMode(Enum):
    """How latency measurements were obtained.

    Different engines have different latency measurement capabilities:
    - Transformers: True per-token timestamps via TextIteratorStreamer
    - vLLM/TensorRT: May use proportional estimation from total time

    This enum makes the measurement semantics explicit in results.
    """

    TRUE_STREAMING = "true_streaming"
    """Actual per-token timestamps captured via streaming API.

    Most accurate method. Each token timestamp represents when that
    specific token was generated. Transformers engine achieves this via
    TextIteratorStreamer callback.
    """

    PER_REQUEST_BATCH = "per_request_batch"
    """Per-request timing without streaming.

    Measures total request latency but may estimate ITL by dividing
    total time by token count. Less accurate than true streaming.
    """

    PROPORTIONAL_ESTIMATE = "proportional"
    """Estimated from total inference time.

    ITL calculated by distributing total time proportionally across
    tokens. Least accurate - used as fallback when streaming not available.
    """


@dataclass
class LatencyStatistics:
    """Computed statistics from raw latency measurements.

    Created at aggregation time from raw TTFT/ITL sample lists. This is the
    final form stored in ExperimentResult and displayed in CLI output.

    Primary metrics use trimmed ITL (excluding first/last tokens per request).
    Full ITL stats are provided for comparison/debugging.
    """

    # TTFT statistics
    ttft_mean_ms: float
    ttft_median_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_min_ms: float
    ttft_max_ms: float
    ttft_samples: int

    # ITL statistics (trimmed - primary metric)
    itl_mean_ms: float | None = None
    itl_median_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    itl_samples: int = 0

    # ITL statistics (full - for comparison)
    itl_full_mean_ms: float | None = None
    itl_full_p99_ms: float | None = None

    # Provenance: how these latency measurements were obtained.
    measurement_mode: LatencyMeasurementMode = LatencyMeasurementMode.TRUE_STREAMING


def compute_latency_statistics(
    ttft_ms: list[float],
    itl_trimmed_ms: list[float] | None = None,
    itl_full_ms: list[float] | None = None,
    measurement_mode: LatencyMeasurementMode = LatencyMeasurementMode.TRUE_STREAMING,
) -> LatencyStatistics | None:
    """Compute TTFT/ITL statistics from flat sample lists.

    Single-process helper: takes raw sample lists collected during one run and
    computes mean/median/p95/p99/min/max plus sample counts. Trimmed ITL is the
    primary metric; full ITL is provided for comparison.

    Args:
        ttft_ms: Per-request time-to-first-token samples in ms.
        itl_trimmed_ms: Trimmed inter-token latency samples (first/last excluded).
        itl_full_ms: All inter-token latency samples.
        measurement_mode: How these measurements were obtained (provenance).
            Defaults to TRUE_STREAMING.

    Returns:
        LatencyStatistics, or None when ttft_ms is empty.
    """
    import numpy as np

    if not ttft_ms:
        return None

    ttft_arr = np.array(ttft_ms)

    # ITL statistics (trimmed - primary metric)
    itl_mean_ms: float | None = None
    itl_median_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    itl_samples = 0

    if itl_trimmed_ms:
        itl_arr = np.array(itl_trimmed_ms)
        itl_mean_ms = float(np.mean(itl_arr))
        itl_median_ms = float(np.median(itl_arr))
        itl_p95_ms = float(np.percentile(itl_arr, 95))
        itl_p99_ms = float(np.percentile(itl_arr, 99))
        itl_samples = len(itl_trimmed_ms)

    # ITL full statistics (for comparison)
    itl_full_mean_ms: float | None = None
    itl_full_p99_ms: float | None = None

    if itl_full_ms:
        itl_full_arr = np.array(itl_full_ms)
        itl_full_mean_ms = float(np.mean(itl_full_arr))
        itl_full_p99_ms = float(np.percentile(itl_full_arr, 99))

    return LatencyStatistics(
        ttft_mean_ms=float(np.mean(ttft_arr)),
        ttft_median_ms=float(np.median(ttft_arr)),
        ttft_p95_ms=float(np.percentile(ttft_arr, 95)),
        ttft_p99_ms=float(np.percentile(ttft_arr, 99)),
        ttft_min_ms=float(np.min(ttft_arr)),
        ttft_max_ms=float(np.max(ttft_arr)),
        ttft_samples=len(ttft_ms),
        itl_mean_ms=itl_mean_ms,
        itl_median_ms=itl_median_ms,
        itl_p95_ms=itl_p95_ms,
        itl_p99_ms=itl_p99_ms,
        itl_samples=itl_samples,
        itl_full_mean_ms=itl_full_mean_ms,
        itl_full_p99_ms=itl_full_p99_ms,
        measurement_mode=measurement_mode,
    )


def collect_itl_measurements(
    token_timestamps_per_request: list[list[float]],
) -> tuple[list[float], list[float], int]:
    """Calculate ITL metrics from per-token timestamps.

    Standard implementation used by all engines for consistent ITL calculation.
    Extracts inter-token latencies from timestamp lists, optionally trimming
    first/last intervals per request for cleaner statistics.

    Args:
        token_timestamps_per_request: Per-request list of token arrival times (ms).
            Each inner list contains cumulative timestamps for one request.

    Returns:
        Tuple of (itl_full, itl_trimmed, excluded_count):
            - itl_full: All inter-token intervals
            - itl_trimmed: Excluding first/last per request (cleaner for percentiles)
            - excluded_count: Number of excluded intervals
    """
    import numpy as np

    itl_full: list[float] = []
    itl_trimmed: list[float] = []
    excluded = 0

    for timestamps in token_timestamps_per_request:
        if len(timestamps) < 2:
            continue

        # Calculate inter-token intervals
        intervals = list(np.diff(timestamps))
        itl_full.extend(intervals)

        # Trim first and last intervals for cleaner statistics
        # First interval may include warmup effects, last may have EOS anomalies
        if len(intervals) >= 3:
            itl_trimmed.extend(intervals[1:-1])
            excluded += 2
        elif len(intervals) >= 1:
            # Too short to trim meaningfully
            excluded += len(intervals)

    return itl_full, itl_trimmed, excluded


def compute_cv(values: list[float]) -> float:
    """Compute the coefficient of variation (std / mean) for *values*.

    Returns 0.0 when the mean is zero or negative (avoids division by zero).
    """
    import numpy as np

    mean = float(np.mean(values))
    if mean <= 0:
        return 0.0
    return float(np.std(values)) / mean
