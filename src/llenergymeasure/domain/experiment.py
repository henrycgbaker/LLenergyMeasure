"""Experiment and study result domain models."""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from llenergymeasure.domain.environment import EnvironmentSnapshot
from llenergymeasure.domain.metrics import (
    EnergyBreakdown,
    ExtendedEfficiencyMetrics,
    LatencyStatistics,
    MultiGPUMetrics,
    ThermalThrottleInfo,
    WarmupResult,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


@functools.lru_cache(maxsize=128)
def _hash_canonical(canonical: str) -> str:
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_declared_config_hash(config: ExperimentConfig) -> str:
    """SHA-256[:16] of ExperimentConfig. Layer 3 fields excluded by design.

    Layer 3 fields (datacenter_pue, grid_carbon_intensity) are not in
    ExperimentConfig (they live in user config only), so model_dump()
    naturally excludes them. No special exclusion logic needed.
    """
    canonical = json.dumps(config.model_dump(), sort_keys=True)
    return _hash_canonical(canonical)


def mj_per_token(energy_j: float, total_tokens: float) -> float | None:
    """Millijoules per token. Returns None when total_tokens is non-positive.

    ``total_tokens`` is usually an integer count, but the windowed/steady-state
    measurement modes attribute a fractional token share to a sub-window, so a float
    is accepted.
    """
    return (energy_j / total_tokens * 1000.0) if total_tokens > 0 else None


class AggregationMetadata(BaseModel):
    """Metadata about the aggregation process."""

    method: str = Field(
        default="sum_energy_avg_throughput",
        description="Aggregation method used",
    )
    num_processes: int = Field(..., description="Number of processes aggregated")
    temporal_overlap_verified: bool = Field(
        default=False, description="Whether process timestamps overlapped"
    )
    gpu_attribution_verified: bool = Field(
        default=False, description="Whether GPU IDs were unique (no double counting)"
    )
    warnings: list[str] = Field(default_factory=list, description="Aggregation warnings")


class ExperimentResult(BaseModel):
    """Experiment result - the user-visible output of a measurement run.

    Produced once per single-process measurement run by the harness. Holds the
    final metrics (energy, throughput, FLOPs, latency) directly; there is no
    per-process breakdown.
    """

    # Identity
    schema_version: str = Field(default="4.0", description="Result schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    measurement_config_hash: str = Field(
        ..., description="SHA-256[:16] of ExperimentConfig (environment excluded)"
    )
    llenergymeasure_version: str | None = Field(
        default=None, description="Package version that produced this result"
    )

    # Engine
    engine: str = Field(default="transformers", description="Inference engine used")
    engine_version: str | None = Field(
        default=None, description="Engine version string for reproducibility"
    )
    model_name: str = Field(default="unknown", description="Model name/path used")

    # Methodology
    measurement_methodology: Literal["total", "steady_state", "windowed"] = Field(
        ..., description="What was measured - total run, steady-state window, or explicit window"
    )
    steady_state_window: tuple[float, float] | None = Field(
        default=None,
        description="(start_sec, end_sec) of measurement window relative to experiment start",
    )
    measurement_window_discard_fraction: float | None = Field(
        default=None,
        description="Warm-up fraction discarded for steady_state methodology. None for "
        "total/windowed.",
    )
    steady_state_not_detected: bool = Field(
        default=False,
        description="True when steady_state auto-detection was requested but found no "
        "stable region and fell back to the fixed warm-up discard.",
    )

    # Core metrics
    total_tokens: int = Field(..., description="Total tokens across all processes")
    total_energy_j: float = Field(..., description="Total energy (sum across processes)")
    total_inference_time_sec: float = Field(..., description="Total inference time")
    avg_tokens_per_second: float = Field(..., description="Average throughput")
    avg_energy_per_token_j: float = Field(..., description="Average energy per token")
    mj_per_tok_adjusted: float | None = Field(
        default=None,
        description="Millijoules per token from adjusted (baseline-subtracted) energy. "
        "None when no baseline measurement was taken.",
    )
    mj_per_tok_total: float | None = Field(
        default=None,
        description="Millijoules per token from total (unadjusted) energy.",
    )
    total_flops: float = Field(..., description="Total FLOPs (reference metadata)")

    # FLOPs derived fields (computed from total_flops + token/time denominators)
    flops_per_output_token: float | None = Field(
        default=None,
        description="FLOPs per output (decode) token. None if total_flops=0 or output_tokens=0.",
    )
    flops_per_input_token: float | None = Field(
        default=None,
        description="FLOPs per input (prefill) token. None if total_flops=0 or input_tokens=0.",
    )
    flops_per_second: float | None = Field(
        default=None,
        description="FLOPs throughput (total_flops / inference_time_sec). None if time=0 or flops=0.",
    )

    # Energy detail
    baseline_power_w: float | None = Field(
        default=None, description="Idle GPU power (W) measured before experiment"
    )
    energy_adjusted_j: float | None = Field(
        default=None, description="Baseline-subtracted energy attributable to inference"
    )
    energy_per_device_j: list[float] | None = Field(
        default=None, description="Per-GPU energy breakdown (Zeus backend only)"
    )
    energy_breakdown: EnergyBreakdown | None = Field(
        default=None, description="Detailed energy breakdown with baseline adjustment"
    )

    # Multi-GPU (from result-schema.md design)
    multi_gpu: MultiGPUMetrics | None = Field(
        default=None, description="Multi-GPU metrics. None for single-GPU runs."
    )

    # Quality
    measurement_warnings: list[str] = Field(
        default_factory=list,
        description="Measurement quality warnings (e.g., short duration, thermal drift)",
    )
    warmup_excluded_samples: int | None = Field(
        default=None,
        description="Warmup iterations run before the measurement window "
        "(from WarmupResult.iterations_completed). None when no warmup result.",
    )
    reproducibility_notes: str = Field(
        default=(
            "Energy measured via NVML polling. Accuracy +/-5%. "
            "Results may vary with thermal state and system load."
        ),
        description="Fixed disclaimer about measurement accuracy",
    )

    # Timeseries sidecar reference
    timeseries: str | None = Field(
        default=None,
        description="Relative filename of timeseries sidecar (e.g. 'timeseries.parquet')",
    )

    # Environment sidecar (loaded from environment.json by load_result; excluded
    # from result.json serialisation - the sidecar is the on-disk home).
    environment: EnvironmentSnapshot | None = Field(
        default=None,
        exclude=True,
        description="Hardware/runtime environment loaded from the environment.json sidecar. "
        "None when no sidecar is present. Not serialised back into result.json.",
    )

    # Timestamps
    start_time: datetime = Field(..., description="Earliest process start time")
    end_time: datetime = Field(..., description="Latest process end time")

    aggregation: AggregationMetadata | None = Field(
        default=None, description="Aggregation metadata (method, num_processes)"
    )

    # Optional detail fields used by aggregation/CLI
    thermal_throttle: ThermalThrottleInfo | None = Field(
        default=None, description="GPU thermal and power throttling information"
    )
    warmup_result: WarmupResult | None = Field(
        default=None, description="Warmup convergence detection result"
    )
    latency_stats: LatencyStatistics | None = Field(
        default=None,
        description="Computed TTFT/ITL statistics from streaming inference",
    )
    extended_metrics: ExtendedEfficiencyMetrics | None = Field(
        default=None, description="Extended efficiency metrics (when computed)"
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def duration_sec(self) -> float:
        """Total experiment duration."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def tokens_per_joule(self) -> float:
        """Overall energy efficiency."""
        if self.total_energy_j > 0:
            return self.total_tokens / self.total_energy_j
        return 0.0


class StudySummary(BaseModel):
    """Computed aggregate statistics for a study run."""

    total_experiments: int = Field(default=0, description="Total experiments in the study")
    completed: int = Field(default=0, description="Number of successfully completed experiments")
    failed: int = Field(default=0, description="Number of failed experiments")
    total_wall_time_s: float = Field(default=0.0, description="Total wall-clock time in seconds")
    total_energy_j: float = Field(default=0.0, description="Total energy consumed in joules")
    unique_configurations: int | None = Field(
        default=None,
        description="Number of distinct experiment configurations (total_experiments / n_cycles)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Runtime warnings (CLI narrowing, failures, etc.)",
    )


class StudyResult(BaseModel):
    """Final return value of a study run.

    Distinct from StudyManifest (the in-progress checkpoint). StudyResult is
    assembled once after all experiments complete (or after interrupt) and returned
    to the caller.
    """

    experiments: list[ExperimentResult] = Field(
        default_factory=list, description="Results for each experiment in the study"
    )
    study_name: str | None = Field(default=None, description="Study name")
    study_design_hash: str | None = Field(
        default=None, description="16-char SHA-256 hex of experiment list"
    )
    measurement_protocol: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Flat dict from ExecutionConfig: n_cycles, experiment_order, experiment_gap_seconds, "
            "cycle_gap_seconds, shuffle_seed, experiment_timeout_seconds"
        ),
    )
    result_files: list[str] = Field(
        default_factory=list,
        description="Paths to per-experiment result.json files (paths, not embedded)",
    )
    summary: StudySummary = Field(
        default_factory=StudySummary,
        description="Computed aggregate statistics (counts, totals, warnings)",
    )
    skipped_experiments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Grid points skipped due to validation errors (raw_config + reason + errors)",
    )
