"""Experiment result domain models for LLM Bench."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llenergymeasure.constants import SCHEMA_VERSION
from llenergymeasure.domain.environment import EnvironmentMetadata
from llenergymeasure.domain.metrics import (
    ComputeMetrics,
    EnergyBreakdown,
    EnergyMetrics,
    ExtendedEfficiencyMetrics,
    InferenceMetrics,
    LatencyStatistics,
    ThermalThrottleInfo,
    WarmupResult,
)


class Timestamps(BaseModel):
    """Timing information for an experiment run."""

    start: datetime = Field(..., description="Experiment start time")
    end: datetime = Field(..., description="Experiment end time")
    duration_sec: float = Field(..., description="Duration in seconds")

    @classmethod
    def from_times(cls, start: datetime, end: datetime) -> "Timestamps":
        """Create Timestamps from start and end times."""
        duration = (end - start).total_seconds()
        return cls(start=start, end=end, duration_sec=duration)


class RawProcessResult(BaseModel):
    """Raw metrics from a single process - never aggregated inline.

    This represents the output from one GPU/process during a distributed
    experiment. Raw results are saved individually and aggregated separately.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Result schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    backend: str = Field(default="pytorch", description="Inference backend used")
    backend_version: str | None = Field(
        default=None, description="Backend version string for reproducibility"
    )
    process_index: int = Field(..., description="Process rank in distributed setup")
    gpu_id: int = Field(..., description="GPU device index")
    gpu_name: str = Field(default="", description="GPU model name")
    gpu_is_mig: bool = Field(default=False, description="Whether GPU is a MIG instance")
    gpu_mig_profile: str | None = Field(default=None, description="MIG profile if applicable")
    energy_measurement_warning: str | None = Field(
        default=None,
        description="Warning about energy measurement accuracy (e.g., MIG limitations)",
    )
    energy_tracking_failed: bool = Field(
        default=False,
        description="True if energy tracking failed during this run (metrics are placeholders)",
    )
    config_name: str = Field(..., description="Configuration name for this experiment")
    model_name: str = Field(..., description="Model name/path used")
    timestamps: Timestamps = Field(..., description="Timing information")
    inference_metrics: InferenceMetrics = Field(..., description="Inference performance metrics")
    energy_metrics: EnergyMetrics = Field(..., description="Energy consumption metrics")
    compute_metrics: ComputeMetrics = Field(..., description="Computational metrics")

    # Effective configuration (for reproducibility)
    effective_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Full resolved config (CLI > config file > preset > defaults)",
    )
    cli_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters that were overridden via CLI flags",
    )
    config_warnings: list[str] = Field(
        default_factory=list,
        description="Config validation warnings that were present at runtime",
    )

    # Parameter provenance tracking (new in schema v2)
    parameter_provenance: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Full provenance for each parameter (path -> {value, source, source_detail})",
    )
    preset_chain: list[str] = Field(
        default_factory=list,
        description="Presets applied in order (for preset inheritance tracking)",
    )

    # Extended efficiency metrics (always present, fields null when not computable)
    extended_metrics: ExtendedEfficiencyMetrics = Field(
        default_factory=ExtendedEfficiencyMetrics,
        description="Extended efficiency metrics (TPOT, memory, GPU utilisation, etc.)",
    )

    # Raw data for late aggregation of extended metrics
    per_request_latencies_ms: list[float] = Field(
        default_factory=list,
        description="Per-request E2E latencies for late aggregation",
    )
    gpu_utilisation_samples: list[float] = Field(
        default_factory=list,
        description="GPU utilisation samples for late aggregation",
    )

    # Schema v3: Environment, energy breakdown, thermal, warmup
    environment: EnvironmentMetadata | None = Field(
        default=None,
        description="Hardware/software environment metadata for reproducibility",
    )
    energy_breakdown: EnergyBreakdown | None = Field(
        default=None,
        description="Detailed energy breakdown with baseline adjustment",
    )
    thermal_throttle: ThermalThrottleInfo | None = Field(
        default=None,
        description="GPU thermal and power throttling information",
    )
    warmup_result: WarmupResult | None = Field(
        default=None,
        description="Warmup convergence detection result",
    )
    timeseries_path: str | None = Field(
        default=None,
        description="Path to time-series data file if saved",
    )

    model_config = {"frozen": True}


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


class AggregatedResult(BaseModel):
    """Aggregated experiment result from multiple processes.

    This combines raw results from all processes in a distributed experiment
    into a single result with proper aggregation (sum energy, average throughput).
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Result schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    backend: str = Field(default="pytorch", description="Inference backend used")
    backend_version: str | None = Field(
        default=None, description="Backend version string for reproducibility"
    )
    aggregation: AggregationMetadata = Field(..., description="Aggregation metadata")

    # Aggregated metrics
    total_tokens: int = Field(..., description="Total tokens across all processes")
    total_energy_j: float = Field(..., description="Total energy (sum across processes)")
    total_inference_time_sec: float = Field(..., description="Total inference time")
    avg_tokens_per_second: float = Field(..., description="Average throughput")
    avg_energy_per_token_j: float = Field(..., description="Average energy per token")
    total_flops: float = Field(..., description="Total FLOPs")

    # Per-process breakdown (for debugging/analysis)
    process_results: list[RawProcessResult] = Field(
        default_factory=list, description="Original per-process results"
    )

    # Timestamps
    start_time: datetime = Field(..., description="Earliest process start time")
    end_time: datetime = Field(..., description="Latest process end time")

    # Effective configuration (for reproducibility)
    effective_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Full resolved config (CLI > config file > preset > defaults)",
    )
    cli_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters that were overridden via CLI flags",
    )
    config_warnings: list[str] = Field(
        default_factory=list,
        description="Config validation warnings that were present at runtime",
    )

    # Parameter provenance tracking (new in schema v2)
    parameter_provenance: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Full provenance for each parameter (path -> {value, source, source_detail})",
    )
    preset_chain: list[str] = Field(
        default_factory=list,
        description="Presets applied in order (for preset inheritance tracking)",
    )

    # Streaming latency statistics (computed at aggregation time from raw measurements)
    latency_stats: LatencyStatistics | None = Field(
        default=None,
        description="Computed TTFT/ITL statistics from streaming inference",
    )

    # Energy tracking status
    energy_tracking_failed: bool = Field(
        default=False,
        description="True if any process had energy tracking failures (metrics may be incomplete)",
    )

    # Extended efficiency metrics (aggregated from per-process metrics)
    extended_metrics: ExtendedEfficiencyMetrics = Field(
        default_factory=ExtendedEfficiencyMetrics,
        description="Aggregated extended efficiency metrics",
    )

    # Schema v3: Environment, energy breakdown, thermal
    environment: EnvironmentMetadata | None = Field(
        default=None,
        description="Hardware/software environment metadata (from first process)",
    )
    energy_breakdown: EnergyBreakdown | None = Field(
        default=None,
        description="Aggregated energy breakdown with baseline adjustment",
    )
    thermal_throttle: ThermalThrottleInfo | None = Field(
        default=None,
        description="GPU thermal and power throttling information",
    )
    timeseries_path: str | None = Field(
        default=None,
        description="Path to time-series data file if saved",
    )

    model_config = {"frozen": True}

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
