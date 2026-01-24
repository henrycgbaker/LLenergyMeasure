"""Experiment result domain models for LLM Bench."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llenergymeasure.constants import SCHEMA_VERSION
from llenergymeasure.domain.metrics import (
    ComputeMetrics,
    EnergyMetrics,
    ExtendedEfficiencyMetrics,
    InferenceMetrics,
    LatencyStatistics,
)


class CycleStatistics(BaseModel):
    """Statistical aggregation across multiple cycles.

    Follows academic benchmarking standards (TokenPowerBench, MLPerf):
    - Typically 3-10 repetitions for statistical robustness
    - Reports mean ± standard deviation
    - 95% confidence intervals for key metrics
    """

    num_cycles: int = Field(..., description="Number of cycles executed")

    # Energy statistics
    energy_mean_j: float = Field(..., description="Mean total energy (Joules)")
    energy_std_j: float = Field(..., description="Standard deviation of energy")
    energy_ci_95_lower: float = Field(..., description="95% CI lower bound")
    energy_ci_95_upper: float = Field(..., description="95% CI upper bound")

    # Throughput statistics
    throughput_mean_tps: float = Field(..., description="Mean throughput (tokens/second)")
    throughput_std_tps: float = Field(..., description="Standard deviation of throughput")
    throughput_ci_95_lower: float = Field(..., description="95% CI lower bound")
    throughput_ci_95_upper: float = Field(..., description="95% CI upper bound")

    # Efficiency statistics (tokens per joule)
    efficiency_mean_tpj: float = Field(..., description="Mean efficiency (tokens/joule)")
    efficiency_std_tpj: float = Field(..., description="Standard deviation of efficiency")

    # Latency statistics
    latency_mean_ms: float = Field(..., description="Mean latency per token (ms)")
    latency_std_ms: float = Field(..., description="Standard deviation of latency")

    # Coefficient of variation (useful for benchmarking)
    energy_cv: float = Field(
        default=0.0,
        description="Coefficient of variation for energy (std/mean), lower is more stable",
    )
    throughput_cv: float = Field(
        default=0.0,
        description="Coefficient of variation for throughput (std/mean), lower is more stable",
    )


class CycleMetadata(BaseModel):
    """Per-cycle metadata for tracking experimental conditions."""

    cycle_id: int = Field(..., description="Cycle index (0-based)")
    timestamp: datetime = Field(..., description="Cycle start timestamp")
    gpu_temperature_c: float | None = Field(
        default=None, description="GPU temperature at cycle start (if available)"
    )
    system_load: float | None = Field(
        default=None, description="System CPU load at cycle start (if available)"
    )


class MultiCycleResult(BaseModel):
    """Aggregated results from multi-cycle experiments.

    Multi-cycle experiments run the same configuration multiple times to
    establish statistical robustness, following academic benchmarking practices.
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Result schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    backend: str = Field(default="pytorch", description="Inference backend used")
    backend_version: str | None = Field(
        default=None, description="Backend version string for reproducibility"
    )
    num_cycles: int = Field(..., description="Total cycles executed")

    # Statistical summary
    statistics: CycleStatistics = Field(..., description="Statistical aggregation")

    # Per-cycle results (for detailed analysis)
    cycle_results: list["AggregatedResult"] = Field(
        default_factory=list, description="Individual cycle results"
    )
    cycle_metadata: list[CycleMetadata] = Field(
        default_factory=list, description="Per-cycle metadata"
    )

    # Overall timestamps
    start_time: datetime = Field(..., description="First cycle start")
    end_time: datetime = Field(..., description="Last cycle end")
    total_duration_sec: float = Field(..., description="Total wall-clock time")

    # Configuration (for reproducibility)
    effective_config: dict[str, Any] = Field(
        default_factory=dict, description="Experiment configuration"
    )
    config_warnings: list[str] = Field(
        default_factory=list,
        description="Config validation warnings that were present at runtime",
    )

    model_config = {"frozen": True}


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
    cycle_run_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Cycle metadata (cycle_id, cycle_count, total_cycles)",
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
