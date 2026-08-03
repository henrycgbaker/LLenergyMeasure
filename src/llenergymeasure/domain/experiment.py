"""Experiment and study result domain models."""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from llenergymeasure.domain.bundle_artefacts import BUNDLE_VERSION
from llenergymeasure.domain.environment import EnvironmentSnapshot
from llenergymeasure.domain.metrics import (
    EnergyBreakdown,
    ExtendedEfficiencyMetrics,
    LatencyStatistics,
    MultiGPUMetrics,
    ThrottleInfo,
    WarmupResult,
)

# Re-exported for import stability: RunnerProvenance now lives in its own shared
# domain module (both experiment and environment carry it, and experiment
# imports environment, so a shared low-level module avoids an import cycle).
from llenergymeasure.domain.provenance import RunnerProvenance
from llenergymeasure.domain.session import SessionBlock

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


@functools.lru_cache(maxsize=128)
def _hash_canonical(canonical: str) -> str:
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_declared_config_hash(config: ExperimentConfig) -> str:
    """SHA-256[:16] of ExperimentConfig, with the sole slo exclusion.

    Layer 3 fields (datacenter_pue, grid_carbon_intensity) are not in
    ExperimentConfig (they live in user config only), so model_dump()
    naturally excludes them.

    ``server.traffic.slo`` is the one field deliberately excluded from this
    wholesale dump (``exclude={"server": {"traffic": {"slo"}}}``): SLO bounds are
    a pure post-hoc overlay (O5.3) - two runs differing only in their slo bounds
    are the same physical experiment, so they must hash identically and
    deduplicate together. The exclusion is applied HERE, at the hash call site,
    NOT via ``Field(exclude=True)`` on the slo field - a field-level exclude would
    also strip slo from the config.json sidecar and result provenance, where the
    bounds a run was judged against MUST stay stamped. This is the wholesale-dump
    half of the dual-family mechanism; the resolved/observed views exclude slo by
    the complementary means of simply not projecting it (see
    ``ConfigHashView.mode_section`` / ``ExperimentConfig.mode_section_identity``).

    Dumps in ``mode="json"`` so numeric coercions match the serialized form:
    a field typed float but defaulted to an int literal (e.g. vllm
    ``cpu_offload_gb = 0``) stays int in a python-mode dump but becomes ``0.0``
    after a JSON round-trip. The host and the container must agree on this hash
    (it names the result file), so both hash the JSON-stable form.
    """
    dumped = config.model_dump(mode="json", exclude={"server": {"traffic": {"slo"}}})
    canonical = json.dumps(dumped, sort_keys=True)
    return _hash_canonical(canonical)


def energy_per_token_mj(energy_j: float, total_tokens: float) -> float | None:
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


class ServerWarmupProvenance(BaseModel):
    """One level's warmup outcome, stamped into each of the level's window bundles.

    The server pre-window protocol's outcome (D6 divergence label): identical
    across a level's windows (one warmup per level) but carried per window so the
    persistence layer, which never learns sessions exist, has it locally.
    """

    mode: str = Field(..., description="Warmup mode: 'composite' (convergence gate) or 'fixed'.")
    converged: bool = Field(
        ..., description="Whether the composite gate was satisfied (or the fixed duration ran)."
    )
    timed_out: bool = Field(
        ...,
        description="Whether composite hit its failsafe timeout and proceeded anyway (a loud "
        "disclosure, never a silent pass).",
    )
    elapsed_s: float = Field(..., description="Wall-clock seconds spent in this level's warmup.")


class ServerWindowProvenance(BaseModel):
    """Server-mode per-window provenance: which level/window, and its pre-window protocol.

    The additive server-provenance block on a server-mode ExperimentResult. It
    locates the window within its rate level and records the warmup outcome and
    the disclosed attribution policy. Derived server metrics (goodput, slo_pass,
    energy-at-operating-point) are a later slice and deliberately absent.
    """

    level_index: int = Field(..., description="0-based rate-level index within the session.")
    window_index: int = Field(..., description="0-based measured-window index within the level.")
    level_window_count: int = Field(
        ..., description="Number of measured windows this level produced."
    )
    level_valid: bool = Field(
        ..., description="Whether the level passed its window-to-window J/token stability gate."
    )
    intra_window_cov: float | None = Field(
        default=None,
        description="Within-window J/token coefficient of variation (the k=4 sub-window "
        "diagnostic): a WITHIN-window stability figure, distinct from the window-to-window gate "
        "that sets level_valid. None when unformable (e.g. a window with no attributed tokens) or "
        "for a degraded abort-core bundle.",
    )
    invalid_reason: str | None = Field(
        default=None,
        description="Why the level was invalid (gate failure or abort site), or None when valid.",
    )
    warmup: ServerWarmupProvenance | None = Field(
        default=None, description="This level's warmup outcome, or None when no warmup ran."
    )
    pre_window_protocol: str = Field(
        ...,
        description="Human-readable description of the server pre-window warmup protocol "
        "(the offline-vs-server comparability label).",
    )
    attribution_policy: str = Field(
        ..., description="The disclosed energy/token attribution policy the window used."
    )


class ExperimentResult(BaseModel):
    """Experiment result - the user-visible output of a measurement run.

    Produced once per single-process measurement run by the harness. Holds the
    final metrics (energy, throughput, FLOPs, latency) directly; there is no
    per-process breakdown.
    """

    # Identity
    bundle_version: str = Field(
        default=BUNDLE_VERSION,
        description="Results-bundle version (layout + artefact set + per-artefact schema, "
        "as one contract). Replaces the retired per-artefact result schema_version.",
    )
    experiment_id: str = Field(..., description="Unique experiment identifier")
    declared_config_hash: str = Field(
        ...,
        description="SHA-256[:16] of the whole declared ExperimentConfig "
        "(compute_declared_config_hash). Environment fields are not part of "
        "ExperimentConfig, so they are naturally excluded. Same term as the "
        "declared_config block in the config.json sidecar.",
    )
    llenergymeasure_version: str | None = Field(
        default=None, description="Package version that produced this result"
    )
    serving_mode: str = Field(
        default="offline",
        description="Serving mode that produced this result: the offline/server "
        'discriminator, mirroring the config-side ExperimentConfig.serving_mode. "offline" for '
        'batch measurement (the only mode today); "server" arrives with server mode (v0.8.0). A '
        "plain string, not a closed vocabulary, so the mode set can grow without a schema break. "
        "Stamped by the assembler from the measurement source (see "
        "harness.result_assembly.SourceMetrics).",
    )

    # Convenience identity copies. Deliberate small duplication so a result.json
    # stays self-describing when separated from its directory; the authoritative
    # home for both is the config.json sidecar.
    engine: str = Field(
        default="transformers",
        description="Inference engine used. Convenience copy; authoritative home "
        "is the config.json sidecar.",
    )
    model_name: str = Field(
        default="unknown",
        description="Model name/path used. Convenience copy; authoritative home "
        "is the config.json sidecar.",
    )

    # Core metrics
    input_tokens: int = Field(
        ...,
        description="Actual input (prefill) tokens as observed by the engine after "
        "tokenisation. total_tokens = input_tokens + output_tokens.",
    )
    output_tokens: int = Field(
        ...,
        description="Actual output (decode) tokens as observed by the engine. "
        "total_tokens = input_tokens + output_tokens.",
    )
    total_tokens: int = Field(..., description="Total tokens across all processes")
    total_energy_j: float = Field(..., description="Total energy (sum across processes)")
    total_inference_time_sec: float = Field(..., description="Total inference time")
    avg_tokens_per_second: float = Field(..., description="Average throughput")
    avg_energy_per_token_j: float = Field(..., description="Average energy per token")
    energy_per_token_mj_adjusted: float | None = Field(
        default=None,
        description="Millijoules per token from adjusted (baseline-subtracted) energy. "
        "None when no baseline measurement was taken.",
    )
    energy_per_token_mj_total: float | None = Field(
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
    model_load_time_sec: float | None = Field(
        default=None,
        description="Wall-clock seconds spent in engine.load_model(): model load plus "
        "any engine build/compile the engine performs there (e.g. the tensorrt trt "
        "backend's TRT engine build, vLLM torch.compile/CUDA-graph capture). "
        "Non-energy run metadata: this phase completes before the NVML energy "
        "measurement window opens and contributes nothing to total_energy_j.",
    )
    engine_build_cache_hit: bool | None = Field(
        default=None,
        description="Whether the tensorrt trt-backend engine build was served from "
        "the on-disk build cache (True) or compiled fresh this run (False). None when "
        "the build cache is not in play: the pytorch backend, other engines, an "
        "engine_path override, or the cache disabled. Detected from TRT-LLM's own "
        "build stats; annotates model_load_time_sec (a cache hit skips the compile).",
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

    # Runner provenance - how this experiment was executed (local vs docker).
    # Persisted into result.json (unlike environment) as reproducibility metadata.
    runner_provenance: RunnerProvenance | None = Field(
        default=None,
        description="How the experiment was executed (local process or Docker container). "
        "None when no runner spec was available.",
    )

    # Session facts - dual-serialised into result.json AND the system.json sidecar
    # (mirrors the runner block). Present in both modes (offline: session id +
    # window_count=1, raws null). None for pre-session-facts bundles (D22 loadable).
    session: SessionBlock | None = Field(
        default=None,
        description="Session facts for the measurement session this window belongs to. "
        "Server mode carries launch/warmup/drain raws + window/level counts; offline carries a "
        "session id and window_count=1. None for older bundles.",
    )

    # Server-mode per-window provenance (which level/window, warmup outcome,
    # pre-window protocol). None for offline results.
    server: ServerWindowProvenance | None = Field(
        default=None,
        description="Server-mode per-window provenance: level/window position, level validity, "
        "warmup outcome, and the pre-window protocol label. None for offline results.",
    )

    # Environment sidecar (loaded from system.json by load_result; excluded
    # from result.json serialisation - the sidecar is the on-disk home).
    environment: EnvironmentSnapshot | None = Field(
        default=None,
        exclude=True,
        description="Hardware/runtime environment loaded from the system.json sidecar. "
        "None when no sidecar is present. Not serialised back into result.json.",
    )

    # Timestamps
    start_time: datetime = Field(..., description="Earliest process start time")
    end_time: datetime = Field(..., description="Latest process end time")

    aggregation: AggregationMetadata | None = Field(
        default=None, description="Aggregation metadata (method, num_processes)"
    )

    # Optional detail fields used by aggregation/CLI
    throttle: ThrottleInfo | None = Field(
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

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_keys(cls, data: Any) -> Any:
        """Read a legacy result.json best-effort by dropping retired top-level keys.

        This model forbids extra fields, so keys retired across a bundle break
        would reject an older result.json outright. Drop them before validation
        (both ``model_validate`` and ``model_validate_json`` run before-validators
        on the parsed structure) so a legacy result.json degrades to the current
        defaults rather than being rejected:

        - ``schema_version`` (pre-``bundle_version`` per-artefact counter).
        - ``baseline_power_w`` (bundle 1.0 top-level copy; the single home is now
          ``energy_breakdown.baseline_power_w``).

        The 2.0-window field renames (``measurement_config_hash`` ->
        ``declared_config_hash``, ``thermal_throttle`` -> ``throttle``,
        ``mj_per_tok_*`` -> ``energy_per_token_mj_*``) are a CLEAN BREAK: their
        old keys are not tolerated, so a pre-rename bundle fails loudly under
        ``extra="forbid"``.
        """
        legacy_keys = {"schema_version", "baseline_power_w"}
        if isinstance(data, dict) and legacy_keys & data.keys():
            data = {k: v for k, v in data.items() if k not in legacy_keys}
        return data

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
