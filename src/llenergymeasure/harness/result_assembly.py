"""Assemble an ExperimentResult from a measured window, split by measurement source.

The assembly is deliberately split into two producers around one seam:

- :func:`build_offline_metrics` is the OFFLINE-batch producer. It reads the
  :class:`~llenergymeasure.engines.protocol.InferenceOutput` (token counts,
  throughput, latency stats, batch stats, FLOPs, extended metrics, measurement
  windowing) and packs everything source-derived into a :class:`SourceMetrics`.
- :func:`assemble_experiment_result` is the MODE-AGNOSTIC assembler. It consumes
  a :class:`SourceMetrics` plus the mode-agnostic inputs (energy breakdown from
  the source energy total + baseline, identity, thermal, provenance, warnings)
  and builds the final ExperimentResult. Its signature contains NO
  InferenceOutput and no engine-specific type - that is the server-admittance
  test. Server mode (v0.8.0) adds a sibling ``build_server_metrics`` over its
  LoadGen summary that produces a :class:`SourceMetrics`, and calls this same
  assembler unchanged.

:func:`build_result` composes the two for the offline path (and is the offline
entry point used by :mod:`llenergymeasure.harness.staging`).

Warnings generation and its orchestration live in
:mod:`llenergymeasure.harness.measurement_warnings` (a separable concern that
runs before assembly); this module only consumes the base warning list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from llenergymeasure._version import __version__
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    compute_declared_config_hash,
    energy_per_token_mj,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.engines.protocol import InferenceOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ConfigMethodology:
    """Methodology fields that live in the config.json sidecar, not result.json.

    Derived during result assembly (they depend on the resolved measurement
    window) and threaded out to the config.json sidecar writer, since they are
    configuration/methodology, not measurement output.
    """

    measurement_methodology: str
    steady_state_window: tuple[float, float] | None
    measurement_window_discard_fraction: float | None
    steady_state_not_detected: bool


@dataclass(frozen=True)
class SourceMetrics:
    """Source-derived metrics: the seam between a measurement source and assembly.

    Everything a measurement source (offline batch today; LoadGen server tomorrow)
    computes about the workload it ran. :func:`assemble_experiment_result` consumes
    exactly this plus the mode-agnostic inputs, so a new source only implements a
    producer that returns one of these - the assembler never changes.
    """

    # Serving mode discriminator ("offline" today; "server" for the future
    # LoadGen producer). The assembler stamps it onto ExperimentResult verbatim,
    # so the mode a result was produced in travels through this one seam.
    serving_mode: str
    # Token counts (raw workload).
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Effective energy (post-windowing for the offline source).
    total_energy_j: float
    energy_duration_sec: float
    per_gpu_energy_j: dict[int, float] | None
    windowed_output_tokens: float
    # Timing + throughput.
    measured_time_sec: float
    avg_tokens_per_second: float
    avg_energy_per_token_j: float
    # FLOPs.
    total_flops: float
    flops_per_output_token: float | None
    flops_per_input_token: float | None
    flops_per_second: float | None
    # Rich sub-objects.
    extended_metrics: Any
    latency_stats: Any
    # Provenance derived from the source.
    warmup_excluded_samples: int | None
    # The warmup result placed verbatim on ExperimentResult.warmup_result. The
    # offline producer packs its WarmupResult here; a measurement source with no
    # warmup phase (e.g. a future server producer) leaves it None.
    warmup_result: Any | None
    engine_build_cache_hit: bool | None
    methodology: _ConfigMethodology
    # Warnings the source appended (latency mode, latency profiling, window), in
    # order; the assembler concatenates base warnings + these.
    source_warnings: list[str]


def experiment_identity(config: ExperimentConfig, start_time: datetime) -> str:
    """Compose the experiment identity string (model + measurement start timestamp)."""
    return f"{config.task.model}_{start_time.strftime('%Y%m%d_%H%M%S')}"


def resolve_measurement_mode(
    declared_mode: str | None,
    measurement_warnings: list[str],
) -> Any:
    """Map an engine-declared latency mode string to LatencyMeasurementMode.

    The engine sets ``output.latency_measurement_mode`` explicitly whenever it
    emits TTFT. If it is missing - or an unrecognised string - that is an engine
    bug: log a warning and fall back to the field's default (TRUE_STREAMING),
    noting it in measurement_warnings. A bad engine string must never crash
    result assembly.
    """
    from llenergymeasure.domain.metrics import LatencyMeasurementMode

    if declared_mode is None:
        logger.warning(
            "Engine emitted TTFT samples but no latency_measurement_mode; "
            "defaulting provenance to TRUE_STREAMING."
        )
        measurement_warnings.append(
            "latency_measurement_mode missing despite TTFT capture; "
            "provenance defaulted to true_streaming (engine should set it explicitly)."
        )
        return LatencyMeasurementMode.TRUE_STREAMING
    try:
        return LatencyMeasurementMode(declared_mode)
    except ValueError:
        logger.warning(
            "Engine emitted unrecognised latency_measurement_mode %r; "
            "defaulting provenance to TRUE_STREAMING.",
            declared_mode,
        )
        measurement_warnings.append(
            f"latency_measurement_mode {declared_mode!r} unrecognised; "
            "provenance defaulted to true_streaming."
        )
        return LatencyMeasurementMode.TRUE_STREAMING


def _append_latency_profiling_warnings(
    config: ExperimentConfig,
    output: InferenceOutput,
    engine_name: str,
    measurement_warnings: list[str],
) -> None:
    """Append latency-profiling provenance warnings to measurement_warnings.

    Only fires when ``config.measurement.latency_profiling`` is enabled. Adds the
    fixed provenance note plus, when relevant, the batch-size-forcing note
    (transformers) and the unsupported-engine note (tensorrt).
    """
    if not config.measurement.latency_profiling:
        return
    measurement_warnings.append(
        "latency_profiling enabled: per-token timing capture (streamer/decode-average "
        "ITL) may perturb energy and latency; energy figures emitted as-is and are not "
        "directly comparable to non-profiled runs."
    )
    if output.extras.get("profiling_forced_batch_size"):
        measurement_warnings.append(
            "latency_profiling forced batch_size=1 for per-token timing capture; "
            "throughput is not comparable to the configured batch size."
        )
    if output.extras.get("latency_profiling_unsupported"):
        measurement_warnings.append(
            f"latency_profiling is not supported by the {engine_name} engine; "
            "no per-token timing was captured."
        )


def _resolve_measurement_window(
    config: ExperimentConfig,
    output: InferenceOutput,
    energy_measurement: Any,
    timeseries_samples: list[PowerThermalSample] | None,
) -> Any:
    """Apply the configured measurement window, or None for total mode.

    Prefers the energy sampler's own power samples (NVML) for re-integration, and
    falls back to the harness PowerThermalSampler timeseries (always present even
    with Zeus/CodeCarbon, which expose no raw samples). Returns a WindowResult, or
    None when the window cannot be applied (keeping the unchanged total figures).
    """
    from llenergymeasure.harness.windowing import apply_measurement_window

    if config.measurement.measurement_methodology == "total":
        return None

    sampler_samples = getattr(energy_measurement, "samples", None) or []
    power_samples = sampler_samples if len(sampler_samples) >= 2 else (timeseries_samples or [])
    return apply_measurement_window(power_samples, config.measurement, output.inference_time_sec)


def build_offline_metrics(
    *,
    experiment_id: str,
    engine_name: str,
    config: ExperimentConfig,
    output: InferenceOutput,
    energy_measurement: Any,
    timeseries_samples: list[PowerThermalSample] | None,
    flops_result: Any,
    model_memory_mb: float,
    warmup_result: Any,
    prompt_count: int,
    duration_sec: float,
) -> SourceMetrics:
    """OFFLINE producer: derive all source metrics from one InferenceOutput.

    Everything here is offline-batch-specific (it reads the InferenceOutput and
    the measurement-window machinery keyed to a single inference run). Server mode
    adds a sibling producer over its LoadGen summary; both return a
    :class:`SourceMetrics` for :func:`assemble_experiment_result`.
    """
    from llenergymeasure.domain.extended_metrics import compute_extended_metrics
    from llenergymeasure.domain.metrics import compute_latency_statistics

    source_warnings: list[str] = []

    # Resolve the measurement window (None for total mode = unchanged path).
    window_result = _resolve_measurement_window(
        config, output, energy_measurement, timeseries_samples
    )

    # Reported inference time: window duration for windowed/steady_state, else full run.
    measured_time_sec = (
        window_result.window_duration_sec
        if window_result is not None
        else output.inference_time_sec
    )

    # Real energy values from energy sampler (windowed re-integration overrides
    # the sampler total when a window is in effect).
    if window_result is not None:
        total_energy_j = window_result.energy_j
    elif energy_measurement is not None:
        total_energy_j = energy_measurement.total_j
    else:
        # No authoritative energy measurement (sampler unavailable or disabled). The
        # schema requires a non-null total_energy_j, so we keep a 0.0 placeholder -
        # but this is absence, NOT a measured zero. It is made loud here and, in the
        # persisted result, via the energy_measurement_unavailable measurement warning
        # (see collect_measurement_warnings); it can never be silent.
        logger.warning(
            "No energy measurement available for %s; reporting total_energy_j=0.0 as a "
            "placeholder for absence, not a measured zero (see the "
            "'energy_measurement_unavailable' measurement warning).",
            experiment_id,
        )
        total_energy_j = 0.0

    # Token counts reported describe the full workload; for a sub-window, energy and
    # throughput are normalised by the window-attributed token share (proportional by
    # time - the harness has no absolute per-token timestamps).
    output_tokens = output.output_tokens if output.output_tokens > 0 else output.total_tokens
    token_fraction = window_result.token_fraction if window_result is not None else 1.0
    windowed_output_tokens = output_tokens * token_fraction
    windowed_total_tokens = output.total_tokens * token_fraction

    avg_tokens_per_second = (
        windowed_total_tokens / measured_time_sec if measured_time_sec > 0 else 0.0
    )

    # Energy per token: output tokens only (input tokens are not "generated")
    avg_energy_per_token_j = (
        total_energy_j / windowed_output_tokens
        if (total_energy_j > 0 and windowed_output_tokens > 0)
        else 0.0
    )

    # Energy-breakdown span (consumed by the assembler for baseline adjustment).
    # Use the energy sampler's window duration, not the harness datetime duration,
    # to avoid CUDA sync latency skew. For a sub-window, the realised window
    # duration is the correct baseline span.
    energy_duration = (
        measured_time_sec
        if window_result is not None
        else (energy_measurement.duration_sec if energy_measurement is not None else duration_sec)
    )

    # Per-GPU energy source. A window re-integrates per-GPU energy; otherwise the
    # sampler's per-GPU totals are used.
    per_gpu_source = (
        window_result.per_gpu_j
        if window_result is not None
        else (energy_measurement.per_gpu_j if energy_measurement is not None else None)
    )

    # FLOPs from PaLM formula (0.0 if estimation unavailable)
    total_flops = flops_result.value if flops_result is not None else 0.0
    flops_per_output_token = (
        total_flops / output.output_tokens
        if (total_flops > 0 and output.output_tokens > 0)
        else None
    )
    flops_per_input_token = (
        total_flops / output.input_tokens if (total_flops > 0 and output.input_tokens > 0) else None
    )
    flops_per_second = (
        total_flops / output.inference_time_sec
        if (total_flops > 0 and output.inference_time_sec > 0)
        else None
    )

    # Memory metrics: inference-window-only peak and derived delta. Both peak and
    # model baseline are 0.0 when neither torch nor NVML could measure them
    # (out-of-process engine with NVML unavailable, or a CPU run); the delta is
    # only meaningful when both are real, otherwise it stays null rather than
    # reporting a silently-wrong number.
    inference_memory_mb: float | None
    if output.peak_memory_mb > 0 and model_memory_mb > 0:
        inference_memory_mb = max(0.0, output.peak_memory_mb - model_memory_mb)
    else:
        inference_memory_mb = None
    logger.debug(
        "Memory: model=%.1fMB, peak_inference=%.1fMB, inference_delta=%s",
        model_memory_mb,
        output.peak_memory_mb,
        f"{inference_memory_mb:.1f}MB" if inference_memory_mb is not None else "null",
    )

    # --- Extended efficiency metrics ---
    samples = timeseries_samples or []
    gpu_utilisation_samples = [s.sm_utilisation for s in samples if s.sm_utilisation is not None]
    memory_bandwidth_samples = [
        s.memory_bandwidth_utilisation
        for s in samples
        if s.memory_bandwidth_utilisation is not None
    ]
    total_vram_mb = max(
        (s.memory_total_mb for s in samples if s.memory_total_mb is not None),
        default=0.0,
    )

    kv_cache_stats = output.kv_cache_stats
    kv_cache_mb = kv_cache_stats.get("kv_cache_mb") if kv_cache_stats else None
    memory_stats: dict[str, float] = {
        "peak_mb": output.peak_memory_mb,
        "model_mb": model_memory_mb,
        "total_vram_mb": total_vram_mb,
    }
    if kv_cache_mb is not None:
        memory_stats["kv_cache_mb"] = kv_cache_mb

    # Batch stats: continuous-batching engines (e.g. vLLM) report num_batches as
    # None, so the truthiness guard skips them. Static-batching engines report
    # num_batches + padding; effective batch size derives from prompt count.
    batch_stats: dict[str, Any] | None = None
    if output.num_batches:
        configured_batch_size = config.static_batch_size()
        effective_batch_size: float | None = None
        if output.num_batches > 0:
            effective_batch_size = prompt_count / output.num_batches
        padding_overhead: float | None = None
        if output.padding_tokens is not None and output.input_tokens > 0:
            total_positions = output.input_tokens + output.padding_tokens
            if total_positions > 0:
                padding_overhead = output.padding_tokens / total_positions
        batch_stats = {
            "num_batches": output.num_batches,
            "effective_batch_size": effective_batch_size,
            "configured_batch_size": configured_batch_size,
            "padding_overhead": padding_overhead,
        }

    # Latency stats from streaming TTFT/ITL. Computed BEFORE extended metrics so
    # the ITL mean can feed tpot_ms. measurement_mode is mapped from the
    # engine-declared capture mode (provenance). vLLM populates TTFT-only stats
    # even without profiling; ITL (and thus tpot_ms) needs profiling.
    latency_stats = None
    if output.ttft_ms:
        measurement_mode = resolve_measurement_mode(
            output.latency_measurement_mode, source_warnings
        )
        latency_stats = compute_latency_statistics(
            output.ttft_ms,
            itl_trimmed_ms=output.itl_ms or None,
            measurement_mode=measurement_mode,
        )

    itl_mean_ms = latency_stats.itl_mean_ms if latency_stats is not None else None

    extended_metrics = compute_extended_metrics(
        output_tokens=output.output_tokens,
        total_energy_j=total_energy_j,
        tokens_per_second=avg_tokens_per_second,
        precision_factor=1.0,  # No precision-scaling applied (default)
        itl_mean_ms=itl_mean_ms,  # populates tpot_ms when ITL was captured
        per_request_latencies_ms=output.per_request_latencies_ms or None,
        gpu_utilisation_samples=gpu_utilisation_samples or None,
        memory_bandwidth_samples=memory_bandwidth_samples or None,
        memory_stats=memory_stats,
        batch_stats=batch_stats,
        kv_cache_stats=kv_cache_stats,
    )
    # Preserve inference-only memory delta (compute_extended_metrics does not know
    # the model baseline split).
    extended_metrics.memory.inference_memory_mb = inference_memory_mb

    # Latency profiling provenance warnings (appended to source_warnings).
    _append_latency_profiling_warnings(config, output, engine_name, source_warnings)

    # Measurement-methodology provenance. For total mode the window spans the whole
    # run (unchanged); for windowed/steady_state the realised window is recorded.
    if window_result is not None:
        measurement_methodology = window_result.methodology
        steady_state_window = window_result.window
        steady_state_not_detected = window_result.steady_state_not_detected
        source_warnings.extend(window_result.warnings)
        discard_fraction = (
            window_result.window[0] / output.inference_time_sec
            if (window_result.methodology == "steady_state" and output.inference_time_sec > 0)
            else None
        )
    else:
        measurement_methodology = "total"
        steady_state_window = (0.0, output.inference_time_sec)
        steady_state_not_detected = False
        discard_fraction = None

    warmup_excluded_samples = (
        warmup_result.iterations_completed if warmup_result is not None else None
    )

    return SourceMetrics(
        serving_mode=config.serving_mode,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
        total_tokens=output.total_tokens,
        total_energy_j=total_energy_j,
        energy_duration_sec=energy_duration,
        per_gpu_energy_j=per_gpu_source,
        windowed_output_tokens=windowed_output_tokens,
        measured_time_sec=measured_time_sec,
        avg_tokens_per_second=avg_tokens_per_second,
        avg_energy_per_token_j=avg_energy_per_token_j,
        total_flops=total_flops,
        flops_per_output_token=flops_per_output_token,
        flops_per_input_token=flops_per_input_token,
        flops_per_second=flops_per_second,
        extended_metrics=extended_metrics,
        latency_stats=latency_stats,
        warmup_excluded_samples=warmup_excluded_samples,
        warmup_result=warmup_result,
        engine_build_cache_hit=output.extras.get("engine_build_cache_hit"),
        methodology=_ConfigMethodology(
            measurement_methodology=measurement_methodology,
            steady_state_window=steady_state_window,
            measurement_window_discard_fraction=discard_fraction,
            steady_state_not_detected=steady_state_not_detected,
        ),
        source_warnings=source_warnings,
    )


def assemble_experiment_result(
    metrics: SourceMetrics,
    *,
    experiment_id: str,
    engine_name: str,
    config: ExperimentConfig,
    baseline: Any,
    start_time: datetime,
    end_time: datetime,
    throttle_info: Any,
    timeseries_path: str | None,
    base_warnings: list[str],
    model_load_time_sec: float | None,
) -> tuple[ExperimentResult, _ConfigMethodology]:
    """MODE-AGNOSTIC assembler: build the ExperimentResult from source metrics.

    Consumes a :class:`SourceMetrics` plus the mode-agnostic inputs: energy
    breakdown (from the source energy total + baseline), identity, thermal,
    provenance, and warnings. There is deliberately NO InferenceOutput here - a
    new measurement source is admitted by writing a sibling producer that yields
    a :class:`SourceMetrics`; this assembler is reused unchanged.

    Returns the result plus the :class:`_ConfigMethodology` the source derived
    (written to the config.json sidecar, not result.json).
    """
    from llenergymeasure.domain.metrics import MultiGPUMetrics
    from llenergymeasure.harness.baseline import create_energy_breakdown

    total_energy_j = metrics.total_energy_j
    windowed_output_tokens = metrics.windowed_output_tokens

    # Energy breakdown with baseline adjustment (mode-agnostic: source energy
    # total + baseline over the realised span).
    energy_breakdown = create_energy_breakdown(
        total_energy_j, baseline, metrics.energy_duration_sec
    )

    # Per-GPU energy breakdown.
    energy_per_device_j = None
    multi_gpu = None
    if metrics.per_gpu_energy_j:
        sorted_indices = sorted(metrics.per_gpu_energy_j.keys())
        energy_per_device_j = [metrics.per_gpu_energy_j[i] for i in sorted_indices]
        if len(sorted_indices) > 1:
            multi_gpu = MultiGPUMetrics(
                num_gpus=len(sorted_indices),
                energy_per_gpu_j=energy_per_device_j,
                energy_total_j=total_energy_j,
                energy_per_output_token_j=(
                    total_energy_j / windowed_output_tokens if windowed_output_tokens > 0 else 0.0
                ),
            )

    # mJ/tok derived fields (energy in millijoules per OUTPUT token, matching
    # avg_energy_per_token_j; input tokens are prefilled, not "generated").
    _mj_total = energy_per_token_mj(total_energy_j, windowed_output_tokens)
    energy_adjusted_j = energy_breakdown.adjusted_j if energy_breakdown else None
    _mj_adjusted = (
        energy_per_token_mj(energy_adjusted_j, windowed_output_tokens)
        if energy_adjusted_j is not None
        else None
    )

    # Base (mode-agnostic) warnings first, then the source-appended ones, in the
    # same order the pre-split assembler produced them.
    measurement_warnings = list(base_warnings) + metrics.source_warnings

    result = ExperimentResult(
        experiment_id=experiment_id,
        declared_config_hash=compute_declared_config_hash(config),
        llenergymeasure_version=__version__,
        serving_mode=metrics.serving_mode,
        # Convenience identity copies; authoritative home is config.json.
        engine=engine_name,
        model_name=config.task.model,
        aggregation=AggregationMetadata(
            method="single_process",
            num_processes=1,
        ),
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        total_tokens=metrics.total_tokens,
        total_energy_j=total_energy_j,
        total_inference_time_sec=metrics.measured_time_sec,
        avg_tokens_per_second=metrics.avg_tokens_per_second,
        avg_energy_per_token_j=metrics.avg_energy_per_token_j,
        total_flops=metrics.total_flops,
        flops_per_output_token=metrics.flops_per_output_token,
        flops_per_input_token=metrics.flops_per_input_token,
        flops_per_second=metrics.flops_per_second,
        start_time=start_time,
        end_time=end_time,
        throttle=throttle_info,
        energy_breakdown=energy_breakdown,
        timeseries=timeseries_path,
        energy_per_token_mj_total=_mj_total,
        energy_per_token_mj_adjusted=_mj_adjusted,
        energy_adjusted_j=energy_adjusted_j,
        energy_per_device_j=energy_per_device_j,
        multi_gpu=multi_gpu,
        warmup_result=metrics.warmup_result,
        measurement_warnings=measurement_warnings,
        extended_metrics=metrics.extended_metrics,
        latency_stats=metrics.latency_stats,
        warmup_excluded_samples=metrics.warmup_excluded_samples,
        model_load_time_sec=model_load_time_sec,
        engine_build_cache_hit=metrics.engine_build_cache_hit,
    )
    return result, metrics.methodology


def build_result(
    *,
    engine_name: str,
    config: ExperimentConfig,
    output: InferenceOutput,
    model_memory_mb: float,
    start_time: datetime,
    end_time: datetime,
    duration_sec: float,
    throttle_info: Any,
    energy_measurement: Any,
    baseline: Any,
    flops_result: Any,
    timeseries_path: str | None,
    measurement_warnings: list[str],
    warmup_result: Any = None,
    timeseries_samples: list[PowerThermalSample] | None = None,
    prompt_count: int = 0,
    model_load_time_sec: float | None = None,
) -> tuple[ExperimentResult, _ConfigMethodology]:
    """Offline entry point: run the offline producer, then the mode-agnostic assembler.

    ``measurement_warnings`` is the base (mode-agnostic) warning list from
    :func:`collect_warnings`; the offline producer appends its source warnings and
    the assembler concatenates the two.
    """
    experiment_id = experiment_identity(config, start_time)
    metrics = build_offline_metrics(
        experiment_id=experiment_id,
        engine_name=engine_name,
        config=config,
        output=output,
        energy_measurement=energy_measurement,
        timeseries_samples=timeseries_samples,
        flops_result=flops_result,
        model_memory_mb=model_memory_mb,
        warmup_result=warmup_result,
        prompt_count=prompt_count,
        duration_sec=duration_sec,
    )
    return assemble_experiment_result(
        metrics,
        experiment_id=experiment_id,
        engine_name=engine_name,
        config=config,
        baseline=baseline,
        start_time=start_time,
        end_time=end_time,
        throttle_info=throttle_info,
        timeseries_path=timeseries_path,
        base_warnings=measurement_warnings,
        model_load_time_sec=model_load_time_sec,
    )
