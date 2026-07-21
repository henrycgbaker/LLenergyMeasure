"""Measurement quality warnings for energy experiments.

Five warning flags, all purely informational (never block experiments).
Each includes actionable remediation advice per CONTEXT.md.

This module owns warnings generation and its orchestration: the flag catalogue
(:func:`collect_measurement_warnings`), the GPU-persistence probe
(:func:`_check_persistence_mode`), and :func:`collect_warnings`, which extracts
the thermal-drift endpoints and persistence state from a run and calls the
catalogue. ``collect_warnings`` runs BEFORE result assembly (the base
mode-agnostic warning list threads into the assembler). Tests that stub the flag
catalogue patch it here at ``llenergymeasure.harness.measurement_warnings.collect_measurement_warnings``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.device.power_thermal import PowerThermalSample


def collect_measurement_warnings(
    duration_sec: float,
    gpu_persistence_mode: bool,
    temp_start_c: float | None,
    temp_end_c: float | None,
    nvml_sample_count: int,
    energy_measurement_present: bool = True,
    energy_sampler_reasons: list[str] | None = None,
    # thermal_drift_threshold default 10C - confidence LOW, no peer citation, flagged for validation
    thermal_drift_threshold_c: float = 10.0,
) -> list[str]:
    """Collect measurement quality warnings for a completed experiment.

    All five warnings are purely informational - they never block experiments.
    Each includes actionable remediation advice.

    Note on the two sample/energy signals, which watch DIFFERENT things:
    - ``nvml_low_sample_count`` watches the harness thermal-telemetry sampler
      (``device/power_thermal.py`` PowerThermalSampler, whose samples feed
      ``nvml_sample_count``). It does NOT observe the authoritative energy backend,
      so it cannot detect that energy measurement itself failed.
    - ``energy_measurement_unavailable`` watches the authoritative energy backend
      (Zeus/NVML/CodeCarbon selected by ``select_energy_sampler``). It fires when
      that backend produced no measurement at all, distinguishing an absent
      measurement from a genuine measured zero.

    Args:
        duration_sec: Total measurement window duration in seconds.
        gpu_persistence_mode: Whether GPU persistence mode was enabled during measurement.
        temp_start_c: GPU temperature at measurement start, or None if unavailable.
        temp_end_c: GPU temperature at measurement end, or None if unavailable.
        nvml_sample_count: Number of thermal-telemetry power samples collected during
            measurement (from PowerThermalSampler, not the authoritative energy backend).
        energy_measurement_present: Whether the authoritative energy backend produced a
            measurement. False means energy was not measured (sampler unavailable or
            disabled); reported energy is absent, not a measured zero.
        energy_sampler_reasons: Per-sampler probe why-chain from auto-selection (e.g.
            "zeus: package not installed"). When energy is absent and this is
            non-empty, the reasons are appended to the energy_measurement_unavailable
            warning so the diagnosis is structured, not only in the log.
        thermal_drift_threshold_c: Maximum acceptable temperature change in Celsius.
            Default 10C - confidence LOW (engineering judgement, no peer citation,
            flagged for validation).

    Returns:
        List of warning strings (empty list = clean measurement).
    """
    # baseline_duration 30s - confidence MEDIUM (similar to VILE paper 22-33s windows)
    warnings: list[str] = []

    # 1. Short measurement duration
    if duration_sec < 10.0:
        warnings.append(
            "short_measurement_duration: measurement < 10s; energy values may be unreliable. "
            "Use more prompts or longer sequences."
        )

    # 2. GPU persistence mode off
    if not gpu_persistence_mode:
        warnings.append(
            "gpu_persistence_mode_off: power state variation may inflate measurements. "
            "Run 'nvidia-smi -pm 1' to enable persistence mode."
        )

    # 3. Thermal drift during measurement
    if temp_start_c is not None and temp_end_c is not None:
        drift = abs(temp_end_c - temp_start_c)
        if drift > thermal_drift_threshold_c:
            warnings.append(
                f"thermal_drift_detected: {drift:.1f}C change during measurement "
                f"(threshold {thermal_drift_threshold_c}C). "
                "Increase thermal_floor_seconds or check cooling."
            )

    # 4. Low NVML sample count (thermal-telemetry sampler, not the energy backend)
    if nvml_sample_count < 10:
        warnings.append(
            "nvml_low_sample_count: fewer than 10 NVML power samples collected; "
            "energy integration may be inaccurate."
        )

    # 5. Authoritative energy measurement absent
    if not energy_measurement_present:
        message = (
            "energy_measurement_unavailable: no energy sampler produced a measurement; "
            "reported energy is absent, not a measured zero. Set an explicit energy "
            "backend or install a supported sampler (zeus/nvml/codecarbon)."
        )
        if energy_sampler_reasons:
            message = f"{message} Sampler probe results: {'; '.join(energy_sampler_reasons)}."
        warnings.append(message)

    return warnings


def _check_persistence_mode(gpu_indices: list[int] | None = None) -> bool:
    """Check whether GPU persistence mode is enabled on all specified GPUs.

    Checks every GPU in gpu_indices (defaults to [0] when None). Returns False
    only if persistence mode is definitively off on at least one GPU.

    Args:
        gpu_indices: GPU device indices to check. Defaults to [0] when None.

    Returns:
        True if persistence mode is on (or unknown) for all GPUs,
        False if definitively off on any GPU.
    """
    indices = gpu_indices if gpu_indices is not None else [0]
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        with nvml_context():
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                if mode == pynvml.NVML_FEATURE_DISABLED:
                    return False
        return True
    except Exception:
        return True  # Unknown - don't generate spurious warning


def collect_warnings(
    duration_sec: float,
    timeseries_samples: list[PowerThermalSample],
    gpu_indices: list[int] | None = None,
    energy_measurement: Any = None,
    energy_sampler_reasons: list[str] | None = None,
) -> list[str]:
    """Extract run signals from the timeseries and call the warning catalogue.

    ``energy_measurement`` is the authoritative energy backend result (or None).
    Its presence is a separate signal from ``timeseries_samples`` (which come from
    the thermal-telemetry sampler) - an absent energy measurement must be flagged
    even when thermal telemetry sampled fine.

    ``energy_sampler_reasons`` is the per-sampler probe why-chain captured at
    selection time; it enriches the energy_measurement_unavailable warning so the
    reason each backend was skipped/rejected is structured, not log-only.
    """
    temp_start: float | None = None
    temp_end: float | None = None
    if timeseries_samples:
        temps = [s.temperature_c for s in timeseries_samples if s.temperature_c is not None]
        if temps:
            temp_start = temps[0]
            temp_end = temps[-1]

    persistence_on = _check_persistence_mode(gpu_indices)
    nvml_count = len(timeseries_samples)

    return collect_measurement_warnings(
        duration_sec=duration_sec,
        gpu_persistence_mode=persistence_on,
        temp_start_c=temp_start,
        temp_end_c=temp_end,
        nvml_sample_count=nvml_count,
        energy_measurement_present=energy_measurement is not None,
        energy_sampler_reasons=energy_sampler_reasons,
    )
