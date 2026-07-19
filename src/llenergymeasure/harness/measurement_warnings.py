"""Measurement quality warnings for energy experiments.

Five warning flags, all purely informational (never block experiments).
Each includes actionable remediation advice per CONTEXT.md.
"""

from __future__ import annotations


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
