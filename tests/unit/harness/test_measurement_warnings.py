"""Unit tests for collect_measurement_warnings().

Tests all five warning flags: short duration, persistence mode off, thermal drift,
low NVML sample count, and absent authoritative energy measurement.
"""

from __future__ import annotations

from llenergymeasure.harness.measurement_warnings import collect_measurement_warnings

# ---------------------------------------------------------------------------
# Warning 1: Short measurement duration
# ---------------------------------------------------------------------------


def test_short_duration_warning_fires_below_10s():
    """Warning fires when measurement duration is under 10 seconds."""
    warnings = collect_measurement_warnings(5.0, True, 40.0, 40.0, 50)
    assert any("short_measurement_duration" in w for w in warnings)


def test_short_duration_warning_absent_at_10s():
    """Warning is absent when duration equals exactly 10 seconds."""
    warnings = collect_measurement_warnings(10.0, True, 40.0, 40.0, 50)
    assert not any("short_measurement_duration" in w for w in warnings)


def test_short_duration_warning_absent_above_10s():
    """Warning is absent when duration is above 10 seconds."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 50)
    assert not any("short_measurement_duration" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 2: GPU persistence mode off
# ---------------------------------------------------------------------------


def test_persistence_mode_warning_fires_when_off():
    """Warning fires when gpu_persistence_mode=False."""
    warnings = collect_measurement_warnings(30.0, False, 40.0, 40.0, 50)
    assert any("gpu_persistence_mode_off" in w for w in warnings)


def test_persistence_mode_warning_absent_when_on():
    """Warning is absent when gpu_persistence_mode=True."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 50)
    assert not any("gpu_persistence_mode_off" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 3: Thermal drift
# ---------------------------------------------------------------------------


def test_thermal_drift_warning_fires_above_threshold():
    """Warning fires when temperature drift exceeds 10C threshold."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 52.0, 50)
    assert any("thermal_drift_detected" in w for w in warnings)


def test_thermal_drift_warning_absent_within_threshold():
    """Warning is absent when temperature drift is within threshold."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 49.9, 50)
    assert not any("thermal_drift_detected" in w for w in warnings)


def test_thermal_drift_warning_absent_when_temps_unavailable():
    """Warning is absent when temperature readings are unavailable (None)."""
    warnings = collect_measurement_warnings(30.0, True, None, None, 50)
    assert not any("thermal_drift_detected" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 4: Low NVML sample count
# ---------------------------------------------------------------------------


def test_low_nvml_sample_warning_fires_below_10():
    """Warning fires when NVML sample count is below 10."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 5)
    assert any("nvml_low_sample_count" in w for w in warnings)


def test_low_nvml_sample_warning_absent_at_10():
    """Warning is absent when NVML sample count equals 10."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 10)
    assert not any("nvml_low_sample_count" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 5: Authoritative energy measurement absent
# ---------------------------------------------------------------------------


def test_energy_unavailable_warning_fires_when_absent():
    """Warning fires when the authoritative energy measurement is absent."""
    warnings = collect_measurement_warnings(
        30.0, True, 40.0, 40.0, 50, energy_measurement_present=False
    )
    assert any("energy_measurement_unavailable" in w for w in warnings)


def test_energy_unavailable_warning_absent_when_present():
    """Warning is absent when the energy measurement is present."""
    warnings = collect_measurement_warnings(
        30.0, True, 40.0, 40.0, 50, energy_measurement_present=True
    )
    assert not any("energy_measurement_unavailable" in w for w in warnings)


def test_energy_unavailable_defaults_to_present():
    """Absent-energy warning does not fire by default (backward-compatible callers)."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 50)
    assert not any("energy_measurement_unavailable" in w for w in warnings)


def test_energy_unavailable_independent_of_sample_count():
    """Absent energy fires even when the thermal sampler collected plenty of samples.

    The two signals watch different subsystems: nvml_low_sample_count watches the
    thermal-telemetry sampler, energy_measurement_unavailable watches the energy backend.
    """
    warnings = collect_measurement_warnings(
        30.0, True, 40.0, 40.0, 600, energy_measurement_present=False
    )
    assert any("energy_measurement_unavailable" in w for w in warnings)
    assert not any("nvml_low_sample_count" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning interactions
# ---------------------------------------------------------------------------


def test_no_warnings_when_nvml_inactive_and_all_conditions_good():
    """With nvml_sample_count=0 and good conditions, only the low-sample warning fires."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 0)
    # Only low sample count fires (nvml_sample_count=0 < 10)
    assert len(warnings) == 1
    assert "nvml_low_sample_count" in warnings[0]


def test_no_warnings_when_all_conditions_good():
    """With NVML active (>=10 samples) and all other conditions good, no warnings fire."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 50)
    assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_list_of_strings():
    """collect_measurement_warnings always returns a list of strings."""
    result = collect_measurement_warnings(30.0, True, 40.0, 40.0, 0)
    assert isinstance(result, list)
    assert all(isinstance(w, str) for w in result)


# ---------------------------------------------------------------------------
# energy_sampler_reasons: the per-sampler probe why-chain enriches the
# energy_measurement_unavailable warning (structured backup for the log-only
# diagnostic).
# ---------------------------------------------------------------------------


def test_energy_reasons_appended_when_absent():
    """Probe reasons are folded into the unavailable warning when energy is absent."""
    reasons = ["zeus: package not installed", "nvml: is_available() returned False"]
    warnings = collect_measurement_warnings(
        30.0, True, 40.0, 40.0, 50, energy_measurement_present=False, energy_sampler_reasons=reasons
    )
    unavailable = [w for w in warnings if "energy_measurement_unavailable" in w]
    assert len(unavailable) == 1
    assert "Sampler probe results:" in unavailable[0]
    assert "zeus: package not installed" in unavailable[0]
    assert "nvml: is_available() returned False" in unavailable[0]


def test_energy_reasons_ignored_when_present():
    """Reasons never surface when energy WAS measured (no unavailable warning)."""
    warnings = collect_measurement_warnings(
        30.0,
        True,
        40.0,
        40.0,
        50,
        energy_measurement_present=True,
        energy_sampler_reasons=["zeus: package not installed"],
    )
    assert not any("energy_measurement_unavailable" in w for w in warnings)
    assert not any("Sampler probe results" in w for w in warnings)


def test_energy_absent_without_reasons_stays_generic():
    """Empty/omitted reasons keep the generic unavailable message (no dangling text)."""
    warnings = collect_measurement_warnings(
        30.0, True, 40.0, 40.0, 50, energy_measurement_present=False, energy_sampler_reasons=[]
    )
    unavailable = [w for w in warnings if "energy_measurement_unavailable" in w]
    assert len(unavailable) == 1
    assert "Sampler probe results" not in unavailable[0]
