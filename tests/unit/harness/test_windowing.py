"""Tests for steady-state / windowed energy measurement.

The energy assertions are hand-computed trapezoids over a known synthetic power
series, so a regression in the windowing or re-integration math fails here loudly.
"""

from __future__ import annotations

import math

import pytest

from llenergymeasure.config.models import MeasurementConfig
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.windowing import (
    _clean_samples,
    _coefficient_of_variation,
    _detect_steady_state,
    apply_measurement_window,
)


def _flat_series(
    n: int = 11, dt: float = 0.1, power: float = 100.0, gpu_index: int = 0
) -> list[PowerThermalSample]:
    """Constant-power series: ``n`` samples ``dt`` apart at ``power`` watts."""
    return [
        PowerThermalSample(timestamp=i * dt, power_w=power, gpu_index=gpu_index) for i in range(n)
    ]


# ---------------------------------------------------------------------------
# total mode: not invoked (caller short-circuits)
# ---------------------------------------------------------------------------


def test_total_mode_returns_none():
    """total methodology must not produce a window (caller keeps full-run figures)."""
    cfg = MeasurementConfig(measurement_methodology="total")
    assert apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0) is None


# ---------------------------------------------------------------------------
# windowed mode: energy equals the hand-computed trapezoid over the window
# ---------------------------------------------------------------------------


def test_windowed_energy_matches_hand_trapezoid():
    """[0.2, 0.7] over a flat 100W series = 0.5s * 100W = 50 J exactly."""
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.2, 0.7))
    r = apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0)
    assert r is not None
    assert r.methodology == "windowed"
    assert r.energy_j == pytest.approx(50.0)
    assert r.window == (0.2, 0.7)
    assert r.window_duration_sec == pytest.approx(0.5)


def test_windowed_edge_interpolation_between_samples():
    """A window cutting between samples integrates exactly via edge interpolation.

    Linear ramp 50W..150W over 0..1.0s (slope 100 W/s, non-zero floor so no sample is
    dropped by pre-clean). Window [0.25, 0.75]: power(0.25)=75W, power(0.75)=125W,
    trapezoid = (75+125)/2 * 0.5 = 50 J.
    """
    samples = [
        PowerThermalSample(timestamp=i * 0.1, power_w=50.0 + 100.0 * (i * 0.1), gpu_index=0)
        for i in range(11)
    ]
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.25, 0.75))
    r = apply_measurement_window(samples, cfg, inference_time_sec=1.0)
    assert r is not None
    assert r.energy_j == pytest.approx(50.0)


def test_windowed_token_fraction_is_window_share():
    """token_fraction is the window's share of the cleaned inference span."""
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 0.6))
    r = apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0)
    assert r is not None
    # cleaned span is 1.0s (0..1.0); window 0.6s -> 0.6 fraction.
    assert r.token_fraction == pytest.approx(0.6)
    assert any("proportionally by time" in w for w in r.warnings)


def test_windowed_clamps_end_to_span():
    """An end beyond the sample span clamps to the realised span, not past it."""
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 5.0))
    r = apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0)
    assert r is not None
    assert r.window[1] == pytest.approx(1.0)
    assert r.energy_j == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# steady_state fixed-discard: discards the right prefix
# ---------------------------------------------------------------------------


def test_steady_state_fixed_fraction_discards_prefix():
    """Fraction 0.3 over a flat 100W 1.0s run -> window [0.3, 1.0] = 70 J."""
    cfg = MeasurementConfig(measurement_methodology="steady_state", warmup_discard_fraction=0.3)
    r = apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0)
    assert r is not None
    assert r.methodology == "steady_state"
    assert r.window[0] == pytest.approx(0.3)
    assert r.window[1] == pytest.approx(1.0)
    assert r.energy_j == pytest.approx(70.0)
    assert r.steady_state_not_detected is False


def test_steady_state_fixed_seconds_takes_precedence():
    """warmup_discard_seconds overrides the fraction: discard 0.4s -> [0.4,1.0]=60 J."""
    cfg = MeasurementConfig(
        measurement_methodology="steady_state",
        warmup_discard_fraction=0.9,  # would discard almost everything if used
        warmup_discard_seconds=0.4,
    )
    r = apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0)
    assert r is not None
    assert r.window[0] == pytest.approx(0.4)
    assert r.energy_j == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# auto-detector: finds plateau onset / falls back and flags on failure
# ---------------------------------------------------------------------------


def test_auto_detector_finds_plateau_onset():
    """A warm-up ramp then a flat plateau: the onset lands in the plateau region."""
    ramp = [
        PowerThermalSample(timestamp=t / 10, power_w=20.0 + (180.0) * (t / 20), gpu_index=0)
        for t in range(20)
    ]
    plateau = [
        PowerThermalSample(timestamp=2.0 + t / 10, power_w=200.0, gpu_index=0) for t in range(41)
    ]
    cfg = MeasurementConfig(
        measurement_methodology="steady_state",
        steady_state_auto_detect=True,
        warmup_discard_fraction=0.1,
    )
    r = apply_measurement_window(ramp + plateau, cfg, inference_time_sec=6.0)
    assert r is not None
    assert r.steady_state_not_detected is False
    # Onset must land at or after the start of the ramp-to-plateau transition and
    # before the plateau ends, i.e. inside roughly [1.5, 2.5]s.
    assert 1.5 <= r.window[0] <= 2.5
    # Energy over the plateau-dominated window is close to 200W * duration.
    expected = 200.0 * (r.window[1] - r.window[0])
    assert r.energy_j == pytest.approx(expected, rel=0.05)


def test_auto_detector_falls_back_when_never_stable():
    """A series that never stabilises sets the flag and falls back to fixed discard."""
    noisy = [
        PowerThermalSample(
            timestamp=t / 10,
            power_w=50.0 + 40.0 * math.sin(t) * (1 + t / 10) + (t * 3),
            gpu_index=0,
        )
        for t in range(60)
    ]
    cfg = MeasurementConfig(
        measurement_methodology="steady_state",
        steady_state_auto_detect=True,
        warmup_discard_fraction=0.1,
    )
    r = apply_measurement_window(noisy, cfg, inference_time_sec=6.0)
    assert r is not None
    assert r.steady_state_not_detected is True
    assert any("auto-detection found no stable region" in w for w in r.warnings)
    # Fell back to fixed discard: start ~ 10% of the cleaned span.
    span = noisy[-1].timestamp - noisy[0].timestamp
    assert r.window[0] == pytest.approx(0.1 * span, rel=0.05)


def test_detect_steady_state_directly():
    """The detector returns an onset for ramp->plateau and None for a pure ramp."""
    times = [t / 10 for t in range(60)]
    plateau_powers = [20.0 + 9.0 * t if t < 20 else 200.0 for t in range(60)]
    onset = _detect_steady_state(plateau_powers, times)
    assert onset is not None
    assert onset >= 1.5

    ramp_powers = [float(t) for t in range(60)]  # strictly increasing, never flat
    assert _detect_steady_state(ramp_powers, times) is None


def test_coefficient_of_variation():
    """CV is std/mean; a constant series has CV 0, a non-positive mean is infinite."""
    assert _coefficient_of_variation([100.0, 100.0, 100.0]) == pytest.approx(0.0)
    assert _coefficient_of_variation([0.0, 0.0]) == float("inf")
    assert _coefficient_of_variation([90.0, 110.0]) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# pre-clean: zero / None dropouts do not corrupt the window
# ---------------------------------------------------------------------------


def test_preclean_zero_and_none_dropouts():
    """A zero sample and a None sample do not corrupt a flat 100W integral."""
    samples = _flat_series()
    samples[5].power_w = 0.0  # physically-impossible dropout
    samples[7].power_w = None  # missing reading
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 1.0))
    r = apply_measurement_window(samples, cfg, inference_time_sec=1.0)
    assert r is not None
    # Median filter repairs the zero; the None is dropped. Energy stays ~100 J.
    assert r.energy_j == pytest.approx(100.0, rel=0.02)


def test_clean_drops_nonpositive_and_smooths():
    """_clean_samples drops None / non-positive power and median-smooths transients."""
    samples = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=0.0, gpu_index=0),  # dropout
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.3, power_w=None, gpu_index=0),  # missing
        PowerThermalSample(timestamp=0.4, power_w=100.0, gpu_index=0),
    ]
    cleaned = _clean_samples(samples)
    # None / zero dropped before filtering -> 3 survivors, all ~100W after smoothing.
    assert len(cleaned) == 3
    assert all(s.power_w == pytest.approx(100.0) for s in cleaned)
    # Timestamps preserved.
    assert [s.timestamp for s in cleaned] == [0.0, 0.2, 0.4]


# ---------------------------------------------------------------------------
# min-duration guard
# ---------------------------------------------------------------------------


def test_min_duration_guard_fires_on_short_window():
    """A sub-second window trips the minimum-duration warning."""
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 0.3))
    r = apply_measurement_window(_flat_series(), cfg, inference_time_sec=1.0)
    assert r is not None
    assert any("minimum-duration floor" in w for w in r.warnings)


def test_min_duration_guard_silent_on_long_window():
    """A window over the floor does not emit the minimum-duration warning."""
    samples = _flat_series(n=31, dt=0.1)  # 3.0s span
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 2.0))
    r = apply_measurement_window(samples, cfg, inference_time_sec=3.0)
    assert r is not None
    assert not any("minimum-duration floor" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# degenerate: too few clean samples -> None (caller keeps total figures)
# ---------------------------------------------------------------------------


def test_too_few_samples_returns_none():
    """Fewer than two usable samples after cleaning falls back to total figures."""
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 1.0))
    one = [PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0)]
    assert apply_measurement_window(one, cfg, inference_time_sec=1.0) is None


def test_multi_gpu_window_integrates_per_gpu():
    """Two GPUs each integrate independently over the window."""
    samples = []
    for i in range(11):
        samples.append(PowerThermalSample(timestamp=i * 0.1, power_w=100.0, gpu_index=0))
        samples.append(PowerThermalSample(timestamp=i * 0.1, power_w=50.0, gpu_index=1))
    cfg = MeasurementConfig(measurement_methodology="windowed", measurement_window=(0.0, 1.0))
    r = apply_measurement_window(samples, cfg, inference_time_sec=1.0)
    assert r is not None
    assert r.per_gpu_j[0] == pytest.approx(100.0)
    assert r.per_gpu_j[1] == pytest.approx(50.0)
    assert r.energy_j == pytest.approx(150.0)
