"""Steady-state / windowed energy measurement.

Restricts the measured region of a run to a sub-window of the NVML power series so
that warm-up transients do not bias energy-per-token and throughput. Three modes,
selected by ``MeasurementConfig.measurement_methodology``:

- ``total`` - the whole run (handled by the caller; this module is not invoked).
- ``windowed`` - an explicit ``(start_sec, end_sec)`` window.
- ``steady_state`` - discard a deterministic warm-up prefix, optionally auto-detecting
  the steady-state onset with a sliding-window stability test.

The energy integration math itself is NOT reimplemented here: the cleaned, windowed
sample subset is fed to the existing trapezoidal integrator
(:func:`llenergymeasure.energy.nvml.integrate_power_samples`).

The methodology survey behind this design is recorded in
``.product/research/steady-state-measurement-methodology.md``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from llenergymeasure.energy.nvml import integrate_power_samples

if TYPE_CHECKING:
    from llenergymeasure.config.models import MeasurementConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample

logger = logging.getLogger(__name__)

#: Median-filter kernel for transient-dropout removal (odd, small per the SOTA).
_MEDIAN_KERNEL = 3

#: Minimum realised window duration (seconds) below which the measurement is flagged
#: as too short to be meaningful. Mirrors MLPerf Power's minimum-duration floor,
#: scaled down because LLenergyMeasure runs are far shorter than datacentre sweeps.
_MIN_WINDOW_SEC = 1.0

#: Default auto-detector tuning. The sliding window spans this fraction of the cleaned
#: series, and a window is "stable" when its coefficient of variation is at or below
#: the threshold. Deliberately lenient: the detector only needs to find a plateau onset,
#: and it always has the fixed-discard fallback.
_AUTO_WINDOW_FRACTION = 0.2
_AUTO_CV_THRESHOLD = 0.05
_AUTO_MIN_WINDOW_SAMPLES = 4


@dataclass
class WindowResult:
    """Outcome of applying a measurement window to a power series.

    Attributes:
        methodology: The methodology actually used ("total", "windowed",
            "steady_state").
        energy_j: Re-integrated total energy over the window (sum across GPUs).
        per_gpu_j: Per-GPU re-integrated energy.
        window: Realised ``(start_sec, end_sec)`` relative to inference start.
        window_duration_sec: ``end_sec - start_sec`` of the realised window.
        token_fraction: Fraction of the cleaned inference span the window covers;
            tokens/throughput are attributed proportionally by this fraction.
        steady_state_not_detected: True when auto-detection was requested but failed
            and the run fell back to the fixed warm-up discard.
        warnings: Provenance / quality warnings to surface in the result.
    """

    methodology: str
    energy_j: float
    per_gpu_j: dict[int, float]
    window: tuple[float, float]
    window_duration_sec: float
    token_fraction: float
    steady_state_not_detected: bool = False
    warnings: list[str] = field(default_factory=list)


def _clean_samples(samples: list[PowerThermalSample]) -> list[PowerThermalSample]:
    """Drop zero / physically-impossible power samples and median-filter dropouts.

    Pre-cleaning is mandatory before windowing or stability detection because NVML
    under-samples transitions and emits occasional zero / spurious readings. Samples
    without a power reading or with non-positive power are dropped; the survivors are
    median-filtered (kernel 3) per GPU to kill single-sample transients.

    Timestamps are preserved (the median filter only smooths ``power_w``), so the
    trapezoidal integrator still sees the real time base.
    """
    by_gpu: dict[int, list[PowerThermalSample]] = {}
    for s in samples:
        if s.power_w is None or s.power_w <= 0.0:
            continue
        by_gpu.setdefault(s.gpu_index, []).append(s)

    cleaned: list[PowerThermalSample] = []
    half = _MEDIAN_KERNEL // 2
    for gpu_samples in by_gpu.values():
        # power_w is guaranteed non-None here (None / non-positive dropped above).
        powers = [s.power_w for s in gpu_samples if s.power_w is not None]
        for i, s in enumerate(gpu_samples):
            lo = max(0, i - half)
            hi = min(len(gpu_samples), i + half + 1)
            window = sorted(powers[lo:hi])
            median = window[len(window) // 2]
            # Replace only the smoothed power; keep timestamp / gpu_index intact.
            cleaned.append(_with_power(s, median))

    cleaned.sort(key=lambda s: s.timestamp)
    return cleaned


def _with_power(sample: PowerThermalSample, power_w: float) -> PowerThermalSample:
    """Return a shallow copy of ``sample`` with ``power_w`` replaced."""
    return replace(sample, power_w=power_w)


def _filter_to_window(
    samples: list[PowerThermalSample], origin: float, start_sec: float, end_sec: float
) -> list[PowerThermalSample]:
    """Return samples whose timestamp falls in ``[origin+start, origin+end]``.

    Endpoints are added by interpolation when the window cuts between samples so the
    trapezoid covers exactly ``[start_sec, end_sec]`` rather than snapping to the
    nearest sample. Interpolation is linear in power, consistent with the trapezoidal
    rule the integrator already assumes between samples.
    """
    by_gpu: dict[int, list[PowerThermalSample]] = {}
    for s in samples:
        by_gpu.setdefault(s.gpu_index, []).append(s)

    abs_start = origin + start_sec
    abs_end = origin + end_sec
    result: list[PowerThermalSample] = []
    for gpu_samples in by_gpu.values():
        gpu_samples = sorted(gpu_samples, key=lambda s: s.timestamp)
        result.extend(_clip_gpu_window(gpu_samples, abs_start, abs_end))
    result.sort(key=lambda s: s.timestamp)
    return result


def _clip_gpu_window(
    gpu_samples: list[PowerThermalSample], abs_start: float, abs_end: float
) -> list[PowerThermalSample]:
    """Clip one GPU's ordered samples to ``[abs_start, abs_end]`` with edge interpolation."""
    inside = [s for s in gpu_samples if abs_start <= s.timestamp <= abs_end]
    clipped: list[PowerThermalSample] = list(inside)

    # Interpolate a leading edge sample at abs_start if the window starts mid-gap.
    start_edge = _interp_edge(gpu_samples, abs_start)
    if start_edge is not None and (not inside or inside[0].timestamp > abs_start):
        clipped.append(start_edge)
    # Interpolate a trailing edge sample at abs_end.
    end_edge = _interp_edge(gpu_samples, abs_end)
    if end_edge is not None and (not inside or inside[-1].timestamp < abs_end):
        clipped.append(end_edge)

    clipped.sort(key=lambda s: s.timestamp)
    return clipped


def _interp_edge(gpu_samples: list[PowerThermalSample], at: float) -> PowerThermalSample | None:
    """Linearly interpolate a sample at timestamp ``at`` between bracketing samples.

    Returns None when ``at`` is outside the sample span (nothing to interpolate).
    """
    for i in range(len(gpu_samples) - 1):
        a = gpu_samples[i]
        b = gpu_samples[i + 1]
        if a.timestamp <= at <= b.timestamp:
            if b.timestamp == a.timestamp or a.power_w is None or b.power_w is None:
                return None
            frac = (at - a.timestamp) / (b.timestamp - a.timestamp)
            power = a.power_w + frac * (b.power_w - a.power_w)
            return _with_power_at(a, power, at)
    return None


def _with_power_at(
    sample: PowerThermalSample, power_w: float, timestamp: float
) -> PowerThermalSample:
    """Return a copy of ``sample`` with ``power_w`` and ``timestamp`` replaced."""
    return replace(sample, power_w=power_w, timestamp=timestamp)


def _detect_steady_state(powers: list[float], times: list[float]) -> float | None:
    """Find the steady-state onset (seconds from series start) via sliding-window CV.

    Implements the lightweight windowed-stability test from the SOTA survey (a
    coefficient-of-variation / variance-ratio test in the Cao-Rhinehart R-statistic
    family): slide a window of ``_AUTO_WINDOW_FRACTION`` of the series and return the
    start time of the EARLIEST window whose coefficient of variation (std / mean) is at
    or below ``_AUTO_CV_THRESHOLD`` and which stays stable through the end of the
    series. Returns None when no such region exists (the caller then falls back to the
    fixed discard and flags ``steady_state_not_detected``).

    No heavy change-point library is used: this is a direct ~40-line stability test, as
    the survey found PELT / ruptures / BOCPD fragile on short noisy autocorrelated series.
    """
    n = len(powers)
    if n < _AUTO_MIN_WINDOW_SAMPLES * 2:
        return None
    win = max(_AUTO_MIN_WINDOW_SAMPLES, int(n * _AUTO_WINDOW_FRACTION))
    if win >= n:
        return None

    for start in range(n - win + 1):
        if _is_stable_through_end(powers, start, win):
            return times[start] - times[0]
    return None


def _is_stable_through_end(powers: list[float], start: int, win: int) -> bool:
    """True when every window of size ``win`` from ``start`` to the end is stable.

    Stability is "coefficient of variation at or below threshold". Requiring stability
    through the end (not just at ``start``) rejects a brief flat spot during the warm-up
    ramp - the plateau must persist, matching how production benchmarks define steady
    state.
    """
    n = len(powers)
    for s in range(start, n - win + 1):
        if _coefficient_of_variation(powers[s : s + win]) > _AUTO_CV_THRESHOLD:
            return False
    return True


def _coefficient_of_variation(values: list[float]) -> float:
    """Std / mean of ``values``. Returns infinity for a non-positive mean."""
    n = len(values)
    mean = sum(values) / n
    if mean <= 0.0:
        return float("inf")
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance) / mean


def _resolve_steady_state_window(
    cleaned: list[PowerThermalSample],
    origin: float,
    span: float,
    config: MeasurementConfig,
) -> tuple[float, float, bool, list[str]]:
    """Resolve the steady-state window: auto-detect if opted in, else fixed discard.

    Returns ``(start_sec, end_sec, not_detected, warnings)`` where the bounds are
    relative to ``origin`` (the raw inference-start anchor). ``span`` is the realised
    end bound; the onset is the resolved discard.
    """
    warnings: list[str] = []

    # Fixed-discard onset (the default and the fallback), relative to the anchor.
    if config.warmup_discard_seconds is not None:
        fixed_start = min(config.warmup_discard_seconds, span)
    else:
        fixed_start = span * config.warmup_discard_fraction

    if not config.steady_state_auto_detect:
        return fixed_start, span, False, warnings

    # Auto-detect over the per-GPU-pooled cleaned series (single GPU is the common case;
    # for multi-GPU the pooled CV is a conservative stability proxy). Onset is returned
    # relative to the cleaned series start, then re-anchored to ``origin``.
    times = [s.timestamp for s in cleaned]
    powers = [s.power_w for s in cleaned if s.power_w is not None]
    onset = _detect_steady_state(powers, times) if len(powers) == len(times) else None

    if onset is None:
        warnings.append(
            "steady_state auto-detection found no stable region; "
            "fell back to fixed warm-up discard."
        )
        return fixed_start, span, True, warnings

    # _detect_steady_state returns onset relative to cleaned start; re-anchor to origin.
    onset += cleaned[0].timestamp - origin
    return onset, span, False, warnings


def apply_measurement_window(
    samples: list[PowerThermalSample],
    config: MeasurementConfig,
    inference_time_sec: float,
) -> WindowResult | None:
    """Apply the configured measurement window and re-integrate energy over it.

    Pre-cleans the NVML power series, resolves the window for the selected
    methodology, filters samples to the window, and re-integrates energy with the
    existing trapezoidal integrator. Token / throughput attribution is proportional
    by time (the harness has no absolute per-token timestamps); ``token_fraction``
    carries the multiplier and a provenance warning is attached.

    Args:
        samples: Raw NVML power samples (``PowerThermalSample``) spanning the run.
        config: Measurement configuration (methodology + window/discard params).
        inference_time_sec: Harness-measured inference span (perf_counter delta),
            used as the window denominator and as the end bound for steady_state.

    Returns:
        A WindowResult, or None when windowing cannot be applied (too few clean
        samples) so the caller keeps the unchanged ``total`` figures.
    """
    methodology = config.measurement_methodology
    if methodology == "total":
        return None

    if len(samples) < 2:
        return None

    # Anchor the window to the RAW series origin (the sampler starts at inference
    # start), so a window relative to inference start is not shifted when cleaning
    # drops leading samples.
    origin = min(s.timestamp for s in samples)
    end_ts = max(s.timestamp for s in samples)
    span = end_ts - origin

    cleaned = _clean_samples(samples)
    if len(cleaned) < 2:
        logger.warning(
            "measurement_methodology=%s requested but only %d usable power "
            "samples after cleaning; keeping total-run figures.",
            methodology,
            len(cleaned),
        )
        return None

    warnings: list[str] = []
    not_detected = False

    if methodology == "windowed":
        assert config.measurement_window is not None  # enforced by config validation
        start_sec, end_sec = config.measurement_window
        end_sec = min(end_sec, span)
        start_sec = min(start_sec, end_sec)
    else:  # steady_state
        start_sec, end_sec, not_detected, ss_warnings = _resolve_steady_state_window(
            cleaned, origin, span, config
        )
        warnings.extend(ss_warnings)

    windowed = _filter_to_window(cleaned, origin, start_sec, end_sec)
    per_gpu_j = integrate_power_samples(windowed)
    energy_j = sum(per_gpu_j.values())

    window_duration = end_sec - start_sec
    if window_duration < _MIN_WINDOW_SEC:
        warnings.append(
            f"measurement window is {window_duration:.2f}s, below the "
            f"{_MIN_WINDOW_SEC:.1f}s minimum-duration floor; energy/throughput over "
            "this window have high relative uncertainty."
        )

    # Proportional-by-time token attribution. The harness captures per-request and
    # inter-token durations but no absolute per-token timestamps, so tokens are
    # credited by the window's share of the cleaned inference span. Documented as an
    # approximation in the result provenance.
    denom = span if span > 0.0 else inference_time_sec
    token_fraction = (window_duration / denom) if denom > 0.0 else 0.0
    token_fraction = max(0.0, min(1.0, token_fraction))
    if methodology in ("windowed", "steady_state"):
        warnings.append(
            "tokens and throughput attributed to the measurement window "
            f"proportionally by time (window covers {token_fraction:.0%} of the "
            "cleaned inference span); the harness captures no absolute per-token "
            "timestamps for exact attribution."
        )

    return WindowResult(
        methodology=methodology,
        energy_j=energy_j,
        per_gpu_j=per_gpu_j,
        window=(start_sec, end_sec),
        window_duration_sec=window_duration,
        token_fraction=token_fraction,
        steady_state_not_detected=not_detected,
        warnings=warnings,
    )
