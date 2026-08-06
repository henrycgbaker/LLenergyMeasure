"""Thermal stabilisation and warmup convergence utilities for MeasurementHarness.

Provides:
- thermal_floor_wait(): sleep after warmup for thermal stabilisation
- warmup_until_converged(): convergence loop with per-iteration on_substep callbacks

Engines implement run_warmup_prompt() as a thin inference primitive.
The harness owns the convergence detection loop (CV thresholds, iteration limits).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from llenergymeasure.config.models import ExperimentConfig, WarmupConfig
from llenergymeasure.domain.metrics import WarmupResult

# REUSE: the convergence check shares windowing.py's coefficient-of-variation
# math with the server-warmup power-plateau observable; only the stable-through-end
# semantics are lifted here, the threshold stays warmup's configurable cv_threshold.
from llenergymeasure.harness.windowing import _coefficient_of_variation

logger = logging.getLogger(__name__)


def _stable_through_end(values: list[float], window: int, threshold: float) -> bool:
    """True iff every ``window``-sized slice of ``values`` (start-to-end) is stable.

    "Stable" is coefficient of variation at or below ``threshold``. Requiring every
    trailing window (not merely the last one) to be stable is the stable-through-end
    convergence semantics shared with the windowing detector and the server-warmup
    gate: a sustained plateau, not a single quiet window. Reuses windowing.py's
    CoV math; the threshold stays warmup's configurable ``cv_threshold``.
    """
    n = len(values)
    if n < window:
        return False
    return all(
        _coefficient_of_variation(values[s : s + window]) <= threshold
        for s in range(n - window + 1)
    )


def thermal_floor_wait(config: ExperimentConfig) -> float:
    """Sleep for thermal_floor_seconds after warmup. Returns actual wait time in seconds.

    Returns 0.0 immediately if warmup is disabled or thermal_floor_seconds <= 0.
    Called by MeasurementHarness after warmup, before energy tracking starts.

    Args:
        config: Experiment configuration (reads warmup.enabled and warmup.thermal_floor_seconds).

    Returns:
        Actual elapsed wait time in seconds (>= 0.0).
    """
    warmup = config.offline_warmup()
    if not warmup.enabled or warmup.thermal_floor_seconds <= 0:
        return 0.0
    logger.debug(
        "Thermal stabilisation: waiting %.0fs...",
        warmup.thermal_floor_seconds,
    )
    t0 = time.monotonic()
    time.sleep(warmup.thermal_floor_seconds)
    return time.monotonic() - t0


def warmup_until_converged(
    run_single_inference: Callable[[], float],
    config: WarmupConfig,
    *,
    on_substep: Callable[[str, float], None] | None = None,
) -> WarmupResult:
    """Run warmup prompts until latency CV stabilises below threshold.

    Owns the convergence detection loop (CV thresholds, iteration limits).
    Engines provide run_warmup_prompt() as a thin inference primitive.

    Args:
        run_single_inference: Callable that runs one warmup prompt and
            returns latency in milliseconds. Decouples warmup from
            specific inference implementation.
        config: WarmupConfig controlling convergence parameters.
        on_substep: Optional callback ``(text, elapsed_sec)`` for reporting
            per-iteration CV progress. Called after each iteration with CV info.

    Returns:
        WarmupResult with convergence status and metrics.
    """
    # Early return if warmup is disabled
    if not config.enabled:
        logger.debug("Warmup disabled, skipping")
        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=0,
            target_cv=config.cv_threshold,
            max_prompts=config.max_prompts,
        )

    from llenergymeasure.domain.metrics import compute_cv

    latencies: list[float] = []
    converged = False
    final_cv: float | None = None

    # Fixed mode: run exactly n_prompts prompts without convergence checking.
    # CV mode: run up to max_prompts with convergence checking after min_prompts.
    fixed_mode = not config.convergence_detection
    iteration_limit = config.n_prompts if fixed_mode else config.max_prompts

    for i in range(iteration_limit):
        try:
            latency_ms = run_single_inference()
        except Exception as exc:
            logger.warning("Warmup prompt %d failed: %s", i + 1, exc)
            continue

        latencies.append(latency_ms)

        # Compute CV whenever we have 2+ samples (minimum for meaningful std dev).
        # In fixed mode this is informational; in convergence mode it drives early-break.
        if len(latencies) >= 2:
            recent = latencies[-config.window_size :]
            final_cv = compute_cv(recent)

            # Stable-through-end convergence (upgraded from a single trailing
            # window): once past the warm-start floor, converge iff the trailing
            # eligible region is a sustained plateau - every window-sized slice
            # stable, not just the last one.
            if not fixed_mode and len(latencies) >= max(config.min_prompts, config.window_size):
                eligible = latencies[-max(config.min_prompts, config.window_size) :]
                if _stable_through_end(eligible, config.window_size, config.cv_threshold):
                    converged = True

        if on_substep:
            cv_str = f"  CV: {final_cv:.3f}" if final_cv is not None else ""
            target_info = f"  (target: {config.cv_threshold:.3f})" if not fixed_mode else ""
            on_substep(f"Iteration {i + 1}/{iteration_limit}{cv_str}{target_info}", 0.0)

        if converged:
            break

    # In fixed mode, mark as converged (user chose fixed iterations)
    if fixed_mode:
        converged = True

    # Normalise for return (WarmupResult.final_cv is non-optional float)
    result_cv = final_cv if final_cv is not None else 0.0

    # Log result
    if converged and not fixed_mode:
        logger.info(
            "Warmup converged after %d prompts (CV=%.3f < %.3f)",
            len(latencies),
            result_cv,
            config.cv_threshold,
        )
    elif fixed_mode:
        cv_info = f" (final CV={result_cv:.3f})" if final_cv is not None else ""
        logger.info("Warmup completed %d fixed iterations%s", len(latencies), cv_info)
    else:
        logger.warning(
            "Warmup did not converge after %d prompts (final CV=%.3f, target=%.3f)",
            iteration_limit,
            result_cv,
            config.cv_threshold,
        )

    return WarmupResult(
        converged=converged,
        final_cv=result_cv,
        iterations_completed=len(latencies),
        target_cv=config.cv_threshold,
        max_prompts=config.max_prompts,
        latencies_ms=latencies,
    )
