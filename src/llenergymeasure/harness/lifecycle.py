"""Per-model-load lifecycle phases: baseline, model+snapshot load, prompts, warmup.

These phases run once per model load and their products are fixed for every
measured window taken against that load. :func:`build_lifetime` sequences them
and returns the :class:`_EngineLifetime` carrier; splitting lifetime state from
per-window state is the server-mode seam (one lifetime, N windows later).

The lazy-import wrappers here (``collect_environment_snapshot``,
``collect_environment_snapshot_async``, ``measure_baseline_power``) plus the
top-level ``load_prompts`` / ``thermal_floor_wait`` / ``importlib`` / ``time``
imports are this module's monkeypatch surface: tests patch them at the use site,
``llenergymeasure.harness.lifecycle.<name>``.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import TIMEOUT_ENV_SNAPSHOT
from llenergymeasure.datasets import load_prompts
from llenergymeasure.domain.progress import STEP_BASELINE, emit_substep
from llenergymeasure.engines.protocol import EnginePlugin
from llenergymeasure.harness.warmup import thermal_floor_wait, warmup_until_converged
from llenergymeasure.utils.formatting import bytes_to_mb

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.metrics import WarmupResult
    from llenergymeasure.domain.progress import ProgressCallback
    from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-import wrappers (top-level here so tests can patch them; the heavy work
# stays in the phase-specific modules imported inside).
# ---------------------------------------------------------------------------


def collect_environment_snapshot() -> EnvironmentSnapshot:  # pragma: no cover
    from llenergymeasure.harness.environment import (
        collect_environment_snapshot as _snap,
    )

    return _snap()


def collect_environment_snapshot_async() -> Future[EnvironmentSnapshot]:  # pragma: no cover
    from llenergymeasure.harness.environment import (
        collect_environment_snapshot_async as _snap_async,
    )

    return _snap_async()


def measure_baseline_power(
    duration_sec: float,
    gpu_indices: list[int] | None = None,
) -> BaselineCache | None:  # pragma: no cover
    from llenergymeasure.harness.baseline import measure_baseline_power as _mbp

    return _mbp(duration_sec=duration_sec, gpu_indices=gpu_indices)


@dataclass
class _EngineLifetime:
    """Per-model-load state, built once by the load/warmup phases.

    Everything here is fixed for a given model load and reused across every
    measured window taken against it. Splitting it from _MeasuredWindow is the
    server-mode seam (one lifetime, N windows); offline takes exactly one
    window per lifetime, so no multi-window machinery exists yet.
    """

    model: Any
    snapshot: EnvironmentSnapshot | None
    baseline: BaselineCache | None
    model_memory_mb: float
    model_load_time_sec: float
    prompts: list[str]
    warmup_result: WarmupResult


def build_lifetime(
    engine: EnginePlugin,
    config: ExperimentConfig,
    *,
    snapshot: EnvironmentSnapshot | None,
    gpu_indices: list[int] | None,
    progress: ProgressCallback | None,
    baseline: BaselineCache | None,
) -> _EngineLifetime:
    """Build the per-model-load lifetime: snapshot, baseline, model, prompts, warmup.

    These phases run once per model load and their products are fixed for every
    window taken against that load. If warmup fails after the model is live, the
    model is released here before re-raising (a load or prompt failure leaves
    nothing to release, matching the pre-refactor path).
    """
    # 1. Environment snapshot - start background thread (before model loading)
    snapshot_future: Future[EnvironmentSnapshot] | None = None
    if snapshot is None:
        logger.debug("Collecting environment snapshot (background thread)")
        snapshot_future = collect_environment_snapshot_async()

    # 2. Baseline power measurement (before model load)
    baseline = _measure_or_reuse_baseline(config, baseline, gpu_indices, progress)

    # 3-4. Load model, join env snapshot, capture model memory baseline
    model, snapshot, model_memory_mb, model_load_time_sec = _load_model_and_snapshot(
        engine, config, snapshot, snapshot_future, gpu_indices, progress
    )

    # 4b. Load prompts - BEFORE the measurement window (methodology fix)
    prompts = _load_prompts(config, progress)

    # 5-6. Warmup convergence + thermal floor wait
    try:
        warmup_result = run_warmup(engine, config, model, prompts, progress)
    except BaseException:
        engine.cleanup(model)
        raise

    return _EngineLifetime(
        model=model,
        snapshot=snapshot,
        baseline=baseline,
        model_memory_mb=model_memory_mb,
        model_load_time_sec=model_load_time_sec,
        prompts=prompts,
        warmup_result=warmup_result,
    )


def _measure_or_reuse_baseline(
    config: ExperimentConfig,
    baseline: BaselineCache | None,
    gpu_indices: list[int] | None,
    progress: ProgressCallback | None,
) -> BaselineCache | None:
    """Measure baseline idle power, or mark a study-supplied baseline as cached.

    Returns the baseline to thread into the energy breakdown (None when baseline
    measurement is disabled or unavailable).
    """
    _p = progress

    if baseline is not None and config.measurement.baseline.enabled:
        # For cached/validated strategies, progress events are emitted by
        # study/runner.py::_get_baseline. Here we only mark from_cache for
        # create_energy_breakdown method attribution.
        from dataclasses import replace as _dc_replace

        baseline = _dc_replace(baseline, from_cache=True)
        logger.debug(
            "Using study-level baseline cache: %.1fW (%d samples)",
            baseline.power_w,
            baseline.sample_count,
        )
    elif config.measurement.baseline.enabled:
        dur = config.measurement.baseline.duration_seconds
        logger.debug("Measuring baseline power (%.0fs)...", dur)
        if _p:
            _p.on_step_start(STEP_BASELINE, "Measuring", f"baseline idle power ({dur:.0f}s)")
            t0 = time.perf_counter()
        baseline = measure_baseline_power(dur, gpu_indices=gpu_indices)
        if baseline is not None:
            cache_label = " (cached)" if baseline.from_cache else ""
            emit_substep(
                _p,
                STEP_BASELINE,
                f"baseline: {baseline.power_w:.1f}W ({baseline.sample_count} samples{cache_label})",
            )
            if _p and baseline.from_cache:
                _p.on_step_update(STEP_BASELINE, f"cached baseline {baseline.power_w:.1f}W")
        if _p:
            _p.on_step_done(STEP_BASELINE, time.perf_counter() - t0)
    elif _p:
        _p.on_step_skip(STEP_BASELINE, "disabled")
    return baseline


def _load_model_and_snapshot(
    engine: EnginePlugin,
    config: ExperimentConfig,
    snapshot: EnvironmentSnapshot | None,
    snapshot_future: Future[EnvironmentSnapshot] | None,
    gpu_indices: list[int] | None,
    progress: ProgressCallback | None,
) -> tuple[Any, EnvironmentSnapshot | None, float, float]:
    """Load the model, join the background environment snapshot, and capture the
    post-load GPU memory baseline.

    Returns (model, snapshot, model_memory_mb, load_time_sec). The memory
    baseline must be captured here - before warmup allocates KV cache.
    load_time_sec brackets engine.load_model() alone: model load plus any
    engine build/compile the plugin performs there. It lands on the result as
    model_load_time_sec (non-energy metadata; this phase precedes the NVML
    measurement window).
    """
    _p = progress

    # 3. Load model via engine plugin
    if _p:
        _p.on_step_start("model", "Loading", f"model {config.task.model}")
    t0 = time.perf_counter()

    # Build model substep callback
    def _on_model_substep(text: str, elapsed: float) -> None:
        emit_substep(_p, "model", text, elapsed)

    model = engine.load_model(config, on_substep=_on_model_substep)

    load_time_sec = time.perf_counter() - t0
    if _p:
        _p.on_step_done("model", load_time_sec)

    # 3b. Join snapshot future - collection hidden behind model loading
    if snapshot_future is not None:
        snapshot = snapshot_future.result(timeout=TIMEOUT_ENV_SNAPSHOT)

    # 4. Capture model memory baseline immediately after model load.
    # Must happen BEFORE warmup, which allocates KV cache.
    model_memory_mb = capture_model_memory_mb(gpu_indices=gpu_indices)
    if model_memory_mb > 0:
        emit_substep(_p, "model", f"model memory: {model_memory_mb:.0f}MB")

    return model, snapshot, model_memory_mb, load_time_sec


def _load_prompts(
    config: ExperimentConfig,
    progress: ProgressCallback | None,
) -> list[str]:
    """Load and tokenise prompts before the measurement window (methodology fix)."""
    _p = progress

    if _p:
        _p.on_step_start(
            "prompts",
            "Loading",
            f"prompts ({config.task.dataset.n_prompts} {config.task.dataset.source})",
        )
        t0_prompts = time.perf_counter()
    prompts = load_prompts(config)
    logger.debug("Loaded %d prompts via dataset loader", len(prompts))
    emit_substep(_p, "prompts", f"tokenised {len(prompts)} prompts")
    if _p:
        _p.on_step_done("prompts", time.perf_counter() - t0_prompts)
    return prompts


def run_warmup(
    engine: EnginePlugin,
    config: ExperimentConfig,
    model: Any,
    prompts: list[str],
    progress: ProgressCallback | None,
) -> WarmupResult:
    """Run engine warmup to convergence, then wait out the thermal floor.

    Returns the WarmupResult with thermal_floor_wait_s populated.
    """
    _p = progress
    # Offline warmup protocol (migrated from measurement.warmup to the offline: mode
    # namespace; defaults apply when the optional offline section is omitted).
    warmup = config.offline_warmup()

    # 5. Warmup
    if _p:
        _p.on_step_start("warmup", "Warming up", f"up to {warmup.max_prompts} prompts")
        t0_warmup = time.perf_counter()

    # Probe call determines warmup strategy:
    # > 0.0 -> CV-based convergence (Transformers), 0.0 -> kernel warmup (vLLM/TRT-LLM)
    first_latency = engine.run_warmup_prompt(config, model, prompts[0]) if warmup.enabled else 0.0

    if first_latency > 0.0:
        warmup_substep = (
            (lambda text, elapsed: _p.on_substep("warmup", text, elapsed)) if _p else None
        )
        warmup_result = warmup_until_converged(
            lambda: engine.run_warmup_prompt(config, model, prompts[0]),
            warmup,
            on_substep=warmup_substep,
        )
        # The probe at line above ran one extra discarded inference that
        # warmup_until_converged() does not count; fold it into the warmup
        # total so warmup_excluded_samples reflects every discarded inference.
        warmup_result.iterations_completed += 1
    else:
        from llenergymeasure.domain.metrics import WarmupResult

        warmup_result = WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=1 if warmup.enabled else 0,
            target_cv=warmup.cv_threshold,
            max_prompts=warmup.max_prompts,
        )

    if _p:
        iters = warmup_result.iterations_completed
        cv_info = f"  CV={warmup_result.final_cv:.1%}" if warmup_result.final_cv > 0 else ""
        converged = "converged" if warmup_result.converged else "not converged"
        iters_label = f"{iters} iteration{'s' if iters != 1 else ''}"
        # Kernel-only warmup (vLLM/TRT-LLM): no CV-based convergence
        if first_latency == 0.0 and warmup.enabled:
            _p.on_step_update("warmup", f"engine kernel warmup ({iters_label})")
        else:
            _p.on_step_update("warmup", f"{iters_label} ({converged}{cv_info})")
        _p.on_step_done("warmup", time.perf_counter() - t0_warmup)
    iters = warmup_result.iterations_completed
    emit_substep(
        _p,
        "warmup",
        f"{iters} iteration{'s' if iters != 1 else ''}"
        + (f"  CV={warmup_result.final_cv:.1%}" if warmup_result.final_cv > 0 else ""),
    )

    # 6. Thermal floor - show step before sleeping
    floor_secs = warmup.thermal_floor_seconds if warmup.enabled else 0
    if _p:
        if floor_secs > 0:
            _p.on_step_start("thermal_floor", "Waiting", f"thermal floor ({floor_secs:.0f}s)")
        else:
            _p.on_step_skip("thermal_floor", "wait=0s")

    wait_s = thermal_floor_wait(config)
    warmup_result.thermal_floor_wait_s = wait_s

    if _p and wait_s > 0:
        _p.on_step_done("thermal_floor", wait_s)

    return warmup_result


def capture_model_memory_mb(gpu_indices: list[int] | None = None) -> float:
    """Capture the post-load model-memory baseline in MB.

    In-process engines (Transformers): torch's per-process allocator sees the
    loaded weights, so ``torch.cuda.max_memory_allocated(device=idx)`` per rank
    (max across ``gpu_indices``, defaulting to ``[0]``) is authoritative and
    captures the tensor-parallel peak.

    Out-of-process engines (vLLM V1 EngineCore, TRT-LLM executor): the weights
    are loaded in a child process, so torch here reports a silent 0.0. Fall back
    to NVML device-used memory (whole-device; see
    :func:`get_nvml_device_memory_mb` for the tenancy caveat).

    Returns 0.0 only when NEITHER source is available (no torch/CUDA and no
    NVML, or a CPU run). The domain layer coerces a 0.0 baseline to null so the
    field is never a silently-wrong zero.
    """
    torch_mb = 0.0
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                indices = gpu_indices if gpu_indices is not None else [0]
                if indices:
                    peak = max(torch.cuda.max_memory_allocated(device=idx) for idx in indices)
                    torch_mb = bytes_to_mb(peak)
        except Exception:
            torch_mb = 0.0
    if torch_mb > 0:
        return torch_mb

    # torch saw nothing: out-of-process engine or no in-process CUDA activity.
    from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

    nvml_mb = get_nvml_device_memory_mb(gpu_indices)
    return nvml_mb if nvml_mb is not None else 0.0
