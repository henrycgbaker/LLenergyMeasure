"""Measured-window orchestration: drive the bracket, then the offline post-window steps.

:func:`run_measured_inference` takes one measured window via
:class:`~llenergymeasure.harness.bracket.MeasurementBracket` (the mode-agnostic
window mechanics) and applies the offline-specific post-window steps:
observed-params capture, the canonical inference timer, and FLOPs estimation. It
returns the :class:`_MeasuredWindow` composing the bracket's core with those
offline extras.

The lazy FLOPs wrappers here (``estimate_flops_palm`` /
``estimate_flops_palm_from_config``) are this module's monkeypatch surface: tests
patch them at the use site, ``llenergymeasure.harness.window.<name>``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llenergymeasure.domain.progress import emit_substep
from llenergymeasure.engines.protocol import EnginePlugin, InferenceOutput
from llenergymeasure.harness.bracket import MeasuredWindowCore, MeasurementBracket
from llenergymeasure.harness.lifecycle import _EngineLifetime

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import FlopsResult
    from llenergymeasure.domain.progress import ProgressCallback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-import wrappers (top-level here so tests can patch them).
# ---------------------------------------------------------------------------


def estimate_flops_palm(
    model: Any, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult:  # pragma: no cover
    from llenergymeasure.harness.flops import estimate_flops_palm as _efp

    return _efp(model=model, n_input_tokens=n_input_tokens, n_output_tokens=n_output_tokens)


def estimate_flops_palm_from_config(
    model_name: str, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult | None:  # pragma: no cover
    from llenergymeasure.harness.flops import estimate_flops_palm_from_config as _efpc

    return _efpc(
        model_name=model_name,
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
    )


def _capture_observed_params_into_output(
    engine: Any,
    config: Any,
    model: Any,
    output: Any,
) -> None:
    """Call ``engine.capture_observed_params`` and merge the result into ``output.extras``.

    Invoked post-window (after ``t_inference_end`` + ``_cuda_sync``) so capture
    overhead does not perturb energy or timing measurements.

    Best-effort: engines that haven't implemented the method yet silently skip.
    The result dict is merged into ``output.extras`` under the standard keys
    ``observed_engine_params``, ``observed_sampling_params``, and
    ``library_version``.
    """
    try:
        capture = getattr(engine, "capture_observed_params", None)
        if capture is None:
            return
        params = capture(config, model, output)
        if isinstance(params, dict):
            output.extras["observed_engine_params"] = params.get("engine", {})
            output.extras["observed_sampling_params"] = params.get("sampling", {})
            output.extras["library_version"] = params.get("library_version", "unknown")
    except Exception as exc:
        logger.debug("capture_observed_params failed (non-fatal): %s", exc)


@dataclass
class _MeasuredWindow:
    """Product of one measured inference window.

    Composes the bracket's mode-agnostic
    :class:`~llenergymeasure.harness.bracket.MeasuredWindowCore` (energy, thermal,
    timeseries, timestamps, sampler-probe reasons) with the offline-specific
    extras (the InferenceOutput and the FLOPs estimate).
    """

    core: MeasuredWindowCore
    output: InferenceOutput
    flops_result: Any


def run_measured_inference(
    engine: EnginePlugin,
    config: ExperimentConfig,
    lifetime: _EngineLifetime,
    gpu_indices: list[int] | None,
    progress: ProgressCallback | None,
) -> _MeasuredWindow:
    """Take one measured window via MeasurementBracket, then apply the
    offline-specific post-window steps: observed-params capture, canonical
    inference timer, and FLOPs estimation.

    The bracket owns the mode-agnostic window mechanics. Observed-params capture
    is interleaved between the bracket's context exit (thermal sampler stopped)
    and ``finish()`` (energy tracker stopped), preserving the deliberate
    energy>thermal window-width gap. See harness/bracket.py and
    tests/unit/harness/test_window_ordering.py.
    """
    _p = progress
    bracket = MeasurementBracket(
        config.measurement,
        gpu_indices,
        progress,
        measure_detail=f"inference ({config.task.dataset.n_prompts} prompts)",
    )

    with bracket:
        output = engine.run_inference(config, lifetime.model, lifetime.prompts)

    # 11a. Capture observed params post-window: after the thermal sampler has
    # stopped (bracket __exit__) but before the energy tracker stops
    # (bracket.finish()), so the capture overhead (~5-50 ms pure Python) lands
    # inside the energy window but outside the thermal timeseries. The model is
    # still alive.
    _capture_observed_params_into_output(engine, config, lifetime.model, output)

    # Harness sets canonical inference timer - overrides engine's elapsed_time_sec
    output.inference_time_sec = bracket.inference_duration_sec

    # 12. Stop energy tracking and finalise the mode-agnostic window core.
    core = bracket.finish()

    # 13. FLOPs estimation (warmup tokens excluded) - offline-specific.
    if _p:
        _p.on_step_start("flops", "Estimating", "FLOPs (PaLM formula)")
        t0_flops = time.perf_counter()
    flops_result = estimate_flops(engine, config, output)
    if _p:
        if flops_result is not None:
            _p.on_step_update("flops", f"FLOPs: {flops_result.value:.2e}")
        _p.on_step_done("flops", time.perf_counter() - t0_flops)
    if flops_result is not None:
        emit_substep(_p, "flops", f"FLOPs: {flops_result.value:.2e}")

    return _MeasuredWindow(core=core, output=output, flops_result=flops_result)


def estimate_flops(
    engine: EnginePlugin,
    config: ExperimentConfig,
    output: InferenceOutput,
) -> Any:
    """Estimate FLOPs from model and token counts.

    Fallback chain (highest confidence first):
    1. hf_model path - uses estimate_flops_palm(hf_model) when extras['hf_model']
       is set. Higher confidence: counts the actual loaded parameters.
    2. AutoConfig path - uses estimate_flops_palm_from_config(config.task.model).
       Works for engines that do not expose a model object (vLLM, TensorRT-LLM).
    3. None - FLOPs unavailable.
    """
    # Step 1: hf_model path (higher confidence - uses actual loaded parameters)
    model_obj = output.extras.get("hf_model")
    if model_obj is not None:
        try:
            return estimate_flops_palm(
                model=model_obj,
                n_input_tokens=output.input_tokens,
                n_output_tokens=output.output_tokens,
            )
        except Exception as e:
            logger.debug("hf_model FLOPs estimation failed: %s", e)

    # Step 2: AutoConfig path (works without a loaded model object)
    try:
        result = estimate_flops_palm_from_config(
            model_name=config.task.model,
            n_input_tokens=output.input_tokens,
            n_output_tokens=output.output_tokens,
        )
        if result is not None:
            return result
    except Exception as e:
        logger.debug("AutoConfig FLOPs estimation failed: %s", e)

    # Step 3: FLOPs unavailable
    return None
