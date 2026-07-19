"""TensorRT-LLM inference engine - thin EnginePlugin.

Implements the EnginePlugin protocol:
  load_model, warmup, run_inference, cleanup, check_hardware

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only TRT-LLM-specific inference: model loading via tensorrt_llm.LLM(),
a minimal 1-prompt warmup, offline batch llm.generate(), and cleanup.

All tensorrt_llm and torch imports are lazy so this module can be imported on
hosts without TRT-LLM or CUDA installed.

Engine compilation must NEVER occur inside the NVML measurement window.
The load_model() call triggers compilation; run_inference() assumes the
engine is ready.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.utils.exceptions import ConfigError, EngineError

logger = logging.getLogger(__name__)


def _validate_engine_directory(engine_path: Path, tp_size: int) -> list[str]:
    """Pre-flight validation for TRT-LLM engine directory.

    Checks directory exists, config.json exists, tp_size matches, and
    rank{N}.engine files exist. Returns list of error strings (empty = valid).
    Does NOT re-implement TRT-LLM's format detection.

    Args:
        engine_path: Path to the engine directory.
        tp_size: Expected tensor-parallel size (number of rank files to check).

    Returns:
        List of error strings. Empty list means the directory is valid.
    """
    errors: list[str] = []

    if not engine_path.is_dir():
        errors.append(f"engine_path does not exist or is not a directory: {engine_path}")
        return errors

    config_path = engine_path / "config.json"
    if not config_path.exists():
        errors.append(f"config.json not found in engine directory: {engine_path}")
    else:
        try:
            with config_path.open() as f:
                config_data = json.load(f)
            engine_tp_size = (
                config_data.get("pretrained_config", {}).get("mapping", {}).get("tp_size")
            )
            if engine_tp_size is not None and engine_tp_size != tp_size:
                errors.append(
                    f"tp_size mismatch: engine was built with tp_size={engine_tp_size} "
                    f"but config requests tp_size={tp_size}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"Failed to parse config.json in engine directory: {exc}")

    for rank in range(tp_size):
        rank_file = engine_path / f"rank{rank}.engine"
        if not rank_file.exists():
            errors.append(f"rank{rank}.engine not found in engine directory: {engine_path}")

    return errors


def _extract_metrics_dict(outputs: Any) -> tuple[list[float], list[float], list[float]]:
    """Per-request (e2e_ms, ttft_ms, tpot_ms) lists from ``RequestOutput.metrics_dict``.

    TRT-LLM 1.x records per-request perf metrics in ``metrics_dict`` - keyed by
    ``MetricNames`` enum members whose ``.value`` is the plain metric name, with
    SECOND-valued timings - when ``return_perf_metrics=True`` was requested.
    Live-verified at 1.2.1 on both backend legs (pytorch and trt): the
    ``SamplingParams(return_perf_metrics=True)`` flag alone populates
    ``ttft`` / ``e2e`` / ``tpot`` (+ ``arrival_timestamp`` /
    ``request_queue_time`` / ``finished_reason``). Without the flag the dict is
    empty, so this extraction contributes nothing on default runs.

    ``tpot`` is the engine-computed per-request average time-per-output-token,
    not a per-token timestamp stream - the caller labels the capture
    ``per_request_batch`` accordingly.
    """
    latencies_ms: list[float] = []
    ttft_ms: list[float] = []
    itl_ms: list[float] = []
    for o in outputs:
        md = getattr(o, "metrics_dict", None) or {}
        normalized = {str(getattr(k, "value", k)): v for k, v in md.items()}
        e2e = normalized.get("e2e")
        ttft = normalized.get("ttft")
        tpot = normalized.get("tpot")
        if isinstance(e2e, (int, float)):
            latencies_ms.append(float(e2e) * 1000.0)
        if isinstance(ttft, (int, float)):
            ttft_ms.append(float(ttft) * 1000.0)
        if isinstance(tpot, (int, float)):
            itl_ms.append(float(tpot) * 1000.0)
    return latencies_ms, ttft_ms, itl_ms


def _apply_default_build_cache(kwargs: dict[str, Any]) -> None:
    """Apply the env-var-gated default TRT-LLM build cache to ``kwargs``.

    The opinionated llenergymeasure default is enabled (engine compilation is
    expensive; the cache is a large time-saver for repeat runs) - shipped via
    ``LLEM_TRT_BUILD_CACHE_ENABLED=1`` in ``.env.example``. The helpers are
    pure passthrough (see :mod:`llenergymeasure.utils.env_config`), so
    removing the line reverts to TRT-LLM's disabled default.

    - Disabled by env → ``enable_build_cache`` is not set (TRT-LLM default is
      False, matching the discovered schema).
    - Enabled with a user-supplied path → build a ``BuildCacheConfig`` whose
      ``cache_root`` is that path. Raises :class:`EngineError` if
      ``tensorrt_llm.llmapi`` cannot be imported: a measurement instrument must
      not silently run without a build cache the user configured.
    - Enabled without a path → set ``enable_build_cache = True`` (preserves the
      pre-env-var behaviour).
    """
    from llenergymeasure.utils.env_config import trt_build_cache_enabled, trt_build_cache_path

    if not trt_build_cache_enabled():
        return

    cache_root = trt_build_cache_path()
    if cache_root is not None:
        try:
            from tensorrt_llm.llmapi import BuildCacheConfig
        except ImportError as exc:
            raise EngineError(
                f"A build cache path was configured (cache_root={cache_root}) but "
                "tensorrt_llm.llmapi.BuildCacheConfig could not be imported "
                f"({exc}). Refusing to silently run without the configured build "
                "cache. Run inside the tensorrt_llm image, or unset "
                "LLEM_TRT_BUILD_CACHE_PATH."
            ) from exc
        kwargs["enable_build_cache"] = BuildCacheConfig(cache_root=cache_root)
        return

    kwargs["enable_build_cache"] = True


class TensorRTEngine:
    """TensorRT-LLM inference engine - offline batch mode, thin plugin.

    Implements EnginePlugin:
    - load_model: Compile/load engine via tensorrt_llm.LLM()
    - warmup: Minimal 1-prompt warmup with 1-token output
    - run_inference: Single llm.generate() call with ALL prompts, returns InferenceOutput
    - cleanup: Delete LLM instance, gc.collect(), clear CUDA cache
    - check_hardware: Check SM >= 7.5 (Turing) and FP8 requires SM >= 8.9
    """

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "tensorrt"

    @property
    def version(self) -> str:
        """TensorRT-LLM version string."""
        try:
            import tensorrt_llm

            return str(tensorrt_llm.__version__)
        except Exception:
            return "unknown"

    # -------------------------------------------------------------------------
    # EnginePlugin: load_model
    # -------------------------------------------------------------------------

    def load_model(
        self,
        config: ExperimentConfig,
        on_substep: Callable[[str, float], None] | None = None,
    ) -> tuple[Any, Any]:
        """Compile/load engine via tensorrt_llm.LLM() and build SamplingParams.

        Engine compilation happens here - BEFORE the NVML measurement window.

        All tensorrt_llm imports are lazy so this module can be imported without TRT-LLM.

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for substep visibility.

        Returns:
            Tuple of (llm, sampling_params).

        Raises:
            EngineError: If TRT-LLM is not installed or model loading fails.
        """
        from llenergymeasure.engines._errors import require_import

        _trt_mod = require_import("tensorrt_llm")
        LLM = self._resolve_llm_class(config)

        kwargs = self._build_llm_kwargs(config)
        logger.info("Loading TRT-LLM model %r (kwargs: %s)", config.task.model, list(kwargs.keys()))

        from llenergymeasure.device.gpu_info import get_gpu_architecture

        gpu_arch = get_gpu_architecture()

        trt_version = getattr(_trt_mod, "__version__", "unknown")

        build_start = time.perf_counter()

        try:
            llm = LLM(**kwargs)
        except Exception as e:
            raise EngineError(f"TensorRT-LLM model loading failed: {e}") from e

        build_time_sec = time.perf_counter() - build_start
        if on_substep is not None:
            on_substep(f"engine compiled ({gpu_arch}, TRT-LLM {trt_version})", build_time_sec)
        logger.debug(
            "TRT-LLM engine built in %.1fs (arch=%s, version=%s)",
            build_time_sec,
            gpu_arch,
            trt_version,
        )

        sampling_params = self._build_sampling_params(config)
        if on_substep is not None:
            on_substep("sampling params built", 0.0)
        return llm, sampling_params

    # -------------------------------------------------------------------------
    # EnginePlugin: warmup
    # -------------------------------------------------------------------------

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt via single-token kernel warmup. Returns 0.0.

        Returns 0.0 to signal the harness to skip CV-based convergence.
        TRT-LLM uses a single-token kernel warmup rather than CV convergence.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompt: Single warmup prompt text.

        Returns:
            0.0 (signals harness to skip convergence loop).
        """
        from tensorrt_llm import SamplingParams

        from llenergymeasure.engines._cuda import warmup_single_token

        llm, _sampling_params = model
        warmup_single_token(llm, [prompt], SamplingParams, max_tokens=1)
        return 0.0  # Signals harness to skip CV loop

    # -------------------------------------------------------------------------
    # EnginePlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run offline batch inference over all prompts.

        Single llm.generate() call with ALL prompts - no streaming.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.

        Raises:
            EngineError: On OOM or other inference failures.
        """
        llm, sampling_params = model

        # Reset peak stats before the measurement loop
        from llenergymeasure.engines._cuda import reset_cuda_peak_memory

        reset_cuda_peak_memory()

        logger.info(
            "Starting TRT-LLM offline batch inference: %d prompts, max_tokens=%s",
            len(prompts),
            config.task.max_output_tokens or "unlimited",
        )

        try:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            from llenergymeasure.engines._errors import raise_engine_error

            raise_engine_error(
                e,
                "TRT-LLM",
                hint="reduce n, use a smaller max_batch_size, or use a smaller model.",
            )

        # Capture peak memory - torch first, NVML fallback for out-of-process runs.
        from llenergymeasure.engines._cuda import (
            get_cuda_peak_memory_mb,
            get_nvml_device_memory_mb,
        )

        peak_mb = get_cuda_peak_memory_mb()
        if peak_mb == 0.0:
            # TRT-LLM runs its model in a separate executor process, so torch's
            # per-process allocator here saw nothing (a silent 0.0). Fall back to
            # NVML device-used memory (whole-device; see get_nvml_device_memory_mb
            # for the tenancy caveat). None stays 0.0 and is nulled downstream.
            nvml_peak = get_nvml_device_memory_mb()
            if nvml_peak is not None:
                peak_mb = nvml_peak

        # Count tokens from RequestOutput objects (same pattern as vLLM)
        from llenergymeasure.engines._observed import count_request_tokens

        input_token_count, output_token_count = count_request_tokens(outputs)

        logger.debug(
            "TRT-LLM inference complete: %d total tokens (in=%d, out=%d) in %.2fs",
            input_token_count + output_token_count,
            input_token_count,
            output_token_count,
            elapsed,
        )

        extras: dict[str, Any] = {}

        # Build-cache hit/miss annotation (trt backend only). Read from TRT-LLM's
        # own build stats; None when the cache is not in play. Harness maps this
        # to ExperimentResult.engine_build_cache_hit.
        extras["engine_build_cache_hit"] = self._detect_build_cache_hit(config, llm)

        # Per-request latency metrics from RequestOutput.metrics_dict. Populated
        # only when SamplingParams carried return_perf_metrics=True - which
        # _build_sampling_kwargs sets exactly when latency_profiling is enabled -
        # so default energy runs pay nothing and extract nothing. (The vLLM-shaped
        # ``RequestOutput.metrics`` namespace does NOT exist at TRT-LLM 1.2.1;
        # metrics_dict is the 1.x surface, live-verified on both backend legs.)
        per_request_latencies_ms, ttft_ms, itl_ms = _extract_metrics_dict(outputs)
        latency_measurement_mode: str | None = None
        if ttft_ms or per_request_latencies_ms:
            # Engine-recorded per-request TTFT/E2E plus an engine-computed
            # average TPOT per request - per-request timing without a
            # per-token stream.
            latency_measurement_mode = "per_request_batch"
        elif config.measurement.latency_profiling:
            # Profiling was requested but the engine returned no per-request
            # metrics (e.g. an engine_path run where the flag is not applied,
            # or an upstream regression): keep the loud degradation flag.
            extras["latency_profiling_unsupported"] = True

        return InferenceOutput(
            elapsed_time_sec=elapsed,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            peak_memory_mb=peak_mb,
            model_memory_mb=0.0,  # Captured by harness before run_inference
            batch_times=[elapsed],
            extras=extras,
            per_request_latencies_ms=per_request_latencies_ms,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            latency_measurement_mode=latency_measurement_mode,
            num_batches=1,  # Single batched generate() call
            padding_tokens=None,  # Not measurable from TRT-LLM RequestOutputs
            kv_cache_stats=None,  # TRT-LLM does not expose paged KV-cache stats here
        )

    @staticmethod
    def _detect_build_cache_hit(config: ExperimentConfig, llm: Any) -> bool | None:
        """Return whether the trt engine build was served from the build cache.

        Detection reads TRT-LLM's own ``llm_build_stats.engine_dir``: at 1.2.1 it
        is populated ONLY on the cache-reuse path (``llm_utils.py`` sets it just
        before the early return in the ``is_cached()`` branch) and left ``None``
        on a fresh build. The sibling ``cache_hitted`` flag is deliberately NOT
        used - it is set ``True`` on BOTH the reuse path AND after a fresh build
        writes the cache, so it cannot discriminate hit from miss.

        Returns ``None`` (annotation not applicable) when the build cache is not
        in play: a non-trt backend, an ``engine_path`` override (loads a prebuilt
        engine, bypassing the cache), or the cache disabled. Also returns ``None``
        defensively if the stats attribute is unavailable - never raises.

        Failure modes: relies on an internal TRT-LLM attribute (not public API),
        re-checked each pin; concurrent processes writing the same cache root do
        not affect this signal (unlike a directory-count delta) because it reads
        this construction's own stats object.
        """
        from llenergymeasure.utils.env_config import trt_build_cache_enabled

        engine_params = config.active_engine_params()
        backend = getattr(engine_params, "backend", None) if engine_params is not None else None
        engine_path = (
            getattr(engine_params, "engine_path", None) if engine_params is not None else None
        )
        if backend != "trt" or engine_path or not trt_build_cache_enabled():
            return None
        try:
            return llm.llm_build_stats.engine_dir is not None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Private: observed-params capture (observed_config_hash)
    # -------------------------------------------------------------------------

    @staticmethod
    def _capture_observed_params(
        config: ExperimentConfig,
        llm: Any,
        sampling_params: Any,
    ) -> dict[str, Any]:
        """Extract post-construction state from the TRT-LLM native types.

        TRT-LLM's ``LlmArgs`` + nested ``BuildConfig`` are Pydantic; accessible
        on ``llm.args`` in current releases. Private fields (if any surface in
        a given TRT-LLM version) are stripped by the default
        ``_``-prefix allowlist behaviour in
        :func:`extract_observed_params`.
        """
        from llenergymeasure.engines._observed import capture_two_part_observed

        return capture_two_part_observed(
            "tensorrt_llm",
            logger=logger,
            sampling_obj=sampling_params,
            engine_obj=getattr(llm, "args", None),
        )

    # -------------------------------------------------------------------------
    # EnginePlugin: capture_observed_params (post-measurement-window)
    # -------------------------------------------------------------------------

    def capture_observed_params(
        self,
        config: ExperimentConfig,
        model: Any,
        output: InferenceOutput,
    ) -> dict[str, Any]:
        """Extract library-observed effective parameters post-measurement-window.

        Called by the harness after ``t_inference_end`` + ``_cuda_sync`` so
        this overhead is outside the NVML energy window.

        The ``sampling_params`` object and ``llm`` are in the model tuple
        from ``load_model()``.
        """
        llm, sampling_params = model
        return self._capture_observed_params(config, llm, sampling_params)

    # -------------------------------------------------------------------------
    # EnginePlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release TRT-LLM model from memory and clear CUDA cache.

        Args:
            model: Tuple of (llm, sampling_params) from load_model().
        """
        from llenergymeasure.engines._cuda import cleanup_model

        llm, _sampling_params = model
        cleanup_model(llm)
        logger.debug("TRT-LLM model cleanup complete")

    # -------------------------------------------------------------------------
    # EnginePlugin: check_hardware
    # -------------------------------------------------------------------------

    @staticmethod
    def check_hardware(config: ExperimentConfig) -> list[str]:
        """Check SM capability + FP8 requirements against the visible GPU.

        Gates:
          - SM >= 7.5 (Turing minimum for TRT-LLM)
          - FP8 weight quant requires SM >= 8.9 (Ada Lovelace / Hopper)
          - FP8 KV-cache quant requires SM >= 8.9

        Returns ``[]`` when no GPU is visible.
        """
        from llenergymeasure.device.gpu_info import get_compute_capability

        sm = get_compute_capability()
        if sm is None:
            return []

        major, minor = sm
        sm_float = major + minor / 10
        errors: list[str] = []

        if sm_float < 7.5:
            errors.append(
                f"TensorRT-LLM requires SM >= 7.5 (Turing). This GPU has SM {major}.{minor}."
            )

        quant_config = config.engine_sub_dict("quant_config")
        if quant_config is not None:
            if quant_config.get("quant_algo") == "FP8" and sm_float < 8.9:
                errors.append(
                    f"FP8 quantisation requires SM >= 8.9 (Ada Lovelace or Hopper). "
                    f"This GPU has SM {major}.{minor} "
                    f"(A100=8.0, H100=9.0, RTX4090=8.9). "
                    f"Use W8A16, W4A16_AWQ, or W4A16_GPTQ instead."
                )
            if quant_config.get("kv_cache_quant_algo") == "FP8" and sm_float < 8.9:
                errors.append(
                    f"FP8 KV cache quantisation requires SM >= 8.9 (Ada Lovelace or Hopper). "
                    f"This GPU has SM {major}.{minor} "
                    f"(A100=8.0, H100=9.0, RTX4090=8.9). "
                    f"Use INT8 KV cache quantisation instead."
                )

        return errors

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_llm_class(config: ExperimentConfig) -> Any:
        """Select the TRT-LLM constructor class from the configured backend.

        At 1.2.1 ``tensorrt_llm.LLM`` is ``_TorchLLM``-based (validates against
        ``TorchLlmArgs``) and REJECTS ``backend='trt'`` at model load; the
        compiled-TensorRT path is the distinct
        ``tensorrt_llm._tensorrt_engine.LLM`` (``_TrtLLM``, validates against
        ``TrtLlmArgs``). We dispatch on the class rather than forwarding the
        upstream-deprecated ``backend`` kwarg (which is never passed to either
        constructor - see :meth:`_build_llm_kwargs`).

        - ``backend`` None or ``'pytorch'`` -> ``tensorrt_llm.LLM``
        - ``backend`` ``'trt'`` -> ``tensorrt_llm._tensorrt_engine.LLM``
        - anything else -> :class:`ConfigError` naming the accepted values.

        Raises:
            ConfigError: If ``backend`` is set to an unsupported value.
            EngineError: If the required TRT-LLM class cannot be imported.
        """
        from llenergymeasure.engines._errors import require_import

        engine_params = config.active_engine_params()
        backend = getattr(engine_params, "backend", None) if engine_params is not None else None
        if backend is None or backend == "pytorch":
            return require_import("tensorrt_llm").LLM
        if backend == "trt":
            return require_import("tensorrt_llm._tensorrt_engine").LLM
        raise ConfigError(
            f"Unsupported tensorrt backend {backend!r}; accepted values are "
            "{pytorch, trt}. backend selects the TRT-LLM constructor class "
            "(pytorch -> tensorrt_llm.LLM, trt -> tensorrt_llm._tensorrt_engine.LLM)."
        )

    def _build_llm_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs dict for the resolved tensorrt_llm LLM() constructor.

        All engine fields live on the generated ``engine_params`` block: typed
        scalars (tensor_parallel_size, max_batch_size, dtype, ...) and Any-typed
        sub-config dicts (quant_config, kv_cache_config, scheduler_config).
        ``backend`` is NEVER forwarded as a kwarg: it selects the constructor
        class instead (see :meth:`_resolve_llm_class`), because at 1.2.1 the
        kwarg is upstream-deprecated and ``backend='trt'`` is rejected by the
        base ``LLM``.

        When engine_path is set (an extra="allow" passthrough field, not
        curated), returns early with only {"model": engine_path}. Compile-time
        kwargs are baked into the engine and must not be re-specified.
        """
        kwargs: dict[str, Any] = {
            "model": config.task.model,
        }

        engine_params = config.active_engine_params()

        # engine_path early-return: pass engine dir as model, skip all compile-time kwargs.
        # engine_path is not a curated field (D1 drop); accessed via extra="allow" passthrough.
        raw_engine_path = (
            getattr(engine_params, "engine_path", None) if engine_params is not None else None
        )
        if engine_params is not None and raw_engine_path is not None:
            engine_path = Path(str(raw_engine_path))
            tp_size = engine_params.tensor_parallel_size or 1
            errors = _validate_engine_directory(engine_path, tp_size=tp_size)
            if errors:
                raise ConfigError(f"engine_path validation failed: {'; '.join(errors)}")
            # Pass engine dir as model - TRT-LLM auto-detects TLLM_ENGINE format.
            # Compile-time kwargs are baked into the engine; don't pass them.
            # backend is not a kwarg - it selects the constructor class.
            return {"model": str(raw_engine_path)}

        backend = getattr(engine_params, "backend", None) if engine_params is not None else None
        is_trt = backend == "trt"

        if engine_params is None:
            # No tensorrt section -> default (pytorch) backend. The TRT-engine-build
            # knobs (build cache, fast_build, quant_config) are absent from the
            # pytorch backend's TorchLlmArgs, so nothing extra is forwarded.
            return kwargs

        # Scalar fields and extras: forward non-None values verbatim. The
        # sub-config dicts and the TRT-build-only knobs are popped and handled
        # separately below.
        dumped: dict[str, Any] = engine_params.model_dump(exclude_none=True)
        quant_config = dumped.pop("quant_config", None)
        kv_cache_config = dumped.pop("kv_cache_config", None)
        scheduler_config = dumped.pop("scheduler_config", None)
        dumped.pop("backend", None)  # backend selects the class, never a kwarg
        fast_build = dumped.pop("fast_build", None)  # TRT-build-only; re-added on trt path

        # fast_build, quant_config and the on-disk build cache are TRT-engine-build
        # concepts absent from the pytorch backend's TorchLlmArgs (extra='forbid').
        # They are popped above so they never forward blindly and are re-added only
        # on the trt path below. Declaring them with a non-trt backend is already
        # rejected at config validation (engine rules + the backend Literal), so no
        # pytorch-path guard is needed here.

        kwargs.update(dumped)  # scalars valid on both backends (tp/pp/max_*/dtype)

        # TRT-build-only surface (re-added here only on the trt path).
        if is_trt:
            if fast_build is not None:
                kwargs["fast_build"] = fast_build

            # Quantisation config -> native ``quant_config`` kwarg. NOT
            # ``quantization``: TrtLlmArgs is extra='forbid' at 1.2.1 and rejects
            # the old ``quantization`` name.
            if isinstance(quant_config, dict) and quant_config:
                try:
                    from tensorrt_llm.llmapi import QuantAlgo, QuantConfig
                except ImportError as exc:
                    raise EngineError(
                        f"quant_config was declared ({quant_config}) but "
                        "tensorrt_llm.llmapi QuantConfig/QuantAlgo could not be "
                        f"imported ({exc}). Refusing to silently measure an "
                        "unquantised model while the config declares quantisation."
                    ) from exc
                qc_kwargs: dict[str, Any] = {}
                if quant_config.get("quant_algo") is not None:
                    qc_kwargs["quant_algo"] = QuantAlgo[quant_config["quant_algo"]]
                if quant_config.get("kv_cache_quant_algo") is not None:
                    qc_kwargs["kv_cache_quant_algo"] = QuantAlgo[
                        quant_config["kv_cache_quant_algo"]
                    ]
                if qc_kwargs:
                    kwargs["quant_config"] = QuantConfig(**qc_kwargs)

            # On-disk engine build cache - env-var-gated default (trt build only).
            _apply_default_build_cache(kwargs)

        # KV cache config (a valid field on both backends)
        if isinstance(kv_cache_config, dict) and kv_cache_config:
            try:
                from tensorrt_llm.llmapi import KvCacheConfig
            except ImportError as exc:
                raise EngineError(
                    f"kv_cache_config was declared ({kv_cache_config}) but "
                    "tensorrt_llm.llmapi.KvCacheConfig could not be imported "
                    f"({exc}). Refusing to silently drop a declared KV-cache "
                    "config that would change what is measured."
                ) from exc
            kwargs["kv_cache_config"] = KvCacheConfig(
                **{k: v for k, v in kv_cache_config.items() if v is not None}
            )

        # Scheduler config
        if isinstance(scheduler_config, dict) and scheduler_config:
            try:
                from tensorrt_llm.llmapi import CapacitySchedulerPolicy, SchedulerConfig
            except ImportError as exc:
                raise EngineError(
                    f"scheduler_config was declared ({scheduler_config}) but "
                    "tensorrt_llm.llmapi SchedulerConfig/CapacitySchedulerPolicy "
                    f"could not be imported ({exc}). Refusing to silently drop a "
                    "declared scheduler config that would change what is measured."
                ) from exc
            sc_kwargs: dict[str, Any] = {}
            if scheduler_config.get("capacity_scheduling_policy") is not None:
                sc_kwargs["capacity_scheduling_policy"] = CapacitySchedulerPolicy[
                    scheduler_config["capacity_scheduling_policy"]
                ]
            if sc_kwargs:
                kwargs["scheduler_config"] = SchedulerConfig(**sc_kwargs)

        return kwargs

    def _build_sampling_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build the effective TRT-LLM SamplingParams kwargs (no constructor call)."""
        sampling = config.active_sampling_params()

        kwargs: dict[str, Any] = (
            sampling.model_dump(exclude_none=True) if sampling is not None else {}
        )
        kwargs["seed"] = config.task.random_seed
        if config.task.max_output_tokens is not None:
            kwargs["max_tokens"] = config.task.max_output_tokens
        if config.measurement.latency_profiling:
            # Ask TRT-LLM to record per-request perf metrics
            # (RequestOutput.metrics_dict: ttft/e2e/tpot). Live-verified at
            # 1.2.1 on both backend legs: the SamplingParams flag alone
            # populates the dict; the mined schema carries the field on both
            # args classes and on SamplingParams. Opt-in only - default energy
            # runs never set it, so the measured configuration is unchanged
            # unless the user asked for latency profiling.
            kwargs["return_perf_metrics"] = True
        return kwargs

    def _build_sampling_params(self, config: ExperimentConfig) -> Any:
        """Build tensorrt_llm.SamplingParams from the generated sampling_params block.

        All sampling fields live on ``config.tensorrt.sampling_params``. None
        values mean "use TRT-LLM's default", so we forward only explicit values.
        User writes top_k=0 to disable (TRT convention, matches HF).
        """
        from tensorrt_llm import SamplingParams

        kwargs = self._build_sampling_kwargs(config)
        return SamplingParams(**kwargs)
