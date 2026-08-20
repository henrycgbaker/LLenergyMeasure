"""vLLM inference engine - thin EnginePlugin (+ ServerCapable adapter).

Implements the offline EnginePlugin protocol (load_model, warmup,
run_inference, cleanup, ...) and, additively, the ServerCapable online-serving
extension (launch, await_ready, shutdown) - an additive sibling of the single-call
offline contract, not a change to it. The serving methods delegate the generic
launch/probe/shutdown mechanics to
:mod:`llenergymeasure.serving.lifecycle` and hold only the vLLM command
knowledge (``vllm serve``) in :mod:`llenergymeasure.engines.vllm._serving`.

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only vLLM-specific inference: model loading via vllm.LLM(), a minimal
1-prompt warmup, offline batch llm.generate(), and cleanup.

All vLLM and torch imports are lazy so this module can be imported on
hosts without vLLM or CUDA installed.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.utils.exceptions import EngineError
from llenergymeasure.utils.formatting import bytes_to_mb

if TYPE_CHECKING:
    from llenergymeasure.serving.types import (
        ProbeRequest,
        ServerHandle,
        ServerPlacement,
    )

logger = logging.getLogger(__name__)


def _capture_kv_cache_stats(llm: Any) -> dict[str, Any] | None:
    """Capture KV-cache stats from a vLLM (V0-era) LLM engine, best-effort.

    Reads:
      - ``blocks_total`` from ``cache_config.num_gpu_blocks`` (reliable in 0.7.3).
      - ``usage`` / ``hit_rate`` from the V0 Stats surface
        (``gpu_cache_usage_sys`` / ``gpu_prefix_cache_hit_rate``) via the engine's
        stat loggers, each guarded independently.
      - ``kv_cache_mb`` derived from ``usage * num_gpu_blocks * block_size_bytes``
        when block size is cheaply exposed; omitted otherwise.

    Any unreadable value is left out of the dict. Returns None only when nothing
    at all could be read.
    """
    stats: dict[str, Any] = {}

    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return None

    cache_config = getattr(engine, "cache_config", None)
    num_gpu_blocks: int | None = None
    block_size: int | None = None
    if cache_config is not None:
        num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
        if isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0:
            stats["blocks_total"] = num_gpu_blocks
        bs = getattr(cache_config, "block_size", None)
        if isinstance(bs, int) and bs > 0:
            block_size = bs

    # V0 Stats surface via stat loggers: gpu_cache_usage_sys, gpu_prefix_cache_hit_rate.
    usage: float | None = None
    try:
        stat_loggers = getattr(engine, "stat_loggers", None)
        if stat_loggers:
            for logger_obj in stat_loggers.values():
                last = getattr(logger_obj, "last_local_log", None)
                if last is None:
                    continue
                u = getattr(last, "gpu_cache_usage_sys", None)
                if u is not None and usage is None:
                    usage = float(u)
                    stats["usage"] = usage
                hr = getattr(last, "gpu_prefix_cache_hit_rate", None)
                if hr is not None and "hit_rate" not in stats:
                    stats["hit_rate"] = float(hr)
    except Exception as exc:  # pragma: no cover - best-effort capture
        logger.debug("vLLM KV-cache stats capture failed: %s", exc)

    # Derive blocks_used and kv_cache_mb when both usage and block geometry known.
    if usage is not None and num_gpu_blocks is not None and num_gpu_blocks > 0:
        stats["blocks_used"] = round(usage * num_gpu_blocks)
        if block_size is not None:
            # block size in bytes is not directly exposed; only derive kv_cache_mb
            # when cache_config carries a precomputed per-block byte size.
            block_bytes = getattr(cache_config, "block_size_bytes", None)
            if isinstance(block_bytes, int) and block_bytes > 0:
                kv_bytes = usage * num_gpu_blocks * block_bytes
                stats["kv_cache_mb"] = bytes_to_mb(kv_bytes)

    return stats or None


def _peak_matches_vllm_prealloc(
    peak_mb: float, total_vram_mb: float, gpu_memory_utilization: float
) -> bool:
    """Return True when a torch peak reading is really vLLM's up-front reservation.

    vLLM reserves ``gpu_memory_utilization * total_vram`` when the engine is
    constructed. A torch peak that lands within 5% of that reservation is the
    pre-allocation, not the inference-window working set, so the caller should
    prefer the NVML device-used proxy over this value.

    Returns False for a non-positive expected reservation (no VRAM reading or a
    zero utilisation), which keeps the caller on the torch value - matching the
    prior inline behaviour where a zero divisor was swallowed and the torch value
    retained.
    """
    expected_prealloc_mb = total_vram_mb * gpu_memory_utilization
    if expected_prealloc_mb <= 0:
        return False
    return abs(peak_mb - expected_prealloc_mb) / expected_prealloc_mb < 0.05


def _extract_request_stats(outputs: Any) -> tuple[list[float], list[float], list[float]]:
    """Per-request (e2e_ms, ttft_ms, decode_itl_ms) from vLLM V1 ``RequestStateStats``.

    vLLM 0.19.x records per-request timing in ``RequestOutput.metrics`` (a
    ``vllm.v1.metrics.stats.RequestStateStats``) ONLY when the engine was built
    with ``disable_log_stats=False`` - which :meth:`_build_llm_kwargs` sets
    exactly when ``latency_profiling`` is enabled. The offline ``LLM``
    entrypoint forces ``disable_log_stats=True`` otherwise
    (``vllm/entrypoints/llm.py``), so ``metrics`` is ``None`` on default energy
    runs and this extraction contributes nothing. (The old vLLM V0
    ``RequestOutput.metrics`` namespace with ``arrival_time`` /
    ``finished_time`` / ``first_token_time`` no longer exists at V1 - hence the
    field switch below. Live-verified at 0.19.1.)

    Fields (live-verified at 0.19.1):

    - ``first_token_latency`` is the engine-recorded wall-clock TTFT in seconds
      (``time.time()`` at the first-token iteration minus frontend arrival).
    - ``first_token_ts`` / ``last_token_ts`` are engine-core MONOTONIC
      timestamps - a DIFFERENT clock from ``arrival_time`` (a wall-clock epoch),
      so the two clocks are NEVER cross-subtracted. Their delta is the decode
      interval.

    Derivations:

    - TTFT_ms = ``first_token_latency * 1000``
    - decode_s = ``last_token_ts - first_token_ts`` (monotonic delta)
    - E2E_ms = ``(first_token_latency + decode_s) * 1000`` (arrival -> last token)
    - decode ITL_ms = ``decode_s / (decode_len - 1) * 1000``, a
      PROPORTIONAL_ESTIMATE (uniform decode spacing). ``decode_len`` is the
      LONGEST single output's token count (n>1 parallel streams share one decode
      window; summing would N-fold understate the ITL), read from ``o.outputs``.

    Best-effort: any request whose ``metrics`` is ``None`` or lacks usable
    timestamps contributes nothing rather than a partial/garbage sample. The
    ``SimpleNamespace`` shape of ``o.metrics`` / ``o.outputs[*].token_ids`` makes
    this testable without a live engine.
    """
    latencies_ms: list[float] = []
    ttft_ms: list[float] = []
    itl_ms: list[float] = []
    for o in outputs:
        metrics = getattr(o, "metrics", None)
        if metrics is None:
            continue
        ttft_s = getattr(metrics, "first_token_latency", None)
        first_ts = getattr(metrics, "first_token_ts", None)
        last_ts = getattr(metrics, "last_token_ts", None)

        ttft_val: float | None = None
        if isinstance(ttft_s, (int, float)) and ttft_s > 0:
            ttft_val = float(ttft_s)
            ttft_ms.append(ttft_val * 1000.0)

        decode_s: float | None = None
        if (
            isinstance(first_ts, (int, float))
            and isinstance(last_ts, (int, float))
            and last_ts >= first_ts
        ):
            decode_s = float(last_ts - first_ts)

        if ttft_val is not None and decode_s is not None:
            latencies_ms.append((ttft_val + decode_s) * 1000.0)

        request_outputs = getattr(o, "outputs", None)
        if decode_s is not None and request_outputs:
            decode_len = max(len(getattr(out, "token_ids", ()) or ()) for out in request_outputs)
            if decode_len > 1:
                itl_ms.append(decode_s * 1000.0 / (decode_len - 1))
    return latencies_ms, ttft_ms, itl_ms


class VLLMEngine:
    """vLLM inference engine - offline batch mode, thin plugin.

    Implements EnginePlugin:
    - load_model: Load model via vllm.LLM(), build SamplingParams
    - warmup: Minimal 1-prompt warmup with 1-token output
    - run_inference: Single llm.generate() call with ALL prompts, returns InferenceOutput
    - cleanup: Delete LLM instance, gc.collect(), clear CUDA cache
    """

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "vllm"

    @property
    def version(self) -> str:
        """vLLM version string."""
        try:
            import vllm

            return str(vllm.__version__)
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
        """Load model via vllm.LLM() and build SamplingParams.

        All vLLM imports are lazy so this module can be imported without vLLM.

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for substep visibility.

        Returns:
            Tuple of (llm, sampling_params).

        Raises:
            EngineError: If vLLM is not installed or model loading fails.
        """
        import os

        from llenergymeasure.engines._errors import require_import

        # The harness touches torch.cuda (hardware preflight, device probes)
        # before this plugin constructs the engine, so CUDA is already
        # initialised in this process. vLLM's default fork start method for
        # its EngineCore worker then dies with "Cannot re-initialize CUDA in
        # forked subprocess" (observed live at 0.19.1 in the containerized
        # study path, 2026-07-14). Spawn is vLLM's own prescription for
        # embedding contexts; setdefault keeps any explicit user override.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        _vllm = require_import("vllm")
        LLM = _vllm.LLM
        SamplingParams = _vllm.SamplingParams

        kwargs = self._build_llm_kwargs(config)
        logger.info(
            "Loading model %r with vllm.LLM (kwargs: %s)", config.task.model, list(kwargs.keys())
        )

        try:
            t0 = time.perf_counter()
            llm = LLM(**kwargs)
            if on_substep is not None:
                on_substep("vLLM engine loaded", time.perf_counter() - t0)
        except Exception as e:
            raise EngineError(f"vLLM model loading failed: {e}") from e

        logger.debug("vLLM model loaded successfully")

        # Build SamplingParams or BeamSearchParams depending on config
        beam_search = config.engine_sub_dict("beam_search")
        if beam_search is not None:
            sampling_params = self._build_beam_search_params(config, beam_search)
        else:
            sampling_params = self._build_sampling_params(config, SamplingParams)
        if on_substep is not None:
            on_substep("sampling params built", 0.0)
        return llm, sampling_params

    # -------------------------------------------------------------------------
    # EnginePlugin: warmup
    # -------------------------------------------------------------------------

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt via single-token kernel warmup. Returns 0.0.

        Returns 0.0 to signal the harness to skip CV-based convergence.
        vLLM uses a single-token kernel warmup rather than CV convergence.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompt: Single warmup prompt text.

        Returns:
            0.0 (signals harness to skip convergence loop).
        """
        from vllm import SamplingParams

        from llenergymeasure.engines._cuda import warmup_single_token

        llm, _sampling_params = model
        warmup_single_token(llm, [prompt], SamplingParams, temperature=0.0, max_tokens=1)
        return 0.0  # Signals harness to skip CV loop

    # -------------------------------------------------------------------------
    # EnginePlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run offline batch inference over all prompts.

        Single llm.generate() call with ALL prompts - no streaming, no
        one-at-a-time loops.

        Args:
            config: Experiment configuration.
            model: Tuple of (llm, sampling_params) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.

        Raises:
            EngineError: On OOM or other inference failures.
        """
        from llenergymeasure.engines._cuda import reset_cuda_peak_memory

        llm, sampling_params = model

        # Reset peak stats before the measurement loop so max_memory_allocated() below
        # captures inference-window peak (KV cache occupancy + activations), not pre-allocation.
        reset_cuda_peak_memory()

        logger.info(
            "Starting vLLM offline batch inference: %d prompts, max_tokens=%s",
            len(prompts),
            config.task.max_output_tokens or "unlimited",
        )

        try:
            t0 = time.perf_counter()
            # BeamSearchParams uses llm.beam_search(), SamplingParams uses llm.generate().
            # Guard import - BeamSearchParams was added in vLLM >=0.8; older versions
            # (e.g. 0.7.3 in the v0.9.0 container image) don't export it.
            try:
                from vllm import BeamSearchParams as _BSP
            except ImportError:
                _BSP = None  # type: ignore[assignment,misc]

            if _BSP is not None and isinstance(sampling_params, _BSP):
                outputs = llm.beam_search(prompts, sampling_params)
            else:
                outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0

        except Exception as e:
            from llenergymeasure.engines._errors import raise_engine_error

            raise_engine_error(
                e,
                "vLLM",
                hint="reduce n, use gpu_memory_utilization=0.8, or use a smaller model.",
            )

        # Capture peak memory - torch first, NVML fallback for out-of-process
        # (V1) runs and pre-allocation detection.
        from llenergymeasure.engines._cuda import (
            get_cuda_peak_memory_mb,
            get_nvml_device_memory_mb,
        )

        peak_mb = get_cuda_peak_memory_mb()

        if peak_mb == 0.0:
            # vLLM V1 runs its model in the EngineCore child process, so torch's
            # per-process allocator here saw nothing (a silent 0.0). Fall back to
            # NVML device-used memory (whole-device; see get_nvml_device_memory_mb
            # for the tenancy caveat). None stays 0.0 and is nulled downstream.
            nvml_peak = get_nvml_device_memory_mb()
            if nvml_peak is not None:
                peak_mb = nvml_peak
        elif peak_mb > 0:
            # torch saw an in-process allocation (V0-era), but it may just be
            # vLLM's pre-allocation. If the peak matches gpu_memory_utilization *
            # total_vram within 5% it is pre-allocation, not actual usage - NVML
            # device-used is the closer proxy (see _peak_matches_vllm_prealloc).
            try:
                import torch

                total_vram = bytes_to_mb(
                    torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                )
                engine_params = config.active_engine_params()
                gpu_util = 0.9  # vLLM default
                if engine_params is not None and engine_params.gpu_memory_utilization is not None:
                    gpu_util = engine_params.gpu_memory_utilization
                if _peak_matches_vllm_prealloc(peak_mb, total_vram, gpu_util):
                    logger.debug(
                        "torch peak (%.1fMB) matches pre-allocation (%.1fMB), trying NVML",
                        peak_mb,
                        total_vram * gpu_util,
                    )
                    nvml_peak = get_nvml_device_memory_mb()
                    if nvml_peak is not None:
                        peak_mb = nvml_peak
            except Exception:
                pass  # Stick with torch value

        # Count tokens from RequestOutput objects.
        # Sum across ALL outputs per request (n>1 or beam search produces multiple).
        from llenergymeasure.engines._observed import count_request_tokens

        input_token_count, output_token_count = count_request_tokens(outputs)

        logger.debug(
            "vLLM inference complete: %d total tokens (in=%d, out=%d) in %.2fs",
            input_token_count + output_token_count,
            input_token_count,
            output_token_count,
            elapsed,
        )

        # Attempt to expose HuggingFace model for FLOPs estimation.
        # vllm.LLM has llm_engine.model_executor.driver_worker.model_runner.model
        # This is an internal API path - stash in extras, harness will attempt FLOPs.
        hf_model = None
        with contextlib.suppress(Exception):
            hf_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        extras: dict[str, Any] = {}
        if hf_model is not None:
            extras["hf_model"] = hf_model

        # Best-effort extended-metrics capture (continuous batching: no static batches).
        from llenergymeasure.domain.metrics import LatencyMeasurementMode

        kv_cache_stats = _capture_kv_cache_stats(llm)

        # Per-request latency from RequestOutput.metrics (RequestStateStats).
        # Populated only when the engine was built with disable_log_stats=False -
        # which _build_llm_kwargs sets exactly when latency_profiling is enabled -
        # so default energy runs pay no stat-logging overhead and extract nothing.
        # The engine-recorded TTFT is real per-request, but the derived decode
        # ITL is a decode-average (PROPORTIONAL_ESTIMATE), so the mode describes
        # that weakest signal.
        per_request_latencies_ms, ttft_ms, itl_ms = _extract_request_stats(outputs)
        latency_measurement_mode: str | None = None
        if ttft_ms or per_request_latencies_ms or itl_ms:
            latency_measurement_mode = LatencyMeasurementMode.PROPORTIONAL_ESTIMATE.value
        elif config.measurement.latency_profiling:
            # Profiling was requested but the engine returned no per-request
            # metrics (e.g. an upstream regression, or a user override forcing
            # disable_log_stats=True): keep the loud degradation flag.
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
            num_batches=None,  # Continuous batching - no static batch count
            padding_tokens=None,  # Continuous batching - no padding
            kv_cache_stats=kv_cache_stats,
        )

    # -------------------------------------------------------------------------
    # Private: observed-params capture (observed_config_hash)
    # -------------------------------------------------------------------------

    @staticmethod
    def _capture_observed_params(
        config: ExperimentConfig,
        llm: Any,
        sampling_params: Any,
    ) -> dict[str, Any]:
        """Extract post-construction state from the vLLM native types.

        Sampling params are captured via :func:`extract_observed_params` with
        no private-field allowlist (the private fields ``_all_stop_token_ids``,
        ``_bad_words_token_ids``, ``_eos_token_id`` default-exclude since they
        vary per-model without affecting measurement-equivalence). Engine
        params derive from ``llm.llm_engine.vllm_config`` when available;
        otherwise we fall back to the declared kwargs dict.
        """
        from llenergymeasure.engines._observed import capture_two_part_observed

        return capture_two_part_observed(
            "vllm",
            logger=logger,
            sampling_obj=sampling_params,
            engine_obj=getattr(getattr(llm, "llm_engine", None), "vllm_config", None),
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

        The ``sampling_params`` object is already in the model tuple from
        ``load_model()``; the ``llm`` object provides the live vllm_config.
        """
        llm, sampling_params = model
        return self._capture_observed_params(config, llm, sampling_params)

    # -------------------------------------------------------------------------
    # EnginePlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release vLLM model from memory and clear CUDA cache.

        Args:
            model: Tuple of (llm, sampling_params) from load_model().
        """
        from llenergymeasure.engines._cuda import cleanup_model

        llm, _sampling_params = model
        cleanup_model(llm)
        logger.debug("vLLM model cleanup complete")

    @staticmethod
    def check_hardware(config: ExperimentConfig) -> list[str]:
        """No preflight hardware rules; vLLM resolves SM x dtype x quant inside EngineArgs."""
        return []

    # -------------------------------------------------------------------------
    # ServerCapable: online-serving lifecycle (additive sibling of the
    # offline run_inference contract; readiness gated by a real probe)
    # -------------------------------------------------------------------------

    def launch(self, config: ExperimentConfig, placement: ServerPlacement) -> ServerHandle:
        """Launch a vLLM server (``vllm serve``) and return a handle to it.

        Container placement runs the pinned upstream ``vllm/vllm-openai`` image
        (resolved via the image registry when ``placement.image`` is None) with
        ``--network host`` and the serve arguments; process placement runs
        ``vllm serve`` as a host subprocess. A free port is allocated here and
        passed to ``vllm serve``; the issuer receives ``handle.base_url``.
        """
        from llenergymeasure.engines.vllm import _serving
        from llenergymeasure.serving import lifecycle as sl

        port = sl.allocate_free_port()
        base_url = sl.base_url_for(port)
        model = config.task.model

        if placement.mode == sl.CONTAINER_MODE:
            from llenergymeasure.infra.image_registry import get_default_image

            image = placement.image or get_default_image(config.engine)
            container_name = sl.server_container_name("vllm")
            argv = sl.build_server_container_argv(
                image=image,
                container_name=container_name,
                gpu_indices=placement.gpu_indices,
                serve_args=_serving.serve_args(model, port),
                labels=placement.labels,
            )
            return sl.launch_container_server(
                argv, base_url=base_url, engine="vllm", container_name=container_name
            )

        argv = _serving.process_argv(model, port)
        log_path = sl.default_server_log_path("vllm", port)
        return sl.launch_process_server(argv, base_url=base_url, engine="vllm", log_path=log_path)

    def await_ready(
        self,
        handle: ServerHandle,
        probe_request: ProbeRequest,
        *,
        timeout: float,
    ) -> None:
        """Wait until vLLM is ready: liveness poll THEN a real probe."""
        from llenergymeasure.serving import lifecycle as sl

        sl.await_ready(handle, probe_request, timeout=timeout)

    def shutdown(self, handle: ServerHandle) -> None:
        """Stop the vLLM server (graceful, escalating to a hard kill); idempotent."""
        from llenergymeasure.serving import lifecycle as sl

        sl.shutdown(handle)

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _build_llm_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs dict for vllm.LLM() constructor.

        All engine fields live on the generated ``engine_params`` block. Typed
        scalars (dtype, gpu_memory_utilization, max_model_len, ...) and Any-typed
        discovery-debt fields (speculative_config, attention, beam_search,
        offload_*, distributed_executor_backend, compilation_config) alike are
        dumped through ``model_dump(exclude_none=True)`` - the engine owns the
        kwargs surface (extra="allow"), so non-None values forward verbatim.
        """
        from llenergymeasure.utils.security import trust_remote_code_enabled

        kwargs: dict[str, Any] = {
            "model": config.task.model,
            "trust_remote_code": trust_remote_code_enabled(),
            "seed": config.task.random_seed,
        }

        if config.measurement.latency_profiling:
            # The offline LLM entrypoint forces disable_log_stats=True
            # (vllm/entrypoints/llm.py), which leaves RequestOutput.metrics None.
            # Flip it so per-request RequestStateStats populate for
            # _extract_request_stats. Opt-in only: log_stats enables per-iteration
            # stat aggregation + default loggers (measurable overhead), so default
            # energy runs never set it and the measured configuration is unchanged
            # unless the user asked for latency profiling.
            kwargs["disable_log_stats"] = False

        engine_params = config.active_engine_params()
        if engine_params is None:
            return kwargs

        dumped: dict[str, Any] = engine_params.model_dump(exclude_none=True)

        # The attention block maps backend -> attention_backend and flattens its
        # remaining keys; beam_search is dispatched separately (sampling path).
        attention = dumped.pop("attention", None)
        dumped.pop("beam_search", None)
        offload_params = dumped.pop("offload_params", None)

        kwargs.update(dumped)

        if isinstance(attention, dict):
            backend = attention.pop("backend", None)
            if backend is not None:
                kwargs["attention_backend"] = backend
            kwargs.update(attention)

        if offload_params is not None:
            kwargs["offload_params"] = set(offload_params)

        return kwargs

    @staticmethod
    def _build_sampling_kwargs(config: ExperimentConfig) -> dict[str, Any]:
        """Build the effective SamplingParams kwargs dict (no constructor call).

        Returns ``{}`` when beam search is active (sampling path preempted); the
        caller dispatches to :meth:`_build_beam_search_params` in that case.
        """
        if config.engine_sub_dict("beam_search") is not None:
            return {}

        sampling = config.active_sampling_params()
        kwargs: dict[str, Any] = (
            sampling.model_dump(exclude_none=True) if sampling is not None else {}
        )
        if config.task.max_output_tokens is not None:
            kwargs["max_tokens"] = config.task.max_output_tokens
        return kwargs

    @staticmethod
    def _build_sampling_params(config: ExperimentConfig, sampling_params_cls: Any) -> Any:
        """Build vllm.SamplingParams from the generated sampling_params block.

        All sampling fields live on ``config.vllm.sampling_params``. None values
        mean "use vLLM's default", so we forward only explicit values. User
        writes top_k=-1 directly to disable (vLLM convention). No translation.
        """
        beam_search = config.engine_sub_dict("beam_search")
        if beam_search is not None:
            return VLLMEngine._build_beam_search_params(config, beam_search)
        kwargs = VLLMEngine._build_sampling_kwargs(config)
        return sampling_params_cls(**kwargs)

    @staticmethod
    def _build_beam_search_params(config: ExperimentConfig, beam_cfg: dict[str, Any]) -> Any:
        """Build vllm.BeamSearchParams from the beam_search engine_params dict.

        ``beam_search`` is an Any-typed engine_params field, so it arrives as a
        plain dict; its keys (beam_width, length_penalty, early_stopping, plus
        any extras) forward verbatim.
        """
        try:
            from vllm import BeamSearchParams
        except ImportError:
            raise EngineError(
                "beam_search config requires vllm.BeamSearchParams which is not "
                "available in the installed vLLM version (added in vLLM >=0.8). "
                "Either upgrade vLLM or remove the beam_search section from "
                "vllm config."
            ) from None

        kwargs = {k: v for k, v in beam_cfg.items() if v is not None}
        if config.task.max_output_tokens is not None:
            kwargs["max_tokens"] = config.task.max_output_tokens
        return BeamSearchParams(**kwargs)
