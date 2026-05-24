"""vLLM inference engine - thin EnginePlugin.

Implements the 4-method EnginePlugin protocol:
  load_model, warmup, run_inference, cleanup

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only vLLM-specific inference: model loading via vllm.LLM(), a minimal
1-prompt warmup, offline batch llm.generate(), and cleanup.

All vLLM and torch imports are lazy so this module can be imported on
hosts without vLLM or CUDA installed.
"""

from __future__ import annotations

import contextlib
import logging
import time as _time
from collections.abc import Callable
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.utils.exceptions import EngineError

logger = logging.getLogger(__name__)


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
        from llenergymeasure.engines._helpers import require_import

        _vllm = require_import("vllm")
        LLM = _vllm.LLM
        SamplingParams = _vllm.SamplingParams

        kwargs = self._build_llm_kwargs(config)
        logger.info(
            "Loading model %r with vllm.LLM (kwargs: %s)", config.task.model, list(kwargs.keys())
        )

        try:
            t0 = _time.perf_counter()
            llm = LLM(**kwargs)
            if on_substep is not None:
                on_substep("vLLM engine loaded", _time.perf_counter() - t0)
        except Exception as e:
            raise EngineError(f"vLLM model loading failed: {e}") from e

        logger.debug("vLLM model loaded successfully")

        # Build SamplingParams or BeamSearchParams depending on config.
        # beam_search sub-config is a Move 1 walker gap (vllm BeamSearchParams
        # not walked yet); for the spike, fall back to standard SamplingParams.
        # Users wanting beam search route via extra='allow' under sampling_params
        # until Move 1 lands the nested class.
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

        from llenergymeasure.engines._helpers import warmup_single_token

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
        import time

        from llenergymeasure.engines._helpers import reset_cuda_peak_memory

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
            from llenergymeasure.engines._helpers import raise_engine_error

            raise_engine_error(
                e,
                "vLLM",
                hint="reduce n, use gpu_memory_utilization=0.8, or use a smaller model.",
            )

        # Capture peak memory - torch first, NVML fallback for pre-allocation detection.
        from llenergymeasure.engines._helpers import get_cuda_peak_memory_mb

        peak_mb = get_cuda_peak_memory_mb()

        # Heuristic: if peak matches gpu_memory_utilization * total_vram within 5%,
        # it's likely pre-allocation, not actual usage. Fall back to NVML.
        if peak_mb > 0:
            try:
                import torch

                total_vram = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory / (1024 * 1024)
                vllm_cfg = config.vllm
                gpu_util = 0.9  # vLLM default
                ep = vllm_cfg.engine_params if vllm_cfg is not None else None
                if ep is not None and getattr(ep, "gpu_memory_utilization", None) is not None:
                    gpu_util = ep.gpu_memory_utilization
                expected_prealloc = total_vram * gpu_util
                if abs(peak_mb - expected_prealloc) / expected_prealloc < 0.05:
                    logger.debug(
                        "torch peak (%.1fMB) matches pre-allocation (%.1fMB), trying NVML",
                        peak_mb,
                        expected_prealloc,
                    )
                    nvml_peak = self._nvml_peak_memory_mb()
                    if nvml_peak is not None:
                        peak_mb = nvml_peak
            except Exception:
                pass  # Stick with torch value

        # Count tokens from RequestOutput objects.
        # Sum across ALL outputs per request (n>1 or beam search produces multiple).
        input_token_count = sum(len(o.prompt_token_ids) for o in outputs)
        output_token_count = sum(
            len(out.token_ids) for o in outputs if o.outputs for out in o.outputs
        )

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

        return InferenceOutput(
            elapsed_time_sec=elapsed,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            peak_memory_mb=peak_mb,
            model_memory_mb=0.0,  # Captured by harness before run_inference
            batch_times=[elapsed],
            extras=extras,
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

        Sampling params are captured via :func:`extract_observed_params` -
        PoC-C allowlist is unset (the private fields ``_all_stop_token_ids``,
        ``_bad_words_token_ids``, ``_eos_token_id`` default-exclude since they
        vary per-model without affecting measurement-equivalence). Engine
        params derive from ``llm.llm_engine.vllm_config`` when available;
        otherwise we fall back to the declared kwargs dict.
        """
        from llenergymeasure.engines._helpers import (
            assemble_observed_params,
            extract_observed_params,
        )

        sampling: dict[str, Any] = {}
        try:
            sampling = extract_observed_params(sampling_params)
        except Exception as exc:  # pragma: no cover - best-effort capture
            logger.debug("vLLM SamplingParams capture failed: %s", exc)

        engine_params: dict[str, Any] = {}
        try:
            vllm_cfg = getattr(llm.llm_engine, "vllm_config", None)
            if vllm_cfg is not None:
                engine_params = extract_observed_params(vllm_cfg)
        except Exception as exc:  # pragma: no cover - best-effort capture
            logger.debug("vLLM config capture failed: %s", exc)

        return assemble_observed_params(engine_params, sampling, "vllm")

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
        from llenergymeasure.engines._helpers import cleanup_model

        llm, _sampling_params = model
        cleanup_model(llm)
        logger.debug("vLLM model cleanup complete")

    @staticmethod
    def check_hardware(config: ExperimentConfig) -> list[str]:
        """No preflight hardware rules; vLLM resolves SM x dtype x quant inside EngineArgs."""
        return []

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _build_llm_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs dict for vllm.LLM() constructor.

        Post-Phase-2-T shape: all engine fields live flat on
        ``config.vllm.engine_params`` (no nested .engine / .attention /
        .speculative_config sub-configs - those were hand-written nested
        classes the generator hasn't walked yet). Users can pass arbitrary
        vllm.LLM kwargs (including ``attention_backend``, ``speculative_config``)
        via extra='allow' on engine_params.
        """
        from llenergymeasure.utils.security import trust_remote_code_enabled

        vllm_cfg = config.vllm
        kwargs: dict[str, Any] = {
            "model": config.task.model,
            "trust_remote_code": trust_remote_code_enabled(),
            "seed": config.task.random_seed,
        }
        ep = vllm_cfg.engine_params if vllm_cfg is not None else None
        if ep is None:
            return kwargs

        # Flat engine_params: dump all non-None fields. Pydantic
        # model_dump(exclude_none=True) handles both declared fields and
        # extras (extra='allow').
        ep_kwargs = ep.model_dump(exclude_none=True)
        # Drop fields that aren't valid vllm.LLM kwargs but might end up here
        # via mining (none today, but defensive against future overlay
        # additions that target documentation rather than the constructor).
        ep_kwargs.pop("compile_config", None)
        kwargs.update(ep_kwargs)

        return kwargs

    @staticmethod
    def _build_sampling_kwargs(config: ExperimentConfig) -> dict[str, Any]:
        """Build the effective SamplingParams kwargs dict (no constructor call).

        Post-Phase-2-T shape: sampling fields live flat on
        ``config.vllm.sampling_params`` (the generated SamplingParams
        class). Beam search nested sub-config is a Move 1 walker gap and
        not surfaced here; users wanting beam search route via extra='allow'.
        """
        vllm_cfg = config.vllm
        sp = vllm_cfg.sampling_params if vllm_cfg is not None else None
        kwargs: dict[str, Any] = (
            sp.model_dump(exclude_none=True) if sp is not None else {}
        )
        if config.task.max_output_tokens is not None:
            kwargs["max_tokens"] = config.task.max_output_tokens
        return kwargs

    @staticmethod
    def _build_sampling_params(config: ExperimentConfig, sampling_params_cls: Any) -> Any:
        """Build vllm.SamplingParams from sampling_params section.

        Post-Phase-2-T shape: all sampling fields live on
        ``config.vllm.sampling_params``. User writes top_k=-1 to disable
        (vLLM convention). No translation.
        """
        kwargs = VLLMEngine._build_sampling_kwargs(config)
        return sampling_params_cls(**kwargs)

    # -------------------------------------------------------------------------
    # Private: inference helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _nvml_peak_memory_mb() -> float | None:
        """Query NVML for current GPU memory used. Returns MB or None on failure."""
        try:
            import pynvml

            from llenergymeasure.device.gpu_info import nvml_context

            mem_mb: float | None = None
            with nvml_context():
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_mb = float(info.used) / (1024 * 1024)
            return mem_mb
        except Exception:
            return None
