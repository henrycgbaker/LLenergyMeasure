"""HuggingFace Transformers inference engine - thin EnginePlugin.

Implements the 4-method EnginePlugin protocol:
  load_model, warmup, run_inference, cleanup

All measurement lifecycle is delegated to MeasurementHarness. This module
owns only Transformers-specific inference: model loading, warmup via
warmup_until_converged(), model.generate() inference loop, and cleanup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.metrics import LatencyMeasurementMode
from llenergymeasure.engines.protocol import InferenceOutput

logger = logging.getLogger(__name__)


class _TimingStreamer:
    """A minimal ``transformers.BaseStreamer`` that records token-arrival times.

    ``model.generate(streamer=...)`` calls ``put()`` once per decode step (and
    once for the prompt tensor, which we drop). Each call records
    ``time.perf_counter() * 1000.0`` (ms). ``end()`` is a no-op. Used only under
    latency profiling, which forces ``batch_size=1`` so each ``put()`` maps to a
    single request's token.
    """

    def __init__(self) -> None:
        self.token_times_ms: list[float] = []
        self._prompt_seen = False

    def put(self, value: Any) -> None:
        import time

        # The first put() carries the prompt input_ids (a 2-D tensor); drop it so
        # only generated-token arrivals are timed.
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        self.token_times_ms.append(time.perf_counter() * 1000.0)

    def end(self) -> None:
        """No-op: nothing to flush."""


class TransformersEngine:
    """HuggingFace Transformers inference engine - thin plugin.

    Implements EnginePlugin:
    - load_model: Load HuggingFace model + tokenizer, apply torch.compile
    - warmup: CV-based warmup via warmup_until_converged()
    - run_inference: Batched model.generate() loop, returns InferenceOutput
    - cleanup: Delete model, clear CUDA cache
    """

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "transformers"

    @property
    def version(self) -> str:
        """Transformers library version string."""
        try:
            import torch

            return torch.__version__  # type: ignore[no-any-return]
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
        """Load model and tokenizer via from_pretrained().

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for substep visibility.

        Returns:
            Tuple of (model, tokenizer).
        """
        import time as _time

        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = self._model_load_kwargs(config)
        logger.info("Loading model %r with kwargs: %s", config.task.model, list(kwargs.keys()))

        t0 = _time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            config.task.model, trust_remote_code=kwargs.get("trust_remote_code", False)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if on_substep is not None:
            on_substep("tokenizer loaded", _time.perf_counter() - t0)

        t0 = _time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(config.task.model, **kwargs)
        model.eval()
        if on_substep is not None:
            on_substep("model weights loaded", _time.perf_counter() - t0)

        # allow_tf32 + torch.compile are llem-orchestration knobs (HarnessConfig),
        # not engine config: llem makes the torch calls around the engine.
        harness = self._harness(config)

        # Apply allow_tf32 (Ampere+ TF32 toggle)
        if harness is not None and harness.allow_tf32 is not None:
            import torch as _torch

            _torch.backends.cuda.matmul.allow_tf32 = harness.allow_tf32

        # Apply torch.compile post-load (must be AFTER from_pretrained + eval)
        if harness is not None and harness.torch_compile:
            import torch as _torch

            mode = harness.torch_compile_mode or "default"
            backend = harness.torch_compile_backend or "inductor"
            try:
                t0 = _time.perf_counter()
                model = _torch.compile(model, mode=mode, backend=backend)  # type: ignore[assignment]
                logger.debug("torch.compile applied (mode=%s, backend=%s)", mode, backend)
                if on_substep is not None:
                    on_substep(f"torch.compile ({mode})", _time.perf_counter() - t0)
            except Exception as e:
                logger.warning("torch.compile failed (non-fatal, continuing without): %s", e)

        logger.debug("Model loaded successfully")
        return model, tokenizer

    # -------------------------------------------------------------------------
    # EnginePlugin: warmup
    # -------------------------------------------------------------------------

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt and return latency in ms.

        Args:
            config: Experiment configuration.
            model: Tuple of (model, tokenizer) from load_model().
            prompt: Single warmup prompt text.

        Returns:
            Latency in milliseconds.
        """
        import time

        import torch

        hf_model, tokenizer = model
        start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            hf_model.generate(**inputs, max_new_tokens=min(config.task.max_output_tokens or 32, 32))
        return (time.perf_counter() - start) * 1000.0

    # -------------------------------------------------------------------------
    # EnginePlugin: run_inference
    # -------------------------------------------------------------------------

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run the batched measurement loop over all prompts.

        Args:
            config: Experiment configuration.
            model: Tuple of (model, tokenizer) from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.

        Raises:
            EngineError: On CUDA OOM or other inference failures.
        """
        hf_model, tokenizer = model

        harness = self._harness(config)
        batch_size = 1
        if harness is not None and harness.batch_size is not None:
            batch_size = harness.batch_size
        else:
            logger.debug("Transformers batch_size not set, defaulting to 1")

        # Latency profiling: per-token timing capture via a custom streamer. The
        # streamer requires batch_size=1 (one put() per decode step maps to a
        # single request), and beam search is incompatible with streaming, so we
        # fall back to the non-profiled path when num_beams > 1.
        profiling = config.measurement.latency_profiling
        profiling_forced_batch_size = False
        _ep = self._engine_params(config)
        num_beams = _ep.num_beams if _ep is not None and _ep.num_beams is not None else 1
        if profiling and num_beams > 1:
            logger.warning(
                "latency_profiling requested but num_beams=%d > 1; beam search is "
                "incompatible with a generation streamer. Falling back to the "
                "non-profiled inference path.",
                num_beams,
            )
            profiling = False
        if profiling and batch_size != 1:
            logger.warning(
                "latency_profiling requested with batch_size=%d; forcing "
                "batch_size=1 so per-token timestamps map to a single request. "
                "This perturbs throughput relative to non-profiled runs.",
                batch_size,
            )
            batch_size = 1
            profiling_forced_batch_size = True

        # Reset peak stats BEFORE the measurement loop so max_memory_allocated()
        # captures inference-window-only peak (KV cache + activations + batch buffers),
        # NOT model weights already allocated by load_model().
        from llenergymeasure.engines._cuda import reset_cuda_peak_memory

        reset_cuda_peak_memory()

        # Seed PyTorch RNG for reproducible sampling (mirrors vLLM's seed= kwarg).
        # manual_seed seeds both CPU and all CUDA devices since PyTorch 1.12+.
        import torch as _torch

        _torch.manual_seed(config.task.random_seed)

        generate_kwargs = self._build_generate_kwargs(config)
        total_input_tokens = 0
        total_output_tokens = 0
        total_time_sec = 0.0
        total_padding_tokens = 0
        batch_times: list[float] = []
        # PER_REQUEST_BATCH approximation (see LatencyMeasurementMode.PER_REQUEST_BATCH):
        # non-streaming generate() gives only per-batch wall time, so each prompt in a
        # batch is attributed batch_time / len(batch). This is an approximation, NOT a
        # true per-request timestamp.
        per_request_latencies_ms: list[float] = []
        # Under profiling, per-request cumulative token-arrival timestamps (ms),
        # starting with t0 (pre-generate) so the first interval is TTFT.
        token_timestamps_per_request: list[list[float]] = []
        ttft_ms_per_request: list[float] = []

        logger.info(
            "Starting measurement: %d prompts, batch_size=%d, max_new_tokens=%s%s",
            len(prompts),
            batch_size,
            config.task.max_output_tokens or "unlimited",
            " (latency profiling)" if profiling else "",
        )

        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start : batch_start + batch_size]
            try:
                if profiling:
                    import time as _time

                    streamer = _TimingStreamer()
                    t0_ms = _time.perf_counter() * 1000.0
                    batch_input, batch_output, batch_time, batch_padding = self._run_batch(
                        hf_model, tokenizer, config, batch, generate_kwargs, streamer=streamer
                    )
                    put_times = streamer.token_times_ms
                    token_timestamps_per_request.append([t0_ms, *put_times])
                    if put_times:
                        ttft_ms_per_request.append(put_times[0] - t0_ms)
                else:
                    batch_input, batch_output, batch_time, batch_padding = self._run_batch(
                        hf_model, tokenizer, config, batch, generate_kwargs
                    )
                total_input_tokens += batch_input
                total_output_tokens += batch_output
                total_time_sec += batch_time
                total_padding_tokens += batch_padding
                batch_times.append(batch_time)
                per_request_ms = (batch_time * 1000.0) / len(batch)
                per_request_latencies_ms.extend([per_request_ms] * len(batch))

                logger.debug(
                    "Batch %d-%d: in=%d out=%d tokens in %.2fs",
                    batch_start,
                    batch_start + len(batch) - 1,
                    batch_input,
                    batch_output,
                    batch_time,
                )
            except Exception as e:
                from llenergymeasure.engines._errors import raise_engine_error

                raise_engine_error(
                    e,
                    "Transformers",
                    hint="reduce batch_size, use dtype=float16, or use a smaller model.",
                )

        # Track peak GPU memory (inference window only - reset above)
        from llenergymeasure.engines._cuda import get_cuda_peak_memory_mb

        peak_memory_mb = get_cuda_peak_memory_mb()

        # model_memory_mb is queried by the harness after load_model(); we report 0.0 here
        # as the harness captures it before warmup (before this method is called).
        logger.debug(
            "Measurement complete: %d total tokens in %.2fs",
            total_input_tokens + total_output_tokens,
            total_time_sec,
        )

        extras: dict[str, Any] = {
            "hf_model": hf_model,  # For FLOPs estimation in harness
            # generate_kwargs stashed so capture_observed_params can read
            # GenerationConfig state post-window without recomputing.
            "generate_kwargs": generate_kwargs,
        }
        if profiling_forced_batch_size:
            extras["profiling_forced_batch_size"] = True

        # Latency profiling capture: trimmed ITL from per-token timestamps + TTFT.
        # The streamer gives true per-token arrivals (batch_size=1 forced), so this
        # is TRUE_STREAMING provenance.
        ttft_ms: list[float] = []
        itl_ms: list[float] = []
        latency_measurement_mode: str | None = None
        if profiling:
            from llenergymeasure.domain.metrics import collect_itl_measurements

            _itl_full, itl_trimmed, _excluded = collect_itl_measurements(
                token_timestamps_per_request
            )
            itl_ms = itl_trimmed
            ttft_ms = ttft_ms_per_request
            latency_measurement_mode = LatencyMeasurementMode.TRUE_STREAMING.value

        return InferenceOutput(
            elapsed_time_sec=total_time_sec,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            peak_memory_mb=peak_memory_mb,
            model_memory_mb=0.0,  # Captured by harness before run_inference
            batch_times=batch_times,
            extras=extras,
            # PER_REQUEST_BATCH approximation - each request gets batch_time/len(batch).
            # Under profiling batch_size=1, so this is the true per-request wall time.
            per_request_latencies_ms=per_request_latencies_ms,
            ttft_ms=ttft_ms,  # Populated only under latency profiling
            itl_ms=itl_ms,  # Populated only under latency profiling
            latency_measurement_mode=latency_measurement_mode,
            num_batches=len(batch_times),
            padding_tokens=total_padding_tokens,
            kv_cache_stats=None,  # Transformers has no paged KV cache
        )

    # -------------------------------------------------------------------------
    # Private: observed-params capture (observed_config_hash)
    # -------------------------------------------------------------------------

    @staticmethod
    def _capture_observed_params(
        config: ExperimentConfig,
        hf_model: Any,
        generate_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract post-construction state from the native types the engine used.

        Transformers splits its native state across ``GenerationConfig`` (the
        sampling shape) and ``BitsAndBytesConfig`` (the engine-side quantisation
        shape, when active). Both are Pydantic-style dumpable objects;
        :func:`extract_observed_params` strips private fields (``_commit_hash``,
        ``_from_model_config``) that would poison observed_config_hash if included.

        Returns a dict with ``engine`` / ``sampling`` / ``library_version``
        entries ready for the observed-config hashing pipeline.
        """
        from llenergymeasure.engines._observed import (
            assemble_observed_params,
            extract_observed_params,
        )

        sampling: dict[str, Any] = {}
        try:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig(**generate_kwargs)
            sampling = extract_observed_params(gen_cfg)
        except Exception as exc:  # pragma: no cover - best-effort capture
            logger.debug("transformers GenerationConfig capture failed: %s", exc)

        engine_params: dict[str, Any] = {}
        ep = config.transformers.engine_params if config.transformers is not None else None
        if ep is not None and (ep.load_in_4bit or ep.load_in_8bit):
            try:
                bnb = getattr(hf_model, "quantization_config", None)
                if bnb is not None:
                    engine_params["quantization_config"] = extract_observed_params(bnb)
            except Exception as exc:  # pragma: no cover - best-effort capture
                logger.debug("transformers BitsAndBytesConfig capture failed: %s", exc)

        return assemble_observed_params(engine_params, sampling, "transformers")

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

        Reads ``generate_kwargs`` from ``output.extras`` (stashed by
        ``run_inference``) and the native model object for BnB config;
        delegates to :func:`_capture_observed_params`.
        """
        hf_model, _tokenizer = model
        generate_kwargs: dict[str, Any] = output.extras.get("generate_kwargs") or {}
        return self._capture_observed_params(config, hf_model, generate_kwargs)

    # -------------------------------------------------------------------------
    # EnginePlugin: cleanup
    # -------------------------------------------------------------------------

    def cleanup(self, model: Any) -> None:
        """Release model from memory and clear CUDA cache.

        Args:
            model: Tuple of (model, tokenizer) from load_model().
        """
        from llenergymeasure.engines._cuda import cleanup_model

        hf_model, _tokenizer = model
        cleanup_model(hf_model, use_gc=False)
        logger.debug("Model cleanup complete")

    @staticmethod
    def check_hardware(config: ExperimentConfig) -> list[str]:
        """No preflight hardware rules; BitsAndBytes/FlashAttention self-check at load time."""
        return []

    # -------------------------------------------------------------------------
    # Private: nested-config accessors (engine_params / sampling_params / harness)
    # -------------------------------------------------------------------------

    @staticmethod
    def _engine_params(config: ExperimentConfig) -> Any:
        """Return the transformers engine_params block, or None."""
        return config.transformers.engine_params if config.transformers is not None else None

    @staticmethod
    def _sampling_params(config: ExperimentConfig) -> Any:
        """Return the transformers sampling_params block, or None."""
        return config.transformers.sampling_params if config.transformers is not None else None

    @staticmethod
    def _harness(config: ExperimentConfig) -> Any:
        """Return the transformers HarnessConfig block (llem-orchestration knobs), or None."""
        return config.harness.transformers if config.harness is not None else None

    # -------------------------------------------------------------------------
    # Private: model loading helpers
    # -------------------------------------------------------------------------

    def _model_load_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build the full kwargs dict for AutoModelForCausalLM.from_pretrained().

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs ready for from_pretrained().
        """
        # All from_pretrained-side fields live on engine_params (dtype, tp_*,
        # device_map, attn_implementation, BnB knobs, max_memory, etc.); extras
        # flow through engine_params.model_extra (extra="allow").
        pt = self._engine_params(config)
        dtype = pt.dtype if pt is not None else None
        kwargs: dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(dtype or "bfloat16"),
        }

        from llenergymeasure.utils.env_config import default_device_map
        from llenergymeasure.utils.security import trust_remote_code_enabled

        # Device placement / tensor parallelism - mutually exclusive
        if pt is not None and pt.tp_plan is not None:
            # Tensor parallelism: tp_plan replaces device_map entirely
            kwargs["tp_plan"] = pt.tp_plan
            if pt.tp_size is not None:
                kwargs["tp_size"] = pt.tp_size
            # Do NOT set device_map - TP handles device placement
        elif pt is not None and pt.device_map is not None:
            kwargs["device_map"] = pt.device_map
        else:
            dm = default_device_map()
            if dm is not None:
                kwargs["device_map"] = dm

        kwargs["trust_remote_code"] = trust_remote_code_enabled()

        # Apply Transformers-specific config options
        if pt is not None:
            if pt.attn_implementation is not None:
                kwargs["attn_implementation"] = self._resolve_attn_implementation(
                    pt.attn_implementation
                )

            # BitsAndBytes quantization - use BitsAndBytesConfig, not raw kwargs
            if pt.load_in_4bit or pt.load_in_8bit:
                from transformers import BitsAndBytesConfig

                bnb_kwargs: dict[str, Any] = {}
                if pt.load_in_4bit:
                    bnb_kwargs["load_in_4bit"] = True
                    if pt.bnb_4bit_compute_dtype is not None:
                        import torch as _torch

                        _dtype_map = {
                            "float16": _torch.float16,
                            "bfloat16": _torch.bfloat16,
                            "float32": _torch.float32,
                        }
                        bnb_kwargs["bnb_4bit_compute_dtype"] = _dtype_map[pt.bnb_4bit_compute_dtype]
                    if pt.bnb_4bit_quant_type is not None:
                        bnb_kwargs["bnb_4bit_quant_type"] = pt.bnb_4bit_quant_type
                    if pt.bnb_4bit_use_double_quant is not None:
                        bnb_kwargs["bnb_4bit_use_double_quant"] = pt.bnb_4bit_use_double_quant
                if pt.load_in_8bit:
                    bnb_kwargs["load_in_8bit"] = True
                kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)

            # Additional from_pretrained() fields
            # revision dropped as typed field (D1); flows through model_extra if set
            if pt.max_memory is not None:
                kwargs["max_memory"] = pt.max_memory
            if pt.low_cpu_mem_usage is not None:
                kwargs["low_cpu_mem_usage"] = pt.low_cpu_mem_usage

        # Transformers extra="allow" passthrough: forward unknown fields to from_pretrained()
        if pt is not None and pt.model_extra:
            kwargs.update(pt.model_extra)

        # passthrough_kwargs merged LAST so researcher can override any default
        if config.passthrough_kwargs:
            kwargs.update(config.passthrough_kwargs)

        return kwargs

    @staticmethod
    def _resolve_attn_implementation(requested: str) -> str:
        """Validate the requested attention implementation is available.

        If flash_attention_2 is requested but the flash_attn package (or any
        of its transitive dependencies such as einops) cannot be imported,
        falls back to sdpa with a warning rather than crashing at model load
        time.

        A simple ``find_spec`` check is insufficient because flash_attn may
        be installed while its dependencies (e.g. einops) are missing.  We
        therefore attempt a real import of the submodule that transformers
        actually uses (``flash_attn.bert_padding``).

        Args:
            requested: The attention implementation string from config.

        Returns:
            The resolved attention implementation string.
        """
        if requested in ("flash_attention_2", "flash_attention_3"):
            fallback = "sdpa"
            try:
                import flash_attn
                import flash_attn.bert_padding  # noqa: F401
            except Exception as exc:
                logger.warning(
                    "attn_implementation=%r requested but flash_attn is not "
                    "fully usable (%s: %s); falling back to %r. "
                    "Install flash-attn and its dependencies (einops) to use "
                    "FlashAttention.",
                    requested,
                    type(exc).__name__,
                    exc,
                    fallback,
                )
                return fallback
            # FA3 additionally requires the flash_attn_interface module
            # (built separately from the flash-attn repo's hopper/ directory)
            if requested == "flash_attention_3":
                try:
                    import flash_attn_interface  # noqa: F401
                except Exception as exc:
                    logger.warning(
                        "attn_implementation='flash_attention_3' requested but "
                        "flash_attn_interface is not installed (%s: %s); "
                        "falling back to %r. Build flash_attn_3 from the "
                        "flash-attn repo's hopper/ directory, or use the "
                        "Docker runner.",
                        type(exc).__name__,
                        exc,
                        fallback,
                    )
                    return fallback
        return requested

    @staticmethod
    def _resolve_torch_dtype(dtype: str) -> Any:
        """Map dtype string to torch dtype object."""
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    # -------------------------------------------------------------------------
    # Private: inference helpers
    # -------------------------------------------------------------------------

    def _run_batch(
        self,
        model: Any,
        tokenizer: Any,
        config: ExperimentConfig,
        batch: list[str],
        generate_kwargs: dict[str, Any],
        streamer: Any | None = None,
    ) -> tuple[int, int, float, int]:
        """Run a single batch through model.generate().

        Returns (input_tokens, output_tokens, time_sec, padding_tokens) where
        padding_tokens = total padded positions in the input
        (``input_ids.numel() - attention_mask.sum()``).

        When ``streamer`` is provided (latency profiling), it is forwarded to
        ``generate(streamer=...)`` so per-token arrival times are recorded.
        """
        import time

        import torch

        tokenizer_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if config.task.max_input_tokens is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = config.task.max_input_tokens

        inputs = tokenizer(batch, **tokenizer_kwargs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_token_count = int(inputs["attention_mask"].sum().item())
        # Padding tokens: total tensor positions minus real (attended) tokens.
        padding_tokens = int(inputs["input_ids"].numel()) - input_token_count

        # Determine autocast settings (HarnessConfig - llem wraps generate()).
        from contextlib import nullcontext

        _h = self._harness(config)
        if _h is not None and _h.autocast_enabled is True and torch.cuda.is_available():
            _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            _amp_ctx = torch.autocast(
                device_type="cuda", dtype=_dtype_map[_h.autocast_dtype or "bfloat16"]
            )
        else:
            _amp_ctx = nullcontext()  # type: ignore[assignment]

        t0 = time.perf_counter()
        with torch.inference_mode(), _amp_ctx:
            gen_kwargs = {**generate_kwargs}
            if config.task.max_output_tokens is not None:
                gen_kwargs["max_new_tokens"] = config.task.max_output_tokens
            if streamer is not None:
                gen_kwargs["streamer"] = streamer
            outputs = model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        # Count only the newly generated tokens per sequence (handles padding correctly)
        input_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch,)
        output_token_count = int(
            sum(max(0, outputs.shape[1] - int(inp_len.item())) for inp_len in input_lengths)
        )
        return input_token_count, output_token_count, elapsed, padding_tokens

    def _build_generate_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build generation kwargs from the nested sampling_params + engine_params.

        None values mean "use HF's default"; only explicit fields are forwarded.
        Sampling fields (temperature, top_k, ...) live on sampling_params; the
        generation-control fields (num_beams, cache_implementation, ...) live on
        engine_params. Greedy decoding (temperature=0 or do_sample=False) strips
        sampling params and forces do_sample=False, matching HF greedy semantics.
        """
        sampling = self._sampling_params(config)
        ep = self._engine_params(config)

        kwargs: dict[str, Any] = (
            sampling.model_dump(exclude_none=True) if sampling is not None else {}
        )

        if ep is not None:
            if ep.use_cache is not None:
                kwargs["use_cache"] = ep.use_cache
            if ep.cache_implementation is not None:
                kwargs["cache_implementation"] = ep.cache_implementation
            if ep.num_beams is not None:
                kwargs["num_beams"] = ep.num_beams
            if ep.early_stopping is not None:
                kwargs["early_stopping"] = ep.early_stopping
            if ep.length_penalty is not None:
                kwargs["length_penalty"] = ep.length_penalty
            if ep.no_repeat_ngram_size is not None:
                kwargs["no_repeat_ngram_size"] = ep.no_repeat_ngram_size
            if ep.prompt_lookup_num_tokens is not None:
                kwargs["prompt_lookup_num_tokens"] = ep.prompt_lookup_num_tokens

        if kwargs.get("temperature") == 0.0 or kwargs.get("do_sample") is False:
            kwargs["do_sample"] = False
            for key in ("temperature", "top_k", "top_p", "min_p"):
                kwargs.pop(key, None)

        return kwargs
