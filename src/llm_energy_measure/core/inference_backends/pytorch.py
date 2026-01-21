"""PyTorch/Transformers inference backend.

This backend wraps the existing HuggingFace Transformers + Accelerate implementation,
providing backward compatibility while conforming to the InferenceBackend protocol.

Design:
- Composes existing ModelLoader and InferenceEngine implementations
- Converts internal InferenceResult to protocol-agnostic BackendResult
- Uses Accelerate for distributed execution and device management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from llm_energy_measure.core.inference_backends.protocols import (
    BackendResult,
    BackendRuntime,
    ConfigWarning,
    CudaManagement,
    LaunchMode,
    RuntimeCapabilities,
)
from llm_energy_measure.core.inference_backends.shared import (
    CORE_SUPPORTED_PARAMS,
    create_precision_metadata,
)
from llm_energy_measure.domain.metrics import (
    LatencyMeasurementMode,
    LatencyMeasurements,
    collect_itl_measurements,
)
from llm_energy_measure.exceptions import (
    BackendInferenceError,
    BackendInitializationError,
)

if TYPE_CHECKING:
    from llm_energy_measure.config.backend_configs import PyTorchConfig
    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.domain.model_info import ModelInfo


# Parameters supported by PyTorch/Transformers backend
# Extends core params with PyTorch-specific options
_SUPPORTED_PARAMS: set[str] = set(CORE_SUPPORTED_PARAMS) | {
    # PyTorch-specific: LoRA adapter support
    "adapter",
    # Extended batching
    "batching.max_tokens_per_batch",
    # Extended decoder options
    "decoder.do_sample",
    "decoder.min_p",
    "decoder.repetition_penalty",
    "decoder.no_repeat_ngram_size",
    # Quantization (BitsAndBytes)
    "quantization.load_in_4bit",
    "quantization.load_in_8bit",
    "quantization.quantization",
    # Sharding (via Accelerate)
    "sharding.strategy",
    "sharding.num_shards",
    # Traffic simulation
    "traffic_simulation.enabled",
    "traffic_simulation.mode",
    "traffic_simulation.target_qps",
}


class PyTorchBackend:
    """HuggingFace Transformers inference backend.

    This backend composes existing implementations:
    - HuggingFaceModelLoader for model/tokenizer loading
    - TransformersInferenceEngine for inference
    - ThroughputMetricsCollector for metrics

    It provides full backward compatibility with existing experiments while
    conforming to the InferenceBackend protocol.

    PyTorch-specific configuration (config.pytorch) controls:
    - Attention implementation (flash_attention_2, sdpa, eager)
    - torch.compile optimization
    - BetterTransformer conversion
    - KV caching, memory management
    - Assisted generation (speculative decoding)
    """

    def __init__(self) -> None:
        """Initialize backend (model loaded lazily in initialize())."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._accelerator: Any = None
        self._config: ExperimentConfig | None = None
        self._runtime: BackendRuntime | None = None
        self._assistant_model: Any = None  # For assisted generation
        self._is_compiled: bool = False  # Track if torch.compile was applied

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "pytorch"

    @property
    def version(self) -> str:
        """Backend version string."""
        import transformers

        return f"transformers={transformers.__version__}, torch={torch.__version__}"

    def is_available(self) -> bool:
        """Check if PyTorch and Transformers are available."""
        try:
            import accelerate  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        """Return PyTorch/Accelerate runtime requirements.

        The orchestration layer manages CUDA context via Accelerate.
        torch.cuda.* calls are safe before this backend initializes.
        """
        return RuntimeCapabilities(
            launch_mode=LaunchMode.ACCELERATE,
            cuda_management=CudaManagement.ORCHESTRATOR,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=False,
            manages_own_batching=False,
        )

    def _build_model_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs for model loading from PyTorchConfig.

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs for AutoModelForCausalLM.from_pretrained().
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        kwargs: dict[str, Any] = {}

        if pytorch_cfg is None:
            return kwargs

        # Attention implementation
        if pytorch_cfg.attn_implementation != "sdpa":
            kwargs["attn_implementation"] = pytorch_cfg.attn_implementation

        # Memory management
        if pytorch_cfg.low_cpu_mem_usage:
            kwargs["low_cpu_mem_usage"] = True
        if pytorch_cfg.max_memory:
            kwargs["max_memory"] = pytorch_cfg.max_memory

        # Escape hatch
        if pytorch_cfg.extra:
            kwargs.update(pytorch_cfg.extra)

        return kwargs

    def _build_generation_kwargs(
        self,
        config: ExperimentConfig,
        input_length: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build comprehensive kwargs for model.generate().

        Consolidates all generation parameters from:
        - Decoder config (temperature, sampling, beam search)
        - PyTorch-specific config (caching, compile options)
        - Token limits

        Args:
            config: Experiment configuration.
            input_length: Current input sequence length (for min_length calculation).
            max_output_tokens: Override for max output tokens (defaults to config value).

        Returns:
            Dict of kwargs for model.generate().
        """
        kwargs: dict[str, Any] = {}
        decoder = config.decoder
        pytorch_cfg: PyTorchConfig | None = config.pytorch

        # Token limits
        effective_max_output = max_output_tokens or config.max_output_tokens
        kwargs["max_new_tokens"] = effective_max_output

        if config.min_output_tokens:
            if input_length is not None:
                kwargs["min_length"] = input_length + config.min_output_tokens
            else:
                kwargs["min_new_tokens"] = config.min_output_tokens

        # Temperature=0 forces greedy regardless of other settings
        if decoder.temperature == 0.0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = decoder.do_sample
            kwargs["temperature"] = decoder.temperature

            if decoder.do_sample:
                # Nucleus/top sampling (only when sampling is enabled)
                if decoder.top_k > 0:
                    kwargs["top_k"] = decoder.top_k
                if decoder.top_p < 1.0:
                    kwargs["top_p"] = decoder.top_p
                if decoder.min_p > 0.0:
                    kwargs["min_p"] = decoder.min_p

                # Repetition control
                if decoder.repetition_penalty != 1.0:
                    kwargs["repetition_penalty"] = decoder.repetition_penalty
                if decoder.no_repeat_ngram_size > 0:
                    kwargs["no_repeat_ngram_size"] = decoder.no_repeat_ngram_size

        # Beam search from unified decoder config
        beam_cfg = decoder.beam_search
        if beam_cfg.enabled and beam_cfg.num_beams > 1:
            kwargs["num_beams"] = beam_cfg.num_beams
            if beam_cfg.early_stopping:
                kwargs["early_stopping"] = True
            if beam_cfg.length_penalty != 1.0:
                kwargs["length_penalty"] = beam_cfg.length_penalty

        # PyTorch-specific options
        if pytorch_cfg is not None:
            # KV caching
            if not pytorch_cfg.use_cache:
                kwargs["use_cache"] = False

            # Cache implementation (static enables CUDA graphs)
            if pytorch_cfg.cache_implementation:
                kwargs["cache_implementation"] = pytorch_cfg.cache_implementation

            # Output configuration
            if pytorch_cfg.output_scores:
                kwargs["output_scores"] = True
            if pytorch_cfg.return_dict_in_generate:
                kwargs["return_dict_in_generate"] = True

        return kwargs

    def _apply_torch_compile(self, config: ExperimentConfig) -> None:
        """Apply torch.compile to model if configured.

        Args:
            config: Experiment configuration.
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        if pytorch_cfg is None or not pytorch_cfg.torch_compile:
            return

        if self._model is None:
            return

        # Determine compilation mode
        mode = "default"
        if isinstance(pytorch_cfg.torch_compile, str):
            mode = pytorch_cfg.torch_compile

        logger.info(f"Applying torch.compile with mode='{mode}'")

        try:
            self._model = torch.compile(self._model, mode=mode)
            self._is_compiled = True
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed (non-fatal): {e}")

    def _apply_bettertransformer(self, config: ExperimentConfig) -> None:
        """Convert model to BetterTransformer if configured.

        Args:
            config: Experiment configuration.
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        if pytorch_cfg is None or not pytorch_cfg.use_bettertransformer:
            return

        if self._model is None:
            return

        logger.info("Converting model to BetterTransformer")

        try:
            self._model = self._model.to_bettertransformer()
            logger.info("BetterTransformer conversion successful")
        except Exception as e:
            logger.warning(f"BetterTransformer conversion failed (non-fatal): {e}")

    def _load_assistant_model(self, config: ExperimentConfig) -> None:
        """Load assistant model for assisted generation if configured.

        Args:
            config: Experiment configuration.
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        if pytorch_cfg is None:
            return

        assisted_cfg = pytorch_cfg.assisted_generation
        if assisted_cfg is None or not assisted_cfg.model:
            return

        from transformers import AutoModelForCausalLM

        logger.info(f"Loading assistant model for speculative decoding: {assisted_cfg.model}")

        try:
            self._assistant_model = AutoModelForCausalLM.from_pretrained(
                assisted_cfg.model,
                torch_dtype=getattr(torch, config.fp_precision),
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info(f"Assistant model loaded: {assisted_cfg.model}")
        except Exception as e:
            logger.warning(f"Failed to load assistant model (non-fatal): {e}")
            self._assistant_model = None

    def _supports_streaming(self) -> bool:
        """Check if TextIteratorStreamer is available for streaming inference.

        Returns:
            True if transformers supports TextIteratorStreamer.
        """
        try:
            from transformers import TextIteratorStreamer  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, config: ExperimentConfig, runtime: BackendRuntime) -> None:
        """Load model and prepare for inference.

        Args:
            config: Experiment configuration.
            runtime: Runtime context with accelerator.

        Raises:
            BackendInitializationError: If model loading fails.
        """
        from llm_energy_measure.core.implementations import HuggingFaceModelLoader

        self._config = config
        self._runtime = runtime
        self._accelerator = runtime.accelerator

        if self._accelerator is None:
            raise BackendInitializationError(
                "PyTorch backend requires an Accelerator instance in BackendRuntime. "
                "This backend uses HuggingFace Accelerate for distributed execution."
            )

        try:
            # Build model loading kwargs from PyTorchConfig
            model_kwargs = self._build_model_kwargs(config)
            if model_kwargs:
                logger.info(f"PyTorch config: {model_kwargs}")

            loader = HuggingFaceModelLoader()
            # Note: HuggingFaceModelLoader.load() currently doesn't accept extra kwargs
            # TODO: Pass model_kwargs to loader when supported
            self._model, self._tokenizer = loader.load(config)
            logger.info(f"Model loaded: {config.model_name}")

            # Apply post-load optimizations
            self._apply_bettertransformer(config)
            self._apply_torch_compile(config)

            # Load assistant model for speculative decoding if configured
            self._load_assistant_model(config)

        except Exception as e:
            raise BackendInitializationError(
                f"Failed to load model '{config.model_name}': {e}"
            ) from e

    def run_inference(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference using Transformers.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with token counts and timing.

        Raises:
            BackendInferenceError: If inference fails.
        """
        if self._model is None or self._tokenizer is None:
            raise BackendInferenceError("Backend not initialized. Call initialize() first.")

        try:
            # Check if streaming mode is enabled for latency measurement
            if config.streaming:
                if not self._supports_streaming():
                    logger.warning(
                        "TextIteratorStreamer unavailable. "
                        "Falling back to batch with TTFT estimation."
                    )
                    return self._run_batch_with_ttft_estimation(prompts, config)

                # Warn about torch.compile incompatibility
                if self._is_compiled:
                    logger.warning(
                        "torch.compile may be incompatible with streaming. "
                        "Using non-compiled inference path."
                    )

                return self._run_streaming_inference(prompts, config)

            return self._run_inference_batch(prompts, config)

        except Exception as e:
            raise BackendInferenceError(f"Inference failed: {e}") from e

    def _run_inference_batch(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run batch inference with full control over generation.

        Self-contained batch processing that:
        - Creates batches using prompts.py utilities
        - Applies traffic simulation if configured
        - Uses unified generation kwargs
        - Tracks per-batch latencies

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with token counts and timing.
        """
        import time

        from llm_energy_measure.core.prompts import (
            tokenize_batch,
        )
        from llm_energy_measure.core.traffic import TrafficGenerator, apply_traffic_delay
        from llm_energy_measure.progress import batch_progress

        device = self._accelerator.device
        max_input_tokens = config.max_input_tokens
        max_output_tokens = config.max_output_tokens

        # Create batches based on strategy
        batches = self._create_batches(prompts, config)

        # Initialise traffic generator if enabled
        traffic_generator: TrafficGenerator | None = None
        if config.traffic_simulation.enabled:
            traffic_generator = TrafficGenerator(config.traffic_simulation)

        # Process batches
        all_outputs: list[torch.Tensor] = []
        latencies: list[float] = []
        total_generated_tokens = 0
        total_input_tokens = 0

        with batch_progress(
            total=len(batches),
            desc="Batches",
            is_main_process=self._accelerator.is_main_process,
        ) as progress:
            for batch_idx, batch in enumerate(batches):
                # Apply traffic simulation delay if configured
                if traffic_generator is not None:
                    apply_traffic_delay(
                        config=config.traffic_simulation,
                        batch_idx=batch_idx,
                        generator=traffic_generator,
                    )

                # Set seed for reproducible sampling
                if config.random_seed is not None:
                    batch_seed = config.random_seed + batch_idx
                    torch.manual_seed(batch_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(batch_seed)

                # Tokenise batch
                tokenized = tokenize_batch(
                    prompts=batch,
                    tokenizer=self._tokenizer,
                    max_length=max_input_tokens,
                    batch_size=len(batch),
                )

                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                input_tokens = input_ids.numel()
                total_input_tokens += input_tokens
                current_length = input_ids.shape[1]

                # Calculate allowed output tokens (respect model max length)
                total_allowed = self._tokenizer.model_max_length
                allowed_new = max(0, total_allowed - current_length)

                # Build generation kwargs
                generation_kwargs = self._build_generation_kwargs(
                    config=config,
                    input_length=current_length,
                    max_output_tokens=min(max_output_tokens, allowed_new),
                )

                gpu_id = device.index if hasattr(device, "index") else 0
                logger.debug(f"[GPU {gpu_id}] Processing batch {batch_idx + 1}/{len(batches)}")

                # Run inference
                start_time = time.perf_counter()

                with torch.no_grad():
                    outputs = self._model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=self._tokenizer.pad_token_id,
                        **generation_kwargs,
                    )

                torch.cuda.synchronize(device)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0
                latencies.append(latency_ms)

                logger.debug(
                    f"[GPU {gpu_id}] Completed batch {batch_idx + 1}/{len(batches)} "
                    f"in {latency_ms:.1f}ms"
                )

                # Count generated tokens
                for j in range(input_ids.size(0)):
                    prompt_len = input_ids[j].shape[0]
                    gen_len = outputs[j].shape[0] - prompt_len
                    total_generated_tokens += gen_len

                # Update progress
                progress.update(1, latency_ms=latency_ms)

                if config.save_outputs:
                    all_outputs.append(outputs)

        # Calculate metrics
        total_time_sec = sum(latencies) / 1000.0 if latencies else 0.0
        total_tokens = total_input_tokens + total_generated_tokens
        tokens_per_sec = total_generated_tokens / total_time_sec if total_time_sec > 0 else 0.0
        latency_per_token_ms = 1000.0 / tokens_per_sec if tokens_per_sec > 0 else 0.0

        # Get actual compute dtype from model
        actual_dtype = str(self._model.dtype) if self._model is not None else None

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_generated_tokens,
            inference_time_sec=total_time_sec,
            batch_latencies_ms=latencies,
            output_texts=None,
            backend_metadata={
                "backend": self.name,
                "version": self.version,
                "lora_adapter": config.adapter,
                "tokens_per_second": tokens_per_sec,
                "latency_per_token_ms": latency_per_token_ms,
                "num_batches": len(batches),
                "batching_strategy": config.batching.strategy,
            },
            precision_metadata=create_precision_metadata(config, self.name, actual_dtype),
        )

    def _create_batches(self, prompts: list[str], config: ExperimentConfig) -> list[list[str]]:
        """Create batches based on config batching strategy.

        Industry-standard strategies (per MLPerf/vLLM terminology):
        - static: Fixed batch size, pads to max_length (MLPerf offline scenario)
        - dynamic: Token-aware batching with max_tokens_per_batch (MLPerf server scenario)
        - sorted_static: Sort by length, then static batches (reduces padding waste)
        - sorted_dynamic: Sort by length + dynamic token budget (optimal packing)

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            List of batches (each batch is a list of prompts).
        """
        from llm_energy_measure.core.prompts import (
            create_adaptive_batches,
            create_fixed_batches,
        )

        batching = config.batching
        strategy = batching.strategy
        max_tokens = batching.max_tokens_per_batch or config.max_input_tokens

        # Sort prompts by length if strategy requires it
        working_prompts = prompts
        if strategy in ("sorted_static", "sorted_dynamic"):
            working_prompts = sorted(prompts, key=len)
            logger.debug(f"Sorted {len(prompts)} prompts by length for {strategy} strategy")

        # Dispatch based on dynamic vs static batching
        if strategy in ("dynamic", "sorted_dynamic"):
            return create_adaptive_batches(
                prompts=working_prompts,
                tokenizer=self._tokenizer,
                max_tokens_per_batch=max_tokens,
                max_prompt_tokens=config.max_input_tokens,
                max_batch_size=batching.batch_size,
            )

        # Default to static batching (static, sorted_static, or unknown)
        if strategy not in ("static", "sorted_static"):
            logger.warning(f"Unknown batching strategy '{strategy}', using static")
        return create_fixed_batches(
            prompts=working_prompts,
            batch_size=batching.batch_size,
        )

    def _run_streaming_inference(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Run inference with streaming for TTFT/ITL latency measurement.

        Uses TextIteratorStreamer with threading to capture per-token timestamps.
        Processes prompts sequentially (one at a time) to measure individual latencies.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with latency_measurements containing raw samples.
        """
        import threading
        import time

        import numpy as np
        from transformers import TextIteratorStreamer

        warmup_count = config.streaming_warmup_requests

        # Split into warmup and measurement prompts
        warmup_prompts = prompts[:warmup_count] if warmup_count > 0 else []
        measurement_prompts = prompts[warmup_count:]

        # Run warmup (results discarded from stats)
        if warmup_prompts:
            logger.info(f"Running {len(warmup_prompts)} streaming warmup requests...")
            for prompt in warmup_prompts:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                self._model.generate(**inputs, max_new_tokens=config.max_output_tokens)
            logger.debug("Streaming warmup complete")

        if not measurement_prompts:
            logger.warning("No prompts remaining after warmup. Increase num_input_prompts.")
            return BackendResult(
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                inference_time_sec=0.0,
                latency_measurements=LatencyMeasurements(
                    ttft_ms=[],
                    itl_full_ms=[],
                    itl_trimmed_ms=[],
                    request_count=0,
                    total_output_tokens=0,
                    excluded_tokens=0,
                    streaming_mode=True,
                    warmup_requests_excluded=warmup_count,
                    measurement_mode=LatencyMeasurementMode.TRUE_STREAMING,
                ),
            )

        # Statistical sufficiency warning
        if len(measurement_prompts) < 30:
            logger.warning(
                f"Only {len(measurement_prompts)} samples for latency statistics. "
                "Consider increasing num_input_prompts for reliable percentiles."
            )

        # Warn about batch_size being ignored
        if config.batching.batch_size and config.batching.batch_size > 1:
            logger.warning(
                f"streaming=True forces sequential processing. "
                f"batch_size={config.batching.batch_size} ignored."
            )

        # Collect per-request timing data
        ttft_samples: list[float] = []
        token_timestamps_per_request: list[list[float]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []

        logger.info(f"Running streaming inference on {len(measurement_prompts)} prompts...")
        start_time = time.perf_counter()

        # Build generation kwargs using unified method
        generation_kwargs = self._build_generation_kwargs(config)

        # Process each prompt individually with streaming
        for prompt in measurement_prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            input_length = inputs["input_ids"].shape[1]
            total_input_tokens += input_length

            # Create streamer for this request
            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Track token timestamps
            token_times: list[float] = []
            first_token_time: float | None = None
            request_start = time.perf_counter()

            # Run generation in background thread
            gen_kwargs = {
                **generation_kwargs,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "streamer": streamer,
            }

            thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
            thread.start()

            # Collect tokens as they arrive
            generated_text = ""
            num_tokens = 0
            for token_text in streamer:
                token_arrival = time.perf_counter()
                token_time_ms = (token_arrival - request_start) * 1000

                if first_token_time is None:
                    first_token_time = token_time_ms
                    ttft_samples.append(first_token_time)

                token_times.append(token_time_ms)
                generated_text += token_text
                num_tokens += 1

            thread.join()

            total_output_tokens += num_tokens
            output_texts.append(generated_text)
            token_timestamps_per_request.append(token_times)

        inference_time = time.perf_counter() - start_time

        # Calculate ITL from token timestamps using shared utility
        itl_full, itl_trimmed, excluded = collect_itl_measurements(token_timestamps_per_request)

        # Build latency measurements
        latency_measurements = LatencyMeasurements(
            ttft_ms=ttft_samples,
            itl_full_ms=itl_full,
            itl_trimmed_ms=itl_trimmed,
            request_count=len(measurement_prompts),
            total_output_tokens=total_output_tokens,
            excluded_tokens=excluded,
            streaming_mode=True,
            warmup_requests_excluded=warmup_count,
            measurement_mode=LatencyMeasurementMode.TRUE_STREAMING,
        )

        # Calculate average TTFT for BackendResult (backward compat)
        avg_ttft_ms: float | None = None
        if ttft_samples:
            avg_ttft_ms = float(np.mean(ttft_samples))

        total_tokens = total_input_tokens + total_output_tokens

        logger.info(
            f"Streaming inference complete: {len(measurement_prompts)} prompts, "
            f"{total_tokens} tokens in {inference_time:.2f}s, "
            f"TTFT samples={len(ttft_samples)}, ITL samples={len(itl_trimmed)}"
        )

        # Get actual compute dtype from model
        actual_dtype = str(self._model.dtype) if self._model is not None else None

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            inference_time_sec=inference_time,
            time_to_first_token_ms=avg_ttft_ms,
            output_texts=output_texts if config.save_outputs else None,
            backend_metadata={
                "backend": self.name,
                "version": self.version,
                "streaming_mode": True,
                "lora_adapter": config.adapter,
                "num_prompts": len(measurement_prompts),
                "warmup_prompts": warmup_count,
                "ttft_samples": len(ttft_samples),
                "itl_samples": len(itl_trimmed),
                "torch_compiled": self._is_compiled,
            },
            latency_measurements=latency_measurements,
            precision_metadata=create_precision_metadata(config, self.name, actual_dtype),
        )

    def _run_batch_with_ttft_estimation(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Fallback: batch inference with TTFT estimation.

        Processes prompts one at a time to estimate TTFT from total request time.
        Used when TextIteratorStreamer is unavailable.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with estimated latency measurements.
        """
        import time

        import numpy as np

        warmup_count = config.streaming_warmup_requests
        warmup_prompts = prompts[:warmup_count] if warmup_count > 0 else []
        measurement_prompts = prompts[warmup_count:]

        # Run warmup
        if warmup_prompts:
            logger.info(f"Running {len(warmup_prompts)} warmup requests...")
            for prompt in warmup_prompts:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                self._model.generate(**inputs, max_new_tokens=config.max_output_tokens)

        if not measurement_prompts:
            return BackendResult(
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                inference_time_sec=0.0,
                latency_measurements=LatencyMeasurements(
                    ttft_ms=[],
                    itl_full_ms=[],
                    itl_trimmed_ms=[],
                    request_count=0,
                    total_output_tokens=0,
                    excluded_tokens=0,
                    streaming_mode=False,
                    warmup_requests_excluded=warmup_count,
                    measurement_mode=LatencyMeasurementMode.PER_REQUEST_BATCH,
                ),
            )

        ttft_samples: list[float] = []
        token_timestamps_per_request: list[list[float]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []

        logger.info(
            f"Running batch inference with TTFT estimation on "
            f"{len(measurement_prompts)} prompts..."
        )
        start_time = time.perf_counter()

        # Build generation kwargs using unified method
        generation_kwargs = self._build_generation_kwargs(config)
        # Force return_dict_in_generate for output parsing
        generation_kwargs["return_dict_in_generate"] = True

        for prompt in measurement_prompts:
            request_start = time.perf_counter()

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            input_length = inputs["input_ids"].shape[1]
            total_input_tokens += input_length

            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )

            request_end = time.perf_counter()
            total_time_ms = (request_end - request_start) * 1000

            # Count output tokens
            output_ids = outputs.sequences[0][input_length:]
            num_tokens = len(output_ids)
            total_output_tokens += num_tokens

            # Decode output
            output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            output_texts.append(output_text)

            # Estimate TTFT as proportional to 1 token of total time
            estimated_ttft = total_time_ms / (num_tokens + 1) if num_tokens > 0 else total_time_ms
            ttft_samples.append(estimated_ttft)

            # Estimate token times evenly distributed
            if num_tokens > 1:
                decode_time_ms = total_time_ms - estimated_ttft
                token_times = [estimated_ttft]
                time_per_token = decode_time_ms / (num_tokens - 1)
                for i in range(1, num_tokens):
                    token_times.append(estimated_ttft + (i * time_per_token))
                token_timestamps_per_request.append(token_times)

        inference_time = time.perf_counter() - start_time

        # Calculate ITL from estimated timestamps
        itl_full, itl_trimmed, excluded = collect_itl_measurements(token_timestamps_per_request)

        latency_measurements = LatencyMeasurements(
            ttft_ms=ttft_samples,
            itl_full_ms=itl_full,
            itl_trimmed_ms=itl_trimmed,
            request_count=len(measurement_prompts),
            total_output_tokens=total_output_tokens,
            excluded_tokens=excluded,
            streaming_mode=False,
            warmup_requests_excluded=warmup_count,
            measurement_mode=LatencyMeasurementMode.PROPORTIONAL_ESTIMATE,
        )

        avg_ttft_ms: float | None = None
        if ttft_samples:
            avg_ttft_ms = float(np.mean(ttft_samples))

        total_tokens = total_input_tokens + total_output_tokens

        logger.info(
            f"Batch inference complete: {len(measurement_prompts)} prompts, "
            f"{total_tokens} tokens in {inference_time:.2f}s"
        )

        # Get actual compute dtype from model
        actual_dtype = str(self._model.dtype) if self._model is not None else None

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            inference_time_sec=inference_time,
            time_to_first_token_ms=avg_ttft_ms,
            output_texts=output_texts if config.save_outputs else None,
            backend_metadata={
                "backend": self.name,
                "version": self.version,
                "streaming_mode": False,
                "lora_adapter": config.adapter,
                "num_prompts": len(measurement_prompts),
                "warmup_prompts": warmup_count,
                "latency_warning": (
                    "ITL values are estimated (uniform distribution), not measured per-token. "
                    "Not suitable for publication-quality research."
                ),
            },
            latency_measurements=latency_measurements,
            precision_metadata=create_precision_metadata(config, self.name, actual_dtype),
        )

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._assistant_model is not None:
            del self._assistant_model
            self._assistant_model = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("PyTorch backend cleaned up")

    def get_model_info(self) -> ModelInfo:
        """Return model metadata.

        Returns:
            ModelInfo with parameter count, architecture details.
        """
        from llm_energy_measure.domain.model_info import ModelInfo, QuantizationSpec

        if self._model is None or self._config is None:
            raise BackendInferenceError("Backend not initialized. Call initialize() first.")

        model_config = self._model.config

        # Determine quantization
        quant_config = self._config.quantization
        if quant_config.load_in_4bit:
            quant = QuantizationSpec(
                enabled=True, bits=4, method="bitsandbytes", compute_dtype="float16"
            )
        elif quant_config.load_in_8bit:
            quant = QuantizationSpec(
                enabled=True, bits=8, method="bitsandbytes", compute_dtype="float16"
            )
        else:
            quant = QuantizationSpec(
                enabled=False, bits=None, method="none", compute_dtype=self._config.fp_precision
            )

        return ModelInfo(
            name=self._config.model_name,
            revision=None,
            num_parameters=sum(p.numel() for p in self._model.parameters()),
            num_layers=getattr(model_config, "num_hidden_layers", 0),
            hidden_size=getattr(model_config, "hidden_size", 0),
            num_attention_heads=getattr(model_config, "num_attention_heads", 0),
            vocab_size=getattr(model_config, "vocab_size", 0),
            model_type=getattr(model_config, "model_type", "unknown"),
            torch_dtype=str(self._model.dtype),
            quantization=quant,
        )

    def get_supported_params(self) -> set[str]:
        """Return parameters supported by this backend."""
        return _SUPPORTED_PARAMS.copy()

    def validate_config(self, config: ExperimentConfig) -> list[ConfigWarning]:
        """Validate config compatibility with PyTorch backend.

        Only returns warnings for actual incompatibilities or problems,
        not informational notes about normal backend behaviour.

        Args:
            config: Configuration to validate.

        Returns:
            List of warnings/errors for config problems. Empty for valid configs.
        """
        warnings: list[ConfigWarning] = []

        pytorch_cfg = config.pytorch
        if pytorch_cfg is None:
            return warnings

        # Check for flash_attention_2 - requires flash-attn package
        if pytorch_cfg.attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                warnings.append(
                    ConfigWarning(
                        param="pytorch.attn_implementation",
                        message=(
                            "flash_attention_2 requires the flash-attn package. "
                            "Falling back to sdpa."
                        ),
                        severity="warning",
                        suggestion="pip install flash-attn or use attn_implementation='sdpa'",
                    )
                )

        # BetterTransformer is deprecated with sdpa/flash_attention
        if pytorch_cfg.use_bettertransformer and pytorch_cfg.attn_implementation != "eager":
            warnings.append(
                ConfigWarning(
                    param="pytorch.use_bettertransformer",
                    message=(
                        "BetterTransformer is deprecated and may conflict with "
                        f"attn_implementation='{pytorch_cfg.attn_implementation}'. "
                        "Consider removing use_bettertransformer."
                    ),
                    severity="warning",
                )
            )

        # torch.compile with BetterTransformer is problematic
        if pytorch_cfg.torch_compile and pytorch_cfg.use_bettertransformer:
            warnings.append(
                ConfigWarning(
                    param="pytorch.torch_compile",
                    message="torch.compile and BetterTransformer together may cause issues.",
                    severity="warning",
                )
            )

        return warnings
