"""vLLM inference backend.

High-performance inference backend using vLLM with PagedAttention
and continuous batching. Provides native tensor parallelism support
for multi-GPU inference.

Key differences from PyTorch backend:
- vLLM manages its own model loading and distribution (not Accelerate)
- Continuous batching means batch_size is a hint, not exact
- Uses SamplingParams for decoder configuration
- Supports PagedAttention for efficient KV cache management

Config Mapping:
- model_name → LLM(model=...)
- fp_precision → dtype
- sharding.strategy: tensor_parallel → tensor_parallel_size
- batching.batch_size → max_num_seqs (hint only)
- decoder.* → SamplingParams(...)
- quantization.load_in_4bit → quantization="bitsandbytes"
- quantization.load_in_8bit → NOT SUPPORTED (warning issued)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from llm_energy_measure.core.inference_backends.protocols import (
    BackendResult,
    BackendRuntime,
    ConfigWarning,
    CudaManagement,
    LatencyMeasurements,
    LaunchMode,
    RuntimeCapabilities,
    collect_itl_measurements,
)
from llm_energy_measure.exceptions import (
    BackendInferenceError,
    BackendInitializationError,
    BackendNotAvailableError,
)
from llm_energy_measure.progress import batch_progress, prompt_progress

if TYPE_CHECKING:
    from llm_energy_measure.config.backend_configs import VLLMConfig
    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.domain.model_info import ModelInfo


# Parameters supported by vLLM backend
_SUPPORTED_PARAMS: set[str] = {
    # Core
    "model_name",
    "adapter",
    "fp_precision",
    "max_input_tokens",
    "max_output_tokens",
    "min_output_tokens",
    "random_seed",
    # Batching (hints only - vLLM uses continuous batching)
    "batch_size",
    "batching.batch_size",
    "batching.max_tokens_per_batch",
    # Note: batching.strategy is accepted but ignored (vLLM always continuous)
    "batching.strategy",
    # Decoder/Generation
    "decoder.preset",
    "decoder.temperature",
    "decoder.top_p",
    "decoder.top_k",
    "decoder.min_p",
    "decoder.repetition_penalty",
    # Note: do_sample not needed (vLLM uses temperature=0 for greedy)
    # Quantization (limited support)
    "quantization.load_in_4bit",
    "quantization.quantization",
    # Sharding (native tensor parallelism)
    "sharding.strategy",
    "sharding.num_shards",
    # Other
    "save_outputs",
    "num_input_prompts",
    "gpus",
    "num_processes",
    # Streaming latency measurement
    "streaming",
    "streaming_warmup_requests",
}

# Parameters that require warnings
_UNSUPPORTED_WITH_WARNING: dict[str, str] = {
    "quantization.load_in_8bit": (
        "8-bit BitsAndBytes quantization not supported by vLLM. "
        "Use load_in_4bit or a pre-quantized GPTQ/AWQ model."
    ),
    "decoder.no_repeat_ngram_size": (
        "no_repeat_ngram_size not supported by vLLM. " "Use repetition_penalty instead."
    ),
}


def _check_vllm_available() -> bool:
    """Check if vLLM is installed and importable."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def _get_vllm_version() -> str:
    """Get vLLM version string."""
    try:
        import vllm

        return f"vllm={vllm.__version__}"
    except (ImportError, AttributeError):
        return "vllm=unknown"


class VLLMBackend:
    """vLLM inference backend with PagedAttention and continuous batching.

    vLLM provides:
    - PagedAttention for efficient KV cache management
    - Continuous batching for optimal throughput
    - Native tensor parallelism for multi-GPU inference

    Note: vLLM manages its own model loading and distribution. It does NOT
    use HuggingFace Accelerate. The BackendRuntime.accelerator field is ignored.
    """

    def __init__(self) -> None:
        """Initialize backend (model loaded lazily in initialize())."""
        self._llm: Any = None
        self._sampling_params: Any = None
        self._config: ExperimentConfig | None = None
        self._runtime: BackendRuntime | None = None
        self._model_info: ModelInfo | None = None
        self._warmup_done: bool = False
        self._lora_request: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "vllm"

    @property
    def version(self) -> str:
        """Backend version string."""
        return _get_vllm_version()

    def is_available(self) -> bool:
        """Check if vLLM is installed."""
        return _check_vllm_available()

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        """Return vLLM runtime requirements.

        vLLM manages its own CUDA context and uses spawn multiprocessing for
        tensor parallelism. The orchestration layer MUST NOT call torch.cuda.*
        functions before vLLM initializes.
        """
        return RuntimeCapabilities(
            launch_mode=LaunchMode.DIRECT,
            cuda_management=CudaManagement.BACKEND,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=True,
            manages_own_batching=True,
        )

    def initialize(self, config: ExperimentConfig, runtime: BackendRuntime) -> None:
        """Load model using vLLM's LLM class.

        Args:
            config: Experiment configuration.
            runtime: Runtime context (device info largely ignored - vLLM manages).

        Raises:
            BackendNotAvailableError: If vLLM is not installed.
            BackendInitializationError: If model loading fails.
        """
        if not self.is_available():
            raise BackendNotAvailableError(
                "vllm", install_hint="pip install llm-energy-measure[vllm]"
            )

        self._config = config
        self._runtime = runtime

        try:
            from vllm import LLM, SamplingParams

            # Build engine kwargs from config (shared + vLLM-specific)
            llm_kwargs = self._build_engine_kwargs(config)

            # Log configuration
            tp_size = llm_kwargs.get("tensor_parallel_size", 1)
            dtype = llm_kwargs.get("dtype", "auto")
            quant = llm_kwargs.get("quantization")
            prefix_caching = llm_kwargs.get("enable_prefix_caching", False)
            speculative = "speculative_config" in llm_kwargs
            lora_enabled = llm_kwargs.get("enable_lora", False)

            logger.info(
                f"Initializing vLLM: model={config.model_name}, "
                f"dtype={dtype}, tp={tp_size}, quant={quant}, "
                f"prefix_caching={prefix_caching}, speculative={speculative}, "
                f"lora={lora_enabled}"
            )

            if config.vllm:
                logger.info(
                    f"vLLM config: max_num_seqs={config.vllm.max_num_seqs}, "
                    f"gpu_mem={config.vllm.gpu_memory_utilization}, "
                    f"kv_dtype={config.vllm.kv_cache_dtype}"
                )

            self._llm = LLM(**llm_kwargs)

            # Create sampling params
            self._sampling_params = self._create_sampling_params(config, SamplingParams)

            # Create LoRARequest if adapter specified
            if config.adapter:
                self._lora_request = self._create_lora_request(config)

            logger.info(f"vLLM model loaded: {config.model_name}")

            # Perform warmup (JIT compilation happens on first inference)
            self._perform_warmup()

        except Exception as e:
            raise BackendInitializationError(
                f"Failed to initialize vLLM with model '{config.model_name}': {e}"
            ) from e

    def _get_tensor_parallel_size(self, config: ExperimentConfig) -> int:
        """Determine tensor parallelism size from config WITHOUT initializing CUDA.

        IMPORTANT: This method must NOT call torch.cuda.* functions as vLLM
        manages its own CUDA context. Calling torch.cuda.device_count() or
        torch.cuda.is_available() here would pre-initialize CUDA and cause
        fork issues when vLLM spawns worker processes.

        Args:
            config: Experiment configuration.

        Returns:
            Number of GPUs for tensor parallelism.
        """
        import os

        sharding = config.sharding_config

        # Check if tensor parallelism is explicitly requested
        if sharding.strategy == "tensor_parallel":
            if sharding.num_shards:
                return sharding.num_shards

            # Use gpu_list if provided
            if config.gpu_list:
                return len(config.gpu_list)

            # Fall back to CUDA_VISIBLE_DEVICES (no torch.cuda.* calls!)
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                return len(cuda_visible.split(","))

            # vLLM will auto-detect at initialization
            logger.warning(
                "tensor_parallel without num_shards or gpu_list specified. "
                "vLLM will auto-detect available GPUs at initialization."
            )
            return 1

        # Check GPU list
        if config.gpu_list:
            return len(config.gpu_list)

        # Default to 1
        return 1

    def _map_dtype(self, fp_precision: str) -> str:
        """Map precision config to vLLM dtype string.

        Args:
            fp_precision: Config precision (float16, bfloat16, float32).

        Returns:
            vLLM dtype string.
        """
        mapping = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "auto": "auto",
        }
        return mapping.get(fp_precision, "auto")

    def _map_quantization(self, config: ExperimentConfig) -> str | None:
        """Map quantization config to vLLM quantization method.

        Args:
            config: Experiment configuration.

        Returns:
            vLLM quantization string or None.
        """
        # Check vLLM-specific quantization method first
        if config.vllm and config.vllm.quantization_method:
            return config.vllm.quantization_method

        quant = config.quantization_config

        # Check for BitsAndBytes 4-bit
        if quant.load_in_4bit:
            return "bitsandbytes"

        # Note: quant.quantization is a bool flag, not a method string
        # vLLM auto-detects quantization for GPTQ/AWQ models from model config
        return None

    def _build_engine_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build vLLM LLM() constructor kwargs from config.

        Combines shared ExperimentConfig params with vLLM-specific VLLMConfig params.
        VLLMConfig params take precedence where both are specified.

        Args:
            config: Experiment configuration (may include config.vllm).

        Returns:
            Dict of kwargs for vLLM LLM() constructor.
        """
        vllm_cfg: VLLMConfig | None = config.vllm

        # Base kwargs from shared config
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "dtype": self._map_dtype(config.fp_precision),
            "tensor_parallel_size": self._get_tensor_parallel_size(config),
            "trust_remote_code": True,
        }

        # Add quantization if specified
        quantization = self._map_quantization(config)
        if quantization:
            kwargs["quantization"] = quantization

        # Set seed if specified
        if config.random_seed is not None:
            kwargs["seed"] = config.random_seed

        # Configure max model length
        if config.max_input_tokens:
            kwargs["max_model_len"] = config.max_input_tokens + config.max_output_tokens

        # If no vLLM-specific config, return base kwargs
        if vllm_cfg is None:
            return kwargs

        # =================================================================
        # Apply vLLM-specific configuration
        # =================================================================

        # Memory & Batching
        if vllm_cfg.max_num_seqs != 256:  # Non-default
            kwargs["max_num_seqs"] = vllm_cfg.max_num_seqs
        if vllm_cfg.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = vllm_cfg.max_num_batched_tokens
        if vllm_cfg.gpu_memory_utilization != 0.9:  # Non-default
            kwargs["gpu_memory_utilization"] = vllm_cfg.gpu_memory_utilization
        if vllm_cfg.swap_space != 4.0:  # Non-default
            kwargs["swap_space"] = vllm_cfg.swap_space
        if vllm_cfg.cpu_offload_gb > 0:
            kwargs["cpu_offload_gb"] = vllm_cfg.cpu_offload_gb

        # KV Cache
        if vllm_cfg.enable_prefix_caching:
            kwargs["enable_prefix_caching"] = True
        if vllm_cfg.enable_chunked_prefill:
            kwargs["enable_chunked_prefill"] = True
        if vllm_cfg.kv_cache_dtype != "auto":
            kwargs["kv_cache_dtype"] = vllm_cfg.kv_cache_dtype
        if vllm_cfg.block_size != 16:  # Non-default
            kwargs["block_size"] = vllm_cfg.block_size

        # Context length (override shared config if specified)
        if vllm_cfg.max_model_len is not None:
            kwargs["max_model_len"] = vllm_cfg.max_model_len
        if vllm_cfg.max_seq_len_to_capture is not None:
            kwargs["max_seq_len_to_capture"] = vllm_cfg.max_seq_len_to_capture

        # Execution mode
        if vllm_cfg.enforce_eager:
            kwargs["enforce_eager"] = True

        # Parallelism
        if vllm_cfg.distributed_backend != "mp":
            kwargs["distributed_executor_backend"] = vllm_cfg.distributed_backend
        if vllm_cfg.disable_custom_all_reduce:
            kwargs["disable_custom_all_reduce"] = True

        # Load format
        if vllm_cfg.load_format != "auto":
            kwargs["load_format"] = vllm_cfg.load_format

        # Attention configuration
        if vllm_cfg.attention:
            attn = vllm_cfg.attention
            if attn.backend != "auto":
                # vLLM uses VLLM_ATTENTION_BACKEND env var or attention_backend arg
                kwargs["attention_backend"] = attn.backend
            if attn.disable_sliding_window:
                kwargs["disable_sliding_window"] = True

        # Speculative decoding
        if vllm_cfg.speculative and vllm_cfg.speculative.model:
            spec = vllm_cfg.speculative
            spec_config: dict[str, Any] = {
                "model": spec.model,
                "num_speculative_tokens": spec.num_tokens,
            }
            if spec.method != "ngram":
                spec_config["method"] = spec.method
            if spec.method == "ngram":
                if spec.ngram_min != 1:
                    spec_config["ngram_prompt_lookup_min"] = spec.ngram_min
                if spec.ngram_max is not None:
                    spec_config["ngram_prompt_lookup_max"] = spec.ngram_max
            if spec.draft_tp_size > 1:
                spec_config["draft_tensor_parallel_size"] = spec.draft_tp_size
            kwargs["speculative_config"] = spec_config

        # LoRA - auto-enable if adapter specified, or use explicit config
        if config.adapter:
            # Auto-enable LoRA when adapter is specified
            kwargs["enable_lora"] = True
            if vllm_cfg and vllm_cfg.lora:
                lora = vllm_cfg.lora
                kwargs["max_loras"] = lora.max_loras
                kwargs["max_lora_rank"] = lora.max_rank
                if lora.extra_vocab_size != 256:
                    kwargs["lora_extra_vocab_size"] = lora.extra_vocab_size
            else:
                # Sensible defaults for single adapter
                kwargs["max_loras"] = 1
                kwargs["max_lora_rank"] = 64
        elif vllm_cfg and vllm_cfg.lora and vllm_cfg.lora.enabled:
            # Explicit LoRA config without adapter (pre-configured engine)
            lora = vllm_cfg.lora
            kwargs["enable_lora"] = True
            kwargs["max_loras"] = lora.max_loras
            kwargs["max_lora_rank"] = lora.max_rank
            if lora.extra_vocab_size != 256:
                kwargs["lora_extra_vocab_size"] = lora.extra_vocab_size

        # Escape hatch: merge any extra kwargs
        if vllm_cfg.extra:
            kwargs.update(vllm_cfg.extra)

        return kwargs

    def _build_sampling_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build additional sampling kwargs from VLLMConfig.

        Args:
            config: Experiment configuration.

        Returns:
            Dict of additional kwargs for SamplingParams.
        """
        vllm_cfg: VLLMConfig | None = config.vllm
        if vllm_cfg is None:
            return {}

        kwargs: dict[str, Any] = {}

        # Advanced sampling params
        if vllm_cfg.best_of is not None and vllm_cfg.best_of > 1:
            kwargs["best_of"] = vllm_cfg.best_of
        if vllm_cfg.use_beam_search:
            kwargs["use_beam_search"] = True
            if vllm_cfg.length_penalty != 1.0:
                kwargs["length_penalty"] = vllm_cfg.length_penalty
        if vllm_cfg.logprobs is not None:
            kwargs["logprobs"] = vllm_cfg.logprobs
        if vllm_cfg.logit_bias:
            kwargs["logit_bias"] = vllm_cfg.logit_bias

        return kwargs

    def _create_lora_request(self, config: ExperimentConfig) -> Any:
        """Create vLLM LoRARequest for adapter inference.

        Args:
            config: Experiment configuration with adapter path.

        Returns:
            LoRARequest instance for generate() calls.

        Raises:
            BackendInitializationError: If adapter loading fails.
        """
        from vllm.lora.request import LoRARequest

        adapter_path = config.adapter
        assert adapter_path is not None  # Caller verified this

        logger.info(f"Creating LoRA request for adapter: {adapter_path}")

        # Generate a unique ID for this adapter
        # Using hash of path for consistency across runs
        lora_id = abs(hash(adapter_path)) % (10**6)

        return LoRARequest(
            lora_name=adapter_path,
            lora_int_id=lora_id,
            lora_path=adapter_path,
        )

    def _create_sampling_params(self, config: ExperimentConfig, SamplingParams: type) -> Any:
        """Create vLLM SamplingParams from config.

        Args:
            config: Experiment configuration.
            SamplingParams: vLLM SamplingParams class.

        Returns:
            Configured SamplingParams instance.
        """
        decoder = config.decoder_config

        params: dict[str, Any] = {
            "max_tokens": config.max_output_tokens,
        }

        # Min tokens
        if config.min_output_tokens:
            params["min_tokens"] = config.min_output_tokens

        # Temperature (0 = greedy in vLLM)
        if decoder.temperature is not None:
            params["temperature"] = decoder.temperature
            # If temperature is 0, explicitly disable sampling
            if decoder.temperature == 0:
                params["top_p"] = 1.0
                params["top_k"] = -1

        # Top-p
        if decoder.top_p is not None:
            params["top_p"] = decoder.top_p

        # Top-k (vLLM uses -1 to disable, HF uses 0)
        if decoder.top_k is not None:
            # Convert HF convention (0 = disabled) to vLLM (-1 = disabled)
            params["top_k"] = -1 if decoder.top_k == 0 else decoder.top_k

        # Min-p
        if decoder.min_p is not None:
            params["min_p"] = decoder.min_p

        # Repetition penalty
        if decoder.repetition_penalty is not None:
            params["repetition_penalty"] = decoder.repetition_penalty

        # Seed (per-request seed)
        if config.random_seed is not None:
            params["seed"] = config.random_seed

        # Add vLLM-specific sampling params (best_of, beam_search, logprobs, etc.)
        vllm_sampling_kwargs = self._build_sampling_kwargs(config)
        params.update(vllm_sampling_kwargs)

        return SamplingParams(**params)

    def _perform_warmup(self) -> None:
        """Perform warmup inference to trigger JIT compilation.

        vLLM has JIT compilation overhead on first inference. Running a
        warmup prompt before measurement ensures this doesn't affect results.
        """
        if self._warmup_done or self._llm is None:
            return

        logger.debug("Performing vLLM warmup inference...")
        warmup_prompt = "Hello"

        try:
            # Run single warmup inference
            from vllm import SamplingParams

            warmup_params = SamplingParams(max_tokens=1, temperature=0)
            self._llm.generate([warmup_prompt], warmup_params)
            self._warmup_done = True
            logger.debug("vLLM warmup complete")
        except Exception as e:
            logger.warning(f"vLLM warmup failed (non-fatal): {e}")

    def _get_tokenizer(self) -> Any:
        """Get tokenizer from vLLM engine (lazy initialization).

        Returns:
            HuggingFace tokenizer used by vLLM.
        """
        if self._tokenizer is None and self._llm is not None:
            try:
                # vLLM exposes tokenizer via get_tokenizer() method
                self._tokenizer = self._llm.get_tokenizer()
            except AttributeError:
                # Fallback for older vLLM versions
                self._tokenizer = self._llm.llm_engine.tokenizer.tokenizer
        return self._tokenizer

    def _truncate_prompts(self, prompts: list[str], max_input_tokens: int | None) -> list[str]:
        """Truncate prompts to max_input_tokens for consistent behaviour with PyTorch backend.

        vLLM doesn't automatically truncate prompts - it either accepts them fully
        or rejects them if they exceed max_model_len. This method ensures prompts
        are truncated to max_input_tokens, matching PyTorch backend behaviour.

        Args:
            prompts: List of prompt strings.
            max_input_tokens: Maximum tokens per prompt (None = no truncation).

        Returns:
            List of (possibly truncated) prompt strings.
        """
        if max_input_tokens is None:
            return prompts

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            logger.warning("Cannot truncate prompts: tokenizer not available")
            return prompts

        truncated: list[str] = []
        truncation_count = 0

        for prompt in prompts:
            # Tokenize the prompt (with special tokens to match vLLM's internal behaviour)
            tokens = tokenizer.encode(prompt, add_special_tokens=True)

            # Truncate if necessary
            if len(tokens) > max_input_tokens:
                tokens = tokens[:max_input_tokens]
                truncation_count += 1

            # Decode back to string, skipping special tokens so vLLM can add its own
            # This prevents double-counting of BOS/EOS tokens
            truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
            truncated.append(truncated_prompt)

        if truncation_count > 0:
            logger.debug(
                f"Truncated {truncation_count}/{len(prompts)} prompts to {max_input_tokens} tokens"
            )

        return truncated

    def run_inference(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference using vLLM.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with token counts and timing.

        Raises:
            BackendInferenceError: If inference fails.
        """
        if self._llm is None:
            raise BackendInferenceError("vLLM not initialized. Call initialize() first.")

        try:
            # Check if streaming mode is enabled for latency measurement
            if config.streaming:
                return self._run_streaming_inference(prompts, config)

            # Check if traffic simulation is enabled
            traffic_config = config.latency_simulation
            if traffic_config.enabled:
                return self._run_inference_with_traffic(prompts, config)

            return self._run_inference_batch(prompts, config)

        except Exception as e:
            raise BackendInferenceError(f"vLLM inference failed: {e}") from e

    def _run_inference_batch(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference on all prompts at once (no traffic simulation)."""
        # Truncate prompts to max_input_tokens (matches PyTorch backend behaviour)
        prompts = self._truncate_prompts(prompts, config.max_input_tokens)

        start_time = time.perf_counter()

        # Run inference (with optional LoRA adapter)
        outputs = self._llm.generate(
            prompts,
            self._sampling_params,
            lora_request=self._lora_request,
        )

        inference_time = time.perf_counter() - start_time

        return self._process_outputs(outputs, config, inference_time, len(prompts))

    def _run_streaming_inference(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Run inference with streaming for TTFT/ITL latency measurement.

        Enables detailed latency metrics by processing outputs token-by-token.
        Collects raw TTFT and ITL samples for late aggregation.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with latency_measurements containing raw samples.
        """
        import numpy as np

        # Truncate prompts to max_input_tokens (matches PyTorch backend behaviour)
        prompts = self._truncate_prompts(prompts, config.max_input_tokens)

        warmup_count = config.streaming_warmup_requests

        # Split into warmup and measurement prompts
        warmup_prompts = prompts[:warmup_count] if warmup_count > 0 else []
        measurement_prompts = prompts[warmup_count:]

        # Run warmup (results discarded from stats)
        if warmup_prompts:
            logger.info(f"Running {len(warmup_prompts)} streaming warmup requests...")
            self._llm.generate(warmup_prompts, self._sampling_params)
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
                ),
            )

        # Collect per-request timing data
        ttft_samples: list[float] = []
        token_timestamps_per_request: list[list[float]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []

        logger.info(f"Running streaming inference on {len(measurement_prompts)} prompts...")
        start_time = time.perf_counter()

        # Process each prompt individually to capture per-token timing
        # Note: This is less efficient than batch mode but enables ITL measurement
        with prompt_progress(
            total=len(measurement_prompts),
            desc="Streaming",
            is_main_process=True,
        ) as progress:
            for prompt in measurement_prompts:
                request_start = time.perf_counter()
                token_times: list[float] = []
                first_token_time: float | None = None

                # Run inference on single prompt
                outputs = self._llm.generate([prompt], self._sampling_params)

                # vLLM returns complete outputs - we extract timing from metrics if available
                if outputs:
                    output = outputs[0]
                    total_input_tokens += len(output.prompt_token_ids)

                    if output.outputs:
                        completion = output.outputs[0]
                        num_tokens = len(completion.token_ids)
                        total_output_tokens += num_tokens
                        output_texts.append(completion.text)

                        # Try to get TTFT from vLLM metrics
                        if hasattr(output, "metrics") and output.metrics is not None:
                            metrics = output.metrics
                            if hasattr(metrics, "time_to_first_token"):
                                first_token_time = metrics.time_to_first_token * 1000  # to ms
                            elif hasattr(metrics, "first_token_time"):
                                first_token_time = metrics.first_token_time * 1000

                        # If no TTFT from metrics, estimate from request time
                        if first_token_time is None:
                            request_end = time.perf_counter()
                            # Rough estimate: TTFT is a fraction of total time based on token count
                            # This is less accurate but provides some data
                            total_time_ms = (request_end - request_start) * 1000
                            if num_tokens > 0:
                                # Estimate TTFT as portion proportional to 1 token
                                first_token_time = total_time_ms / (num_tokens + 1)
                            else:
                                first_token_time = total_time_ms

                        ttft_samples.append(first_token_time)

                        # For ITL, estimate token times evenly distributed
                        # (vLLM batch mode doesn't provide per-token timestamps)
                        if num_tokens > 1:
                            request_end = time.perf_counter()
                            total_time_ms = (request_end - request_start) * 1000
                            decode_time_ms = total_time_ms - first_token_time

                            # Distribute decode time evenly across tokens
                            # token_times[0] = first token (TTFT), token_times[1:] = subsequent
                            token_times = [first_token_time]
                            time_per_token = (
                                decode_time_ms / (num_tokens - 1) if num_tokens > 1 else 0
                            )
                            for i in range(1, num_tokens):
                                token_times.append(first_token_time + (i * time_per_token))

                        token_timestamps_per_request.append(token_times)

                # Update progress
                progress.update(1, latency_ms=first_token_time)

        inference_time = time.perf_counter() - start_time

        # Calculate ITL from token timestamps using shared utility
        itl_full, itl_trimmed, excluded = collect_itl_measurements(token_timestamps_per_request)

        # Build latency measurements
        # Note: vLLM doesn't have true streaming API - we estimate ITL from per-request timing
        latency_measurements = LatencyMeasurements(
            ttft_ms=ttft_samples,
            itl_full_ms=itl_full,
            itl_trimmed_ms=itl_trimmed,
            request_count=len(measurement_prompts),
            total_output_tokens=total_output_tokens,
            excluded_tokens=excluded,
            streaming_mode=True,
            warmup_requests_excluded=warmup_count,
            measurement_method="proportional_estimate",  # ITL estimated, not true streaming
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

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            inference_time_sec=inference_time,
            time_to_first_token_ms=avg_ttft_ms,
            output_texts=output_texts if config.save_outputs else None,
            backend_metadata={
                "backend": "vllm",
                "streaming_mode": True,
                "continuous_batching": False,  # Streaming processes one at a time
                "num_prompts": len(measurement_prompts),
                "warmup_prompts": warmup_count,
                "ttft_samples": len(ttft_samples),
                "itl_samples": len(itl_trimmed),
            },
            latency_measurements=latency_measurements,
        )

    def _run_inference_with_traffic(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Run inference with MLPerf-style traffic simulation.

        Submits prompts in sub-batches with inter-arrival delays to simulate
        realistic request patterns (Poisson or constant arrivals).
        """
        from llm_energy_measure.core.traffic import TrafficGenerator

        # Truncate prompts to max_input_tokens (matches PyTorch backend behaviour)
        prompts = self._truncate_prompts(prompts, config.max_input_tokens)

        traffic_config = config.latency_simulation
        generator = TrafficGenerator(traffic_config, seed=config.random_seed)

        # Determine batch size for sub-batches
        batch_size = config.batching_options.batch_size or len(prompts)
        batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

        logger.info(
            f"vLLM traffic simulation: {len(batches)} batches, "
            f"mode={traffic_config.mode}, target_qps={traffic_config.target_qps}"
        )

        all_outputs: list[Any] = []
        start_time = time.perf_counter()

        with batch_progress(
            total=len(batches),
            desc="Batches",
            is_main_process=True,
        ) as progress:
            for batch_idx, batch in enumerate(batches):
                # Apply traffic delay (skip first batch)
                if batch_idx > 0:
                    delay = generator.wait_for_next_request()
                    if delay > 0.1:
                        logger.debug(f"Traffic delay: {delay:.3f}s before batch {batch_idx + 1}")

                # Run this batch (with optional LoRA adapter)
                batch_start = time.perf_counter()
                batch_outputs = self._llm.generate(
                    batch,
                    self._sampling_params,
                    lora_request=self._lora_request,
                )
                batch_latency_ms = (time.perf_counter() - batch_start) * 1000
                all_outputs.extend(batch_outputs)

                # Update progress
                progress.update(1, latency_ms=batch_latency_ms)

        inference_time = time.perf_counter() - start_time

        return self._process_outputs(
            all_outputs,
            config,
            inference_time,
            len(prompts),
            extra_metadata={"traffic_simulation": True, "num_batches": len(batches)},
        )

    def _process_outputs(
        self,
        outputs: list[Any],
        config: ExperimentConfig,
        inference_time: float,
        num_prompts: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Process vLLM outputs into BackendResult."""
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []
        ttft_values: list[float] = []

        for output in outputs:
            # Input tokens from prompt
            total_input_tokens += len(output.prompt_token_ids)

            # Output tokens from first completion
            if output.outputs:
                completion = output.outputs[0]
                total_output_tokens += len(completion.token_ids)
                output_texts.append(completion.text)

            # Extract TTFT if available (vLLM metrics)
            if hasattr(output, "metrics") and output.metrics is not None:
                metrics = output.metrics
                if hasattr(metrics, "time_to_first_token"):
                    ttft_values.append(metrics.time_to_first_token * 1000)
                elif hasattr(metrics, "first_token_time"):
                    ttft_values.append(metrics.first_token_time * 1000)

        total_tokens = total_input_tokens + total_output_tokens

        # Calculate average TTFT
        avg_ttft_ms: float | None = None
        if ttft_values:
            avg_ttft_ms = sum(ttft_values) / len(ttft_values)
            logger.debug(f"vLLM TTFT: avg={avg_ttft_ms:.2f}ms from {len(ttft_values)} samples")

        logger.info(
            f"vLLM inference complete: {num_prompts} prompts, "
            f"{total_tokens} tokens in {inference_time:.2f}s"
        )

        metadata: dict[str, Any] = {
            "backend": "vllm",
            "continuous_batching": True,
            "lora_adapter": self._config.adapter if self._config else None,
            "num_prompts": num_prompts,
            "ttft_samples": len(ttft_values) if ttft_values else 0,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            inference_time_sec=inference_time,
            time_to_first_token_ms=avg_ttft_ms,
            output_texts=output_texts if config.save_outputs else None,
            backend_metadata=metadata,
        )

    def cleanup(self) -> None:
        """Release vLLM resources.

        Note: vLLM's LLM class manages its own cleanup. We just clear references.
        """
        if self._llm is not None:
            logger.debug("Cleaning up vLLM resources")
            # vLLM doesn't have an explicit cleanup method
            # Setting to None allows garbage collection
            self._llm = None
            self._sampling_params = None
            self._warmup_done = False

    def get_model_info(self) -> ModelInfo:
        """Return model metadata.

        Returns:
            ModelInfo with model details.
        """
        from llm_energy_measure.domain.model_info import ModelInfo

        if self._config is None:
            return ModelInfo(
                name="unknown",
                num_parameters=0,
                num_layers=0,
                hidden_size=0,
                num_attention_heads=0,
                vocab_size=0,
                model_type="unknown",
                torch_dtype="float16",
            )

        # Get parameter count from model if available
        param_count = 0
        if self._llm is not None:
            try:
                # vLLM's LLM exposes the model
                model = self._llm.llm_engine.model_executor.driver_worker.model_runner.model
                param_count = sum(p.numel() for p in model.parameters())
            except Exception:
                # Fall back to config or estimate
                pass

        return ModelInfo(
            name=self._config.model_name,
            num_parameters=param_count,
            num_layers=0,
            hidden_size=0,
            num_attention_heads=0,
            vocab_size=0,
            model_type="unknown",
            torch_dtype=self._config.fp_precision,
        )

    def get_supported_params(self) -> set[str]:
        """Return parameters supported by vLLM backend."""
        return _SUPPORTED_PARAMS.copy()

    def validate_config(self, config: ExperimentConfig) -> list[ConfigWarning]:
        """Validate config compatibility with vLLM.

        Args:
            config: Configuration to validate.

        Returns:
            List of warnings for incompatible or semantically different params.
        """
        warnings: list[ConfigWarning] = []

        # Check for unsupported params that require warnings
        quant = config.quantization_config
        if quant.load_in_8bit:
            warnings.append(
                ConfigWarning(
                    param="quantization.load_in_8bit",
                    message=_UNSUPPORTED_WITH_WARNING["quantization.load_in_8bit"],
                    severity="error",
                    suggestion="Use load_in_4bit=true or a pre-quantized model",
                )
            )

        decoder = config.decoder_config
        if decoder.no_repeat_ngram_size and decoder.no_repeat_ngram_size > 0:
            warnings.append(
                ConfigWarning(
                    param="decoder.no_repeat_ngram_size",
                    message=_UNSUPPORTED_WITH_WARNING["decoder.no_repeat_ngram_size"],
                    severity="warning",
                )
            )

        # Inform about semantic differences
        batching = config.batching_options
        if batching.strategy in ("static", "sorted_static"):
            warnings.append(
                ConfigWarning(
                    param="batching.strategy",
                    message=(
                        f"Strategy '{batching.strategy}' requested but vLLM uses "
                        "continuous batching. batch_size becomes max_num_seqs hint."
                    ),
                    severity="warning",
                )
            )

        return warnings
