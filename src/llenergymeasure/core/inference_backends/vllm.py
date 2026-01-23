"""vLLM inference backend.

High-performance inference backend using vLLM with PagedAttention
and continuous batching. Provides native tensor parallelism support
for multi-GPU inference.

Backend-Native Architecture:
- All vLLM-specific params read from config.vllm
- Tier 1 (universal) params from top-level config (model_name, decoder, etc.)
- Tier 2 (vLLM-specific) params from config.vllm section

Key differences from PyTorch backend:
- vLLM manages its own model loading and distribution (not Accelerate)
- Continuous batching - no external batch_size concept
- Uses SamplingParams for decoder configuration
- Supports PagedAttention for efficient KV cache management

Config Mapping:
    Tier 1 (Universal):
        - model_name → LLM(model=...)
        - fp_precision → dtype
        - decoder.temperature, decoder.top_p → SamplingParams(...)
        - max_output_tokens → SamplingParams(max_tokens=...)

    Tier 2 (vLLM-specific from config.vllm):
        - tensor_parallel_size → tensor parallelism
        - max_num_seqs → continuous batching limit
        - quantization → pre-quantized models (awq, gptq, etc.)
        - min_p → SamplingParams extension (top_k moved to universal decoder)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from llenergymeasure.core.inference_backends.protocols import (
    BackendResult,
    BackendRuntime,
    ConfigWarning,
    CudaManagement,
    LaunchMode,
    RuntimeCapabilities,
)
from llenergymeasure.core.inference_backends.shared import (
    create_precision_metadata,
)
from llenergymeasure.domain.metrics import (
    LatencyMeasurementMode,
    LatencyMeasurements,
    collect_itl_measurements,
)
from llenergymeasure.exceptions import (
    BackendInferenceError,
    BackendInitializationError,
    BackendNotAvailableError,
)
from llenergymeasure.progress import batch_progress, prompt_progress

if TYPE_CHECKING:
    from llenergymeasure.config.backend_configs import VLLMConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.model_info import ModelInfo


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

    Configuration (Backend-Native Architecture):
        Tier 1 (Universal) - from top-level config:
            - model_name, adapter, fp_precision
            - decoder.temperature, decoder.do_sample, decoder.top_p, decoder.repetition_penalty
            - max_input_tokens, max_output_tokens, min_output_tokens
            - streaming, streaming_warmup_requests
            - traffic_simulation.*

        Tier 2 (vLLM-specific) - from config.vllm:
            - tensor_parallel_size, pipeline_parallel_size
            - max_num_seqs, max_num_batched_tokens
            - gpu_memory_utilization, swap_space
            - enable_prefix_caching, kv_cache_dtype
            - quantization (awq, gptq, fp8, marlin, etc.)
            - min_p (sampling extension - top_k is now universal in decoder)
            - speculative.*, lora.*
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
            raise BackendNotAvailableError("vllm", install_hint="pip install lem[vllm]")

        self._config = config
        self._runtime = runtime

        try:
            from vllm import LLM, SamplingParams

            # Build engine kwargs from config (Tier 1 + Tier 2)
            llm_kwargs = self._build_engine_kwargs(config)

            # Log configuration
            vllm_cfg = config.vllm
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

            if vllm_cfg:
                logger.info(
                    f"vLLM config: max_num_seqs={vllm_cfg.max_num_seqs}, "
                    f"gpu_mem={vllm_cfg.gpu_memory_utilization}, "
                    f"kv_dtype={vllm_cfg.kv_cache_dtype}"
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
        manages its own CUDA context.

        Priority:
        1. config.vllm.tensor_parallel_size (Tier 2 - explicit)
        2. len(config.gpus) (Tier 1 - from GPU list)
        3. CUDA_VISIBLE_DEVICES environment variable
        4. Default to 1

        Args:
            config: Experiment configuration.

        Returns:
            Number of GPUs for tensor parallelism.
        """
        import os

        vllm_cfg = config.vllm

        # Check vLLM-specific config first (Tier 2)
        if vllm_cfg is not None and vllm_cfg.tensor_parallel_size > 1:
            return vllm_cfg.tensor_parallel_size

        # Use gpu_list if provided (Tier 1)
        if config.gpus and len(config.gpus) > 1:
            return len(config.gpus)

        # Fall back to CUDA_VISIBLE_DEVICES (no torch.cuda.* calls!)
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and "," in cuda_visible:
            return len(cuda_visible.split(","))

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

    def _build_engine_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build vLLM LLM() constructor kwargs from config.

        Combines Tier 1 (universal) params with Tier 2 (vLLM-specific) params.
        Tier 2 params take precedence where both are specified.

        Args:
            config: Experiment configuration (may include config.vllm).

        Returns:
            Dict of kwargs for vLLM LLM() constructor.
        """
        vllm_cfg: VLLMConfig | None = config.vllm

        # Tier 1: Base kwargs from universal config
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "dtype": self._map_dtype(config.fp_precision),
            "tensor_parallel_size": self._get_tensor_parallel_size(config),
            "trust_remote_code": True,
        }

        # Set seed if specified (Tier 1)
        if config.random_seed is not None:
            kwargs["seed"] = config.random_seed

        # Configure max model length (Tier 1 token limits)
        if config.max_input_tokens:
            kwargs["max_model_len"] = config.max_input_tokens + config.max_output_tokens

        # If no vLLM-specific config, return base kwargs
        if vllm_cfg is None:
            return kwargs

        # =================================================================
        # Tier 2: Apply vLLM-specific configuration
        # =================================================================

        # Parallelism (vLLM-native)
        if vllm_cfg.tensor_parallel_size > 1:
            kwargs["tensor_parallel_size"] = vllm_cfg.tensor_parallel_size
        if vllm_cfg.pipeline_parallel_size > 1:
            kwargs["pipeline_parallel_size"] = vllm_cfg.pipeline_parallel_size
        if vllm_cfg.distributed_backend != "mp":
            kwargs["distributed_executor_backend"] = vllm_cfg.distributed_backend
        if vllm_cfg.disable_custom_all_reduce:
            kwargs["disable_custom_all_reduce"] = True

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

        # Context length (override Tier 1 if specified)
        if vllm_cfg.max_model_len is not None:
            kwargs["max_model_len"] = vllm_cfg.max_model_len
        if vllm_cfg.max_seq_len_to_capture is not None:
            kwargs["max_seq_len_to_capture"] = vllm_cfg.max_seq_len_to_capture

        # Execution mode
        if vllm_cfg.enforce_eager:
            kwargs["enforce_eager"] = True

        # Quantization (vLLM-native)
        if vllm_cfg.quantization is not None:
            kwargs["quantization"] = vllm_cfg.quantization

        # Load format
        if vllm_cfg.load_format != "auto":
            kwargs["load_format"] = vllm_cfg.load_format

        # Attention configuration
        if vllm_cfg.attention:
            attn = vllm_cfg.attention
            if attn.backend != "auto":
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
                if spec.prompt_lookup_min != 1:
                    spec_config["ngram_prompt_lookup_min"] = spec.prompt_lookup_min
                if spec.prompt_lookup_max is not None:
                    spec_config["ngram_prompt_lookup_max"] = spec.prompt_lookup_max
            if spec.draft_tp_size > 1:
                spec_config["draft_tensor_parallel_size"] = spec.draft_tp_size
            kwargs["speculative_config"] = spec_config

        # LoRA - auto-enable if adapter specified, or use explicit config
        if config.adapter:
            kwargs["enable_lora"] = True
            if vllm_cfg.lora:
                lora = vllm_cfg.lora
                kwargs["max_loras"] = lora.max_loras
                kwargs["max_lora_rank"] = lora.max_rank
                if lora.extra_vocab_size != 256:
                    kwargs["lora_extra_vocab_size"] = lora.extra_vocab_size
            else:
                kwargs["max_loras"] = 1
                kwargs["max_lora_rank"] = 64
        elif vllm_cfg.lora and vllm_cfg.lora.enabled:
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

    def _create_lora_request(self, config: ExperimentConfig) -> Any:
        """Create vLLM LoRARequest for adapter inference.

        Args:
            config: Experiment configuration with adapter path.

        Returns:
            LoRARequest instance for generate() calls.
        """
        from vllm.lora.request import LoRARequest

        adapter_path = config.adapter
        assert adapter_path is not None

        logger.info(f"Creating LoRA request for adapter: {adapter_path}")

        lora_id = abs(hash(adapter_path)) % (10**6)

        return LoRARequest(
            lora_name=adapter_path,
            lora_int_id=lora_id,
            lora_path=adapter_path,
        )

    def _create_sampling_params(self, config: ExperimentConfig, SamplingParams: type) -> Any:
        """Create vLLM SamplingParams from config.

        Combines Tier 1 (universal decoder) with Tier 2 (vLLM-specific) params.

        Args:
            config: Experiment configuration.
            SamplingParams: vLLM SamplingParams class.

        Returns:
            Configured SamplingParams instance.
        """
        decoder = config.decoder
        vllm_cfg = config.vllm

        params: dict[str, Any] = {
            "max_tokens": config.max_output_tokens,
        }

        # Min tokens (Tier 1)
        if config.min_output_tokens:
            params["min_tokens"] = config.min_output_tokens

        # Tier 1: Universal decoder params
        # Temperature (0 = greedy in vLLM)
        if decoder.temperature is not None:
            params["temperature"] = decoder.temperature
            if decoder.temperature == 0:
                params["top_p"] = 1.0
                params["top_k"] = -1

        # Top-p (Tier 1)
        if decoder.top_p is not None:
            params["top_p"] = decoder.top_p

        # Repetition penalty (Tier 1)
        if decoder.repetition_penalty is not None:
            params["repetition_penalty"] = decoder.repetition_penalty

        # Top-k (Tier 1 - universal, but vLLM uses -1 to disable instead of 0)
        # Convert from universal convention (0 = disabled) to vLLM convention (-1 = disabled)
        if decoder.top_k > 0:
            params["top_k"] = decoder.top_k
        else:
            params["top_k"] = -1  # vLLM disabled convention

        # Tier 2: vLLM-specific sampling extensions
        if vllm_cfg is not None:
            # Min-p
            if vllm_cfg.min_p > 0.0:
                params["min_p"] = vllm_cfg.min_p

            # Advanced sampling (vLLM-specific)
            if vllm_cfg.best_of is not None and vllm_cfg.best_of > 1:
                params["best_of"] = vllm_cfg.best_of
            if vllm_cfg.logprobs is not None:
                params["logprobs"] = vllm_cfg.logprobs
            if vllm_cfg.logit_bias:
                params["logit_bias"] = vllm_cfg.logit_bias

        # Seed (per-request seed)
        if config.random_seed is not None:
            params["seed"] = config.random_seed

        return SamplingParams(**params)

    def _perform_warmup(self) -> None:
        """Perform warmup inference to trigger JIT compilation."""
        if self._warmup_done or self._llm is None:
            return

        logger.debug("Performing vLLM warmup inference...")
        warmup_prompt = "Hello"

        try:
            from vllm import SamplingParams

            warmup_params = SamplingParams(max_tokens=1, temperature=0)
            self._llm.generate([warmup_prompt], warmup_params)
            self._warmup_done = True
            logger.debug("vLLM warmup complete")
        except Exception as e:
            logger.warning(f"vLLM warmup failed (non-fatal): {e}")

    def _get_tokenizer(self) -> Any:
        """Get tokenizer from vLLM engine (lazy initialization)."""
        if self._tokenizer is None and self._llm is not None:
            try:
                self._tokenizer = self._llm.get_tokenizer()
            except AttributeError:
                self._tokenizer = self._llm.llm_engine.tokenizer.tokenizer
        return self._tokenizer

    def _truncate_prompts(self, prompts: list[str], max_input_tokens: int | None) -> list[str]:
        """Truncate prompts to max_input_tokens for consistent behaviour with PyTorch backend."""
        if max_input_tokens is None:
            return prompts

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            logger.warning("Cannot truncate prompts: tokenizer not available")
            return prompts

        truncated: list[str] = []
        truncation_count = 0

        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=True)

            if len(tokens) > max_input_tokens:
                tokens = tokens[:max_input_tokens]
                truncation_count += 1

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
            traffic_config = config.traffic_simulation
            if traffic_config.enabled:
                return self._run_inference_with_traffic(prompts, config)

            return self._run_inference_batch(prompts, config)

        except Exception as e:
            raise BackendInferenceError(f"vLLM inference failed: {e}") from e

    def _run_inference_batch(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference on all prompts at once (no traffic simulation)."""
        prompts = self._truncate_prompts(prompts, config.max_input_tokens)

        start_time = time.perf_counter()

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
        """Run inference with streaming for TTFT/ITL latency measurement."""
        import numpy as np

        prompts = self._truncate_prompts(prompts, config.max_input_tokens)

        warmup_count = config.streaming_warmup_requests

        warmup_prompts = prompts[:warmup_count] if warmup_count > 0 else []
        measurement_prompts = prompts[warmup_count:]

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
                precision_metadata=create_precision_metadata(config, self.name),
            )

        ttft_samples: list[float] = []
        token_timestamps_per_request: list[list[float]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []

        logger.info(f"Running streaming inference on {len(measurement_prompts)} prompts...")
        start_time = time.perf_counter()

        with prompt_progress(
            total=len(measurement_prompts),
            desc="Streaming",
            is_main_process=True,
        ) as progress:
            for prompt in measurement_prompts:
                request_start = time.perf_counter()
                token_times: list[float] = []
                first_token_time: float | None = None

                outputs = self._llm.generate([prompt], self._sampling_params)

                if outputs:
                    output = outputs[0]
                    total_input_tokens += len(output.prompt_token_ids)

                    if output.outputs:
                        completion = output.outputs[0]
                        num_tokens = len(completion.token_ids)
                        total_output_tokens += num_tokens
                        output_texts.append(completion.text)

                        if hasattr(output, "metrics") and output.metrics is not None:
                            metrics = output.metrics
                            if hasattr(metrics, "time_to_first_token"):
                                first_token_time = metrics.time_to_first_token * 1000
                            elif hasattr(metrics, "first_token_time"):
                                first_token_time = metrics.first_token_time * 1000

                        if first_token_time is None:
                            request_end = time.perf_counter()
                            total_time_ms = (request_end - request_start) * 1000
                            if num_tokens > 0:
                                first_token_time = total_time_ms / (num_tokens + 1)
                            else:
                                first_token_time = total_time_ms

                        ttft_samples.append(first_token_time)

                        if num_tokens > 1:
                            request_end = time.perf_counter()
                            total_time_ms = (request_end - request_start) * 1000
                            decode_time_ms = total_time_ms - first_token_time

                            token_times = [first_token_time]
                            time_per_token = (
                                decode_time_ms / (num_tokens - 1) if num_tokens > 1 else 0
                            )
                            for i in range(1, num_tokens):
                                token_times.append(first_token_time + (i * time_per_token))

                        token_timestamps_per_request.append(token_times)

                progress.update(1, latency_ms=first_token_time)

        inference_time = time.perf_counter() - start_time

        itl_full, itl_trimmed, excluded = collect_itl_measurements(token_timestamps_per_request)

        latency_measurements = LatencyMeasurements(
            ttft_ms=ttft_samples,
            itl_full_ms=itl_full,
            itl_trimmed_ms=itl_trimmed,
            request_count=len(measurement_prompts),
            total_output_tokens=total_output_tokens,
            excluded_tokens=excluded,
            streaming_mode=True,
            warmup_requests_excluded=warmup_count,
            measurement_mode=LatencyMeasurementMode.PROPORTIONAL_ESTIMATE,
        )

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
                "continuous_batching": False,
                "num_prompts": len(measurement_prompts),
                "warmup_prompts": warmup_count,
                "ttft_samples": len(ttft_samples),
                "itl_samples": len(itl_trimmed),
            },
            latency_measurements=latency_measurements,
            precision_metadata=create_precision_metadata(config, self.name),
        )

    def _run_inference_with_traffic(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Run inference with MLPerf-style traffic simulation."""
        from llenergymeasure.core.traffic import TrafficGenerator

        prompts = self._truncate_prompts(prompts, config.max_input_tokens)

        traffic_config = config.traffic_simulation
        generator = TrafficGenerator(traffic_config, seed=config.random_seed)

        # vLLM uses continuous batching - we submit in small batches to simulate arrivals
        # Use max_num_seqs from vLLM config as batch hint, or default to all prompts
        vllm_cfg = config.vllm
        batch_size = vllm_cfg.max_num_seqs if vllm_cfg else len(prompts)
        batch_size = min(batch_size, len(prompts))  # Don't exceed prompt count

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
                if batch_idx > 0:
                    delay = generator.wait_for_next_request()
                    if delay > 0.1:
                        logger.debug(f"Traffic delay: {delay:.3f}s before batch {batch_idx + 1}")

                batch_start = time.perf_counter()
                batch_outputs = self._llm.generate(
                    batch,
                    self._sampling_params,
                    lora_request=self._lora_request,
                )
                batch_latency_ms = (time.perf_counter() - batch_start) * 1000
                all_outputs.extend(batch_outputs)

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
            total_input_tokens += len(output.prompt_token_ids)

            if output.outputs:
                completion = output.outputs[0]
                total_output_tokens += len(completion.token_ids)
                output_texts.append(completion.text)

            if hasattr(output, "metrics") and output.metrics is not None:
                metrics = output.metrics
                if hasattr(metrics, "time_to_first_token"):
                    ttft_values.append(metrics.time_to_first_token * 1000)
                elif hasattr(metrics, "first_token_time"):
                    ttft_values.append(metrics.first_token_time * 1000)

        total_tokens = total_input_tokens + total_output_tokens

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
            precision_metadata=create_precision_metadata(config, self.name),
        )

    def cleanup(self) -> None:
        """Release vLLM resources."""
        if self._llm is not None:
            logger.debug("Cleaning up vLLM resources")
            self._llm = None
            self._sampling_params = None
            self._warmup_done = False

    def get_model_info(self) -> ModelInfo:
        """Return model metadata."""
        from llenergymeasure.domain.model_info import ModelInfo

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

        param_count = 0
        if self._llm is not None:
            try:
                model = self._llm.llm_engine.model_executor.driver_worker.model_runner.model
                param_count = sum(p.numel() for p in model.parameters())
            except Exception:
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

    def validate_config(self, config: ExperimentConfig) -> list[ConfigWarning]:
        """Validate config compatibility with vLLM.

        Args:
            config: Configuration to validate.

        Returns:
            List of warnings for incompatible or semantically different params.
        """
        warnings: list[ConfigWarning] = []

        vllm_cfg = config.vllm
        if vllm_cfg is None:
            return warnings

        # Warn about potential memory issues
        if vllm_cfg.gpu_memory_utilization > 0.95:
            warnings.append(
                ConfigWarning(
                    field="vllm.gpu_memory_utilization",
                    message=(
                        f"gpu_memory_utilization={vllm_cfg.gpu_memory_utilization} is very high. "
                        "May cause OOM errors. Consider 0.9 or lower."
                    ),
                    severity="warning",
                )
            )

        # Warn about speculative decoding requirements
        if vllm_cfg.speculative and vllm_cfg.speculative.model and vllm_cfg.enforce_eager:
            warnings.append(
                ConfigWarning(
                    field="vllm.speculative",
                    message=(
                        "Speculative decoding with enforce_eager=true may have reduced "
                        "performance. CUDA graphs improve speculation efficiency."
                    ),
                    severity="info",
                )
            )

        # Warn about LoRA + tensor parallelism
        if (
            config.adapter
            and vllm_cfg.tensor_parallel_size > 1
            and (not vllm_cfg.lora or vllm_cfg.lora.max_loras < 1)
        ):
            warnings.append(
                ConfigWarning(
                    field="vllm.lora",
                    message=(
                        "LoRA adapter with tensor_parallel_size > 1 requires explicit "
                        "vllm.lora configuration for optimal performance."
                    ),
                    severity="info",
                )
            )

        return warnings
