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
)
from llm_energy_measure.exceptions import (
    BackendInferenceError,
    BackendInitializationError,
    BackendNotAvailableError,
)

if TYPE_CHECKING:
    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.domain.model_info import ModelInfo


# Parameters supported by vLLM backend
_SUPPORTED_PARAMS: set[str] = {
    # Core
    "model_name",
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

            # Determine tensor parallelism
            tensor_parallel_size = self._get_tensor_parallel_size(config)

            # Map precision
            dtype = self._map_dtype(config.fp_precision)

            # Determine quantization
            quantization = self._map_quantization(config)

            logger.info(
                f"Initializing vLLM: model={config.model_name}, "
                f"dtype={dtype}, tp={tensor_parallel_size}, quant={quantization}"
            )

            # Create LLM instance
            llm_kwargs: dict[str, Any] = {
                "model": config.model_name,
                "dtype": dtype,
                "tensor_parallel_size": tensor_parallel_size,
                "trust_remote_code": True,
            }

            # Add quantization if specified
            if quantization:
                llm_kwargs["quantization"] = quantization

            # Set seed if specified
            if config.random_seed is not None:
                llm_kwargs["seed"] = config.random_seed

            # Configure max model length if specified
            if config.max_input_tokens:
                llm_kwargs["max_model_len"] = config.max_input_tokens + config.max_output_tokens

            self._llm = LLM(**llm_kwargs)

            # Create sampling params
            self._sampling_params = self._create_sampling_params(config, SamplingParams)

            logger.info(f"vLLM model loaded: {config.model_name}")

            # Perform warmup (JIT compilation happens on first inference)
            self._perform_warmup()

        except Exception as e:
            raise BackendInitializationError(
                f"Failed to initialize vLLM with model '{config.model_name}': {e}"
            ) from e

    def _get_tensor_parallel_size(self, config: ExperimentConfig) -> int:
        """Determine tensor parallelism size from config.

        Args:
            config: Experiment configuration.

        Returns:
            Number of GPUs for tensor parallelism.
        """
        sharding = config.sharding_config

        # Check if tensor parallelism is explicitly requested
        if sharding.strategy == "tensor_parallel":
            if sharding.num_shards:
                return sharding.num_shards
            # Use all available GPUs
            import torch

            return torch.cuda.device_count() if torch.cuda.is_available() else 1

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
        quant = config.quantization_config

        # Check for BitsAndBytes 4-bit
        if quant.load_in_4bit:
            return "bitsandbytes"

        # Note: quant.quantization is a bool flag, not a method string
        # vLLM auto-detects quantization for GPTQ/AWQ models from model config
        return None

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
            # Check if traffic simulation is enabled
            traffic_config = config.latency_simulation
            if traffic_config.enabled:
                return self._run_inference_with_traffic(prompts, config)

            return self._run_inference_batch(prompts, config)

        except Exception as e:
            raise BackendInferenceError(f"vLLM inference failed: {e}") from e

    def _run_inference_batch(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference on all prompts at once (no traffic simulation)."""
        start_time = time.perf_counter()

        # Run inference
        outputs = self._llm.generate(prompts, self._sampling_params)

        inference_time = time.perf_counter() - start_time

        return self._process_outputs(outputs, config, inference_time, len(prompts))

    def _run_inference_with_traffic(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Run inference with MLPerf-style traffic simulation.

        Submits prompts in sub-batches with inter-arrival delays to simulate
        realistic request patterns (Poisson or constant arrivals).
        """
        from llm_energy_measure.core.traffic import TrafficGenerator

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

        for batch_idx, batch in enumerate(batches):
            # Apply traffic delay (skip first batch)
            if batch_idx > 0:
                delay = generator.wait_for_next_request()
                if delay > 0.1:
                    logger.debug(f"Traffic delay: {delay:.3f}s before batch {batch_idx + 1}")

            # Run this batch
            batch_outputs = self._llm.generate(batch, self._sampling_params)
            all_outputs.extend(batch_outputs)

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
