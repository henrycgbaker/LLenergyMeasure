"""TensorRT-LLM inference backend.

High-performance inference backend using NVIDIA TensorRT-LLM with compiled
execution plans. Supports both pre-compiled engines and on-demand compilation
from HuggingFace checkpoints.

Key features:
- Compiled inference plans optimised for specific GPU + configuration
- Inflight batching for optimal throughput
- Native tensor/pipeline parallelism
- FP8/INT8/INT4 quantization support

Config Mapping (backend-native architecture):
- model_name → HF checkpoint for engine building
- fp_precision → Build-time dtype
- tensorrt.tp_size → Tensor parallel size
- tensorrt.pp_size → Pipeline parallel size
- tensorrt.engine_path → Pre-compiled engine (optional)
- tensorrt.quantization → Quantization method during build
- decoder.top_k → Top-k sampling (universal, moved from tensorrt.top_k)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
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
    CORE_SUPPORTED_PARAMS,
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
from llenergymeasure.progress import prompt_progress

if TYPE_CHECKING:
    from llenergymeasure.config.backend_configs import TensorRTConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.model_info import ModelInfo


# Default cache directory for compiled engines
DEFAULT_ENGINE_CACHE_DIR = Path.home() / ".cache" / "lem" / "tensorrt-engines"

# Parameters supported by TensorRT backend
# Extends core params with TensorRT-specific options
# Note: TensorRT does NOT support LoRA adapters
_SUPPORTED_PARAMS: set[str] = set(CORE_SUPPORTED_PARAMS) | {
    # All tensorrt.* section params
    "tensorrt.engine_path",
    "tensorrt.force_rebuild",
    "tensorrt.engine_cache_dir",
    "tensorrt.max_batch_size",
    "tensorrt.max_input_len",
    "tensorrt.max_output_len",
    "tensorrt.builder_opt_level",
    "tensorrt.strongly_typed",
    "tensorrt.multiple_profiles",
    "tensorrt.tp_size",
    "tensorrt.pp_size",
    "tensorrt.quantization",
    "tensorrt.calibration",
    "tensorrt.kv_cache_type",
    "tensorrt.enable_chunked_context",
    "tensorrt.enable_kv_cache_reuse",
    "tensorrt.gpu_memory_utilization",
    "tensorrt.max_num_tokens",
    # Note: top_k is now universal in decoder config (decoder.top_k)
    "tensorrt.draft_model",
    "tensorrt.num_draft_tokens",
}

# Parameters that require warnings (informational for users who might expect these)
# Note: These are documented limitations, not config validation errors
_UNSUPPORTED_FEATURES: dict[str, str] = {
    "min_p": "min_p not supported by TensorRT-LLM. Use top_k or top_p instead.",
    "no_repeat_ngram_size": (
        "no_repeat_ngram_size not supported by TensorRT-LLM. Use repetition_penalty instead."
    ),
    "bitsandbytes": (
        "BitsAndBytes quantization not supported. "
        "Use tensorrt.quantization=int8_sq, int4_awq, or int4_gptq."
    ),
}


def _check_tensorrt_available() -> bool:
    """Check if TensorRT-LLM is installed and importable."""
    try:
        import tensorrt_llm  # noqa: F401

        return True
    except ImportError:
        return False


def _get_tensorrt_version() -> str:
    """Get TensorRT-LLM version string."""
    try:
        import tensorrt_llm

        return f"tensorrt_llm={tensorrt_llm.__version__}"
    except (ImportError, AttributeError):
        return "tensorrt_llm=unknown"


def _get_gpu_architecture() -> str:
    """Get GPU compute capability string for cache key.

    Delegates to SSOT in gpu_info module.
    """
    from llenergymeasure.core.gpu_info import get_gpu_architecture

    return get_gpu_architecture(device_index=0)


class EngineCacheManager:
    """Manages TensorRT engine caching.

    Engines are cached based on a composite key including:
    - Model name and revision
    - GPU architecture
    - TensorRT-LLM version
    - Build configuration (dtype, TP size, max lengths, quantization)
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialise cache manager.

        Args:
            cache_dir: Directory for engine cache. Defaults to ~/.cache/lem/tensorrt-engines/
        """
        self.cache_dir = cache_dir or DEFAULT_ENGINE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, config: ExperimentConfig) -> str:
        """Generate cache key from configuration.

        Args:
            config: Experiment configuration.

        Returns:
            Hash string uniquely identifying this engine configuration.
        """
        trt_cfg: TensorRTConfig | None = config.tensorrt

        key_components = {
            "model": config.model_name,
            "dtype": config.fp_precision,
            "gpu_arch": _get_gpu_architecture(),
            "trt_version": _get_tensorrt_version(),
            "max_input": config.max_input_tokens,
            "max_output": config.max_output_tokens,
        }

        if trt_cfg:
            key_components.update(
                {
                    "max_batch": trt_cfg.max_batch_size,
                    "tp_size": trt_cfg.tp_size,
                    "pp_size": trt_cfg.pp_size,
                    "quant_method": trt_cfg.quantization,
                    "builder_opt": trt_cfg.builder_opt_level,
                }
            )

        # Create deterministic hash
        key_str = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get_engine_path(self, config: ExperimentConfig) -> Path:
        """Get path for cached engine.

        Args:
            config: Experiment configuration.

        Returns:
            Path to engine directory.
        """
        cache_key = self.get_cache_key(config)
        model_name_safe = config.model_name.replace("/", "_")
        return self.cache_dir / f"{model_name_safe}_{cache_key}"

    def has_cached_engine(self, config: ExperimentConfig) -> bool:
        """Check if a valid cached engine exists.

        Args:
            config: Experiment configuration.

        Returns:
            True if cached engine exists and is valid.
        """
        engine_path = self.get_engine_path(config)
        if not engine_path.exists():
            return False

        # Check for required files
        required_files = ["config.json", "rank0.engine"]
        return all((engine_path / f).exists() for f in required_files)

    def save_metadata(self, config: ExperimentConfig, build_time_sec: float) -> None:
        """Save engine metadata.

        Args:
            config: Experiment configuration.
            build_time_sec: Time taken to build the engine.
        """
        engine_path = self.get_engine_path(config)
        metadata = {
            "model_name": config.model_name,
            "cache_key": self.get_cache_key(config),
            "build_time_sec": build_time_sec,
            "trt_version": _get_tensorrt_version(),
            "gpu_arch": _get_gpu_architecture(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(engine_path / "cache_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


class TensorRTBackend:
    """TensorRT-LLM inference backend with compiled execution plans.

    TensorRT-LLM provides:
    - Compiled inference plans optimised for specific GPU + configuration
    - Inflight batching for optimal throughput
    - Native tensor parallelism for multi-GPU inference
    - FP8/INT8/INT4 quantization

    Note: TensorRT-LLM manages its own model loading and CUDA context.
    The BackendRuntime.accelerator field is ignored.
    """

    def __init__(self) -> None:
        """Initialise backend (model loaded lazily in initialize())."""
        self._executor: Any = None
        self._tokenizer: Any = None
        self._config: ExperimentConfig | None = None
        self._runtime: BackendRuntime | None = None
        self._model_info: ModelInfo | None = None
        self._warmup_done: bool = False
        self._cache_manager: EngineCacheManager | None = None
        self._engine_path: Path | None = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "tensorrt"

    @property
    def version(self) -> str:
        """Backend version string."""
        return _get_tensorrt_version()

    def is_available(self) -> bool:
        """Check if TensorRT-LLM is installed."""
        return _check_tensorrt_available()

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        """Return TensorRT-LLM runtime requirements.

        TensorRT-LLM manages its own CUDA context and uses MPI for
        tensor parallelism. The orchestration layer MUST NOT call torch.cuda.*
        functions before TensorRT initializes.
        """
        return RuntimeCapabilities(
            launch_mode=LaunchMode.DIRECT,
            cuda_management=CudaManagement.BACKEND,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=False,  # Not in MVP
            manages_own_batching=True,  # Inflight batching
        )

    def initialize(self, config: ExperimentConfig, runtime: BackendRuntime) -> None:
        """Load or build TensorRT engine.

        Args:
            config: Experiment configuration.
            runtime: Runtime context (device info largely ignored - TRT manages).

        Raises:
            BackendNotAvailableError: If TensorRT-LLM is not installed.
            BackendInitializationError: If engine loading/building fails.
        """
        if not self.is_available():
            raise BackendNotAvailableError("tensorrt", install_hint="pip install lem[tensorrt]")

        self._config = config
        self._runtime = runtime

        trt_cfg: TensorRTConfig | None = config.tensorrt

        # Determine cache directory
        cache_dir = None
        if trt_cfg and trt_cfg.engine_cache_dir:
            cache_dir = Path(trt_cfg.engine_cache_dir)
        self._cache_manager = EngineCacheManager(cache_dir)

        try:
            # Check for pre-compiled engine path
            if trt_cfg and trt_cfg.engine_path:
                engine_path = Path(trt_cfg.engine_path)
                if not engine_path.exists():
                    raise BackendInitializationError(
                        f"Specified engine path does not exist: {engine_path}"
                    )
                self._engine_path = engine_path
                logger.info(f"Using pre-compiled engine: {engine_path}")

            # Check for cached engine
            elif not (trt_cfg and trt_cfg.force_rebuild) and self._cache_manager.has_cached_engine(
                config
            ):
                self._engine_path = self._cache_manager.get_engine_path(config)
                logger.info(f"Using cached engine: {self._engine_path}")

            # Build engine from HuggingFace checkpoint
            else:
                logger.info(f"Building TensorRT engine for {config.model_name}...")
                self._engine_path = self._build_engine(config)
                logger.info(f"Engine built and cached: {self._engine_path}")

            # Load tokenizer
            self._load_tokenizer(config)

            # Initialize executor
            self._initialize_executor(config)

            # Perform warmup
            self._perform_warmup()

        except Exception as e:
            raise BackendInitializationError(
                f"Failed to initialize TensorRT with model '{config.model_name}': {e}"
            ) from e

    def _build_engine(self, config: ExperimentConfig) -> Path:
        """Build TensorRT engine from HuggingFace checkpoint.

        Args:
            config: Experiment configuration.

        Returns:
            Path to built engine directory.
        """
        try:
            from tensorrt_llm import LLM, BuildConfig
        except ImportError as e:
            raise BackendInitializationError(
                "TensorRT-LLM not installed. Install with: pip install tensorrt-llm"
            ) from e

        trt_cfg: TensorRTConfig | None = config.tensorrt
        assert self._cache_manager is not None, "Cache manager not initialized"
        output_dir = self._cache_manager.get_engine_path(config)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build configuration
        build_config = BuildConfig()
        build_config.max_batch_size = trt_cfg.max_batch_size if trt_cfg else 8
        build_config.max_input_len = (
            trt_cfg.max_input_len if trt_cfg and trt_cfg.max_input_len else config.max_input_tokens
        )
        build_config.max_seq_len = build_config.max_input_len + (
            trt_cfg.max_output_len
            if trt_cfg and trt_cfg.max_output_len
            else config.max_output_tokens
        )

        if trt_cfg:
            build_config.builder_opt = trt_cfg.builder_opt_level
            if trt_cfg.strongly_typed:
                build_config.strongly_typed = True

        # Map precision
        dtype_map = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
        }
        dtype = dtype_map.get(config.fp_precision, "float16")

        # Tensor parallelism - read directly from tensorrt config
        tp_size = trt_cfg.tp_size if trt_cfg else 1

        # Quantization
        quantization = None
        if trt_cfg and trt_cfg.quantization != "none":
            quantization = self._map_quantization(trt_cfg.quantization)

        logger.info(
            f"Building engine: dtype={dtype}, tp_size={tp_size}, "
            f"max_batch={build_config.max_batch_size}, quant={quantization}"
        )

        start_time = time.perf_counter()

        # Use TensorRT-LLM's LLM class for building
        # Note: TensorRT-LLM 0.21.0+ uses different API for quantization
        # For now, build without quantization (use default settings)
        llm = LLM(
            model=config.model_name,
            dtype=dtype,
            tensor_parallel_size=tp_size,
        )

        # Save the engine
        llm.save(str(output_dir))

        build_time = time.perf_counter() - start_time
        self._cache_manager.save_metadata(config, build_time)
        logger.info(f"Engine built in {build_time:.1f}s")

        return output_dir

    def _map_quantization(self, method: str) -> str | None:
        """Map quantization method to TensorRT-LLM format.

        Args:
            method: Quantization method from config.

        Returns:
            TensorRT-LLM quantization string or None.
        """
        mapping = {
            "none": None,
            "fp8": "fp8",
            "int8_sq": "int8_sq",
            "int8_weight_only": "int8_wo",
            "int4_awq": "int4_awq",
            "int4_gptq": "int4_gptq",
        }
        return mapping.get(method)

    def _load_tokenizer(self, config: ExperimentConfig) -> None:
        """Load tokenizer from HuggingFace.

        Args:
            config: Experiment configuration.
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _initialize_executor(self, config: ExperimentConfig) -> None:
        """Initialize TensorRT-LLM executor.

        Args:
            config: Experiment configuration.
        """
        try:
            from tensorrt_llm import LLM
        except ImportError as e:
            raise BackendInitializationError(
                "TensorRT-LLM not installed. Install with: pip install tensorrt-llm"
            ) from e

        # Load from engine path
        self._executor = LLM(model=str(self._engine_path))

        logger.info(f"TensorRT executor initialized from {self._engine_path}")

    def _perform_warmup(self) -> None:
        """Perform warmup inference to trigger any JIT compilation."""
        if self._warmup_done or self._executor is None:
            return

        logger.debug("Performing TensorRT warmup inference...")
        warmup_prompt = "Hello"

        try:
            from tensorrt_llm import SamplingParams

            warmup_params = SamplingParams(max_tokens=1)
            self._executor.generate([warmup_prompt], warmup_params)
            self._warmup_done = True
            logger.debug("TensorRT warmup complete")
        except Exception as e:
            logger.warning(f"TensorRT warmup failed (non-fatal): {e}")

    def _supports_streaming(self) -> bool:
        """Check if TensorRT-LLM version supports streaming.

        TRT-LLM streaming requires version 0.9+.

        Returns:
            True if streaming is supported.
        """
        try:
            import tensorrt_llm

            version = getattr(tensorrt_llm, "__version__", "0.0.0")
            major, minor = map(int, version.split(".")[:2])
            return (major, minor) >= (0, 9)
        except (ImportError, ValueError):
            return False

    def run_inference(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference using TensorRT-LLM.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with token counts and timing.

        Raises:
            BackendInferenceError: If inference fails.
        """
        if self._executor is None:
            raise BackendInferenceError("TensorRT not initialized. Call initialize() first.")

        try:
            # Check if streaming mode is enabled for latency measurement
            if config.streaming:
                if self._supports_streaming():
                    return self._run_streaming_inference(prompts, config)
                else:
                    logger.warning(
                        "TRT-LLM version doesn't support streaming. "
                        "Using batch mode with TTFT estimation."
                    )
                    return self._run_batch_with_ttft_estimation(prompts, config)

            return self._run_inference_batch(prompts, config)
        except Exception as e:
            raise BackendInferenceError(f"TensorRT inference failed: {e}") from e

    def _run_inference_batch(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference on all prompts."""

        # Create sampling params
        sampling_params = self._create_sampling_params(config)

        start_time = time.perf_counter()

        # Run inference
        outputs = self._executor.generate(prompts, sampling_params)

        inference_time = time.perf_counter() - start_time

        return self._process_outputs(outputs, config, inference_time, len(prompts))

    def _create_sampling_params(self, config: ExperimentConfig) -> Any:
        """Create TensorRT-LLM sampling params from config.

        Reads universal decoder params from config.decoder and TensorRT-specific
        params from config.tensorrt.

        Args:
            config: Experiment configuration.

        Returns:
            Configured SamplingParams instance.
        """
        from tensorrt_llm import SamplingParams

        decoder = config.decoder

        params: dict[str, Any] = {
            "max_tokens": config.max_output_tokens,
        }

        # Temperature (universal)
        if decoder.temperature is not None:
            params["temperature"] = decoder.temperature

        # Top-p (universal)
        if decoder.top_p is not None:
            params["top_p"] = decoder.top_p

        # Top-k (universal - 0 = disabled for TensorRT)
        if decoder.top_k > 0:
            params["top_k"] = decoder.top_k

        # Repetition penalty (universal)
        if decoder.repetition_penalty is not None and decoder.repetition_penalty != 1.0:
            params["repetition_penalty"] = decoder.repetition_penalty

        # Seed
        if config.random_seed is not None:
            params["random_seed"] = config.random_seed

        return SamplingParams(**params)

    def _process_outputs(
        self,
        outputs: list[Any],
        config: ExperimentConfig,
        inference_time: float,
        num_prompts: int,
    ) -> BackendResult:
        """Process TensorRT-LLM outputs into BackendResult."""
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []

        for output in outputs:
            # Count tokens
            if hasattr(output, "prompt_token_ids"):
                total_input_tokens += len(output.prompt_token_ids)
            if hasattr(output, "outputs") and output.outputs:
                completion = output.outputs[0]
                if hasattr(completion, "token_ids"):
                    total_output_tokens += len(completion.token_ids)
                if hasattr(completion, "text"):
                    output_texts.append(completion.text)

        total_tokens = total_input_tokens + total_output_tokens

        logger.info(
            f"TensorRT inference complete: {num_prompts} prompts, "
            f"{total_tokens} tokens in {inference_time:.2f}s"
        )

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            inference_time_sec=inference_time,
            output_texts=output_texts if config.save_outputs else None,
            backend_metadata={
                "backend": "tensorrt",
                "version": self.version,
                "engine_path": str(self._engine_path) if self._engine_path else None,
                "inflight_batching": True,
                "num_prompts": num_prompts,
            },
        )

    def _run_streaming_inference(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Run inference with streaming for TTFT/ITL latency measurement.

        Processes prompts sequentially to capture per-token timing using TRT-LLM's
        streaming API where available.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with latency_measurements containing raw samples.
        """
        import numpy as np

        warmup_count = config.streaming_warmup_requests
        sampling_params = self._create_sampling_params(config)

        # Warmup reuses first N prompts; measurement uses ALL prompts
        # (warmup is additional overhead, not subtracted from measurement budget)
        warmup_prompts = prompts[: min(warmup_count, len(prompts))] if warmup_count > 0 else []
        measurement_prompts = prompts

        # Run warmup (results discarded from stats)
        if warmup_prompts:
            logger.info(f"Running {len(warmup_prompts)} streaming warmup requests...")
            self._executor.generate(warmup_prompts, sampling_params)
            logger.debug("Streaming warmup complete")

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

        # Note: TensorRT-LLM uses inflight batching internally
        # streaming=True forces sequential processing for latency measurement

        # Collect per-request timing data
        ttft_samples: list[float] = []
        token_timestamps_per_request: list[list[float]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        output_texts: list[str] = []

        logger.info(f"Running streaming inference on {len(measurement_prompts)} prompts...")
        start_time = time.perf_counter()

        # Process each prompt individually with progress tracking
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
                outputs = self._executor.generate([prompt], sampling_params)

                request_end = time.perf_counter()
                total_time_ms = (request_end - request_start) * 1000

                # Process outputs
                if outputs:
                    output = outputs[0]

                    if hasattr(output, "prompt_token_ids"):
                        total_input_tokens += len(output.prompt_token_ids)

                    # Try to get TTFT from TRT-LLM metrics if available (works regardless of outputs)
                    if hasattr(output, "metrics") and output.metrics is not None:
                        metrics = output.metrics
                        if hasattr(metrics, "time_to_first_token"):
                            first_token_time = metrics.time_to_first_token * 1000

                    if hasattr(output, "outputs") and output.outputs:
                        completion = output.outputs[0]
                        num_tokens = (
                            len(completion.token_ids) if hasattr(completion, "token_ids") else 0
                        )
                        total_output_tokens += num_tokens

                        if hasattr(completion, "text"):
                            output_texts.append(completion.text)

                        # If no TTFT from metrics, estimate from request time
                        if first_token_time is None and num_tokens > 0:
                            # Estimate TTFT as proportion of total time
                            first_token_time = total_time_ms / (num_tokens + 1)
                        elif first_token_time is None:
                            first_token_time = total_time_ms

                        ttft_samples.append(first_token_time)

                        # Estimate token times evenly distributed
                        # (TRT-LLM batch mode doesn't provide per-token timestamps)
                        if num_tokens > 1:
                            decode_time_ms = total_time_ms - first_token_time
                            token_times = [first_token_time]
                            time_per_token = decode_time_ms / (num_tokens - 1)
                            for i in range(1, num_tokens):
                                token_times.append(first_token_time + (i * time_per_token))

                        token_timestamps_per_request.append(token_times)
                    else:
                        # Handle empty outputs - TensorRT returned no completion tokens
                        # Use request-level timing as fallback TTFT estimate
                        if first_token_time is None:
                            first_token_time = total_time_ms
                        ttft_samples.append(first_token_time)
                        token_timestamps_per_request.append([first_token_time])
                        input_len = (
                            len(output.prompt_token_ids)
                            if hasattr(output, "prompt_token_ids")
                            else 0
                        )
                        logger.warning(
                            f"TensorRT returned empty outputs for prompt "
                            f"(input_length={input_len}). "
                            f"Using request timing as TTFT estimate: {first_token_time:.1f}ms"
                        )
                else:
                    # Handle complete failure - no outputs at all
                    logger.warning(
                        f"TensorRT generate() returned no outputs for prompt. "
                        f"Request time: {total_time_ms:.1f}ms"
                    )

                # Update progress
                progress.update(1, latency_ms=first_token_time)

        inference_time = time.perf_counter() - start_time

        # Calculate ITL from token timestamps using shared utility
        itl_full, itl_trimmed, excluded = collect_itl_measurements(token_timestamps_per_request)

        # Build latency measurements
        # Note: TRT-LLM doesn't provide true streaming timestamps - we estimate ITL
        latency_measurements = LatencyMeasurements(
            ttft_ms=ttft_samples,
            itl_full_ms=itl_full,
            itl_trimmed_ms=itl_trimmed,
            request_count=len(measurement_prompts),
            total_output_tokens=total_output_tokens,
            excluded_tokens=excluded,
            streaming_mode=True,
            warmup_requests_excluded=warmup_count,
            measurement_mode=LatencyMeasurementMode.PROPORTIONAL_ESTIMATE,  # ITL estimated, not true streaming
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
                "backend": "tensorrt",
                "version": self.version,
                "streaming_mode": True,
                "engine_path": str(self._engine_path) if self._engine_path else None,
                "num_prompts": len(measurement_prompts),
                "warmup_prompts": warmup_count,
                "ttft_samples": len(ttft_samples),
                "itl_samples": len(itl_trimmed),
            },
            latency_measurements=latency_measurements,
        )

    def _run_batch_with_ttft_estimation(
        self, prompts: list[str], config: ExperimentConfig
    ) -> BackendResult:
        """Fallback: batch inference with TTFT estimation.

        Processes prompts one at a time to estimate TTFT from total request time.
        Used when TRT-LLM streaming is unavailable.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with estimated latency measurements.
        """
        import numpy as np

        warmup_count = config.streaming_warmup_requests
        sampling_params = self._create_sampling_params(config)
        # Warmup reuses first N prompts; measurement uses ALL prompts
        warmup_prompts = prompts[: min(warmup_count, len(prompts))] if warmup_count > 0 else []
        measurement_prompts = prompts

        # Run warmup
        if warmup_prompts:
            logger.info(f"Running {len(warmup_prompts)} warmup requests...")
            self._executor.generate(warmup_prompts, sampling_params)

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

        with prompt_progress(
            total=len(measurement_prompts),
            desc="Prompts",
            is_main_process=True,
        ) as progress:
            for prompt in measurement_prompts:
                request_start = time.perf_counter()

                outputs = self._executor.generate([prompt], sampling_params)

                request_end = time.perf_counter()
                total_time_ms = (request_end - request_start) * 1000
                estimated_ttft: float | None = None

                if outputs:
                    output = outputs[0]

                    if hasattr(output, "prompt_token_ids"):
                        total_input_tokens += len(output.prompt_token_ids)

                    if hasattr(output, "outputs") and output.outputs:
                        completion = output.outputs[0]
                        num_tokens = (
                            len(completion.token_ids) if hasattr(completion, "token_ids") else 0
                        )
                        total_output_tokens += num_tokens

                        if hasattr(completion, "text"):
                            output_texts.append(completion.text)

                        # Estimate TTFT as proportional to 1 token of total time
                        if num_tokens > 0:
                            estimated_ttft = total_time_ms / (num_tokens + 1)
                        else:
                            estimated_ttft = total_time_ms
                        ttft_samples.append(estimated_ttft)

                        # Estimate token times evenly distributed
                        if num_tokens > 1:
                            decode_time_ms = total_time_ms - estimated_ttft
                            token_times = [estimated_ttft]
                            time_per_token = decode_time_ms / (num_tokens - 1)
                            for i in range(1, num_tokens):
                                token_times.append(estimated_ttft + (i * time_per_token))
                            token_timestamps_per_request.append(token_times)

                # Update progress
                progress.update(1, latency_ms=estimated_ttft)

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

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            inference_time_sec=inference_time,
            time_to_first_token_ms=avg_ttft_ms,
            output_texts=output_texts if config.save_outputs else None,
            backend_metadata={
                "backend": "tensorrt",
                "version": self.version,
                "streaming_mode": False,
                "engine_path": str(self._engine_path) if self._engine_path else None,
                "num_prompts": len(measurement_prompts),
                "warmup_prompts": warmup_count,
                "latency_warning": (
                    "ITL values are estimated (uniform distribution), not measured per-token. "
                    "Not suitable for publication-quality research."
                ),
            },
            latency_measurements=latency_measurements,
        )

    def cleanup(self) -> None:
        """Release TensorRT resources."""
        if self._executor is not None:
            logger.debug("Cleaning up TensorRT resources")
            self._executor = None
            self._tokenizer = None
            self._warmup_done = False

    def get_model_info(self) -> ModelInfo:
        """Return model metadata.

        Returns:
            ModelInfo with model details.
        """
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

        # Try to get model config from HuggingFace
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                self._config.model_name,
                trust_remote_code=True,
            )

            # Estimate parameter count from config
            param_count = 0
            if hasattr(hf_config, "num_parameters"):
                param_count = hf_config.num_parameters

            return ModelInfo(
                name=self._config.model_name,
                num_parameters=param_count,
                num_layers=getattr(hf_config, "num_hidden_layers", 0),
                hidden_size=getattr(hf_config, "hidden_size", 0),
                num_attention_heads=getattr(hf_config, "num_attention_heads", 0),
                vocab_size=getattr(hf_config, "vocab_size", 0),
                model_type=getattr(hf_config, "model_type", "unknown"),
                torch_dtype=self._config.fp_precision,
            )
        except Exception:
            return ModelInfo(
                name=self._config.model_name,
                num_parameters=0,
                num_layers=0,
                hidden_size=0,
                num_attention_heads=0,
                vocab_size=0,
                model_type="unknown",
                torch_dtype=self._config.fp_precision,
            )

    def get_supported_params(self) -> set[str]:
        """Return parameters supported by TensorRT backend."""
        return _SUPPORTED_PARAMS.copy()

    def validate_config(self, config: ExperimentConfig) -> list[ConfigWarning]:
        """Validate config compatibility with TensorRT.

        Args:
            config: Configuration to validate.

        Returns:
            List of warnings for incompatible or semantically different params.
        """
        warnings: list[ConfigWarning] = []
        trt_cfg = config.tensorrt

        if not trt_cfg:
            warnings.append(
                ConfigWarning(
                    field="tensorrt",
                    message="No tensorrt config section. Using default TensorRT settings.",
                    severity="info",
                )
            )
            return warnings

        # Inform about inflight batching semantics
        warnings.append(
            ConfigWarning(
                field="tensorrt.max_batch_size",
                message=(
                    f"TensorRT uses compile-time max_batch_size={trt_cfg.max_batch_size}. "
                    "Runtime batching is managed by inflight batching."
                ),
                severity="info",
            )
        )

        # Check INT8 calibration requirements
        if trt_cfg.quantization == "int8_sq" and trt_cfg.calibration is None:
            warnings.append(
                ConfigWarning(
                    field="tensorrt.quantization",
                    message="INT8 SmoothQuant requires calibration data for optimal accuracy.",
                    severity="warning",
                    suggestion="Add tensorrt.calibration with dataset and num_samples",
                )
            )

        # Check for FP8 on non-Hopper GPUs
        if trt_cfg.quantization == "fp8":
            gpu_arch = _get_gpu_architecture()
            if not gpu_arch.startswith("sm_9"):
                warnings.append(
                    ConfigWarning(
                        field="tensorrt.quantization",
                        message=f"FP8 quantization requires Hopper+ GPU (sm_90+). Detected: {gpu_arch}",
                        severity="warning",
                    )
                )

        # Check tensor parallel configuration against available GPUs
        if trt_cfg.tp_size > 1:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    if trt_cfg.tp_size > gpu_count:
                        warnings.append(
                            ConfigWarning(
                                field="tensorrt.tp_size",
                                message=(
                                    f"tp_size={trt_cfg.tp_size} exceeds available GPUs ({gpu_count})"
                                ),
                                severity="error",
                            )
                        )
            except ImportError:
                pass

        return warnings
