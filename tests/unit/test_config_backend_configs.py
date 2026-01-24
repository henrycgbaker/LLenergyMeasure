"""Tests for backend-specific configuration models."""

import pytest
from pydantic import ValidationError

from llenergymeasure.config.backend_configs import (
    PyTorchAssistedGenerationConfig,
    PyTorchConfig,
    TensorRTCalibrationConfig,
    TensorRTConfig,
    VLLMAttentionConfig,
    VLLMConfig,
    VLLMLoRAConfig,
    VLLMSpeculativeConfig,
)
from llenergymeasure.config.models import ExperimentConfig


class TestVLLMAttentionConfig:
    """Tests for VLLMAttentionConfig."""

    def test_defaults(self):
        config = VLLMAttentionConfig()
        assert config.backend == "auto"
        assert config.flash_version is None
        assert config.disable_sliding_window is False

    def test_valid_backends(self):
        # Note: TORCH_SDPA was removed in vLLM v1 (not registered as valid backend)
        for backend in ["auto", "FLASH_ATTN", "FLASHINFER"]:
            config = VLLMAttentionConfig(backend=backend)
            assert config.backend == backend

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValidationError):
            VLLMAttentionConfig(backend="invalid")

    def test_flash_version_2(self):
        config = VLLMAttentionConfig(flash_version=2)
        assert config.flash_version == 2

    def test_flash_version_3(self):
        config = VLLMAttentionConfig(flash_version=3)
        assert config.flash_version == 3

    def test_invalid_flash_version_rejected(self):
        with pytest.raises(ValidationError):
            VLLMAttentionConfig(flash_version=1)


class TestVLLMSpeculativeConfig:
    """Tests for VLLMSpeculativeConfig."""

    def test_defaults(self):
        config = VLLMSpeculativeConfig()
        assert config.model is None
        assert config.num_tokens == 5
        assert config.method == "ngram"
        assert config.prompt_lookup_min == 1
        assert config.prompt_lookup_max is None
        assert config.draft_tp_size == 1

    def test_valid_methods(self):
        for method in ["ngram", "eagle", "eagle3", "medusa", "mlp", "lookahead"]:
            config = VLLMSpeculativeConfig(method=method)
            assert config.method == method

    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            VLLMSpeculativeConfig(method="invalid")

    def test_num_tokens_range(self):
        # Valid range 1-10
        VLLMSpeculativeConfig(num_tokens=1)
        VLLMSpeculativeConfig(num_tokens=10)

        # Invalid
        with pytest.raises(ValidationError):
            VLLMSpeculativeConfig(num_tokens=0)

        with pytest.raises(ValidationError):
            VLLMSpeculativeConfig(num_tokens=11)

    def test_prompt_lookup_min_positive(self):
        config = VLLMSpeculativeConfig(prompt_lookup_min=1)
        assert config.prompt_lookup_min == 1

        with pytest.raises(ValidationError):
            VLLMSpeculativeConfig(prompt_lookup_min=0)

    def test_draft_tp_size_positive(self):
        config = VLLMSpeculativeConfig(draft_tp_size=4)
        assert config.draft_tp_size == 4

        with pytest.raises(ValidationError):
            VLLMSpeculativeConfig(draft_tp_size=0)


class TestVLLMLoRAConfig:
    """Tests for VLLMLoRAConfig."""

    def test_defaults(self):
        config = VLLMLoRAConfig()
        assert config.enabled is False
        assert config.max_loras == 1
        assert config.max_rank == 16
        assert config.extra_vocab_size == 256

    def test_enabled(self):
        config = VLLMLoRAConfig(enabled=True, max_loras=4)
        assert config.enabled is True
        assert config.max_loras == 4

    def test_max_loras_positive(self):
        with pytest.raises(ValidationError):
            VLLMLoRAConfig(max_loras=0)

    def test_max_rank_positive(self):
        with pytest.raises(ValidationError):
            VLLMLoRAConfig(max_rank=0)

    def test_extra_vocab_size_non_negative(self):
        VLLMLoRAConfig(extra_vocab_size=0)

        with pytest.raises(ValidationError):
            VLLMLoRAConfig(extra_vocab_size=-1)


class TestVLLMConfig:
    """Tests for VLLMConfig."""

    def test_defaults(self):
        config = VLLMConfig()
        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens is None
        assert config.gpu_memory_utilization == 0.9
        assert config.swap_space == 4.0
        assert config.cpu_offload_gb == 0.0
        assert config.enable_prefix_caching is False
        assert config.enable_chunked_prefill is False
        assert config.kv_cache_dtype == "auto"
        assert config.block_size == 16
        assert config.max_model_len is None
        assert config.enforce_eager is False
        assert config.distributed_backend == "mp"
        assert config.attention is None
        assert config.speculative is None
        assert config.lora is None
        assert config.quantization is None
        assert config.load_format == "auto"
        # Note: best_of was removed in vLLM v1 (use beam search or repetition instead)
        assert config.logprobs is None
        assert config.logit_bias is None
        assert config.extra == {}

    def test_max_num_seqs_range(self):
        # Valid range 1-1024
        VLLMConfig(max_num_seqs=1)
        VLLMConfig(max_num_seqs=1024)

        with pytest.raises(ValidationError):
            VLLMConfig(max_num_seqs=0)

        with pytest.raises(ValidationError):
            VLLMConfig(max_num_seqs=1025)

    def test_gpu_memory_utilization_range(self):
        # Valid range 0.5-0.99
        VLLMConfig(gpu_memory_utilization=0.5)
        VLLMConfig(gpu_memory_utilization=0.99)

        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=0.49)

        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=1.0)

    def test_swap_space_non_negative(self):
        VLLMConfig(swap_space=0.0)

        with pytest.raises(ValidationError):
            VLLMConfig(swap_space=-1.0)

    def test_cpu_offload_non_negative(self):
        VLLMConfig(cpu_offload_gb=0.0)

        with pytest.raises(ValidationError):
            VLLMConfig(cpu_offload_gb=-1.0)

    def test_kv_cache_dtype_values(self):
        # Only auto and fp8 supported - vLLM v1 handles dtype automatically
        for dtype in ["auto", "fp8"]:
            config = VLLMConfig(kv_cache_dtype=dtype)
            assert config.kv_cache_dtype == dtype

        # Explicit float16/bfloat16 removed - vLLM v1 handles this automatically
        with pytest.raises(ValidationError):
            VLLMConfig(kv_cache_dtype="float16")
        with pytest.raises(ValidationError):
            VLLMConfig(kv_cache_dtype="bfloat16")
        with pytest.raises(ValidationError):
            VLLMConfig(kv_cache_dtype="float32")

    def test_block_size_values(self):
        # Only 16 and 32 supported - 8 removed as most model/attention configs don't support it
        for size in [16, 32]:
            config = VLLMConfig(block_size=size)
            assert config.block_size == size

        # block_size=8 removed - not supported by most model/attention configs
        with pytest.raises(ValidationError):
            VLLMConfig(block_size=8)
        with pytest.raises(ValidationError):
            VLLMConfig(block_size=64)

    def test_distributed_backend_values(self):
        for backend in ["mp", "ray"]:
            config = VLLMConfig(distributed_backend=backend)
            assert config.distributed_backend == backend

        with pytest.raises(ValidationError):
            VLLMConfig(distributed_backend="invalid")

    def test_load_format_values(self):
        # Only auto and safetensors supported - most HuggingFace models use safetensors
        for fmt in ["auto", "safetensors"]:
            config = VLLMConfig(load_format=fmt)
            assert config.load_format == fmt

        # pt/gguf removed - most HuggingFace models use safetensors
        with pytest.raises(ValidationError):
            VLLMConfig(load_format="pt")
        with pytest.raises(ValidationError):
            VLLMConfig(load_format="gguf")
        with pytest.raises(ValidationError):
            VLLMConfig(load_format="invalid")

    def test_logprobs_range(self):
        # Valid range 1-20
        VLLMConfig(logprobs=1)
        VLLMConfig(logprobs=20)

        with pytest.raises(ValidationError):
            VLLMConfig(logprobs=0)

        with pytest.raises(ValidationError):
            VLLMConfig(logprobs=21)

    # Note: test_best_of_positive removed - best_of was removed in vLLM v1

    def test_nested_attention_config(self):
        config = VLLMConfig(attention=VLLMAttentionConfig(backend="FLASH_ATTN", flash_version=2))
        assert config.attention is not None
        assert config.attention.backend == "FLASH_ATTN"
        assert config.attention.flash_version == 2

    def test_nested_speculative_config(self):
        config = VLLMConfig(
            speculative=VLLMSpeculativeConfig(
                model="TinyLlama/TinyLlama-1.1B",
                method="ngram",
                num_tokens=5,
            )
        )
        assert config.speculative is not None
        assert config.speculative.model == "TinyLlama/TinyLlama-1.1B"
        assert config.speculative.method == "ngram"

    def test_nested_lora_config(self):
        config = VLLMConfig(lora=VLLMLoRAConfig(enabled=True, max_loras=4))
        assert config.lora is not None
        assert config.lora.enabled is True
        assert config.lora.max_loras == 4

    def test_extra_escape_hatch(self):
        config = VLLMConfig(extra={"custom_param": "value", "another": 123})
        assert config.extra["custom_param"] == "value"
        assert config.extra["another"] == 123

    def test_serialization_roundtrip(self):
        config = VLLMConfig(
            max_num_seqs=512,
            enable_prefix_caching=True,
            speculative=VLLMSpeculativeConfig(method="ngram", num_tokens=5),
        )
        json_str = config.model_dump_json()
        restored = VLLMConfig.model_validate_json(json_str)
        assert restored.max_num_seqs == 512
        assert restored.enable_prefix_caching is True
        assert restored.speculative is not None
        assert restored.speculative.method == "ngram"


class TestPyTorchAssistedGenerationConfig:
    """Tests for PyTorchAssistedGenerationConfig."""

    def test_defaults(self):
        config = PyTorchAssistedGenerationConfig()
        assert config.model is None
        assert config.num_tokens == 5

    def test_with_model(self):
        config = PyTorchAssistedGenerationConfig(model="TinyLlama/TinyLlama-1.1B", num_tokens=3)
        assert config.model == "TinyLlama/TinyLlama-1.1B"
        assert config.num_tokens == 3

    def test_num_tokens_range(self):
        # Valid range 1-10
        PyTorchAssistedGenerationConfig(num_tokens=1)
        PyTorchAssistedGenerationConfig(num_tokens=10)

        with pytest.raises(ValidationError):
            PyTorchAssistedGenerationConfig(num_tokens=0)

        with pytest.raises(ValidationError):
            PyTorchAssistedGenerationConfig(num_tokens=11)


class TestPyTorchConfig:
    """Tests for PyTorchConfig."""

    def test_defaults(self):
        config = PyTorchConfig()
        assert config.attn_implementation == "sdpa"
        assert config.torch_compile is False
        assert config.use_bettertransformer is False
        assert config.use_cache is True
        # Note: cache_implementation is a new field for static/dynamic KV cache
        assert config.cache_implementation is None
        assert config.low_cpu_mem_usage is True
        assert config.max_memory is None
        assert config.assisted_generation is None
        # Note: num_beams, early_stopping, length_penalty moved to DecoderConfig.beam_search
        assert config.output_scores is False
        assert config.return_dict_in_generate is False
        assert config.extra == {}

    def test_attn_implementation_values(self):
        for impl in ["sdpa", "flash_attention_2", "eager"]:
            config = PyTorchConfig(attn_implementation=impl)
            assert config.attn_implementation == impl

        with pytest.raises(ValidationError):
            PyTorchConfig(attn_implementation="invalid")

    def test_torch_compile_bool(self):
        PyTorchConfig(torch_compile=False)
        PyTorchConfig(torch_compile=True)

    def test_torch_compile_string_modes(self):
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            config = PyTorchConfig(torch_compile=mode)
            assert config.torch_compile == mode

    def test_cache_implementation_values(self):
        """Test valid cache implementation values."""
        for impl in ["dynamic", "static", "hybrid", "sliding_window"]:
            config = PyTorchConfig(cache_implementation=impl)
            assert config.cache_implementation == impl

        # None is allowed (default)
        config = PyTorchConfig(cache_implementation=None)
        assert config.cache_implementation is None

    def test_max_memory_dict(self):
        config = PyTorchConfig(max_memory={"0": "20GiB", "cpu": "30GiB"})
        assert config.max_memory == {"0": "20GiB", "cpu": "30GiB"}

    def test_nested_assisted_generation(self):
        config = PyTorchConfig(
            assisted_generation=PyTorchAssistedGenerationConfig(
                model="TinyLlama/TinyLlama-1.1B", num_tokens=5
            )
        )
        assert config.assisted_generation is not None
        assert config.assisted_generation.model == "TinyLlama/TinyLlama-1.1B"
        assert config.assisted_generation.num_tokens == 5

    def test_extra_escape_hatch(self):
        config = PyTorchConfig(extra={"custom_param": "value"})
        assert config.extra["custom_param"] == "value"

    def test_serialization_roundtrip(self):
        config = PyTorchConfig(
            attn_implementation="flash_attention_2",
            torch_compile="reduce-overhead",
            assisted_generation=PyTorchAssistedGenerationConfig(num_tokens=5),
        )
        json_str = config.model_dump_json()
        restored = PyTorchConfig.model_validate_json(json_str)
        assert restored.attn_implementation == "flash_attention_2"
        assert restored.torch_compile == "reduce-overhead"
        assert restored.assisted_generation is not None


class TestExperimentConfigBackendIntegration:
    """Tests for backend config integration in ExperimentConfig."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "config_name": "test_config",
            "model_name": "test-model",
        }

    def test_vllm_config_with_vllm_backend(self, minimal_config):
        """vllm config with backend='vllm' is valid."""
        config = ExperimentConfig(
            **minimal_config,
            backend="vllm",
            vllm=VLLMConfig(max_num_seqs=512, enable_prefix_caching=True),
        )
        assert config.backend == "vllm"
        assert config.vllm is not None
        assert config.vllm.max_num_seqs == 512
        assert config.vllm.enable_prefix_caching is True

    def test_pytorch_config_with_pytorch_backend(self, minimal_config):
        """pytorch config with backend='pytorch' is valid."""
        config = ExperimentConfig(
            **minimal_config,
            backend="pytorch",
            pytorch=PyTorchConfig(
                attn_implementation="flash_attention_2", torch_compile="reduce-overhead"
            ),
        )
        assert config.backend == "pytorch"
        assert config.pytorch is not None
        assert config.pytorch.attn_implementation == "flash_attention_2"
        assert config.pytorch.torch_compile == "reduce-overhead"

    def test_vllm_config_with_pytorch_backend_rejected(self, minimal_config):
        """vllm config with backend='pytorch' is rejected."""
        with pytest.raises(ValidationError, match="vllm.*config.*but backend is.*pytorch"):
            ExperimentConfig(
                **minimal_config,
                backend="pytorch",
                vllm=VLLMConfig(max_num_seqs=512),
            )

    def test_pytorch_config_with_vllm_backend_rejected(self, minimal_config):
        """pytorch config with backend='vllm' is rejected."""
        with pytest.raises(ValidationError, match="pytorch.*config.*but backend is.*vllm"):
            ExperimentConfig(
                **minimal_config,
                backend="vllm",
                pytorch=PyTorchConfig(attn_implementation="flash_attention_2"),
            )

    def test_no_backend_config_default_pytorch(self, minimal_config):
        """No backend config with default backend='pytorch' is valid."""
        config = ExperimentConfig(**minimal_config)
        assert config.backend == "pytorch"
        assert config.vllm is None
        assert config.pytorch is None

    def test_vllm_config_from_dict(self, minimal_config):
        """vllm config can be specified as dict."""
        config = ExperimentConfig(
            **minimal_config,
            backend="vllm",
            vllm={"max_num_seqs": 256, "enable_prefix_caching": True},
        )
        assert config.vllm is not None
        assert config.vllm.max_num_seqs == 256

    def test_pytorch_config_from_dict(self, minimal_config):
        """pytorch config can be specified as dict."""
        config = ExperimentConfig(
            **minimal_config,
            backend="pytorch",
            pytorch={"attn_implementation": "sdpa", "torch_compile": False},
        )
        assert config.pytorch is not None
        assert config.pytorch.attn_implementation == "sdpa"

    def test_nested_vllm_speculative_from_dict(self, minimal_config):
        """Nested vllm speculative config works from dict."""
        config = ExperimentConfig(
            **minimal_config,
            backend="vllm",
            vllm={
                "speculative": {
                    "model": "TinyLlama/TinyLlama-1.1B",
                    "method": "ngram",
                    "num_tokens": 5,
                }
            },
        )
        assert config.vllm is not None
        assert config.vllm.speculative is not None
        assert config.vllm.speculative.model == "TinyLlama/TinyLlama-1.1B"
        assert config.vllm.speculative.method == "ngram"

    def test_nested_pytorch_assisted_from_dict(self, minimal_config):
        """Nested pytorch assisted_generation config works from dict."""
        config = ExperimentConfig(
            **minimal_config,
            backend="pytorch",
            pytorch={
                "assisted_generation": {
                    "model": "TinyLlama/TinyLlama-1.1B",
                    "num_tokens": 5,
                }
            },
        )
        assert config.pytorch is not None
        assert config.pytorch.assisted_generation is not None
        assert config.pytorch.assisted_generation.model == "TinyLlama/TinyLlama-1.1B"

    def test_serialization_roundtrip_vllm(self, minimal_config):
        """Full vllm config survives serialization roundtrip."""
        config = ExperimentConfig(
            **minimal_config,
            backend="vllm",
            vllm=VLLMConfig(
                max_num_seqs=512,
                enable_prefix_caching=True,
                speculative=VLLMSpeculativeConfig(method="ngram"),
            ),
        )
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.vllm is not None
        assert restored.vllm.max_num_seqs == 512
        assert restored.vllm.speculative is not None
        assert restored.vllm.speculative.method == "ngram"

    def test_serialization_roundtrip_pytorch(self, minimal_config):
        """Full pytorch config survives serialization roundtrip."""
        config = ExperimentConfig(
            **minimal_config,
            backend="pytorch",
            pytorch=PyTorchConfig(
                attn_implementation="flash_attention_2",
                torch_compile="reduce-overhead",
                assisted_generation=PyTorchAssistedGenerationConfig(num_tokens=5),
            ),
        )
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.pytorch is not None
        assert restored.pytorch.attn_implementation == "flash_attention_2"
        assert restored.pytorch.assisted_generation is not None


# =============================================================================
# TensorRT Config Tests
# =============================================================================


class TestTensorRTCalibrationConfig:
    """Tests for TensorRTCalibrationConfig."""

    def test_defaults(self):
        config = TensorRTCalibrationConfig()
        assert config.dataset == "wikitext"
        assert config.split == "train"
        assert config.num_samples == 512
        assert config.max_length == 2048

    def test_valid_num_samples_range(self):
        TensorRTCalibrationConfig(num_samples=64)
        TensorRTCalibrationConfig(num_samples=4096)

        with pytest.raises(ValidationError):
            TensorRTCalibrationConfig(num_samples=63)

        with pytest.raises(ValidationError):
            TensorRTCalibrationConfig(num_samples=4097)

    def test_custom_dataset(self):
        config = TensorRTCalibrationConfig(
            dataset="custom/dataset",
            split="validation",
            num_samples=1024,
        )
        assert config.dataset == "custom/dataset"
        assert config.split == "validation"
        assert config.num_samples == 1024


class TestTensorRTConfig:
    """Tests for TensorRTConfig."""

    def test_defaults(self):
        config = TensorRTConfig()
        assert config.engine_path is None
        assert config.max_batch_size == 8
        assert config.max_input_len is None
        assert config.max_output_len is None
        assert config.builder_opt_level == 3
        assert config.strongly_typed is True
        # Note: tp_size removed - use parallelism.degree with strategy=tensor_parallel
        assert config.pp_size == 1
        assert config.kv_cache_type == "paged"
        assert config.enable_chunked_context is True
        assert config.gpu_memory_utilization == 0.9
        assert config.force_rebuild is False
        # New energy-impacting options
        assert config.multiple_profiles is False
        assert config.enable_kv_cache_reuse is False

    def test_valid_batch_sizes(self):
        TensorRTConfig(max_batch_size=1)
        TensorRTConfig(max_batch_size=256)

        with pytest.raises(ValidationError):
            TensorRTConfig(max_batch_size=0)

        with pytest.raises(ValidationError):
            TensorRTConfig(max_batch_size=257)

    def test_valid_builder_opt_levels(self):
        for level in range(6):
            config = TensorRTConfig(builder_opt_level=level)
            assert config.builder_opt_level == level

        with pytest.raises(ValidationError):
            TensorRTConfig(builder_opt_level=-1)

        with pytest.raises(ValidationError):
            TensorRTConfig(builder_opt_level=6)

    def test_valid_kv_cache_types(self):
        for kv_type in ["paged", "continuous"]:
            config = TensorRTConfig(kv_cache_type=kv_type)
            assert config.kv_cache_type == kv_type

        with pytest.raises(ValidationError):
            TensorRTConfig(kv_cache_type="invalid")

    def test_gpu_memory_utilization_range(self):
        TensorRTConfig(gpu_memory_utilization=0.5)
        TensorRTConfig(gpu_memory_utilization=0.99)

        with pytest.raises(ValidationError):
            TensorRTConfig(gpu_memory_utilization=0.49)

        with pytest.raises(ValidationError):
            TensorRTConfig(gpu_memory_utilization=1.0)

    def test_pipeline_parallelism(self):
        """Test pipeline parallelism config."""
        # Note: tp_size removed - use parallelism.degree with strategy=tensor_parallel
        config = TensorRTConfig(pp_size=2)
        assert config.pp_size == 2

    def test_energy_impacting_options(self):
        """Test new energy-impacting options."""
        config = TensorRTConfig(
            multiple_profiles=True,
            enable_kv_cache_reuse=True,
        )
        assert config.multiple_profiles is True
        assert config.enable_kv_cache_reuse is True

    def test_with_quantization(self):
        """Quantization is now a flat string literal."""
        config = TensorRTConfig(quantization="fp8")
        assert config.quantization == "fp8"

    def test_with_calibration(self):
        """Calibration is now a separate field for INT8 SmoothQuant."""
        config = TensorRTConfig(
            quantization="int8_sq",
            calibration=TensorRTCalibrationConfig(
                dataset="wikitext",
                num_samples=1024,
            ),
        )
        assert config.quantization == "int8_sq"
        assert config.calibration is not None
        assert config.calibration.num_samples == 1024

    def test_valid_quantization_methods(self):
        """All quantization methods are valid."""
        for method in ["none", "fp8", "int8_sq", "int8_weight_only", "int4_awq", "int4_gptq"]:
            config = TensorRTConfig(quantization=method)
            assert config.quantization == method

    def test_invalid_quantization_rejected(self):
        """Invalid quantization method is rejected."""
        with pytest.raises(ValidationError):
            TensorRTConfig(quantization="invalid")

    def test_extra_args(self):
        config = TensorRTConfig(
            extra_build_args={"custom_flag": True},
            extra_runtime_args={"max_beam_width": 4},
        )
        assert config.extra_build_args["custom_flag"] is True
        assert config.extra_runtime_args["max_beam_width"] == 4


class TestTensorRTExperimentConfigIntegration:
    """Tests for TensorRTConfig integration with ExperimentConfig."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "config_name": "test",
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }

    def test_tensorrt_backend_with_config(self, minimal_config):
        """TensorRT backend accepts tensorrt config."""
        config = ExperimentConfig(
            **minimal_config,
            backend="tensorrt",
            tensorrt=TensorRTConfig(
                max_batch_size=16,
                quantization="fp8",
            ),
        )
        assert config.backend == "tensorrt"
        assert config.tensorrt is not None
        assert config.tensorrt.max_batch_size == 16
        assert config.tensorrt.quantization == "fp8"

    def test_tensorrt_config_with_wrong_backend_rejected(self, minimal_config):
        """TensorRT config rejected when backend is not tensorrt."""
        with pytest.raises(ValueError, match="tensorrt config provided but backend is"):
            ExperimentConfig(
                **minimal_config,
                backend="pytorch",
                tensorrt=TensorRTConfig(),
            )

    def test_tensorrt_config_from_dict(self, minimal_config):
        """TensorRT config can be specified as dict."""
        config = ExperimentConfig(
            **minimal_config,
            backend="tensorrt",
            tensorrt={
                "max_batch_size": 32,
                "builder_opt_level": 5,
                "quantization": "fp8",
            },
        )
        assert config.tensorrt is not None
        assert config.tensorrt.max_batch_size == 32
        assert config.tensorrt.builder_opt_level == 5
        assert config.tensorrt.quantization == "fp8"

    def test_serialization_roundtrip_tensorrt(self, minimal_config):
        """Full tensorrt config survives serialization roundtrip."""
        config = ExperimentConfig(
            **minimal_config,
            backend="tensorrt",
            tensorrt=TensorRTConfig(
                max_batch_size=16,
                builder_opt_level=4,
                pp_size=2,
                enable_kv_cache_reuse=True,
                quantization="int8_sq",
                calibration=TensorRTCalibrationConfig(num_samples=256),
            ),
        )
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.tensorrt is not None
        assert restored.tensorrt.max_batch_size == 16
        assert restored.tensorrt.builder_opt_level == 4
        assert restored.tensorrt.pp_size == 2
        assert restored.tensorrt.enable_kv_cache_reuse is True
        assert restored.tensorrt.quantization == "int8_sq"
        assert restored.tensorrt.calibration is not None
        assert restored.tensorrt.calibration.num_samples == 256
