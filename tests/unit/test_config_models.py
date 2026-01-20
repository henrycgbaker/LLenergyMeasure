"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from llm_energy_measure.config.models import (
    BUILTIN_DATASETS,
    BatchingConfig,
    DecoderConfig,
    ExperimentConfig,
    FilePromptSource,
    HuggingFacePromptSource,
    LatencySimulation,
    QuantizationConfig,
    TrafficSimulation,
)


class TestBatchingConfig:
    """Tests for BatchingConfig with industry-standard strategies."""

    def test_defaults(self):
        config = BatchingConfig()
        assert config.batch_size == 1
        assert config.strategy == "static"
        assert config.max_tokens_per_batch is None
        assert config.dynamic_batching is False

    def test_static_strategy(self):
        """Static strategy: fixed batch size (MLPerf offline)."""
        config = BatchingConfig(strategy="static", batch_size=8)
        assert config.strategy == "static"
        assert config.batch_size == 8

    def test_dynamic_strategy(self):
        """Dynamic strategy: token-aware batching (MLPerf server)."""
        config = BatchingConfig(strategy="dynamic", batch_size=8, max_tokens_per_batch=2048)
        assert config.strategy == "dynamic"
        assert config.max_tokens_per_batch == 2048

    def test_sorted_static_strategy(self):
        """Sorted static: sort by length then fixed batches."""
        config = BatchingConfig(strategy="sorted_static", batch_size=4)
        assert config.strategy == "sorted_static"

    def test_sorted_dynamic_strategy(self):
        """Sorted dynamic: sort by length + token budget."""
        config = BatchingConfig(strategy="sorted_dynamic", batch_size=8, max_tokens_per_batch=1024)
        assert config.strategy == "sorted_dynamic"
        assert config.max_tokens_per_batch == 1024

    def test_legacy_dynamic_batching_flag(self):
        """Legacy dynamic_batching=True maps to strategy='dynamic'."""
        config = BatchingConfig(dynamic_batching=True)
        assert config.strategy == "dynamic"

    def test_legacy_flag_does_not_override_explicit_strategy(self):
        """Explicit strategy takes precedence over legacy flag."""
        config = BatchingConfig(strategy="sorted_static", dynamic_batching=True)
        # sorted_static is not "static", so flag should not override
        assert config.strategy == "sorted_static"

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            BatchingConfig(batch_size=0)

    def test_invalid_strategy_rejected(self):
        """Invalid strategy value is rejected."""
        with pytest.raises(ValidationError):
            BatchingConfig(strategy="invalid")


class TestTrafficSimulation:
    """Tests for TrafficSimulation (MLPerf-style traffic patterns)."""

    def test_defaults(self):
        config = TrafficSimulation()
        assert config.enabled is False
        assert config.mode == "poisson"
        assert config.target_qps == 1.0
        assert config.seed is None

    def test_poisson_mode(self):
        """Poisson mode: MLPerf server scenario."""
        config = TrafficSimulation(enabled=True, mode="poisson", target_qps=10.0)
        assert config.mode == "poisson"
        assert config.target_qps == 10.0

    def test_constant_mode(self):
        """Constant mode: fixed inter-arrival time."""
        config = TrafficSimulation(enabled=True, mode="constant", target_qps=5.0)
        assert config.mode == "constant"
        assert config.target_qps == 5.0

    def test_target_qps_must_be_positive(self):
        """target_qps must be > 0."""
        with pytest.raises(ValidationError):
            TrafficSimulation(target_qps=0)
        with pytest.raises(ValidationError):
            TrafficSimulation(target_qps=-1.0)

    def test_seed_for_reproducibility(self):
        """seed allows reproducible Poisson arrivals."""
        config = TrafficSimulation(enabled=True, mode="poisson", seed=42)
        assert config.seed == 42

    def test_invalid_mode_rejected(self):
        """Invalid mode is rejected."""
        with pytest.raises(ValidationError):
            TrafficSimulation(mode="invalid")

    def test_backwards_compat_alias(self):
        """LatencySimulation is an alias for TrafficSimulation."""
        config = LatencySimulation()
        assert isinstance(config, TrafficSimulation)


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_defaults(self):
        config = QuantizationConfig()
        assert config.quantization is False
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False

    def test_4bit_config(self):
        config = QuantizationConfig(load_in_4bit=True)
        assert config.quantization is True  # Auto-enabled
        assert config.load_in_4bit is True

    def test_8bit_config(self):
        config = QuantizationConfig(load_in_8bit=True)
        assert config.quantization is True  # Auto-enabled

    def test_mutual_exclusivity(self):
        with pytest.raises(ValidationError, match="Cannot enable both 4-bit and 8-bit"):
            QuantizationConfig(load_in_4bit=True, load_in_8bit=True)


class TestDecoderConfig:
    """Tests for DecoderConfig with sampling presets."""

    def test_defaults(self):
        """Default values match specification."""
        config = DecoderConfig()
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.top_k == 50
        assert config.do_sample is True
        assert config.min_p == 0.0
        assert config.repetition_penalty == 1.0
        assert config.no_repeat_ngram_size == 0
        assert config.preset is None

    def test_top_p_bounds(self):
        """top_p must be in [0, 1]."""
        with pytest.raises(ValidationError):
            DecoderConfig(top_p=1.5)

        with pytest.raises(ValidationError):
            DecoderConfig(top_p=-0.1)

    def test_temperature_bounds(self):
        """temperature must be in [0, 2]."""
        # Valid range
        DecoderConfig(temperature=0.0)
        DecoderConfig(temperature=2.0)

        # Invalid
        with pytest.raises(ValidationError):
            DecoderConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            DecoderConfig(temperature=2.1)

    def test_repetition_penalty_bounds(self):
        """repetition_penalty must be in [0.1, 10]."""
        # Valid range
        DecoderConfig(repetition_penalty=0.1)
        DecoderConfig(repetition_penalty=10.0)

        # Invalid
        with pytest.raises(ValidationError):
            DecoderConfig(repetition_penalty=0.05)

        with pytest.raises(ValidationError):
            DecoderConfig(repetition_penalty=10.5)

    def test_min_p_bounds(self):
        """min_p must be in [0, 1]."""
        # Valid range
        DecoderConfig(min_p=0.0)
        DecoderConfig(min_p=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            DecoderConfig(min_p=-0.1)

        with pytest.raises(ValidationError):
            DecoderConfig(min_p=1.1)

    def test_no_repeat_ngram_non_negative(self):
        """no_repeat_ngram_size must be >= 0."""
        DecoderConfig(no_repeat_ngram_size=0)
        DecoderConfig(no_repeat_ngram_size=3)

        with pytest.raises(ValidationError):
            DecoderConfig(no_repeat_ngram_size=-1)

    def test_preset_deterministic(self):
        """preset: deterministic expands to greedy settings."""
        config = DecoderConfig(preset="deterministic")
        assert config.temperature == 0.0
        assert config.do_sample is False
        assert config.preset == "deterministic"

    def test_preset_standard(self):
        """preset: standard expands to balanced sampling settings."""
        config = DecoderConfig(preset="standard")
        assert config.temperature == 1.0
        assert config.do_sample is True
        assert config.top_p == 0.95
        assert config.top_k == 50

    def test_preset_creative(self):
        """preset: creative expands to higher variance settings."""
        config = DecoderConfig(preset="creative")
        assert config.temperature == 0.8
        assert config.do_sample is True
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1

    def test_preset_factual(self):
        """preset: factual expands to lower variance settings."""
        config = DecoderConfig(preset="factual")
        assert config.temperature == 0.3
        assert config.do_sample is True
        assert config.top_k == 10

    def test_preset_with_override(self):
        """Explicit params override preset values."""
        config = DecoderConfig(preset="deterministic", temperature=0.5)
        # temperature from explicit param, not preset
        assert config.temperature == 0.5
        # do_sample from preset
        assert config.do_sample is False
        assert config.preset == "deterministic"

    def test_preset_invalid_rejected(self):
        """Invalid preset name is rejected."""
        with pytest.raises(ValidationError):
            DecoderConfig(preset="invalid_preset")

    def test_is_deterministic_temp_zero(self):
        """is_deterministic True when temperature=0."""
        config = DecoderConfig(temperature=0.0)
        assert config.is_deterministic is True

    def test_is_deterministic_do_sample_false(self):
        """is_deterministic True when do_sample=False."""
        config = DecoderConfig(temperature=1.0, do_sample=False)
        assert config.is_deterministic is True

    def test_is_deterministic_sampling_enabled(self):
        """is_deterministic False when sampling with temp > 0."""
        config = DecoderConfig(temperature=1.0, do_sample=True)
        assert config.is_deterministic is False

    def test_is_deterministic_preset(self):
        """is_deterministic correct for preset."""
        deterministic = DecoderConfig(preset="deterministic")
        assert deterministic.is_deterministic is True

        standard = DecoderConfig(preset="standard")
        assert standard.is_deterministic is False


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "config_name": "test_config",
            "model_name": "test-model",
        }

    def test_minimal_config(self, minimal_config):
        config = ExperimentConfig(**minimal_config)
        assert config.config_name == "test_config"
        assert config.model_name == "test-model"
        assert config.gpu_list == [0]
        assert config.num_processes == 1

    def test_full_config(self, minimal_config):
        full = {
            **minimal_config,
            "max_input_tokens": 1024,
            "max_output_tokens": 256,
            "gpu_list": [0, 1, 2, 3],
            "num_processes": 4,
            "fp_precision": "float16",
        }
        config = ExperimentConfig(**full)
        assert config.max_input_tokens == 1024
        assert config.num_processes == 4

    def test_num_processes_validation(self, minimal_config):
        with pytest.raises(ValidationError, match="num_processes.*must be <=.*gpu_list"):
            ExperimentConfig(
                **minimal_config,
                gpu_list=[0, 1],
                num_processes=4,
            )

    def test_min_max_tokens_validation(self, minimal_config):
        with pytest.raises(ValidationError, match="min_output_tokens.*must be <="):
            ExperimentConfig(
                **minimal_config,
                min_output_tokens=200,
                max_output_tokens=100,
            )

    def test_gpu_list_from_int(self, minimal_config):
        config = ExperimentConfig(**minimal_config, gpu_list=0)
        assert config.gpu_list == [0]

    def test_nested_configs(self, minimal_config):
        config = ExperimentConfig(
            **minimal_config,
            quantization_config={"load_in_4bit": True},
            decoder_config={"temperature": 0.7},
        )
        assert config.quantization_config.load_in_4bit is True
        assert config.decoder_config.temperature == 0.7

    def test_extra_metadata(self, minimal_config):
        config = ExperimentConfig(**minimal_config, extra_metadata={"custom": "value"})
        assert config.extra_metadata["custom"] == "value"

    def test_config_name_required(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(model_name="test")

    def test_model_name_required(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(config_name="test")

    def test_serialization_roundtrip(self, minimal_config):
        config = ExperimentConfig(**minimal_config)
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.config_name == config.config_name

    def test_prompt_source_file(self, minimal_config):
        """Config with file prompt source."""
        config = ExperimentConfig(
            **minimal_config,
            prompt_source={"type": "file", "path": "/path/to/prompts.txt"},
        )
        assert config.prompt_source is not None
        assert isinstance(config.prompt_source, FilePromptSource)
        assert config.prompt_source.path == "/path/to/prompts.txt"

    def test_prompt_source_huggingface(self, minimal_config):
        """Config with HuggingFace prompt source."""
        config = ExperimentConfig(
            **minimal_config,
            prompt_source={
                "type": "huggingface",
                "dataset": "alpaca",
                "sample_size": 1000,
            },
        )
        assert config.prompt_source is not None
        assert isinstance(config.prompt_source, HuggingFacePromptSource)
        # Alias should be resolved
        assert config.prompt_source.dataset == "tatsu-lab/alpaca"
        assert config.prompt_source.sample_size == 1000

    def test_prompt_source_none_default(self, minimal_config):
        """Config without prompt source defaults to None."""
        config = ExperimentConfig(**minimal_config)
        assert config.prompt_source is None

    def test_pytorch_pipeline_parallel_rejected(self, minimal_config):
        """PyTorch backend with pipeline_parallel strategy is rejected.

        Pipeline parallelism requires full model access for generate() which
        PyTorch's pipelining abstraction can't provide for autoregressive
        generation. Users should use vLLM or TensorRT for PP inference.
        """
        with pytest.raises(
            ValidationError,
            match="Pipeline parallelism is not supported with PyTorch backend",
        ):
            ExperimentConfig(
                **minimal_config,
                backend="pytorch",
                gpu_list=[0, 1],
                sharding_config={"strategy": "pipeline_parallel", "num_shards": 2},
            )

    def test_vllm_pipeline_parallel_allowed(self, minimal_config):
        """vLLM backend supports pipeline_parallel strategy."""
        config = ExperimentConfig(
            **minimal_config,
            backend="vllm",
            gpu_list=[0, 1],
            sharding_config={"strategy": "pipeline_parallel", "num_shards": 2},
        )
        assert config.sharding_config.strategy == "pipeline_parallel"
        assert config.backend == "vllm"

    def test_pytorch_tensor_parallel_allowed(self, minimal_config):
        """PyTorch backend supports tensor_parallel strategy."""
        config = ExperimentConfig(
            **minimal_config,
            backend="pytorch",
            gpu_list=[0, 1],
            sharding_config={"strategy": "tensor_parallel", "num_shards": 2},
        )
        assert config.sharding_config.strategy == "tensor_parallel"
        assert config.backend == "pytorch"


class TestFilePromptSource:
    """Tests for FilePromptSource config."""

    def test_required_path(self):
        """Path is required."""
        with pytest.raises(ValidationError):
            FilePromptSource()

    def test_type_literal(self):
        """Type is fixed to 'file'."""
        source = FilePromptSource(path="/test.txt")
        assert source.type == "file"


class TestHuggingFacePromptSource:
    """Tests for HuggingFacePromptSource config."""

    def test_required_dataset(self):
        """Dataset is required."""
        with pytest.raises(ValidationError):
            HuggingFacePromptSource()

    def test_type_literal(self):
        """Type is fixed to 'huggingface'."""
        source = HuggingFacePromptSource(dataset="test")
        assert source.type == "huggingface"

    def test_defaults(self):
        """Check default values."""
        source = HuggingFacePromptSource(dataset="test/ds")
        assert source.split == "train"
        assert source.subset is None
        assert source.column is None
        assert source.sample_size is None
        assert source.shuffle is False
        assert source.seed == 42

    def test_builtin_alpaca_resolution(self):
        """Alpaca alias resolves to full path."""
        source = HuggingFacePromptSource(dataset="alpaca")
        assert source.dataset == BUILTIN_DATASETS["alpaca"]["path"]
        assert source.column == BUILTIN_DATASETS["alpaca"]["column"]

    def test_builtin_gsm8k_resolution(self):
        """GSM8K alias resolves with subset."""
        source = HuggingFacePromptSource(dataset="gsm8k")
        assert source.dataset == "gsm8k"
        assert source.subset == "main"
        assert source.column == "question"

    def test_explicit_column_preserved(self):
        """Explicit column is not overwritten by alias."""
        source = HuggingFacePromptSource(dataset="alpaca", column="output")
        assert source.column == "output"

    def test_sample_size_must_be_positive(self):
        """Sample size must be >= 1."""
        with pytest.raises(ValidationError):
            HuggingFacePromptSource(dataset="test", sample_size=0)

        with pytest.raises(ValidationError):
            HuggingFacePromptSource(dataset="test", sample_size=-1)


class TestExperimentConfigRandomSeed:
    """Tests for random_seed field in ExperimentConfig."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "config_name": "test_config",
            "model_name": "test-model",
        }

    def test_random_seed_default_none(self, minimal_config):
        """random_seed defaults to None."""
        config = ExperimentConfig(**minimal_config)
        assert config.random_seed is None

    def test_random_seed_accepts_integer(self, minimal_config):
        """random_seed accepts integer values."""
        config = ExperimentConfig(**minimal_config, random_seed=42)
        assert config.random_seed == 42

    def test_random_seed_accepts_zero(self, minimal_config):
        """random_seed accepts 0 as a valid seed."""
        config = ExperimentConfig(**minimal_config, random_seed=0)
        assert config.random_seed == 0

    def test_random_seed_accepts_large_integer(self, minimal_config):
        """random_seed accepts large integers."""
        config = ExperimentConfig(**minimal_config, random_seed=2**32 - 1)
        assert config.random_seed == 2**32 - 1

    def test_random_seed_field_exists(self):
        """random_seed field is present on ExperimentConfig."""
        assert "random_seed" in ExperimentConfig.model_fields

    def test_random_seed_in_serialization(self, minimal_config):
        """random_seed is included in JSON serialization."""
        config = ExperimentConfig(**minimal_config, random_seed=123)
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.random_seed == 123

    def test_random_seed_none_in_serialization(self, minimal_config):
        """random_seed=None is preserved in serialization."""
        config = ExperimentConfig(**minimal_config)
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.random_seed is None

    def test_random_seed_negative_allowed(self, minimal_config):
        """random_seed allows negative integers (no constraint)."""
        # Note: Whether negative seeds are allowed depends on your validation
        # This test documents the current behavior
        config = ExperimentConfig(**minimal_config, random_seed=-1)
        assert config.random_seed == -1
