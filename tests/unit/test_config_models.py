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
)


class TestBatchingConfig:
    """Tests for BatchingConfig."""

    def test_defaults(self):
        config = BatchingConfig()
        assert config.batch_size == 1
        assert config.dynamic_batching is False

    def test_custom_values(self):
        config = BatchingConfig(batch_size=8, dynamic_batching=True)
        assert config.batch_size == 8
        assert config.dynamic_batching is True

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            BatchingConfig(batch_size=0)


class TestLatencySimulation:
    """Tests for LatencySimulation."""

    def test_defaults(self):
        config = LatencySimulation()
        assert config.enabled is False
        assert config.delay_min_ms == 0.0

    def test_delay_range_validation(self):
        with pytest.raises(ValidationError, match="delay_min_ms must be <= delay_max_ms"):
            LatencySimulation(delay_min_ms=100, delay_max_ms=50)

    def test_valid_delay_range(self):
        config = LatencySimulation(enabled=True, delay_min_ms=10, delay_max_ms=100)
        assert config.delay_min_ms == 10
        assert config.delay_max_ms == 100


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
    """Tests for DecoderConfig."""

    def test_defaults(self):
        config = DecoderConfig()
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.do_sample is True

    def test_top_p_bounds(self):
        with pytest.raises(ValidationError):
            DecoderConfig(top_p=1.5)

        with pytest.raises(ValidationError):
            DecoderConfig(top_p=-0.1)


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
