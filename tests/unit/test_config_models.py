"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from llm_energy_measure.config.models import (
    BatchingConfig,
    DecoderConfig,
    ExperimentConfig,
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
