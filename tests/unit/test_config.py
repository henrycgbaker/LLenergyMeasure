"""Unit tests for configuration models."""

import pytest
from pydantic import ValidationError

from llm_efficiency.config import (
    ExperimentConfig,
    BatchingConfig,
    QuantizationConfig,
    DecoderConfig,
    LatencySimulationConfig,
)


class TestBatchingConfig:
    """Tests for BatchingConfig."""

    def test_default_config(self):
        """Test default batching configuration."""
        config = BatchingConfig()
        assert config.batch_size == 16
        assert config.adaptive is False

    def test_custom_batch_size(self):
        """Test custom batch size."""
        config = BatchingConfig(batch_size=32)
        assert config.batch_size == 32

    def test_invalid_batch_size(self):
        """Test that batch size must be positive."""
        with pytest.raises(ValidationError):
            BatchingConfig(batch_size=0)

        with pytest.raises(ValidationError):
            BatchingConfig(batch_size=-1)

    def test_adaptive_batching_requires_params(self):
        """Test that adaptive batching requires additional parameters."""
        with pytest.raises(ValidationError):
            BatchingConfig(adaptive=True)  # Missing adaptive_max_tokens and max_batch_size

    def test_adaptive_batching_valid(self):
        """Test valid adaptive batching configuration."""
        config = BatchingConfig(
            batch_size=16, adaptive=True, adaptive_max_tokens=2048, max_batch_size=64
        )
        assert config.adaptive is True
        assert config.adaptive_max_tokens == 2048


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_no_quantization(self):
        """Test default (no quantization)."""
        config = QuantizationConfig()
        assert config.enabled is False
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False

    def test_4bit_quantization(self):
        """Test 4-bit quantization."""
        config = QuantizationConfig(load_in_4bit=True)
        assert config.enabled is True
        assert config.load_in_4bit is True
        assert config.load_in_8bit is False

    def test_8bit_quantization(self):
        """Test 8-bit quantization."""
        config = QuantizationConfig(load_in_8bit=True)
        assert config.enabled is True
        assert config.load_in_8bit is True
        assert config.load_in_4bit is False

    def test_cannot_enable_both_4bit_and_8bit(self):
        """Test that both 4-bit and 8-bit cannot be enabled simultaneously."""
        with pytest.raises(ValidationError, match="Cannot enable both"):
            QuantizationConfig(load_in_4bit=True, load_in_8bit=True)


class TestDecoderConfig:
    """Tests for DecoderConfig."""

    def test_default_greedy(self):
        """Test default greedy decoding."""
        config = DecoderConfig()
        assert config.mode == "greedy"
        assert config.temperature == 1.0
        assert config.do_sample is False

    def test_top_k_sampling(self):
        """Test top-k sampling configuration."""
        config = DecoderConfig(mode="top_k", top_k=50, temperature=0.7)
        assert config.mode == "top_k"
        assert config.top_k == 50
        assert config.temperature == 0.7
        assert config.do_sample is True  # Auto-enabled

    def test_top_p_sampling(self):
        """Test top-p (nucleus) sampling configuration."""
        config = DecoderConfig(mode="top_p", top_p=0.95, temperature=0.7)
        assert config.mode == "top_p"
        assert config.top_p == 0.95
        assert config.temperature == 0.7
        assert config.do_sample is True

    def test_top_k_requires_k_parameter(self):
        """Test that top_k mode requires top_k parameter."""
        with pytest.raises(ValidationError, match="top_k must be specified"):
            DecoderConfig(mode="top_k")

    def test_top_p_requires_p_parameter(self):
        """Test that top_p mode requires top_p parameter."""
        with pytest.raises(ValidationError, match="top_p must be specified"):
            DecoderConfig(mode="top_p")

    def test_invalid_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            DecoderConfig(temperature=0.0)  # Must be > 0

        with pytest.raises(ValidationError):
            DecoderConfig(temperature=3.0)  # Must be <= 2.0


class TestLatencySimulationConfig:
    """Tests for LatencySimulationConfig."""

    def test_default_no_simulation(self):
        """Test default (no latency simulation)."""
        config = LatencySimulationConfig()
        assert config.enabled is False

    def test_constant_latency(self):
        """Test constant latency configuration."""
        config = LatencySimulationConfig(
            enabled=True, delay_min=0.1, delay_max=0.1  # 100ms constant
        )
        assert config.enabled is True
        assert config.delay_min == 0.1
        assert config.delay_max == 0.1

    def test_variable_latency(self):
        """Test variable latency range."""
        config = LatencySimulationConfig(enabled=True, delay_min=0.05, delay_max=0.3)
        assert config.delay_min == 0.05
        assert config.delay_max == 0.3

    def test_delay_max_must_be_gte_delay_min(self):
        """Test that delay_max must be >= delay_min."""
        with pytest.raises(ValidationError):
            LatencySimulationConfig(enabled=True, delay_min=0.5, delay_max=0.1)

    def test_bursty_traffic(self):
        """Test bursty traffic configuration."""
        config = LatencySimulationConfig(
            enabled=True,
            delay_min=0.05,
            delay_max=0.2,
            simulate_burst=True,
            burst_interval=2.0,
            burst_size=10,
        )
        assert config.simulate_burst is True
        assert config.burst_interval == 2.0
        assert config.burst_size == 10


class TestExperimentConfig:
    """Tests for complete ExperimentConfig."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = ExperimentConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.precision == "float16"  # Default
        assert config.num_processes == 1

    def test_full_config(self):
        """Test complete configuration."""
        config = ExperimentConfig(
            model_name="meta-llama/Llama-3.2-1B",
            config_name="test_config",
            suite="testing",
            precision="bfloat16",
            num_processes=4,
            gpu_list=[0, 1, 2, 3],
            batching=BatchingConfig(batch_size=32),
            quantization=QuantizationConfig(load_in_8bit=True),
            decoder=DecoderConfig(mode="top_k", top_k=50, temperature=0.7),
        )
        assert config.model_name == "meta-llama/Llama-3.2-1B"
        assert config.precision == "bfloat16"
        assert config.num_processes == 4
        assert config.batching.batch_size == 32
        assert config.quantization.load_in_8bit is True

    def test_num_processes_validation(self):
        """Test that num_processes cannot exceed number of GPUs."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                model_name="test-model", num_processes=4, gpu_list=[0, 1]  # Only 2 GPUs
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                model_name="test-model", unknown_field="value"  # Extra field
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ExperimentConfig(
            model_name="test-model", precision="float32", num_processes=2
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        assert config_dict["precision"] == "float32"

    def test_from_legacy_dict(self):
        """Test migration from v1.0 dictionary format."""
        legacy_config = {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "fp_precision": "float16",
            "num_processes": 4,
            "batching_options": {
                "batch_size___fixed_batching": 32,
                "adaptive_batching": False,
            },
            "decoder_config": {
                "decoding_mode": "top_k",
                "decoder_temperature": 0.7,
                "decoder_top_k": 50,
                "decoder_top_p": None,
            },
        }

        config = ExperimentConfig.from_legacy_dict(legacy_config)
        assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.precision == "float16"
        assert config.num_processes == 4
        assert config.batching.batch_size == 32
        assert config.decoder.mode == "top_k"
        assert config.decoder.top_k == 50
