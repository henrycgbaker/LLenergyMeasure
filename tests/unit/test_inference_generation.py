"""Tests for PyTorchBackend._build_generation_kwargs."""

import pytest

from llenergymeasure.config.backend_configs import PyTorchConfig
from llenergymeasure.config.models import DecoderConfig, ExperimentConfig
from llenergymeasure.core.inference_backends.pytorch import PyTorchBackend


class TestBuildGenerationKwargs:
    """Tests for PyTorchBackend._build_generation_kwargs method."""

    @pytest.fixture
    def backend(self):
        """Create a PyTorchBackend instance for testing."""
        return PyTorchBackend()

    @pytest.fixture
    def base_config(self):
        """Create a base ExperimentConfig for testing."""
        return ExperimentConfig(
            config_name="test",
            model_name="test-model",
            max_output_tokens=128,
            min_output_tokens=0,
        )

    def test_greedy_decoding_temp_zero(self, backend, base_config):
        """temperature=0 forces greedy decoding."""
        base_config.decoder = DecoderConfig(temperature=0.0)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs
        assert "top_k" not in kwargs
        assert "top_p" not in kwargs

    def test_greedy_decoding_ignores_sampling_params(self, backend, base_config):
        """temperature=0 ignores all other sampling params."""
        base_config.decoder = DecoderConfig(
            temperature=0.0,
            top_p=0.9,
            top_k=100,
            repetition_penalty=1.5,
        )
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["do_sample"] is False
        # No sampling params should be present
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs
        assert "repetition_penalty" not in kwargs

    def test_sampling_enabled_with_temperature(self, backend, base_config):
        """Sampling enabled when temperature > 0 and do_sample=True."""
        base_config.decoder = DecoderConfig(temperature=0.7, do_sample=True)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7

    def test_respects_do_sample_config(self, backend, base_config):
        """do_sample config value is respected."""
        base_config.decoder = DecoderConfig(temperature=1.0, do_sample=False)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["do_sample"] is False
        assert kwargs["temperature"] == 1.0
        # No sampling params when do_sample=False
        assert "top_k" not in kwargs

    def test_top_k_applied_when_positive(self, backend, base_config):
        """top_k is applied when > 0."""
        base_config.decoder = DecoderConfig(temperature=1.0, top_k=100)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["top_k"] == 100

    def test_top_k_not_applied_when_zero(self, backend, base_config):
        """top_k is not applied when == 0."""
        base_config.decoder = DecoderConfig(temperature=1.0, top_k=0)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert "top_k" not in kwargs

    def test_top_p_applied_when_less_than_one(self, backend, base_config):
        """top_p is applied when < 1.0."""
        base_config.decoder = DecoderConfig(temperature=1.0, top_p=0.9)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["top_p"] == 0.9

    def test_top_p_not_applied_when_one(self, backend, base_config):
        """top_p is not applied when == 1.0 (disabled)."""
        base_config.decoder = DecoderConfig(temperature=1.0, top_p=1.0)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert "top_p" not in kwargs

    def test_repetition_penalty_applied_when_not_one(self, backend, base_config):
        """repetition_penalty is applied when != 1.0."""
        base_config.decoder = DecoderConfig(temperature=1.0, repetition_penalty=1.2)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["repetition_penalty"] == 1.2

    def test_repetition_penalty_not_applied_when_one(self, backend, base_config):
        """repetition_penalty is not applied when == 1.0 (disabled)."""
        base_config.decoder = DecoderConfig(temperature=1.0, repetition_penalty=1.0)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert "repetition_penalty" not in kwargs

    def test_min_p_applied_when_positive(self, backend, base_config):
        """min_p is applied when > 0 (from PyTorchConfig, Tier 2 param)."""
        base_config.decoder = DecoderConfig(temperature=1.0)
        base_config.pytorch = PyTorchConfig(min_p=0.05)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["min_p"] == 0.05

    def test_min_p_not_applied_when_zero(self, backend, base_config):
        """min_p is not applied when == 0.0 (disabled, Tier 2 param)."""
        base_config.decoder = DecoderConfig(temperature=1.0)
        base_config.pytorch = PyTorchConfig(min_p=0.0)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert "min_p" not in kwargs

    def test_no_repeat_ngram_applied_when_positive(self, backend, base_config):
        """no_repeat_ngram_size is applied when > 0 (from PyTorchConfig, Tier 2 param)."""
        base_config.decoder = DecoderConfig(temperature=1.0)
        base_config.pytorch = PyTorchConfig(no_repeat_ngram_size=3)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["no_repeat_ngram_size"] == 3

    def test_no_repeat_ngram_not_applied_when_zero(self, backend, base_config):
        """no_repeat_ngram_size is not applied when == 0 (disabled, Tier 2 param)."""
        base_config.decoder = DecoderConfig(temperature=1.0)
        base_config.pytorch = PyTorchConfig(no_repeat_ngram_size=0)
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert "no_repeat_ngram_size" not in kwargs

    def test_min_length_respects_min_output_tokens(self, backend, base_config):
        """min_length includes min_output_tokens when input_length provided."""
        base_config.min_output_tokens = 10
        kwargs = backend._build_generation_kwargs(
            config=base_config, input_length=100, max_output_tokens=128
        )
        assert kwargs["min_length"] == 110  # input_length + min_output_tokens

    def test_max_new_tokens_from_config(self, backend, base_config):
        """max_new_tokens from config when specified."""
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=50)
        assert kwargs["max_new_tokens"] == 50

    def test_deterministic_preset_produces_greedy(self, backend, base_config):
        """Deterministic preset produces greedy decoding kwargs."""
        base_config.decoder = DecoderConfig(preset="deterministic")
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs

    def test_sampling_params_not_added_when_do_sample_false(self, backend, base_config):
        """Sampling params not added when do_sample=False even with temp > 0."""
        base_config.decoder = DecoderConfig(
            temperature=1.0,
            do_sample=False,
            top_k=100,
            top_p=0.9,
        )
        kwargs = backend._build_generation_kwargs(config=base_config, max_output_tokens=128)
        assert kwargs["do_sample"] is False
        # Sampling params should not be present
        assert "top_k" not in kwargs
        assert "top_p" not in kwargs
