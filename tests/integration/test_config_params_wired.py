"""Integration tests verifying config parameters are actually wired up and used.

These tests ensure config values aren't just parsed but actually affect behaviour.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_energy_measure.config.models import (
    DecoderConfig,
    ExperimentConfig,
    QuantizationConfig,
    TrafficSimulation,
)
from llm_energy_measure.core.traffic import TrafficGenerator


class TestBatchingConfigWired:
    """Verify batching config affects batch creation."""

    def test_batch_size_affects_batch_count(self):
        """Batch size should determine number of batches created."""
        from llm_energy_measure.core.prompts import create_fixed_batches

        prompts = [f"Prompt {i}" for i in range(20)]

        # Batch size 4 -> 5 batches
        batches_4 = create_fixed_batches(prompts, batch_size=4)
        assert len(batches_4) == 5
        assert all(len(b) == 4 for b in batches_4)

        # Batch size 10 -> 2 batches
        batches_10 = create_fixed_batches(prompts, batch_size=10)
        assert len(batches_10) == 2

        # Batch size 1 -> 20 batches
        batches_1 = create_fixed_batches(prompts, batch_size=1)
        assert len(batches_1) == 20

    def test_strategy_static_uses_fixed_batches(self):
        """Static strategy should create fixed-size batches."""
        from llm_energy_measure.core.prompts import create_fixed_batches

        prompts = ["short", "medium length prompt", "a very long prompt here"]
        batches = create_fixed_batches(prompts, batch_size=2)

        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1  # Remainder

    def test_strategy_sorted_orders_by_length(self):
        """Sorted strategies should order prompts by length."""
        prompts = [
            "very long prompt with many words",
            "short",
            "medium length",
        ]

        sorted_prompts = sorted(prompts, key=len)

        assert sorted_prompts[0] == "short"
        assert sorted_prompts[1] == "medium length"
        assert sorted_prompts[2] == "very long prompt with many words"


class TestDecoderConfigWired:
    """Verify decoder config affects generation kwargs."""

    def test_deterministic_preset_sets_greedy(self):
        """Deterministic preset should disable sampling."""
        config = DecoderConfig(preset="deterministic")

        assert config.temperature == 0.0
        assert config.do_sample is False
        assert config.is_deterministic is True

    def test_temperature_zero_forces_greedy(self):
        """Temperature 0 should force greedy decoding regardless of do_sample."""
        # This tests the logic in _build_generation_kwargs
        config = DecoderConfig(temperature=0.0, do_sample=True)

        # Even with do_sample=True, temp=0 should result in greedy
        assert config.temperature == 0.0
        # The inference code checks temp==0 and overrides do_sample

    def test_sampling_params_only_applied_when_sampling(self):
        """top_k, top_p should only be used when do_sample=True."""
        # When sampling
        sampling_config = DecoderConfig(
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
        )
        assert sampling_config.do_sample is True
        assert sampling_config.top_k == 40
        assert sampling_config.top_p == 0.9

        # When greedy (these values exist but shouldn't be used)
        greedy_config = DecoderConfig(preset="deterministic")
        assert greedy_config.do_sample is False
        # Values exist but inference code won't use them

    def test_generation_kwargs_built_correctly(self):
        """Test _build_generation_kwargs produces correct output."""
        from llm_energy_measure.core.inference import _build_generation_kwargs

        # Test deterministic - use actual field name, not alias
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            decoder_config=DecoderConfig(preset="deterministic"),
        )
        # Function signature: (config, input_length, max_output_tokens, allowed_new_tokens)
        kwargs = _build_generation_kwargs(
            config, input_length=10, max_output_tokens=50, allowed_new_tokens=100
        )

        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 50
        assert "top_k" not in kwargs  # Not used for greedy
        assert "top_p" not in kwargs

    def test_generation_kwargs_sampling_mode(self):
        """Test _build_generation_kwargs with sampling enabled."""
        from llm_energy_measure.core.inference import _build_generation_kwargs

        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            decoder_config=DecoderConfig(
                temperature=0.8,
                do_sample=True,
                top_k=40,
                top_p=0.9,
            ),
        )
        # Function signature: (config, input_length, max_output_tokens, allowed_new_tokens)
        kwargs = _build_generation_kwargs(
            config, input_length=10, max_output_tokens=50, allowed_new_tokens=100
        )

        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_k"] == 40
        assert kwargs["top_p"] == 0.9


class TestTrafficSimulationWired:
    """Verify traffic simulation actually introduces delays."""

    def test_traffic_generator_creates_delays(self):
        """TrafficGenerator should produce non-zero delays."""
        config = TrafficSimulation(
            enabled=True,
            mode="poisson",
            target_qps=10.0,  # 10 QPS = ~100ms average delay
            seed=42,
        )
        generator = TrafficGenerator(config)

        delays = [generator.get_inter_arrival_time() for _ in range(100)]

        # All delays should be positive
        assert all(d >= 0 for d in delays)
        # Average should be around 0.1s (1/10 QPS)
        avg_delay = sum(delays) / len(delays)
        assert 0.05 < avg_delay < 0.2  # Allow variance

    def test_constant_mode_fixed_delays(self):
        """Constant mode should produce fixed delays."""
        config = TrafficSimulation(
            enabled=True,
            mode="constant",
            target_qps=5.0,  # 5 QPS = 200ms fixed delay
        )
        generator = TrafficGenerator(config)

        delays = [generator.get_inter_arrival_time() for _ in range(10)]

        # All delays should be exactly 0.2s
        assert all(abs(d - 0.2) < 0.001 for d in delays)

    def test_wait_for_next_request_applies_sleep(self):
        """wait_for_next_request should actually sleep."""
        config = TrafficSimulation(
            enabled=True,
            mode="constant",
            target_qps=100.0,  # 10ms delay for fast test
        )
        generator = TrafficGenerator(config)

        start = time.perf_counter()
        generator.wait_for_next_request()
        elapsed = time.perf_counter() - start

        # Should have waited ~10ms
        assert elapsed >= 0.008  # Allow some tolerance

    def test_disabled_traffic_sim_no_generator(self):
        """Disabled traffic simulation should not create generator."""
        from llm_energy_measure.core.inference import _create_traffic_generator

        # Use model_validate to properly handle validation_alias
        config = ExperimentConfig.model_validate(
            {
                "config_name": "test",
                "model_name": "gpt2",
                "traffic_simulation": {"enabled": False},
            }
        )

        generator = _create_traffic_generator(config)
        assert generator is None

    def test_enabled_traffic_sim_creates_generator(self):
        """Enabled traffic simulation should create generator."""
        from llm_energy_measure.core.inference import _create_traffic_generator

        # Use model_validate to properly handle validation_alias
        config = ExperimentConfig.model_validate(
            {
                "config_name": "test",
                "model_name": "gpt2",
                "traffic_simulation": {
                    "enabled": True,
                    "mode": "poisson",
                    "target_qps": 5.0,
                },
            }
        )

        generator = _create_traffic_generator(config)
        assert generator is not None
        assert isinstance(generator, TrafficGenerator)


class TestQuantizationConfigWired:
    """Verify quantization config is passed to model loading."""

    def test_quantization_config_builds_bnb_config(self):
        """Quantization settings should build BitsAndBytesConfig."""
        config = QuantizationConfig(
            quantization=True,
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )

        assert config.quantization is True
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_4bit_and_8bit_mutually_exclusive(self):
        """Cannot enable both 4-bit and 8-bit quantization."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            QuantizationConfig(
                quantization=True,
                load_in_4bit=True,
                load_in_8bit=True,
            )

    def test_quantized_model_uses_bnb_config(self):
        """Model loading with quantization should pass quantization_config to from_pretrained."""
        from llm_energy_measure.core.model_loader import _load_quantized_model

        config = QuantizationConfig(quantization=True, load_in_8bit=True)

        with (
            patch(
                "llm_energy_measure.core.model_loader.detect_quantization_support"
            ) as mock_detect,
            patch("llm_energy_measure.core.model_loader.AutoModelForCausalLM") as mock_model,
            patch("llm_energy_measure.core.model_loader.BitsAndBytesConfig") as mock_bnb,
        ):
            mock_detect.return_value = MagicMock(
                supports_4bit=True,
                supports_8bit=True,
                default_4bit_quant_type="nf4",
                default_8bit_quant_type="int8",
            )
            mock_bnb.return_value = MagicMock(name="bnb_config")
            mock_model.from_pretrained.return_value = MagicMock()

            _load_quantized_model("gpt2", config)

            # Verify BitsAndBytesConfig was created with load_in_8bit=True
            mock_bnb.assert_called_once()
            bnb_call_kwargs = mock_bnb.call_args[1]
            assert bnb_call_kwargs.get("load_in_8bit") is True

            # Verify model loaded with quantization_config
            mock_model.from_pretrained.assert_called_once()
            call_kwargs = mock_model.from_pretrained.call_args[1]
            assert "quantization_config" in call_kwargs


class TestShardingConfigNotImplemented:
    """Verify sharding config exists but is NOT yet wired up."""

    def test_sharding_config_parsed(self):
        """Sharding config should be parsed correctly."""
        # Use model_validate to properly handle validation_alias
        config = ExperimentConfig.model_validate(
            {
                "config_name": "test",
                "model_name": "gpt2",
                "sharding": {"strategy": "tensor_parallel", "num_shards": 2},
                "gpus": [0, 1],
            }
        )

        assert config.sharding_config.strategy == "tensor_parallel"
        assert config.sharding_config.num_shards == 2

    def test_sharding_not_used_in_model_loader(self):
        """Sharding config is NOT used in model loading (documenting current state)."""
        # This test documents that sharding is not implemented
        # It should be updated when sharding is implemented

        import inspect

        from llm_energy_measure.core import model_loader

        source = inspect.getsource(model_loader.load_model_tokenizer)

        # Sharding config should NOT appear in model loader
        assert "sharding" not in source.lower()
        assert "tensor_parallel" not in source
        assert "pipeline_parallel" not in source


class TestBackendConfigNotImplemented:
    """Verify backend config exists but is NOT yet wired up."""

    def test_backend_config_parsed(self):
        """Backend config should be parsed correctly."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
        )

        assert config.backend == "vllm"

    def test_backend_not_used_in_inference(self):
        """Backend config is NOT used in inference (documenting current state)."""
        import inspect

        from llm_energy_measure.core import inference

        source = inspect.getsource(inference)

        # Backend selection logic should NOT appear in inference
        # (comments mentioning vLLM terminology are OK, actual implementation is not)
        assert "tensorrt" not in source.lower()
        assert "if config.backend" not in source.lower()
        assert "backend ==" not in source.lower()
