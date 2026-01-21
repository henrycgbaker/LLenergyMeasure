"""Tests for parallelism strategies.

Tests the parallelism module which provides tensor parallelism (TP) and
pipeline parallelism (PP) strategies for multi-GPU inference.
"""

import pytest

from llm_energy_measure.config.models import ShardingConfig
from llm_energy_measure.core.parallelism import (
    NoParallelism,
    PipelineParallelStrategy,
    TensorParallelStrategy,
    get_parallelism_strategy,
    is_model_tp_compatible,
)


class TestGetParallelismStrategy:
    """Tests for the factory function."""

    def test_none_strategy_returns_no_parallelism(self):
        config = ShardingConfig(strategy="none")
        strategy = get_parallelism_strategy(config)
        assert isinstance(strategy, NoParallelism)

    def test_tensor_parallel_strategy(self):
        config = ShardingConfig(strategy="tensor_parallel", num_shards=2)
        strategy = get_parallelism_strategy(config)
        assert isinstance(strategy, TensorParallelStrategy)

    def test_pipeline_parallel_strategy(self):
        config = ShardingConfig(strategy="pipeline_parallel", num_shards=2)
        strategy = get_parallelism_strategy(config)
        assert isinstance(strategy, PipelineParallelStrategy)

    def test_unknown_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        # Create config with valid strategy first, then modify
        config = ShardingConfig(strategy="none")
        # Manually set invalid strategy (bypassing validation for test)
        object.__setattr__(config, "strategy", "invalid")

        with pytest.raises(ValueError, match="Unknown sharding strategy"):
            get_parallelism_strategy(config)


class TestNoParallelism:
    """Tests for NoParallelism strategy (default behaviour)."""

    def test_setup_is_noop(self):
        """Setup should be a no-op for default strategy."""
        strategy = NoParallelism()
        config = ShardingConfig(strategy="none")
        # Should not raise
        strategy.setup(config, gpus=[0])

    def test_prepare_model_kwargs_returns_device_map_auto(self):
        """Should return device_map='auto' for automatic placement."""
        strategy = NoParallelism()
        kwargs = strategy.prepare_model_kwargs()
        assert kwargs == {"device_map": "auto"}

    def test_wrap_model_returns_unchanged(self):
        """Wrap model should return the model unchanged."""
        strategy = NoParallelism()
        mock_model = object()  # Just need any object
        result = strategy.wrap_model(mock_model)
        assert result is mock_model

    def test_does_not_require_torchrun(self):
        """Default strategy uses accelerate, not torchrun."""
        strategy = NoParallelism()
        assert strategy.requires_torchrun is False


class TestTensorParallelStrategy:
    """Tests for TensorParallelStrategy (HuggingFace native TP)."""

    def test_setup_validates_gpu_count(self):
        """Should raise if num_shards exceeds available GPUs."""
        strategy = TensorParallelStrategy()
        config = ShardingConfig(strategy="tensor_parallel", num_shards=4)

        with pytest.raises(ValueError, match="exceeds available GPUs"):
            strategy.setup(config, gpus=[0, 1])  # Only 2 GPUs

    def test_setup_accepts_valid_config(self):
        """Should succeed when GPUs are sufficient."""
        strategy = TensorParallelStrategy()
        config = ShardingConfig(strategy="tensor_parallel", num_shards=2)

        # Should not raise
        strategy.setup(config, gpus=[0, 1])

    def test_prepare_model_kwargs_includes_tp_plan(self):
        """Should return tp_plan='auto' for HF native TP."""
        strategy = TensorParallelStrategy()
        kwargs = strategy.prepare_model_kwargs()
        assert "tp_plan" in kwargs
        assert kwargs["tp_plan"] == "auto"

    def test_wrap_model_returns_unchanged(self):
        """TP is applied at load time, so wrap is a no-op."""
        strategy = TensorParallelStrategy()
        mock_model = object()
        result = strategy.wrap_model(mock_model)
        assert result is mock_model

    def test_requires_torchrun(self):
        """TP requires torchrun launcher."""
        strategy = TensorParallelStrategy()
        assert strategy.requires_torchrun is True


class TestPipelineParallelStrategy:
    """Tests for PipelineParallelStrategy (PyTorch native PP)."""

    def test_setup_validates_gpu_count(self):
        """Should raise if num_shards exceeds available GPUs."""
        strategy = PipelineParallelStrategy()
        config = ShardingConfig(strategy="pipeline_parallel", num_shards=8)

        with pytest.raises(ValueError, match="exceeds available GPUs"):
            strategy.setup(config, gpus=[0, 1])  # Only 2 GPUs

    def test_prepare_model_kwargs_returns_cpu_device_map(self):
        """PP loads model on CPU first for splitting."""
        strategy = PipelineParallelStrategy()
        kwargs = strategy.prepare_model_kwargs()
        assert kwargs["device_map"] == "cpu"
        assert kwargs["low_cpu_mem_usage"] is True

    def test_requires_torchrun(self):
        """PP requires torchrun launcher."""
        strategy = PipelineParallelStrategy()
        assert strategy.requires_torchrun is True


class TestShardingConfigExtensions:
    """Tests for ShardingConfig extended fields."""

    def test_defaults(self):
        """Default sharding config should be none strategy."""
        config = ShardingConfig()
        assert config.strategy == "none"
        assert config.num_shards == 1
        assert config.tp_plan is None

    def test_tp_plan_defaults_to_auto_for_tensor_parallel(self):
        """tp_plan should default to 'auto' when strategy is tensor_parallel."""
        config = ShardingConfig(strategy="tensor_parallel", num_shards=2)
        assert config.tp_plan == "auto"

    def test_tp_plan_stays_none_for_other_strategies(self):
        """tp_plan should stay None for non-TP strategies."""
        config = ShardingConfig(strategy="none")
        assert config.tp_plan is None

        config = ShardingConfig(strategy="pipeline_parallel", num_shards=2)
        assert config.tp_plan is None

    def test_pipeline_parallel_config(self):
        """Pipeline parallel config should just need strategy and num_shards."""
        config = ShardingConfig(strategy="pipeline_parallel", num_shards=4)
        assert config.strategy == "pipeline_parallel"
        assert config.num_shards == 4
        assert config.tp_plan is None  # TP options not used for PP


class TestModelTPCompatibility:
    """Tests for model TP compatibility checking."""

    def test_llama_is_compatible(self):
        assert is_model_tp_compatible("meta-llama/Llama-2-7b-hf") is True
        assert is_model_tp_compatible("meta-llama/Meta-Llama-3-8B") is True

    def test_mistral_is_compatible(self):
        assert is_model_tp_compatible("mistralai/Mistral-7B-v0.1") is True
        assert is_model_tp_compatible("mistralai/Mixtral-8x7B-v0.1") is True

    def test_qwen_is_compatible(self):
        assert is_model_tp_compatible("Qwen/Qwen2-7B") is True

    def test_phi_is_compatible(self):
        assert is_model_tp_compatible("microsoft/phi-3-mini-4k-instruct") is True

    def test_gemma_is_compatible(self):
        assert is_model_tp_compatible("google/gemma-2b") is True

    def test_gpt2_is_not_compatible(self):
        """GPT-2 doesn't have tp_plan support."""
        assert is_model_tp_compatible("gpt2") is False
        assert is_model_tp_compatible("openai-gpt") is False

    def test_unknown_model_is_not_compatible(self):
        """Unknown models should return False."""
        assert is_model_tp_compatible("some-random/model-name") is False

    def test_case_insensitive(self):
        """Model name matching should be case-insensitive."""
        assert is_model_tp_compatible("META-LLAMA/LLAMA-2-7B-HF") is True
        assert is_model_tp_compatible("MISTRALAI/MISTRAL-7B") is True
