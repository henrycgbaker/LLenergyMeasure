"""Tests for configuration models.

Tests the backend-native configuration architecture with:
- Tier 1: Universal params (config_name, model_name, decoder, etc.)
- Tier 2: Backend-specific params (pytorch, vllm, tensorrt sections)
"""

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import (
    BUILTIN_DATASETS,
    DecoderConfig,
    ExperimentConfig,
    FilePromptSource,
    HuggingFacePromptSource,
    TrafficSimulation,
)


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


class TestDecoderConfig:
    """Tests for DecoderConfig with sampling presets.

    Note: In the backend-native architecture, DecoderConfig only contains
    universal params (temperature, do_sample, top_p, repetition_penalty, preset).
    Backend-specific decoder extensions (top_k, min_p, etc.) are in backend configs.
    """

    def test_defaults(self):
        """Default values match specification."""
        config = DecoderConfig()
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.do_sample is True
        assert config.repetition_penalty == 1.0
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
    """Tests for ExperimentConfig with backend-native architecture."""

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
        assert config.gpus == [0]

    def test_full_config(self, minimal_config):
        full = {
            **minimal_config,
            "max_input_tokens": 1024,
            "max_output_tokens": 256,
            "gpus": [0, 1, 2, 3],
            "fp_precision": "float16",
        }
        config = ExperimentConfig(**full)
        assert config.max_input_tokens == 1024

    def test_min_max_tokens_validation(self, minimal_config):
        with pytest.raises(ValidationError, match="min_output_tokens.*must be <="):
            ExperimentConfig(
                **minimal_config,
                min_output_tokens=200,
                max_output_tokens=100,
            )

    def test_gpus_from_int(self, minimal_config):
        config = ExperimentConfig(**minimal_config, gpus=0)
        assert config.gpus == [0]

    def test_nested_decoder_config(self, minimal_config):
        """Decoder config can be set as dict."""
        config = ExperimentConfig(
            **minimal_config,
            decoder={"temperature": 0.7},
        )
        assert config.decoder.temperature == 0.7

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

    def test_prompts_file(self, minimal_config):
        """Config with file prompt source."""
        config = ExperimentConfig(
            **minimal_config,
            prompts={"type": "file", "path": "/path/to/prompts.txt"},
        )
        assert config.prompts is not None
        assert isinstance(config.prompts, FilePromptSource)
        assert config.prompts.path == "/path/to/prompts.txt"

    def test_prompts_huggingface(self, minimal_config):
        """Config with HuggingFace prompt source."""
        config = ExperimentConfig(
            **minimal_config,
            prompts={
                "type": "huggingface",
                "dataset": "alpaca",
                "sample_size": 1000,
            },
        )
        assert config.prompts is not None
        assert isinstance(config.prompts, HuggingFacePromptSource)
        # Alias should be resolved
        assert config.prompts.dataset == "tatsu-lab/alpaca"
        assert config.prompts.sample_size == 1000

    def test_prompts_none_default(self, minimal_config):
        """Config without prompts defaults to None."""
        config = ExperimentConfig(**minimal_config)
        assert config.prompts is None

    def test_backend_selection(self, minimal_config):
        """Backend can be selected."""
        for backend in ["pytorch", "vllm", "tensorrt"]:
            config = ExperimentConfig(**minimal_config, backend=backend)
            assert config.backend == backend

    def test_pytorch_config_section(self, minimal_config):
        """PyTorch backend config section works."""
        config = ExperimentConfig(
            **minimal_config,
            backend="pytorch",
            decoder={"top_k": 50},  # top_k is now universal in decoder
            pytorch={
                "batch_size": 4,
                "batching_strategy": "dynamic",
                "load_in_4bit": True,
                "min_p": 0.1,  # min_p is still backend-specific
            },
        )
        assert config.pytorch is not None
        assert config.pytorch.batch_size == 4
        assert config.pytorch.batching_strategy == "dynamic"
        assert config.pytorch.load_in_4bit is True
        assert config.decoder.top_k == 50
        assert config.pytorch.min_p == 0.1

    def test_vllm_config_section(self, minimal_config):
        """vLLM backend config section works."""
        config = ExperimentConfig(
            **minimal_config,
            backend="vllm",
            decoder={"top_k": 50},  # top_k is now universal in decoder
            vllm={
                "max_num_seqs": 128,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.8,
                "min_p": 0.1,  # min_p is still backend-specific
            },
        )
        assert config.vllm is not None
        assert config.vllm.max_num_seqs == 128
        assert config.vllm.tensor_parallel_size == 2
        assert config.vllm.gpu_memory_utilization == 0.8
        assert config.decoder.top_k == 50
        assert config.vllm.min_p == 0.1

    def test_tensorrt_config_section(self, minimal_config):
        """TensorRT backend config section works."""
        config = ExperimentConfig(
            **minimal_config,
            backend="tensorrt",
            decoder={"top_k": 50},  # top_k is now universal in decoder
            tensorrt={
                "max_batch_size": 16,
                "tp_size": 2,
                "quantization": "fp8",
            },
        )
        assert config.tensorrt is not None
        assert config.tensorrt.max_batch_size == 16
        assert config.tensorrt.tp_size == 2
        assert config.tensorrt.quantization == "fp8"
        assert config.decoder.top_k == 50


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
        config = ExperimentConfig(**minimal_config, random_seed=-1)
        assert config.random_seed == -1
