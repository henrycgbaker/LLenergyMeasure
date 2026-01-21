"""Tests for configuration loader."""

import pytest

from llm_energy_measure.config.loader import (
    deep_merge,
    load_config,
    load_config_dict,
    resolve_inheritance,
    validate_config,
)
from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        overlay = {"a": {"y": 5, "z": 6}}
        result = deep_merge(base, overlay)
        assert result == {"a": {"x": 1, "y": 5, "z": 6}, "b": 3}

    def test_overlay_replaces_non_dict(self):
        base = {"a": {"x": 1}}
        overlay = {"a": "replaced"}
        result = deep_merge(base, overlay)
        assert result == {"a": "replaced"}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        deep_merge(base, overlay)
        assert base == {"a": {"x": 1}}
        assert overlay == {"a": {"y": 2}}


class TestLoadConfigDict:
    """Tests for load_config_dict function."""

    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("config_name: test\nmodel_name: test-model\n")
        result = load_config_dict(config_file)
        assert result["config_name"] == "test"

    def test_load_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"config_name": "test", "model_name": "test-model"}')
        result = load_config_dict(config_file)
        assert result["config_name"] == "test"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not found"):
            load_config_dict(tmp_path / "nonexistent.yaml")

    def test_unsupported_format(self, tmp_path):
        config_file = tmp_path / "config.txt"
        config_file.write_text("config_name: test")
        with pytest.raises(ConfigurationError, match="Unsupported config format"):
            load_config_dict(config_file)

    def test_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content:")
        with pytest.raises(ConfigurationError, match="Failed to parse"):
            load_config_dict(config_file)


class TestResolveInheritance:
    """Tests for resolve_inheritance function."""

    def test_no_inheritance(self, tmp_path):
        config = {"config_name": "test", "model_name": "test-model"}
        result = resolve_inheritance(config, tmp_path / "config.yaml")
        assert result == config

    def test_single_inheritance(self, tmp_path):
        # Create base config
        base_file = tmp_path / "base.yaml"
        base_file.write_text("max_input_tokens: 512\nmax_output_tokens: 128\n")

        # Create child config
        child_config = {
            "_extends": "base.yaml",
            "config_name": "child",
            "model_name": "test-model",
        }

        result = resolve_inheritance(child_config, tmp_path / "child.yaml")
        assert result["config_name"] == "child"
        assert result["max_input_tokens"] == 512
        assert "_extends" not in result

    def test_circular_inheritance_detected(self, tmp_path):
        # Create configs that reference each other
        a_file = tmp_path / "a.yaml"
        b_file = tmp_path / "b.yaml"
        a_file.write_text("_extends: b.yaml\nconfig_name: a\n")
        b_file.write_text("_extends: a.yaml\nconfig_name: b\n")

        with pytest.raises(ConfigurationError, match="Circular"):
            load_config(a_file)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: meta-llama/Llama-2-7b-hf
max_input_tokens: 512
max_output_tokens: 256
""")
        config = load_config(config_file)
        assert isinstance(config, ExperimentConfig)
        assert config.config_name == "test_experiment"
        assert config.max_input_tokens == 512

    def test_load_with_inheritance(self, tmp_path):
        # Base config
        base = tmp_path / "base.yaml"
        base.write_text("""
max_input_tokens: 1024
decoder:
  temperature: 0.7
  top_p: 0.9
""")

        # Child config
        child = tmp_path / "child.yaml"
        child.write_text("""
_extends: base.yaml
config_name: child_experiment
model_name: test-model
decoder:
  temperature: 0.5
""")

        config = load_config(child)
        assert config.config_name == "child_experiment"
        assert config.max_input_tokens == 1024
        assert config.decoder.temperature == 0.5
        assert config.decoder.top_p == 0.9  # Inherited

    def test_load_invalid_config(self, tmp_path):
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("config_name: test\n")  # Missing model_name
        with pytest.raises(ConfigurationError, match="Invalid config"):
            load_config(config_file)


class TestValidateConfig:
    """Tests for validate_config function."""

    @pytest.fixture
    def base_config(self):
        return ExperimentConfig(
            config_name="test",
            model_name="test-model",
        )

    def test_no_warnings_for_valid_config(self, base_config):
        warnings = validate_config(base_config)
        assert len(warnings) == 0

    def test_warning_for_multi_process_single_gpu(self):
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            gpus=[0],
            num_processes=1,  # Valid, but let's test multi-process
        )
        # This is valid, no warning
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_warning_for_high_output_tokens(self):
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            max_output_tokens=4096,
        )
        warnings = validate_config(config)
        assert any("max_output_tokens" in w.field for w in warnings)

    def test_warning_for_quant_with_fp32(self):
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            fp_precision="float32",
            quantization={"quantization": True, "load_in_4bit": True},
        )
        warnings = validate_config(config)
        assert any("quantization" in w.field for w in warnings)

    def test_warning_for_temp_and_top_p_modified(self):
        """Warning when both temperature and top_p are modified from defaults."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            decoder={"temperature": 0.7, "top_p": 0.9},
        )
        warnings = validate_config(config)
        assert any("decoder" in w.field and "temperature" in w.message.lower() for w in warnings)

    def test_warning_for_do_sample_with_temp_zero(self):
        """Info warning when do_sample=True with temperature=0."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            decoder={"temperature": 0.0, "do_sample": True},
        )
        warnings = validate_config(config)
        assert any("do_sample" in w.field and w.severity == "info" for w in warnings)

    def test_no_warning_for_deterministic_preset(self):
        """No warning for deterministic preset (greedy decoding)."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            decoder={"preset": "deterministic"},
        )
        warnings = validate_config(config)
        # Should have no decoder-related warnings (temp=0 with do_sample=False is fine)
        decoder_warnings = [w for w in warnings if "decoder" in w.field]
        assert len(decoder_warnings) == 0

    def test_no_warning_for_default_decoder_config(self, base_config):
        """No warnings for default decoder config."""
        warnings = validate_config(base_config)
        decoder_warnings = [w for w in warnings if "decoder" in w.field]
        assert len(decoder_warnings) == 0

    def test_config_warning_str_format(self):
        """ConfigWarning __str__ produces expected format."""
        from llm_energy_measure.config.loader import ConfigWarning

        warning = ConfigWarning(
            field="test_field",
            message="Test message",
            severity="warning",
        )
        assert "[WARNING]" in str(warning)
        assert "test_field" in str(warning)
        assert "Test message" in str(warning)

    def test_config_warning_info_severity(self):
        """ConfigWarning with info severity."""
        from llm_energy_measure.config.loader import ConfigWarning

        warning = ConfigWarning(
            field="test_field",
            message="Info message",
            severity="info",
        )
        assert "[INFO]" in str(warning)
        assert warning.severity == "info"

    def test_config_warning_error_severity(self):
        """ConfigWarning with error severity."""
        from llm_energy_measure.config.loader import ConfigWarning

        warning = ConfigWarning(
            field="test_field",
            message="Error message",
            severity="error",
        )
        assert "[ERROR]" in str(warning)
        assert warning.severity == "error"

    def test_config_warning_to_result_string(self):
        """ConfigWarning.to_result_string() formats for embedding in results."""
        from llm_energy_measure.config.loader import ConfigWarning

        warning = ConfigWarning(
            field="decoder",
            message="Sampling params ignored",
            severity="error",
        )
        result_str = warning.to_result_string()
        assert result_str == "error: decoder - Sampling params ignored"

    def test_error_for_quantization_without_bit_mode(self):
        """Error when quantization=True but no bit mode specified."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            quantization={"quantization": True},  # No load_in_4bit or load_in_8bit
        )
        warnings = validate_config(config)
        error_warnings = [w for w in warnings if w.severity == "error"]
        assert any("quantization" in w.field and "4bit" in w.message for w in error_warnings)

    def test_error_for_sharding_exceeds_gpus(self):
        """Error when num_shards exceeds available GPUs.

        Note: With the new parallelism migration, this now fails at model
        construction time rather than in validate_config(). We verify the
        Pydantic validation error is raised.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="parallelism.degree.*must be <=.*gpus"):
            ExperimentConfig(
                config_name="test",
                model_name="test-model",
                gpus=[0, 1],
                sharding={"strategy": "tensor_parallel", "num_shards": 4},
            )

    def test_error_for_sampling_params_in_deterministic_mode(self):
        """Error when sampling params set in deterministic mode."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            decoder={"temperature": 0.0, "top_k": 100, "top_p": 0.9},
        )
        warnings = validate_config(config)
        error_warnings = [w for w in warnings if w.severity == "error"]
        assert any("decoder" in w.field and "deterministic" in w.message for w in error_warnings)

    def test_no_error_for_default_sampling_params_in_deterministic_mode(self):
        """No error when only default sampling params in deterministic mode."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            decoder={"preset": "deterministic"},  # Default params, temp=0
        )
        warnings = validate_config(config)
        error_warnings = [w for w in warnings if w.severity == "error"]
        assert len(error_warnings) == 0


class TestHasBlockingWarnings:
    """Tests for has_blocking_warnings function."""

    def test_no_blocking_with_empty_list(self):
        from llm_energy_measure.config.loader import has_blocking_warnings

        assert has_blocking_warnings([]) is False

    def test_no_blocking_with_info_warnings(self):
        from llm_energy_measure.config.loader import ConfigWarning, has_blocking_warnings

        warnings = [
            ConfigWarning(field="a", message="info 1", severity="info"),
            ConfigWarning(field="b", message="info 2", severity="info"),
        ]
        assert has_blocking_warnings(warnings) is False

    def test_no_blocking_with_warning_warnings(self):
        from llm_energy_measure.config.loader import ConfigWarning, has_blocking_warnings

        warnings = [
            ConfigWarning(field="a", message="warning 1", severity="warning"),
            ConfigWarning(field="b", message="info 1", severity="info"),
        ]
        assert has_blocking_warnings(warnings) is False

    def test_blocking_with_error_warning(self):
        from llm_energy_measure.config.loader import ConfigWarning, has_blocking_warnings

        warnings = [
            ConfigWarning(field="a", message="info 1", severity="info"),
            ConfigWarning(field="b", message="error 1", severity="error"),
        ]
        assert has_blocking_warnings(warnings) is True

    def test_blocking_with_multiple_errors(self):
        from llm_energy_measure.config.loader import ConfigWarning, has_blocking_warnings

        warnings = [
            ConfigWarning(field="a", message="error 1", severity="error"),
            ConfigWarning(field="b", message="error 2", severity="error"),
        ]
        assert has_blocking_warnings(warnings) is True
