"""Unit tests for user configuration loading (llenergymeasure.config.user_config).

Tests XDG path, missing file graceful defaults, valid file loading, and the
removed-field error messages. Loading is all this module does: ranking the file's
values against env vars and the study file is the precedence chain's job, tested
in test_precedence.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.config.user_config import UserConfig, get_user_config_path, load_user_config
from llenergymeasure.utils.exceptions import ConfigError

# ---------------------------------------------------------------------------
# get_user_config_path
# ---------------------------------------------------------------------------


def test_get_user_config_path_returns_path():
    """get_user_config_path() returns a Path object."""
    path = get_user_config_path()
    assert isinstance(path, Path)


def test_get_user_config_path_ends_with_config_yaml():
    """get_user_config_path() ends with 'config.yaml'."""
    path = get_user_config_path()
    assert path.name == "config.yaml"


def test_get_user_config_path_contains_llenergymeasure():
    """get_user_config_path() includes 'llenergymeasure' in the path."""
    path = get_user_config_path()
    assert "llenergymeasure" in str(path)


# ---------------------------------------------------------------------------
# Missing file → defaults
# ---------------------------------------------------------------------------


def test_load_user_config_missing_file_returns_defaults(tmp_path):
    """load_user_config() with nonexistent file returns UserConfig with all defaults."""
    nonexistent = tmp_path / "nonexistent.yaml"
    config = load_user_config(config_path=nonexistent)
    assert isinstance(config, UserConfig)
    # Default values from the model
    assert config.output.results_dir == "./results"
    assert config.measurement.energy_sampler == "auto"
    assert config.ui.progress_mode == "auto"


def test_load_user_config_missing_file_no_error(tmp_path):
    """load_user_config() with missing file does not raise any exception."""
    nonexistent = tmp_path / "missing.yaml"
    # Should not raise
    config = load_user_config(config_path=nonexistent)
    assert config is not None


# ---------------------------------------------------------------------------
# Valid file loading
# ---------------------------------------------------------------------------


def test_load_user_config_valid_file(tmp_path):
    """load_user_config() with valid YAML returns UserConfig with overridden values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("output:\n  results_dir: /custom/results\nui:\n  progress_mode: plain\n")
    config = load_user_config(config_path=config_file)
    assert config.output.results_dir == "/custom/results"
    assert config.ui.progress_mode == "plain"


def test_load_user_config_partial_file_uses_defaults_for_missing(tmp_path):
    """Partial user config file merges with defaults for unspecified fields."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ui:\n  progress_mode: quiet\n")
    config = load_user_config(config_path=config_file)
    assert config.ui.progress_mode == "quiet"
    # Unspecified fields retain defaults
    assert config.output.results_dir == "./results"


def test_load_user_config_energy_sampler_override(tmp_path):
    """User config can override energy sampler preference."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("measurement:\n  energy_sampler: nvml\n")
    config = load_user_config(config_path=config_file)
    assert config.measurement.energy_sampler == "nvml"


# ---------------------------------------------------------------------------
# Removed fields error loudly and name their replacement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_text", "field", "hint"),
    [
        ("output:\n  model_cache_dir: ~/.cache/hf\n", "output.model_cache_dir", "HF_HOME"),
        (
            "measurement:\n  carbon_intensity_gco2_kwh: 300\n",
            "measurement.carbon_intensity_gco2_kwh",
            "CodeCarbon",
        ),
        ("measurement:\n  datacenter_pue: 1.2\n", "measurement.datacenter_pue", "CodeCarbon"),
        ("ui:\n  log_level: DEBUG\n", "ui.log_level", "LLEM_LOG_LEVEL"),
        ("ui:\n  prompt: false\n", "ui.prompt", "no interactive prompts"),
    ],
)
def test_removed_field_errors_loudly_and_names_the_field(tmp_path, yaml_text, field, hint):
    """A config file still carrying a removed field fails with the removal reason.

    The removed fields were never read by anything; each error names the mechanism
    that covers the need instead, so the fix is in the message.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    with pytest.raises(ConfigError) as exc:
        load_user_config(config_path=config_file)
    message = str(exc.value)
    assert field in message
    assert "removed" in message
    assert hint in message


def test_unknown_field_still_errors_plainly(tmp_path):
    """A field that never existed gets the plain extra-input error, not a removal note."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("output:\n  made_up_field: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_user_config(config_path=config_file)
    message = str(exc.value)
    assert "output.made_up_field" in message
    assert "removed" not in message


# ---------------------------------------------------------------------------
# UserRunnersConfig "auto" default behaviour
# ---------------------------------------------------------------------------


def test_user_runners_config_defaults_to_auto():
    """UserRunnersConfig() with no args has all three runner fields default to 'auto'."""
    from llenergymeasure.config.user_config import UserRunnersConfig

    config = UserRunnersConfig()
    assert config.transformers == "auto"
    assert config.vllm == "auto"
    assert config.tensorrt == "auto"


def test_user_runners_config_accepts_auto_in_file(tmp_path):
    """Config file with runners.transformers: auto loads without error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("runners:\n  transformers: auto\n")
    config = load_user_config(config_path=config_file)
    assert config.runners.transformers == "auto"


def test_user_runners_config_validator_accepts_auto():
    """UserRunnersConfig(transformers='auto') does not raise ValidationError."""
    from llenergymeasure.config.user_config import UserRunnersConfig

    config = UserRunnersConfig(transformers="auto")
    assert config.transformers == "auto"


def test_user_runners_config_validator_accepts_canonical_values():
    """The canonical process/container/container:<image> values validate."""
    from llenergymeasure.config.user_config import UserRunnersConfig

    config = UserRunnersConfig(
        transformers="process", vllm="container", tensorrt="container:img:v1"
    )
    assert config.transformers == "process"
    assert config.vllm == "container"
    assert config.tensorrt == "container:img:v1"


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        ("local", "use 'process'"),
        ("docker", "use 'container'"),
        ("docker:my/img:v1", "use 'container:my/img:v1'"),
    ],
)
def test_user_runners_config_validator_rejects_legacy_vocabulary(value, hint):
    """The renamed-in-v0.7 legacy values are rejected with a migration hint.

    The message must name the user's ACTUAL input value, the field context, and the
    canonical replacement.
    """
    import pydantic

    from llenergymeasure.config.user_config import UserRunnersConfig

    with pytest.raises(
        pydantic.ValidationError,
        match=rf"'{value}' was renamed in v0.7 \(runners.transformers\).*{hint}",
    ):
        UserRunnersConfig(transformers=value)
