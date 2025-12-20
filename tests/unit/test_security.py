"""Tests for security utilities."""

from pathlib import Path

import pytest

from llm_bench.exceptions import ConfigurationError
from llm_bench.security import (
    check_env_for_secrets,
    is_safe_path,
    sanitize_experiment_id,
    validate_path,
)


class TestValidatePath:
    """Tests for validate_path function."""

    def test_relative_path_allowed(self, tmp_path):
        relative = Path("some/relative/path")
        result = validate_path(relative, must_exist=False, allow_relative=True)
        assert result.is_absolute()

    def test_relative_path_disallowed(self):
        relative = Path("some/relative/path")
        with pytest.raises(ConfigurationError, match="Absolute path required"):
            validate_path(relative, allow_relative=False)

    def test_must_exist_fails(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.txt"
        with pytest.raises(ConfigurationError, match="does not exist"):
            validate_path(nonexistent, must_exist=True)

    def test_must_exist_succeeds(self, tmp_path):
        existing = tmp_path / "exists.txt"
        existing.touch()
        result = validate_path(existing, must_exist=True)
        assert result.exists()


class TestIsSafePath:
    """Tests for is_safe_path function."""

    def test_path_within_base(self, tmp_path):
        target = tmp_path / "subdir" / "file.txt"
        assert is_safe_path(tmp_path, target) is True

    def test_path_outside_base(self, tmp_path):
        target = tmp_path.parent / "other_dir" / "file.txt"
        assert is_safe_path(tmp_path, target) is False

    def test_path_traversal_attempt(self, tmp_path):
        target = tmp_path / ".." / ".." / "etc" / "passwd"
        assert is_safe_path(tmp_path, target) is False


class TestSanitizeExperimentId:
    """Tests for sanitize_experiment_id function."""

    def test_valid_id_unchanged(self):
        assert sanitize_experiment_id("exp_001") == "exp_001"

    def test_alphanumeric_allowed(self):
        assert sanitize_experiment_id("Exp123Test") == "Exp123Test"

    def test_special_chars_replaced(self):
        assert sanitize_experiment_id("exp/with/slashes") == "exp_with_slashes"
        assert sanitize_experiment_id("exp with spaces") == "exp_with_spaces"

    def test_hyphen_allowed(self):
        assert sanitize_experiment_id("exp-with-hyphens") == "exp-with-hyphens"

    def test_dot_allowed(self):
        assert sanitize_experiment_id("exp.v1.0") == "exp.v1.0"

    def test_empty_id_raises(self):
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            sanitize_experiment_id("")


class TestCheckEnvForSecrets:
    """Tests for check_env_for_secrets function."""

    def test_existing_var_detected(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_EXISTS", "value")
        result = check_env_for_secrets(["TEST_VAR_EXISTS", "NONEXISTENT_VAR"])
        assert result["TEST_VAR_EXISTS"] is True
        assert result["NONEXISTENT_VAR"] is False

    def test_empty_list(self):
        assert check_env_for_secrets([]) == {}
