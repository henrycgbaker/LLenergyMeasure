"""Tests for experiment launcher utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.exceptions import ConfigurationError
from llenergymeasure.orchestration.launcher import (
    get_config_file_path,
    log_failed_experiment,
    run_from_config,
)


class TestGetConfigFilePath:
    """Tests for get_config_file_path function."""

    def test_dict_creates_temp_file(self, tmp_path):
        config = {"model_name": "test/model", "gpus": [0]}

        with patch("tempfile.mkstemp") as mock_mkstemp:
            # Create a real temp file for the test
            temp_file = tmp_path / "config.json"
            mock_mkstemp.return_value = (0, str(temp_file))

            with patch("os.fdopen") as mock_fdopen:
                mock_file = MagicMock()
                mock_fdopen.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_fdopen.return_value.__exit__ = MagicMock(return_value=False)

                result = get_config_file_path(config)

                assert result == temp_file
                mock_mkstemp.assert_called_once()

    def test_string_path_returned_as_path(self):
        result = get_config_file_path("/some/path/config.json")
        assert result == Path("/some/path/config.json")

    def test_path_object_returned(self):
        path = Path("/some/path/config.json")
        result = get_config_file_path(path)
        assert result == path

    def test_invalid_type_raises_error(self):
        with pytest.raises(ConfigurationError, match="Config must be dict or path"):
            get_config_file_path(123)  # type: ignore[arg-type]


class TestLogFailedExperiment:
    """Tests for log_failed_experiment function."""

    def test_creates_csv_with_header(self, tmp_path):
        output_file = tmp_path / "failed.csv"
        config = {"model_name": "test/model", "suite": "test_suite"}

        log_failed_experiment("exp_001", config, "Test error", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "experiment_id" in content
        assert "timestamp" in content
        assert "exp_001" in content
        assert "Test error" in content

    def test_appends_to_existing_csv(self, tmp_path):
        output_file = tmp_path / "failed.csv"
        config = {"model_name": "test/model"}

        log_failed_experiment("exp_001", config, "Error 1", output_file)
        log_failed_experiment("exp_002", config, "Error 2", output_file)

        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 entries
        assert "exp_001" in lines[1]
        assert "exp_002" in lines[2]

    def test_uses_unknown_for_missing_suite(self, tmp_path):
        output_file = tmp_path / "failed.csv"
        config = {"model_name": "test/model"}  # No suite

        log_failed_experiment("exp_001", config, "Error", output_file)

        content = output_file.read_text()
        assert "unknown" in content


class TestRunFromConfig:
    """Tests for run_from_config function."""

    @patch("llenergymeasure.orchestration.lifecycle.ensure_clean_start")
    @patch("llenergymeasure.orchestration.context.ExperimentContext.create")
    def test_success_returns_true(self, mock_create, mock_clean):
        mock_ctx = MagicMock()
        mock_ctx.experiment_id = "0001"
        mock_ctx.cleanup = MagicMock()
        mock_create.return_value = mock_ctx

        config_data = {
            "config_name": "test",
            "model_name": "test/model",
            "gpus": [0],
            "num_processes": 1,
        }

        success, result = run_from_config(config_data, ["test prompt"])

        assert success is True
        assert result == "0001"
        mock_clean.assert_called_once()

    @patch("llenergymeasure.orchestration.lifecycle.ensure_clean_start")
    @patch("llenergymeasure.orchestration.context.ExperimentContext.create")
    def test_failure_returns_false(self, mock_create, mock_clean):
        mock_create.side_effect = RuntimeError("Failed")

        config_data = {
            "config_name": "test",
            "model_name": "test/model",
            "gpus": [0],
            "num_processes": 1,
        }

        success, result = run_from_config(config_data, ["test"], max_retries=1)

        assert success is False
        assert result is None

    @patch("llenergymeasure.orchestration.launcher.time.sleep")
    @patch("llenergymeasure.orchestration.lifecycle.ensure_clean_start")
    @patch("llenergymeasure.orchestration.context.ExperimentContext.create")
    def test_retries_on_failure(self, mock_create, mock_clean, mock_sleep):
        mock_ctx = MagicMock()
        mock_ctx.experiment_id = "0001"
        mock_ctx.cleanup = MagicMock()

        # Fail first, succeed second
        mock_create.side_effect = [RuntimeError("Fail"), mock_ctx]

        config_data = {
            "config_name": "test",
            "model_name": "test/model",
            "gpus": [0],
            "num_processes": 1,
        }

        success, result = run_from_config(config_data, ["test"], max_retries=2, retry_delay=0)

        assert success is True
        mock_sleep.assert_called_once()
