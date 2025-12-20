"""Tests for constants module."""

from pathlib import Path

from llm_bench.constants import (
    AGGREGATED_RESULTS_SUBDIR,
    DEFAULT_ACCELERATE_PORT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SAMPLING_INTERVAL_SEC,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_WARMUP_RUNS,
    RAW_RESULTS_SUBDIR,
    SCHEMA_VERSION,
)


class TestResultsConstants:
    """Tests for results-related constants."""

    def test_default_results_dir_is_path(self):
        assert isinstance(DEFAULT_RESULTS_DIR, Path)

    def test_subdir_names_are_strings(self):
        assert isinstance(RAW_RESULTS_SUBDIR, str)
        assert isinstance(AGGREGATED_RESULTS_SUBDIR, str)


class TestExperimentDefaults:
    """Tests for experiment default values."""

    def test_warmup_runs_positive(self):
        assert DEFAULT_WARMUP_RUNS > 0

    def test_sampling_interval_positive(self):
        assert DEFAULT_SAMPLING_INTERVAL_SEC > 0

    def test_accelerate_port_valid(self):
        assert 1024 <= DEFAULT_ACCELERATE_PORT <= 65535


class TestInferenceDefaults:
    """Tests for inference default values."""

    def test_max_new_tokens_positive(self):
        assert DEFAULT_MAX_NEW_TOKENS > 0

    def test_temperature_valid(self):
        assert DEFAULT_TEMPERATURE >= 0

    def test_top_p_valid(self):
        assert 0 <= DEFAULT_TOP_P <= 1


class TestSchemaVersion:
    """Tests for schema version constant."""

    def test_schema_version_format(self):
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
