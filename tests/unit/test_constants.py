"""Tests for constants module."""

from pathlib import Path

from llm_energy_measure.constants import (
    AGGREGATED_RESULTS_SUBDIR,
    DEFAULT_ACCELERATE_PORT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SAMPLING_INTERVAL_SEC,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_WARMUP_RUNS,
    PRESETS,
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


class TestPresets:
    """Tests for built-in experiment presets."""

    def test_presets_has_expected_keys(self):
        """PRESETS dict has expected preset names."""
        expected_keys = {"quick-test", "benchmark", "throughput"}
        assert expected_keys == set(PRESETS.keys())

    def test_presets_is_dict(self):
        """PRESETS is a dict type."""
        assert isinstance(PRESETS, dict)

    def test_quick_test_preset_structure(self):
        """quick-test preset has required fields."""
        preset = PRESETS["quick-test"]
        assert "max_input_tokens" in preset
        assert "max_output_tokens" in preset
        assert "batching" in preset
        assert "batch_size" in preset["batching"]
        assert preset["max_input_tokens"] > 0
        assert preset["max_output_tokens"] > 0

    def test_benchmark_preset_structure(self):
        """benchmark preset has required fields."""
        preset = PRESETS["benchmark"]
        assert "max_input_tokens" in preset
        assert "max_output_tokens" in preset
        assert "fp_precision" in preset
        assert "batching" in preset
        assert preset["fp_precision"] in ("float16", "float32", "bfloat16")

    def test_throughput_preset_structure(self):
        """throughput preset has required fields."""
        preset = PRESETS["throughput"]
        assert "max_input_tokens" in preset
        assert "max_output_tokens" in preset
        assert "batching" in preset
        # Throughput preset should have larger batch size
        assert preset["batching"]["batch_size"] > 1

    def test_all_presets_have_token_limits(self):
        """All presets have max_input_tokens and max_output_tokens."""
        for name, preset in PRESETS.items():
            assert "max_input_tokens" in preset, f"{name} missing max_input_tokens"
            assert "max_output_tokens" in preset, f"{name} missing max_output_tokens"
            assert preset["max_input_tokens"] > 0, f"{name} has invalid max_input_tokens"
            assert preset["max_output_tokens"] > 0, f"{name} has invalid max_output_tokens"

    def test_all_presets_have_batching(self):
        """All presets have batching."""
        for name, preset in PRESETS.items():
            assert "batching" in preset, f"{name} missing batching"
            assert "batch_size" in preset["batching"], f"{name} missing batch_size"

    def test_preset_values_are_valid_types(self):
        """Preset values have correct types."""
        for name, preset in PRESETS.items():
            assert isinstance(preset["max_input_tokens"], int), f"{name}: max_input_tokens not int"
            assert isinstance(
                preset["max_output_tokens"], int
            ), f"{name}: max_output_tokens not int"
            assert isinstance(preset["batching"]["batch_size"], int), f"{name}: batch_size not int"

    def test_quick_test_is_minimal(self):
        """quick-test preset uses minimal values for fast testing."""
        preset = PRESETS["quick-test"]
        # Quick test should have small token limits
        assert preset["max_input_tokens"] <= 128
        assert preset["max_output_tokens"] <= 64
        assert preset["batching"]["batch_size"] == 1

    def test_benchmark_is_deterministic(self):
        """benchmark preset has deterministic settings."""
        preset = PRESETS["benchmark"]
        # Benchmark should be deterministic for reproducibility
        if "decoder" in preset:
            # Either explicit do_sample=False or deterministic preset
            is_deterministic = (
                preset["decoder"].get("do_sample") is False
                or preset["decoder"].get("preset") == "deterministic"
            )
            assert is_deterministic

    def test_throughput_has_dynamic_batching(self):
        """throughput preset enables dynamic batching."""
        preset = PRESETS["throughput"]
        assert preset["batching"].get("dynamic_batching") is True
