"""Integration tests for error handling scenarios.

Tests that the system handles various error conditions gracefully.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_energy_measure.cli import app
from llm_energy_measure.config.loader import load_config
from llm_energy_measure.exceptions import ConfigurationError
from llm_energy_measure.results.repository import FileSystemRepository

runner = CliRunner()


class TestConfigErrorHandling:
    """Test config-related error handling."""

    def test_missing_config_file(self, tmp_path: Path):
        """Test handling of missing config file."""
        missing = tmp_path / "nonexistent.yaml"

        result = runner.invoke(app, ["config", "validate", str(missing)])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "Invalid" in result.stdout

    def test_invalid_yaml_syntax(self, tmp_path: Path):
        """Test handling of invalid YAML syntax."""
        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("""
config_name: test
  bad_indent: value
    another: bad
""")

        result = runner.invoke(app, ["config", "validate", str(invalid)])
        assert result.exit_code == 1

    def test_circular_inheritance(self, tmp_path: Path):
        """Test detection of circular config inheritance."""
        # Create circular reference
        config_a = tmp_path / "a.yaml"
        config_b = tmp_path / "b.yaml"

        config_a.write_text("""
_extends: b.yaml
config_name: a
model_name: test
""")
        config_b.write_text("""
_extends: a.yaml
config_name: b
model_name: test
""")

        with pytest.raises(ConfigurationError, match="[Cc]ircular"):
            load_config(config_a)

    def test_missing_parent_config(self, tmp_path: Path):
        """Test handling of missing parent in inheritance."""
        child = tmp_path / "child.yaml"
        child.write_text("""
_extends: nonexistent_parent.yaml
config_name: child
model_name: test
""")

        with pytest.raises(ConfigurationError, match="not found"):
            load_config(child)

    def test_invalid_constraint_violation(self, tmp_path: Path):
        """Test config constraint violations."""
        invalid = tmp_path / "bad_constraints.yaml"
        invalid.write_text("""
config_name: bad
model_name: test
min_output_tokens: 100
max_output_tokens: 50
""")

        result = runner.invoke(app, ["config", "validate", str(invalid)])
        assert result.exit_code == 1

    def test_invalid_quantization_combo(self, tmp_path: Path):
        """Test mutually exclusive quantization options."""
        invalid = tmp_path / "bad_quant.yaml"
        invalid.write_text("""
config_name: bad-quant
model_name: test
quantization_config:
  quantization: true
  load_in_4bit: true
  load_in_8bit: true
""")

        result = runner.invoke(app, ["config", "validate", str(invalid)])
        assert result.exit_code == 1


class TestRepositoryErrorHandling:
    """Test repository error handling."""

    def test_corrupted_result_file(self, tmp_path: Path):
        """Test handling of corrupted result files."""
        repo = FileSystemRepository(tmp_path)

        # Create corrupted file manually
        raw_dir = tmp_path / "raw" / "corrupted_exp"
        raw_dir.mkdir(parents=True)
        bad_file = raw_dir / "process_0.json"
        bad_file.write_text("{ this is not valid json }")

        # load_raw should raise
        with pytest.raises(ConfigurationError, match="Failed to load"):
            repo.load_raw(bad_file)

        # load_all_raw should also fail
        with pytest.raises(ConfigurationError):
            repo.load_all_raw("corrupted_exp")

    def test_partial_result_set(self, tmp_path: Path):
        """Test handling when some result files are missing."""
        from datetime import datetime

        from llm_energy_measure.domain.experiment import RawProcessResult, Timestamps
        from llm_energy_measure.domain.metrics import (
            ComputeMetrics,
            EnergyMetrics,
            InferenceMetrics,
        )

        repo = FileSystemRepository(tmp_path)

        # Only save process 0 and 2 (skip 1)
        for proc in [0, 2]:
            result = RawProcessResult(
                experiment_id="partial_exp",
                process_index=proc,
                gpu_id=proc,
                config_name="test",
                model_name="test-model",
                timestamps=Timestamps(
                    start=datetime(2024, 1, 1, 10, 0, 0),
                    end=datetime(2024, 1, 1, 10, 1, 0),
                    duration_sec=60.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=100,
                    input_tokens=50,
                    output_tokens=50,
                    inference_time_sec=60.0,
                    tokens_per_second=1.67,
                    latency_per_token_ms=600.0,
                ),
                energy_metrics=EnergyMetrics(total_energy_j=50.0, duration_sec=60.0),
                compute_metrics=ComputeMetrics(
                    flops_total=1e10,
                    flops_method="parameter_estimate",
                    flops_confidence="low",
                ),
            )
            repo.save_raw("partial_exp", result)

        # Should still load what exists
        results = repo.load_all_raw("partial_exp")
        assert len(results) == 2
        assert results[0].process_index == 0
        assert results[1].process_index == 2


class TestCLIErrorHandling:
    """Test CLI error message handling."""

    def test_aggregate_no_experiment_id(self):
        """Test aggregate without experiment ID."""
        result = runner.invoke(app, ["aggregate"])
        assert result.exit_code == 1
        assert "Provide experiment ID or use --all" in result.stdout

    def test_aggregate_nonexistent(self, tmp_path: Path):
        """Test aggregating non-existent experiment."""
        result = runner.invoke(app, ["aggregate", "nonexistent", "--results-dir", str(tmp_path)])
        assert "No raw results found" in result.stdout

    def test_show_nonexistent(self, tmp_path: Path):
        """Test showing non-existent experiment."""
        result = runner.invoke(
            app, ["results", "show", "nonexistent", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "No aggregated result" in result.stdout

    def test_run_missing_config(self):
        """Test run with missing config file."""
        result = runner.invoke(app, ["run", "/nonexistent/path/config.yaml"])
        assert result.exit_code == 1


class TestSecurityErrorHandling:
    """Test security-related error handling."""

    def test_path_traversal_in_experiment_id(self, tmp_path: Path):
        """Test that path traversal characters are neutralized in experiment IDs."""
        from llm_energy_measure.security import sanitize_experiment_id

        # These should have path separators removed (replaced with _)
        dangerous_ids = [
            ("../../../etc/passwd", ".._.._.._etc_passwd"),
            ("..\\..\\windows\\system32", ".._.._windows_system32"),
            ("exp/../../../sensitive", "exp_.._.._.._sensitive"),
        ]

        for dangerous_id, expected in dangerous_ids:
            safe_id = sanitize_experiment_id(dangerous_id)
            # Path separators should be replaced with underscores
            assert "/" not in safe_id
            assert "\\" not in safe_id
            # The sanitized version should match expected pattern
            assert safe_id == expected

    def test_experiment_id_with_special_chars(self, tmp_path: Path):
        """Test handling of special characters in experiment IDs."""
        from datetime import datetime

        from llm_energy_measure.domain.experiment import RawProcessResult, Timestamps
        from llm_energy_measure.domain.metrics import (
            ComputeMetrics,
            EnergyMetrics,
            InferenceMetrics,
        )

        repo = FileSystemRepository(tmp_path)

        # Create result with special chars in experiment ID
        result = RawProcessResult(
            experiment_id="exp@2024#test!",  # Special chars
            process_index=0,
            gpu_id=0,
            config_name="test",
            model_name="test-model",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 10, 0, 0),
                end=datetime(2024, 1, 1, 10, 1, 0),
                duration_sec=60.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=100,
                input_tokens=50,
                output_tokens=50,
                inference_time_sec=60.0,
                tokens_per_second=1.67,
                latency_per_token_ms=600.0,
            ),
            energy_metrics=EnergyMetrics(total_energy_j=50.0, duration_sec=60.0),
            compute_metrics=ComputeMetrics(
                flops_total=1e10,
                flops_method="parameter_estimate",
                flops_confidence="low",
            ),
        )

        # Should sanitize and save successfully
        path = repo.save_raw("exp@2024#test!", result)
        assert path.exists()

        # Directory name should be sanitized
        assert "@" not in path.parent.name
        assert "#" not in path.parent.name
        assert "!" not in path.parent.name


class TestAggregationErrorHandling:
    """Test aggregation error handling."""

    def test_aggregate_empty_results(self, tmp_path: Path):
        """Test aggregating empty results list."""
        from llm_energy_measure.exceptions import AggregationError
        from llm_energy_measure.results.aggregation import aggregate_results

        with pytest.raises(AggregationError, match="[Ee]mpty"):
            aggregate_results("empty_exp", [])

    def test_aggregate_with_cli_empty(self, tmp_path: Path):
        """Test CLI aggregate with no raw results."""
        # Create empty experiment directory
        exp_dir = tmp_path / "raw" / "empty_exp"
        exp_dir.mkdir(parents=True)

        result = runner.invoke(app, ["aggregate", "empty_exp", "--results-dir", str(tmp_path)])
        assert "No raw results found" in result.stdout
