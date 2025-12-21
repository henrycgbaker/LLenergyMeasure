"""Integration tests for repository file operations.

Tests the filesystem repository lifecycle including concurrent operations,
data integrity, and cleanup.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from llm_energy_measure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
    Timestamps,
)
from llm_energy_measure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llm_energy_measure.exceptions import ConfigurationError
from llm_energy_measure.results.repository import FileSystemRepository


@pytest.fixture
def sample_raw_result() -> RawProcessResult:
    """Create a sample raw result for testing."""
    return RawProcessResult(
        experiment_id="repo_test_001",
        process_index=0,
        gpu_id=0,
        config_name="test_config",
        model_name="test-model",
        timestamps=Timestamps(
            start=datetime(2024, 1, 1, 10, 0, 0),
            end=datetime(2024, 1, 1, 10, 1, 0),
            duration_sec=60.0,
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=500,
            input_tokens=200,
            output_tokens=300,
            inference_time_sec=60.0,
            tokens_per_second=8.33,
            latency_per_token_ms=120.0,
        ),
        energy_metrics=EnergyMetrics(
            total_energy_j=75.0,
            gpu_energy_j=60.0,
            cpu_energy_j=15.0,
            duration_sec=60.0,
        ),
        compute_metrics=ComputeMetrics(
            flops_total=5e11,
            flops_per_second=8.33e9,
            flops_method="calflops",
            flops_confidence="high",
        ),
    )


@pytest.fixture
def sample_aggregated_result(sample_raw_result) -> AggregatedResult:
    """Create a sample aggregated result for testing."""
    return AggregatedResult(
        experiment_id="repo_test_001",
        aggregation=AggregationMetadata(
            method="sum_energy_avg_throughput",
            num_processes=1,
            temporal_overlap_verified=True,
            gpu_attribution_verified=True,
        ),
        total_tokens=500,
        total_energy_j=75.0,
        total_inference_time_sec=60.0,
        avg_tokens_per_second=8.33,
        avg_energy_per_token_j=0.15,
        total_flops=5e11,
        process_results=[sample_raw_result],
        start_time=datetime(2024, 1, 1, 10, 0, 0),
        end_time=datetime(2024, 1, 1, 10, 1, 0),
    )


class TestRepositoryLifecycle:
    """Test full repository lifecycle operations."""

    def test_save_load_raw_result(self, tmp_path: Path, sample_raw_result):
        """Test saving and loading a raw result."""
        repo = FileSystemRepository(tmp_path)

        # Save
        path = repo.save_raw("exp_001", sample_raw_result)
        assert path.exists()
        assert "process_0.json" in str(path)

        # Load
        loaded = repo.load_raw(path)
        assert loaded.experiment_id == sample_raw_result.experiment_id
        assert loaded.inference_metrics.total_tokens == 500
        assert loaded.energy_metrics.total_energy_j == 75.0

    def test_save_load_aggregated_result(self, tmp_path: Path, sample_aggregated_result):
        """Test saving and loading an aggregated result."""
        repo = FileSystemRepository(tmp_path)

        # Save
        path = repo.save_aggregated(sample_aggregated_result)
        assert path.exists()

        # Load
        loaded = repo.load_aggregated("repo_test_001")
        assert loaded is not None
        assert loaded.total_tokens == 500
        assert loaded.aggregation.num_processes == 1

    def test_list_experiments(self, tmp_path: Path, sample_raw_result):
        """Test listing experiments."""
        repo = FileSystemRepository(tmp_path)

        # Initially empty
        assert repo.list_experiments() == []

        # Add some experiments
        for exp_id in ["exp_a", "exp_b", "exp_c"]:
            result = RawProcessResult(**{**sample_raw_result.model_dump(), "experiment_id": exp_id})
            repo.save_raw(exp_id, result)

        experiments = repo.list_experiments()
        assert len(experiments) == 3
        assert set(experiments) == {"exp_a", "exp_b", "exp_c"}

    def test_has_raw_and_aggregated(
        self, tmp_path: Path, sample_raw_result, sample_aggregated_result
    ):
        """Test checking for raw and aggregated results."""
        repo = FileSystemRepository(tmp_path)

        # Initially nothing
        assert not repo.has_raw("exp_001")
        assert not repo.has_aggregated("exp_001")

        # Add raw
        repo.save_raw("exp_001", sample_raw_result)
        assert repo.has_raw("exp_001")
        assert not repo.has_aggregated("exp_001")

        # Add aggregated
        agg = AggregatedResult(
            **{**sample_aggregated_result.model_dump(), "experiment_id": "exp_001"}
        )
        repo.save_aggregated(agg)
        assert repo.has_raw("exp_001")
        assert repo.has_aggregated("exp_001")

    def test_delete_experiment(self, tmp_path: Path, sample_raw_result, sample_aggregated_result):
        """Test deleting experiment data."""
        repo = FileSystemRepository(tmp_path)

        # Create experiment
        repo.save_raw("exp_delete", sample_raw_result)
        agg = AggregatedResult(
            **{**sample_aggregated_result.model_dump(), "experiment_id": "exp_delete"}
        )
        repo.save_aggregated(agg)

        assert repo.has_raw("exp_delete")
        assert repo.has_aggregated("exp_delete")

        # Delete
        deleted = repo.delete_experiment("exp_delete")
        assert deleted is True
        assert not repo.has_raw("exp_delete")
        assert not repo.has_aggregated("exp_delete")

        # Delete non-existent returns False
        assert repo.delete_experiment("nonexistent") is False


class TestMultiProcessResults:
    """Test handling multiple process results."""

    def test_save_multiple_processes(self, tmp_path: Path):
        """Test saving results from multiple processes."""
        repo = FileSystemRepository(tmp_path)
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Save 4 process results
        for i in range(4):
            result = RawProcessResult(
                experiment_id="multi_proc_exp",
                process_index=i,
                gpu_id=i,
                config_name="test",
                model_name="test-model",
                timestamps=Timestamps(
                    start=base_time,
                    end=base_time + timedelta(seconds=60),
                    duration_sec=60.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=100 * (i + 1),
                    input_tokens=50,
                    output_tokens=50 * (i + 1),
                    inference_time_sec=60.0,
                    tokens_per_second=1.67 * (i + 1),
                    latency_per_token_ms=600.0 / (i + 1),
                ),
                energy_metrics=EnergyMetrics(
                    total_energy_j=25.0 * (i + 1),
                    duration_sec=60.0,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=1e10 * (i + 1),
                    flops_method="parameter_estimate",
                    flops_confidence="low",
                ),
            )
            repo.save_raw("multi_proc_exp", result)

        # Verify all saved
        paths = repo.list_raw("multi_proc_exp")
        assert len(paths) == 4

        # Paths should be sorted by process index
        assert "process_0" in str(paths[0])
        assert "process_3" in str(paths[3])

    def test_load_all_raw(self, tmp_path: Path):
        """Test loading all raw results for an experiment."""
        repo = FileSystemRepository(tmp_path)
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Save 3 process results
        for i in range(3):
            result = RawProcessResult(
                experiment_id="load_all_test",
                process_index=i,
                gpu_id=i,
                config_name="test",
                model_name="test-model",
                timestamps=Timestamps(
                    start=base_time,
                    end=base_time + timedelta(seconds=60),
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
                energy_metrics=EnergyMetrics(
                    total_energy_j=25.0,
                    duration_sec=60.0,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=1e10,
                    flops_method="parameter_estimate",
                    flops_confidence="low",
                ),
            )
            repo.save_raw("load_all_test", result)

        # Load all
        results = repo.load_all_raw("load_all_test")
        assert len(results) == 3

        # Should be sorted by process index
        for i, result in enumerate(results):
            assert result.process_index == i


class TestRepositoryErrorHandling:
    """Test error handling in repository operations."""

    def test_load_nonexistent_raw(self, tmp_path: Path):
        """Test loading a non-existent raw result."""
        repo = FileSystemRepository(tmp_path)

        with pytest.raises(ConfigurationError, match="not found"):
            repo.load_raw(tmp_path / "nonexistent.json")

    def test_load_nonexistent_aggregated(self, tmp_path: Path):
        """Test loading a non-existent aggregated result."""
        repo = FileSystemRepository(tmp_path)

        result = repo.load_aggregated("nonexistent")
        assert result is None  # Returns None, not raises

    def test_load_corrupted_json(self, tmp_path: Path):
        """Test loading a corrupted JSON file."""
        repo = FileSystemRepository(tmp_path)

        # Create corrupted file
        raw_dir = tmp_path / "raw" / "corrupted_exp"
        raw_dir.mkdir(parents=True)
        corrupted_file = raw_dir / "process_0.json"
        corrupted_file.write_text("{ invalid json }")

        with pytest.raises(ConfigurationError, match="Failed to load"):
            repo.load_raw(corrupted_file)

    def test_list_raw_nonexistent_experiment(self, tmp_path: Path):
        """Test listing raw results for non-existent experiment."""
        repo = FileSystemRepository(tmp_path)

        paths = repo.list_raw("nonexistent")
        assert paths == []


class TestRepositoryDirectoryStructure:
    """Test that repository creates correct directory structure."""

    def test_creates_directories(self, tmp_path: Path, sample_raw_result):
        """Test that save operations create necessary directories."""
        repo = FileSystemRepository(tmp_path)

        # Directories shouldn't exist yet
        raw_dir = tmp_path / "raw"
        agg_dir = tmp_path / "aggregated"
        assert not raw_dir.exists()
        assert not agg_dir.exists()

        # Save raw - should create directories
        repo.save_raw("dir_test", sample_raw_result)
        assert raw_dir.exists()
        assert (raw_dir / "dir_test").exists()

    def test_experiment_isolation(self, tmp_path: Path, sample_raw_result):
        """Test that experiments are isolated in separate directories."""
        repo = FileSystemRepository(tmp_path)

        # Save to different experiments
        for exp_id in ["exp_alpha", "exp_beta"]:
            result = RawProcessResult(**{**sample_raw_result.model_dump(), "experiment_id": exp_id})
            repo.save_raw(exp_id, result)

        # Each should have its own directory
        assert (tmp_path / "raw" / "exp_alpha").is_dir()
        assert (tmp_path / "raw" / "exp_beta").is_dir()

        # Files should be in correct directories
        alpha_files = list((tmp_path / "raw" / "exp_alpha").glob("*.json"))
        beta_files = list((tmp_path / "raw" / "exp_beta").glob("*.json"))
        assert len(alpha_files) == 1
        assert len(beta_files) == 1
