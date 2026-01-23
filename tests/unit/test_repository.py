"""Tests for results repository."""

from datetime import datetime

import pytest

from llenergymeasure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
    Timestamps,
)
from llenergymeasure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llenergymeasure.exceptions import ConfigurationError
from llenergymeasure.results.repository import FileSystemRepository


@pytest.fixture
def sample_raw_result():
    """Create a sample RawProcessResult for testing."""
    return RawProcessResult(
        experiment_id="exp_001",
        process_index=0,
        gpu_id=0,
        config_name="test_config",
        model_name="test/model",
        timestamps=Timestamps(
            start=datetime(2024, 1, 1, 12, 0, 0),
            end=datetime(2024, 1, 1, 12, 0, 10),
            duration_sec=10.0,
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=500,
            input_tokens=50,
            output_tokens=450,
            inference_time_sec=10.0,
            tokens_per_second=50.0,
            latency_per_token_ms=20.0,
        ),
        energy_metrics=EnergyMetrics(
            total_energy_j=250.0,
            gpu_power_w=100.0,
            duration_sec=10.0,
        ),
        compute_metrics=ComputeMetrics(flops_total=5e11),
    )


@pytest.fixture
def sample_aggregated_result():
    """Create a sample AggregatedResult for testing."""
    return AggregatedResult(
        experiment_id="exp_001",
        aggregation=AggregationMetadata(num_processes=2),
        total_tokens=1000,
        total_energy_j=500.0,
        total_inference_time_sec=10.0,
        avg_tokens_per_second=100.0,
        avg_energy_per_token_j=0.5,
        total_flops=1e12,
        start_time=datetime(2024, 1, 1, 12, 0, 0),
        end_time=datetime(2024, 1, 1, 12, 0, 10),
    )


class TestFileSystemRepository:
    """Tests for FileSystemRepository."""

    @pytest.fixture
    def repo(self, tmp_path):
        return FileSystemRepository(base_path=tmp_path / "results")

    def test_save_and_load_raw(self, repo, sample_raw_result):
        path = repo.save_raw("exp_001", sample_raw_result)
        assert path.exists()
        assert "process_0.json" in str(path)

        loaded = repo.load_raw(path)
        assert loaded.experiment_id == "exp_001"
        assert loaded.process_index == 0
        assert loaded.inference_metrics.total_tokens == 500

    def test_save_multiple_processes(self, repo, sample_raw_result):
        # Save multiple process results
        for i in range(4):
            result = sample_raw_result.model_copy(update={"process_index": i, "gpu_id": i})
            repo.save_raw("exp_001", result)

        paths = repo.list_raw("exp_001")
        assert len(paths) == 4

        # Verify sorted by process index
        for i, path in enumerate(paths):
            assert f"process_{i}.json" in str(path)

    def test_load_all_raw(self, repo, sample_raw_result):
        for i in range(3):
            result = sample_raw_result.model_copy(update={"process_index": i, "gpu_id": i})
            repo.save_raw("exp_001", result)

        results = repo.load_all_raw("exp_001")
        assert len(results) == 3
        assert results[0].process_index == 0
        assert results[2].process_index == 2

    def test_list_raw_empty(self, repo):
        paths = repo.list_raw("nonexistent")
        assert paths == []

    def test_save_and_load_aggregated(self, repo, sample_aggregated_result):
        path = repo.save_aggregated(sample_aggregated_result)
        assert path.exists()

        loaded = repo.load_aggregated("exp_001")
        assert loaded is not None
        assert loaded.experiment_id == "exp_001"
        assert loaded.total_tokens == 1000

    def test_load_aggregated_nonexistent(self, repo):
        result = repo.load_aggregated("nonexistent")
        assert result is None

    def test_list_experiments(self, repo, sample_raw_result):
        for exp_id in ["exp_001", "exp_002", "exp_003"]:
            result = sample_raw_result.model_copy(update={"experiment_id": exp_id})
            repo.save_raw(exp_id, result)

        experiments = repo.list_experiments()
        assert len(experiments) == 3
        assert "exp_001" in experiments

    def test_list_aggregated(self, repo, sample_aggregated_result):
        for exp_id in ["exp_001", "exp_002"]:
            result = sample_aggregated_result.model_copy(update={"experiment_id": exp_id})
            repo.save_aggregated(result)

        aggregated = repo.list_aggregated()
        assert len(aggregated) == 2

    def test_has_raw(self, repo, sample_raw_result):
        assert repo.has_raw("exp_001") is False

        repo.save_raw("exp_001", sample_raw_result)
        assert repo.has_raw("exp_001") is True

    def test_has_aggregated(self, repo, sample_aggregated_result):
        assert repo.has_aggregated("exp_001") is False

        repo.save_aggregated(sample_aggregated_result)
        assert repo.has_aggregated("exp_001") is True

    def test_delete_experiment(self, repo, sample_raw_result, sample_aggregated_result):
        repo.save_raw("exp_001", sample_raw_result)
        repo.save_aggregated(sample_aggregated_result)

        assert repo.has_raw("exp_001") is True
        assert repo.has_aggregated("exp_001") is True

        deleted = repo.delete_experiment("exp_001")
        assert deleted is True
        assert repo.has_raw("exp_001") is False
        assert repo.has_aggregated("exp_001") is False

    def test_delete_nonexistent(self, repo):
        deleted = repo.delete_experiment("nonexistent")
        assert deleted is False

    def test_load_invalid_file(self, repo, tmp_path):
        # Create an invalid JSON file
        bad_path = tmp_path / "results" / "raw" / "exp_001" / "process_0.json"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_text("invalid json")

        with pytest.raises(ConfigurationError, match="Failed to load"):
            repo.load_raw(bad_path)

    def test_directory_structure(self, repo, sample_raw_result, sample_aggregated_result):
        """Verify the expected directory structure is created."""
        repo.save_raw("exp_001", sample_raw_result)
        repo.save_aggregated(sample_aggregated_result)

        # Check structure
        assert (repo._base / "raw" / "exp_001" / "process_0.json").exists()
        assert (repo._base / "aggregated" / "exp_001.json").exists()
