"""
Unit tests for results management.

Tests results storage, loading, and aggregation.
"""

import pytest
import json
from pathlib import Path

from llm_efficiency.storage.results import (
    ExperimentResults,
    InferenceMetrics,
    ComputeMetrics,
    EnergyMetrics,
    ModelInfo,
    ResultsManager,
    create_results,
)


class TestInferenceMetrics:
    """Tests for InferenceMetrics dataclass."""

    def test_creation(self):
        """Test creating inference metrics."""
        metrics = InferenceMetrics(
            total_time_seconds=10.0,
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            num_prompts=10,
            tokens_per_second=15.0,
            queries_per_second=1.0,
            avg_latency_per_query=1.0,
            avg_tokens_per_prompt=15.0,
        )
        
        assert metrics.total_time_seconds == 10.0
        assert metrics.total_tokens == 150
        assert metrics.tokens_per_second == 15.0


class TestComputeMetrics:
    """Tests for ComputeMetrics dataclass."""

    def test_creation_with_defaults(self):
        """Test creating compute metrics with defaults."""
        metrics = ComputeMetrics(
            flops=1000000,
            gpu_memory_allocated_mb=512.0,
            gpu_memory_peak_mb=1024.0,
        )
        
        assert metrics.flops == 1000000
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.gpu_utilization_percent == []


class TestEnergyMetrics:
    """Tests for EnergyMetrics dataclass."""

    def test_creation(self):
        """Test creating energy metrics."""
        metrics = EnergyMetrics(
            duration_seconds=120.0,
            total_energy_kwh=0.1,
            cpu_energy_kwh=0.03,
            gpu_energy_kwh=0.05,
            ram_energy_kwh=0.02,
            emissions_kg_co2=0.05,
        )
        
        assert metrics.duration_seconds == 120.0
        assert metrics.total_energy_kwh == 0.1
        assert metrics.emissions_kg_co2 == 0.05


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_creation(self):
        """Test creating model info."""
        info = ModelInfo(
            model_name="test-model",
            total_parameters=1000000,
            trainable_parameters=1000000,
            precision="float16",
        )
        
        assert info.model_name == "test-model"
        assert info.total_parameters == 1000000


class TestExperimentResults:
    """Tests for ExperimentResults dataclass."""

    def test_creation_minimal(self):
        """Test creating results with minimal data."""
        results = ExperimentResults(
            experiment_id="0001",
            timestamp="2025-01-01T00:00:00",
        )
        
        assert results.experiment_id == "0001"
        assert results.timestamp == "2025-01-01T00:00:00"

    def test_efficiency_calculation(self):
        """Test automatic efficiency metrics calculation."""
        results = ExperimentResults(
            experiment_id="0001",
            timestamp="2025-01-01T00:00:00",
            inference=InferenceMetrics(
                total_time_seconds=10.0,
                total_input_tokens=100,
                total_output_tokens=50,
                total_tokens=150,
                num_prompts=10,
                tokens_per_second=15.0,
                queries_per_second=1.0,
                avg_latency_per_query=1.0,
                avg_tokens_per_prompt=15.0,
            ),
            compute=ComputeMetrics(
                flops=1000000,
                gpu_memory_allocated_mb=512.0,
                gpu_memory_peak_mb=1024.0,
            ),
            energy=EnergyMetrics(
                duration_seconds=10.0,
                total_energy_kwh=0.001,
                cpu_energy_kwh=0.0003,
                gpu_energy_kwh=0.0005,
                ram_energy_kwh=0.0002,
                emissions_kg_co2=0.0005,
                cpu_power_w=10.0,
                gpu_power_w=20.0,
                ram_power_w=5.0,
            ),
        )
        
        # Check efficiency metrics were calculated
        assert "tokens_per_joule" in results.efficiency
        assert "flops_per_joule" in results.efficiency
        assert "tokens_per_second_per_watt" in results.efficiency
        assert "kwh_per_query" in results.efficiency
        assert "co2_per_query_g" in results.efficiency
        
        assert results.efficiency["tokens_per_joule"] > 0
        assert results.efficiency["kwh_per_query"] > 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        results = ExperimentResults(
            experiment_id="0001",
            timestamp="2025-01-01T00:00:00",
        )
        
        data = results.to_dict()
        
        assert isinstance(data, dict)
        assert data["experiment_id"] == "0001"

    def test_to_json(self):
        """Test converting to JSON."""
        results = ExperimentResults(
            experiment_id="0001",
            timestamp="2025-01-01T00:00:00",
        )
        
        json_str = results.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["experiment_id"] == "0001"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "experiment_id": "0001",
            "timestamp": "2025-01-01T00:00:00",
            "config_name": None,
            "model": {
                "model_name": "test",
                "total_parameters": 1000,
                "trainable_parameters": 1000,
                "precision": "float16",
                "quantization": None,
                "model_size_mb": 0.0,
            },
            "inference": None,
            "compute": None,
            "energy": None,
            "config": {},
            "outputs": None,
            "efficiency": {},
        }
        
        results = ExperimentResults.from_dict(data)
        
        assert results.experiment_id == "0001"
        assert isinstance(results.model, ModelInfo)

    def test_from_json(self):
        """Test creating from JSON."""
        data = {
            "experiment_id": "0001",
            "timestamp": "2025-01-01T00:00:00",
            "config_name": None,
            "model": None,
            "inference": None,
            "compute": None,
            "energy": None,
            "config": {},
            "outputs": None,
            "efficiency": {},
        }
        json_str = json.dumps(data)
        
        results = ExperimentResults.from_json(json_str)
        
        assert results.experiment_id == "0001"


class TestResultsManager:
    """Tests for ResultsManager class."""

    def test_initialization(self, tmp_path):
        """Test manager initialization."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        assert manager.results_dir.exists()
        assert manager.experiments_dir.exists()

    def test_save_experiment(self, tmp_path):
        """Test saving experiment."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        results = ExperimentResults(
            experiment_id="0001",
            timestamp="2025-01-01T00:00:00",
        )
        
        saved_path = manager.save_experiment(results)
        
        assert saved_path.exists()
        assert saved_path.name == "0001.json"

    def test_load_experiment(self, tmp_path):
        """Test loading experiment."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        # Save first
        results = ExperimentResults(
            experiment_id="0001",
            timestamp="2025-01-01T00:00:00",
        )
        manager.save_experiment(results)
        
        # Load
        loaded = manager.load_experiment("0001")
        
        assert loaded is not None
        assert loaded.experiment_id == "0001"

    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent experiment."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        loaded = manager.load_experiment("9999")
        
        assert loaded is None

    def test_list_experiments(self, tmp_path):
        """Test listing experiments."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        # Create multiple experiments
        for i in range(3):
            results = ExperimentResults(
                experiment_id=f"000{i}",
                timestamp="2025-01-01T00:00:00",
            )
            manager.save_experiment(results)
        
        exp_ids = manager.list_experiments()
        
        assert len(exp_ids) == 3
        assert "0000" in exp_ids

    def test_aggregate_experiments(self, tmp_path):
        """Test aggregating experiments."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        # Create experiments
        for i in range(2):
            results = create_results(
                experiment_id=f"000{i}",
                config={"model_name": "test"},
                inference_metrics={
                    "total_time_seconds": 10.0,
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                    "total_tokens": 150,
                    "num_prompts": 10,
                    "tokens_per_second": 15.0,
                    "queries_per_second": 1.0,
                    "avg_latency_per_query": 1.0,
                    "avg_tokens_per_prompt": 15.0,
                },
            )
            manager.save_experiment(results)
        
        aggregated = manager.aggregate_experiments()
        
        assert len(aggregated) == 2

    def test_export_to_csv(self, tmp_path):
        """Test exporting to CSV."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        # Create experiment
        results = create_results(
            experiment_id="0001",
            config={"model_name": "test"},
            inference_metrics={
                "total_time_seconds": 10.0,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_tokens": 150,
                "num_prompts": 10,
                "tokens_per_second": 15.0,
                "queries_per_second": 1.0,
                "avg_latency_per_query": 1.0,
                "avg_tokens_per_prompt": 15.0,
            },
        )
        manager.save_experiment(results)
        
        csv_path = tmp_path / "results.csv"
        manager.export_to_csv(csv_path)
        
        assert csv_path.exists()

    def test_generate_summary(self, tmp_path):
        """Test generating summary statistics."""
        manager = ResultsManager(results_dir=tmp_path / "results")
        
        # Create experiments
        for i in range(3):
            results = create_results(
                experiment_id=f"000{i}",
                config={"model_name": "test"},
                inference_metrics={
                    "total_time_seconds": 10.0,
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                    "total_tokens": 150,
                    "num_prompts": 10,
                    "tokens_per_second": 15.0 + i,
                    "queries_per_second": 1.0,
                    "avg_latency_per_query": 1.0,
                    "avg_tokens_per_prompt": 15.0,
                },
                energy_metrics={
                    "duration_seconds": 10.0,
                    "total_energy_kwh": 0.001,
                    "cpu_energy_kwh": 0.0003,
                    "gpu_energy_kwh": 0.0005,
                    "ram_energy_kwh": 0.0002,
                    "emissions_kg_co2": 0.0005,
                },
            )
            manager.save_experiment(results)
        
        summary = manager.generate_summary()
        
        assert summary["total_experiments"] == 3
        assert "throughput" in summary
        assert "energy" in summary


class TestCreateResults:
    """Tests for create_results helper function."""

    def test_create_minimal(self):
        """Test creating minimal results."""
        results = create_results(
            experiment_id="0001",
            config={"model_name": "test"},
        )
        
        assert results.experiment_id == "0001"
        assert results.config["model_name"] == "test"

    def test_create_complete(self):
        """Test creating complete results."""
        results = create_results(
            experiment_id="0001",
            config={"model_name": "test"},
            model_info={
                "model_name": "test",
                "total_parameters": 1000,
                "trainable_parameters": 1000,
                "precision": "float16",
            },
            inference_metrics={
                "total_time_seconds": 10.0,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_tokens": 150,
                "num_prompts": 10,
                "tokens_per_second": 15.0,
                "queries_per_second": 1.0,
                "avg_latency_per_query": 1.0,
                "avg_tokens_per_prompt": 15.0,
            },
            compute_metrics={
                "flops": 1000000,
                "gpu_memory_allocated_mb": 512.0,
                "gpu_memory_peak_mb": 1024.0,
            },
            energy_metrics={
                "duration_seconds": 10.0,
                "total_energy_kwh": 0.001,
                "cpu_energy_kwh": 0.0003,
                "gpu_energy_kwh": 0.0005,
                "ram_energy_kwh": 0.0002,
                "emissions_kg_co2": 0.0005,
            },
            outputs=["output1", "output2"],
        )
        
        assert results.model is not None
        assert results.inference is not None
        assert results.compute is not None
        assert results.energy is not None
        assert len(results.outputs) == 2
