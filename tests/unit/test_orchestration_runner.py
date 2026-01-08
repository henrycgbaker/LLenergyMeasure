"""Tests for experiment orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.domain.metrics import (
    CombinedMetrics,
    ComputeMetrics,
    EnergyMetrics,
    InferenceMetrics,
)
from llm_energy_measure.orchestration.runner import ExperimentOrchestrator


@pytest.fixture
def sample_config():
    """Create a sample experiment config."""
    return ExperimentConfig(
        config_name="test_config",
        model_name="test/model",
        gpu_list=[0],
        num_processes=1,
    )


@pytest.fixture
def mock_context(sample_config):
    """Create a mock ExperimentContext."""
    ctx = MagicMock()
    ctx.experiment_id = "0001"
    ctx.config = sample_config
    ctx.process_index = 0
    # Use PropertyMock for device.index
    mock_device = MagicMock()
    type(mock_device).index = PropertyMock(return_value=0)
    ctx.device = mock_device
    # Provide proper dicts for effective_config and cli_overrides
    ctx.effective_config = {}
    ctx.cli_overrides = {}
    return ctx


@pytest.fixture
def mock_components():
    """Create mock components for the orchestrator."""
    loader = MagicMock()
    loader.load.return_value = (MagicMock(), MagicMock())

    inference_engine = MagicMock()
    inference_engine.run.return_value = MagicMock()

    energy_metrics = EnergyMetrics(
        total_energy_j=10.0,
        gpu_power_w=100.0,
        duration_sec=1.0,
    )

    metrics_collector = MagicMock()
    metrics_collector.collect.return_value = CombinedMetrics(
        inference=InferenceMetrics(
            total_tokens=100,
            input_tokens=20,
            output_tokens=80,
            inference_time_sec=1.0,
            tokens_per_second=100.0,
            latency_per_token_ms=10.0,
        ),
        energy=energy_metrics,
        compute=ComputeMetrics(flops_total=1e12),
    )

    energy_backend = MagicMock()
    energy_backend.start_tracking.return_value = MagicMock()
    energy_backend.stop_tracking.return_value = energy_metrics

    repository = MagicMock()
    repository.save_raw.return_value = Path("/tmp/result.json")

    return {
        "model_loader": loader,
        "inference_engine": inference_engine,
        "metrics_collector": metrics_collector,
        "energy_backend": energy_backend,
        "repository": repository,
    }


class TestExperimentOrchestrator:
    """Tests for ExperimentOrchestrator class."""

    def test_init(self, mock_components):
        """Test orchestrator initialization."""
        orchestrator = ExperimentOrchestrator(**mock_components)

        assert orchestrator._loader is mock_components["model_loader"]
        assert orchestrator._inference is mock_components["inference_engine"]
        assert orchestrator._metrics is mock_components["metrics_collector"]
        assert orchestrator._energy is mock_components["energy_backend"]
        assert orchestrator._repository is mock_components["repository"]

    def test_run_calls_components_in_order(self, mock_context, mock_components):
        """Test that run() calls components in correct order."""
        orchestrator = ExperimentOrchestrator(**mock_components)
        prompts = ["Hello", "World"]

        result_path = orchestrator.run(mock_context, prompts)

        # Verify call order
        mock_components["model_loader"].load.assert_called_once()
        mock_components["energy_backend"].start_tracking.assert_called_once()
        mock_components["inference_engine"].run.assert_called_once()
        mock_components["energy_backend"].stop_tracking.assert_called_once()
        mock_components["metrics_collector"].collect.assert_called_once()
        mock_components["repository"].save_raw.assert_called_once()

        assert result_path == Path("/tmp/result.json")

    def test_run_passes_correct_arguments(self, mock_context, mock_components):
        """Test that run() passes correct arguments to components."""
        orchestrator = ExperimentOrchestrator(**mock_components)
        prompts = ["Test prompt"]

        orchestrator.run(mock_context, prompts)

        # Check model loader
        mock_components["model_loader"].load.assert_called_with(mock_context.config)

        # Check inference engine
        model, tokenizer = mock_components["model_loader"].load.return_value
        mock_components["inference_engine"].run.assert_called_with(
            model, tokenizer, prompts, mock_context.config
        )

    def test_run_saves_raw_result(self, mock_context, mock_components):
        """Test that run() saves raw result with correct data."""
        orchestrator = ExperimentOrchestrator(**mock_components)
        prompts = ["Test"]

        orchestrator.run(mock_context, prompts)

        # Verify save_raw was called with experiment_id and a RawProcessResult
        call_args = mock_components["repository"].save_raw.call_args
        assert call_args[0][0] == "0001"  # experiment_id

        raw_result = call_args[0][1]
        assert raw_result.experiment_id == "0001"
        assert raw_result.process_index == 0
        assert raw_result.config_name == "test_config"
        assert raw_result.model_name == "test/model"

    def test_run_handles_gpu_id_none(self, sample_config, mock_components):
        """Test run() when device.index is None (CPU)."""
        # Create context with device.index = None
        ctx = MagicMock()
        ctx.experiment_id = "0001"
        ctx.config = sample_config
        ctx.process_index = 0
        mock_device = MagicMock()
        type(mock_device).index = PropertyMock(return_value=None)
        ctx.device = mock_device
        # Provide proper dicts for effective_config and cli_overrides
        ctx.effective_config = {}
        ctx.cli_overrides = {}

        orchestrator = ExperimentOrchestrator(**mock_components)
        orchestrator.run(ctx, ["test"])

        raw_result = mock_components["repository"].save_raw.call_args[0][1]
        assert raw_result.gpu_id == 0  # Falls back to 0
