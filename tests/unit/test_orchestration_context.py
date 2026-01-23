"""Tests for experiment context management."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_energy_measure.config.backend_configs import PyTorchConfig
from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.orchestration.context import (
    ExperimentContext,
    experiment_context,
)


@pytest.fixture
def sample_config():
    """Create a sample experiment config for testing."""
    return ExperimentConfig(
        config_name="test_config",
        model_name="test/model",
        gpus=[0],
        # num_processes now derived from backend parallelism config
        pytorch=PyTorchConfig(parallelism_strategy="none", parallelism_degree=1),
    )


@pytest.fixture
def mock_accelerator():
    """Create a mock Accelerator."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.is_main_process = True
    accelerator.process_index = 0
    accelerator.num_processes = 1
    return accelerator


class TestExperimentContext:
    """Tests for ExperimentContext dataclass."""

    def test_create_context(self, sample_config, mock_accelerator):
        """Test basic context creation."""
        ctx = ExperimentContext(
            experiment_id="0001",
            config=sample_config,
            accelerator=mock_accelerator,
            device=torch.device("cpu"),
            is_main_process=True,
            process_index=0,
        )

        assert ctx.experiment_id == "0001"
        assert ctx.config is sample_config
        assert ctx.is_main_process is True
        assert ctx.process_index == 0
        assert ctx.device == torch.device("cpu")

    def test_elapsed_time(self, sample_config, mock_accelerator):
        """Test elapsed_time property."""
        ctx = ExperimentContext(
            experiment_id="0001",
            config=sample_config,
            accelerator=mock_accelerator,
            device=torch.device("cpu"),
            is_main_process=True,
            process_index=0,
            start_time=datetime.now(),
        )

        # Elapsed time should be small (< 1 second)
        assert ctx.elapsed_time >= 0
        assert ctx.elapsed_time < 1.0

    def test_num_processes_property(self, sample_config, mock_accelerator):
        """Test num_processes property delegates to accelerator."""
        mock_accelerator.num_processes = 4
        ctx = ExperimentContext(
            experiment_id="0001",
            config=sample_config,
            accelerator=mock_accelerator,
            device=torch.device("cpu"),
            is_main_process=True,
            process_index=0,
        )

        assert ctx.num_processes == 4

    @patch("llm_energy_measure.orchestration.context.cleanup_distributed")
    def test_cleanup_calls_distributed_cleanup(self, mock_cleanup, sample_config, mock_accelerator):
        """Test cleanup calls cleanup_distributed."""
        ctx = ExperimentContext(
            experiment_id="0001",
            config=sample_config,
            accelerator=mock_accelerator,
            device=torch.device("cpu"),
            is_main_process=True,
            process_index=0,
        )

        ctx.cleanup()

        mock_cleanup.assert_called_once()


class TestExperimentContextCreate:
    """Tests for ExperimentContext.create factory method."""

    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_create_sets_up_accelerator(
        self, mock_get_accel, mock_get_id, sample_config, mock_accelerator
    ):
        """Test create sets up accelerator correctly."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0042"

        ctx = ExperimentContext.create(sample_config)

        mock_get_accel.assert_called_once_with(
            gpus=[0],
            num_processes=1,
        )
        assert ctx.accelerator is mock_accelerator
        assert ctx.experiment_id == "0042"

    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_create_propagates_accelerator_properties(
        self, mock_get_accel, mock_get_id, sample_config, mock_accelerator
    ):
        """Test create propagates accelerator properties to context."""
        mock_accelerator.is_main_process = False
        mock_accelerator.process_index = 2
        mock_accelerator.device = torch.device("cuda:1")
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"

        ctx = ExperimentContext.create(sample_config)

        assert ctx.is_main_process is False
        assert ctx.process_index == 2
        assert ctx.device == torch.device("cuda:1")

    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_create_sets_start_time(
        self, mock_get_accel, mock_get_id, sample_config, mock_accelerator
    ):
        """Test create sets start_time to approximately now."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"

        before = datetime.now()
        ctx = ExperimentContext.create(sample_config)
        after = datetime.now()

        assert before <= ctx.start_time <= after

    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_create_with_custom_id_file(
        self, mock_get_accel, mock_get_id, sample_config, mock_accelerator, tmp_path
    ):
        """Test create uses custom id_file."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"
        id_file = tmp_path / "custom_id.txt"

        ExperimentContext.create(sample_config, id_file=id_file)

        mock_get_id.assert_called_once_with(mock_accelerator, id_file)


class TestExperimentContextManager:
    """Tests for experiment_context context manager."""

    @patch("llm_energy_measure.orchestration.context.cleanup_distributed")
    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_context_manager_yields_context(
        self, mock_get_accel, mock_get_id, mock_cleanup, sample_config, mock_accelerator
    ):
        """Test context manager yields ExperimentContext."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"

        with experiment_context(sample_config) as ctx:
            assert isinstance(ctx, ExperimentContext)
            assert ctx.experiment_id == "0001"
            assert ctx.config is sample_config

    @patch("llm_energy_measure.orchestration.context.cleanup_distributed")
    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_context_manager_cleans_up_on_exit(
        self, mock_get_accel, mock_get_id, mock_cleanup, sample_config, mock_accelerator
    ):
        """Test context manager cleans up on normal exit."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"

        with experiment_context(sample_config):
            pass

        mock_cleanup.assert_called_once()

    @patch("llm_energy_measure.orchestration.context.cleanup_distributed")
    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_context_manager_cleans_up_on_exception(
        self, mock_get_accel, mock_get_id, mock_cleanup, sample_config, mock_accelerator
    ):
        """Test context manager cleans up even when exception occurs."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"

        with pytest.raises(ValueError), experiment_context(sample_config):
            raise ValueError("test error")

        mock_cleanup.assert_called_once()

    @patch("llm_energy_measure.orchestration.context.cleanup_distributed")
    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_context_manager_with_custom_id_file(
        self,
        mock_get_accel,
        mock_get_id,
        mock_cleanup,
        sample_config,
        mock_accelerator,
        tmp_path,
    ):
        """Test context manager accepts custom id_file."""
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"
        id_file = tmp_path / "id.txt"

        with experiment_context(sample_config, id_file=id_file):
            pass

        mock_get_id.assert_called_once_with(mock_accelerator, id_file)


class TestExperimentContextMultiProcess:
    """Tests for multi-process scenarios."""

    @patch("llm_energy_measure.orchestration.context.get_shared_unique_id")
    @patch("llm_energy_measure.orchestration.context.get_accelerator")
    def test_multi_process_config(self, mock_get_accel, mock_get_id):
        """Test context creation with multi-process config.

        In backend-native architecture, num_processes is derived from the
        backend's parallelism config (e.g., pytorch.parallelism_degree).
        """
        config = ExperimentConfig(
            config_name="multi_gpu",
            model_name="test/model",
            gpus=[0, 1, 2, 3],
            # Parallelism degree now from backend config
            pytorch=PyTorchConfig(
                parallelism_strategy="tensor_parallel",
                parallelism_degree=4,
            ),
        )
        mock_accelerator = MagicMock()
        mock_accelerator.device = torch.device("cuda:2")
        mock_accelerator.is_main_process = False
        mock_accelerator.process_index = 2
        mock_accelerator.num_processes = 4
        mock_get_accel.return_value = mock_accelerator
        mock_get_id.return_value = "0001"

        ctx = ExperimentContext.create(config)

        # num_processes derived from pytorch.parallelism_degree
        mock_get_accel.assert_called_once_with(
            gpus=[0, 1, 2, 3],
            num_processes=4,
        )
        assert ctx.is_main_process is False
        assert ctx.process_index == 2
        assert ctx.num_processes == 4
