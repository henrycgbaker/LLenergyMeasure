"""Tests for DI factory in orchestration/factory.py."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.orchestration.factory import (
    ExperimentComponents,
    create_components,
    create_orchestrator,
)
from llm_energy_measure.protocols import (
    InferenceEngine,
    MetricsCollector,
    ModelLoader,
    ResultsRepository,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """Create a sample experiment config."""
    return ExperimentConfig(
        config_name="test_config",
        model_name="test/model",
        gpus=[0],
        num_processes=1,
    )


@pytest.fixture
def mock_accelerator() -> MagicMock:
    """Create a mock Accelerator instance."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.is_main_process = True
    accelerator.process_index = 0
    accelerator.num_processes = 1
    return accelerator


@pytest.fixture
def mock_context(sample_config, mock_accelerator) -> MagicMock:
    """Create a mock ExperimentContext."""
    ctx = MagicMock()
    ctx.experiment_id = "0001"
    ctx.config = sample_config
    ctx.accelerator = mock_accelerator
    ctx.process_index = 0
    ctx.is_main_process = True

    mock_device = MagicMock()
    type(mock_device).index = PropertyMock(return_value=0)
    ctx.device = mock_device

    return ctx


# ============================================================
# ExperimentComponents Tests
# ============================================================


class TestExperimentComponents:
    """Tests for ExperimentComponents dataclass."""

    def test_can_create_with_all_fields(self):
        """Verify dataclass can be instantiated with all fields."""
        components = ExperimentComponents(
            model_loader=MagicMock(),
            inference_engine=MagicMock(),
            metrics_collector=MagicMock(),
            energy_backend=MagicMock(),
            repository=MagicMock(),
        )

        assert components.model_loader is not None
        assert components.inference_engine is not None
        assert components.metrics_collector is not None
        assert components.energy_backend is not None
        assert components.repository is not None

    def test_is_dataclass(self):
        """Verify ExperimentComponents is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ExperimentComponents)


# ============================================================
# create_components Tests
# ============================================================


class TestCreateComponents:
    """Tests for create_components factory function.

    Note: The factory now uses backend adapters (BackendModelLoaderAdapter, etc.)
    instead of direct implementations (HuggingFaceModelLoader, etc.).
    """

    @patch("llm_energy_measure.results.repository.FileSystemRepository")
    @patch("llm_energy_measure.core.energy_backends.get_backend")
    @patch("llm_energy_measure.core.inference_backends.get_backend")
    def test_creates_all_components(
        self,
        mock_get_inference_backend,
        mock_get_energy_backend,
        mock_repo_cls,
        mock_context,
    ):
        """Verify create_components creates all required components."""
        # Setup mock backend
        mock_backend = MagicMock()
        mock_backend.name = "pytorch"
        mock_backend.version = "1.0.0"
        mock_backend.is_available.return_value = True
        mock_get_inference_backend.return_value = mock_backend

        components = create_components(mock_context)

        assert isinstance(components, ExperimentComponents)
        # get_backend is called twice: once for validation, once for the actual backend
        assert mock_get_inference_backend.call_count == 2
        mock_get_energy_backend.assert_called_once_with("codecarbon")
        mock_repo_cls.assert_called_once()

    @patch("llm_energy_measure.results.repository.FileSystemRepository")
    @patch("llm_energy_measure.core.energy_backends.get_backend")
    @patch("llm_energy_measure.core.inference_backends.get_backend")
    def test_components_have_correct_types(
        self,
        mock_get_inference_backend,
        mock_get_energy_backend,
        mock_repo_cls,
        mock_context,
    ):
        """Verify components implement correct protocols."""
        # Setup mock backend
        mock_backend = MagicMock()
        mock_backend.name = "pytorch"
        mock_backend.version = "1.0.0"
        mock_get_inference_backend.return_value = mock_backend

        components = create_components(mock_context)

        # Components should be adapters wrapping the backend
        assert components.model_loader is not None
        assert components.inference_engine is not None
        assert components.metrics_collector is not None
        assert components.backend_name == "pytorch"

    @patch("llm_energy_measure.results.repository.FileSystemRepository")
    @patch("llm_energy_measure.core.energy_backends.get_backend")
    @patch("llm_energy_measure.core.inference_backends.get_backend")
    def test_uses_codecarbon_backend(
        self,
        mock_get_inference_backend,
        mock_get_energy_backend,
        mock_repo_cls,
        mock_context,
    ):
        """Verify CodeCarbon is selected as the energy backend."""
        # Setup mock backend
        mock_backend = MagicMock()
        mock_backend.name = "pytorch"
        mock_backend.version = "1.0.0"
        mock_get_inference_backend.return_value = mock_backend

        create_components(mock_context)

        mock_get_energy_backend.assert_called_once_with("codecarbon")

    @patch("llm_energy_measure.results.repository.FileSystemRepository")
    @patch("llm_energy_measure.core.energy_backends.get_backend")
    @patch("llm_energy_measure.core.inference_backends.get_backend")
    def test_returns_correct_component_instances(
        self,
        mock_get_inference_backend,
        mock_get_energy_backend,
        mock_repo_cls,
        mock_context,
    ):
        """Verify returned components contain expected objects."""
        mock_inference_backend = MagicMock()
        mock_inference_backend.name = "pytorch"
        mock_inference_backend.version = "1.0.0"
        mock_energy_backend = MagicMock()
        mock_repo = MagicMock()

        mock_get_inference_backend.return_value = mock_inference_backend
        mock_get_energy_backend.return_value = mock_energy_backend
        mock_repo_cls.return_value = mock_repo

        components = create_components(mock_context)

        # Adapters wrap the backend, so check they exist
        assert components.model_loader is not None
        assert components.inference_engine is not None
        assert components.metrics_collector is not None
        # These are the direct returns from mocks
        assert components.energy_backend is mock_energy_backend
        assert components.repository is mock_repo


# ============================================================
# create_orchestrator Tests
# ============================================================


class TestCreateOrchestrator:
    """Tests for create_orchestrator factory function."""

    @patch("llm_energy_measure.orchestration.runner.ExperimentOrchestrator")
    @patch("llm_energy_measure.orchestration.factory.create_components")
    def test_creates_orchestrator_with_components(
        self, mock_create_components, mock_orchestrator_cls, mock_context
    ):
        """Verify create_orchestrator creates orchestrator with components."""
        mock_components = ExperimentComponents(
            model_loader=MagicMock(),
            inference_engine=MagicMock(),
            metrics_collector=MagicMock(),
            energy_backend=MagicMock(),
            repository=MagicMock(),
            backend_name="pytorch",
            backend_version="transformers=4.50.0",
        )
        mock_create_components.return_value = mock_components

        create_orchestrator(mock_context)

        # create_components is called with (context, results_dir=None)
        mock_create_components.assert_called_once_with(mock_context, None)
        mock_orchestrator_cls.assert_called_once_with(
            model_loader=mock_components.model_loader,
            inference_engine=mock_components.inference_engine,
            metrics_collector=mock_components.metrics_collector,
            energy_backend=mock_components.energy_backend,
            repository=mock_components.repository,
            backend_name=mock_components.backend_name,
            backend_version=mock_components.backend_version,
        )

    @patch("llm_energy_measure.orchestration.runner.ExperimentOrchestrator")
    @patch("llm_energy_measure.orchestration.factory.create_components")
    def test_returns_orchestrator_instance(
        self, mock_create_components, mock_orchestrator_cls, mock_context
    ):
        """Verify create_orchestrator returns the orchestrator instance."""
        mock_components = ExperimentComponents(
            model_loader=MagicMock(),
            inference_engine=MagicMock(),
            metrics_collector=MagicMock(),
            energy_backend=MagicMock(),
            repository=MagicMock(),
            backend_name="pytorch",
            backend_version="transformers=4.50.0",
        )
        mock_create_components.return_value = mock_components

        mock_orchestrator = MagicMock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        result = create_orchestrator(mock_context)

        assert result is mock_orchestrator


# ============================================================
# Integration Tests (with real implementations, mocked deps)
# ============================================================


class TestFactoryIntegration:
    """Integration tests for factory with real implementations."""

    @patch("llm_energy_measure.core.energy_backends.get_backend")
    def test_create_components_produces_protocol_compliant_objects(
        self, mock_get_backend, mock_context
    ):
        """Verify created components implement their protocols."""
        # Mock the energy backend since it has external dependencies
        mock_backend = MagicMock()
        mock_backend.name = "mock"
        mock_backend.start_tracking.return_value = MagicMock()
        mock_backend.is_available.return_value = True
        mock_get_backend.return_value = mock_backend

        components = create_components(mock_context)

        # Verify protocol compliance via isinstance (runtime_checkable)
        assert isinstance(components.model_loader, ModelLoader)
        assert isinstance(components.inference_engine, InferenceEngine)
        assert isinstance(components.metrics_collector, MetricsCollector)
        assert isinstance(components.repository, ResultsRepository)
        # EnergyBackend check - mock won't satisfy it, but real would

    @patch("llm_energy_measure.core.energy_backends.get_backend")
    def test_create_orchestrator_returns_functional_orchestrator(
        self, mock_get_backend, mock_context
    ):
        """Verify create_orchestrator returns a functional orchestrator."""
        from llm_energy_measure.orchestration.runner import ExperimentOrchestrator

        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        orchestrator = create_orchestrator(mock_context)

        assert isinstance(orchestrator, ExperimentOrchestrator)
        # Verify it has the expected interface
        assert hasattr(orchestrator, "run")
        assert callable(orchestrator.run)
