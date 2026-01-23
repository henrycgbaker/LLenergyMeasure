"""Tests for protocol implementations in core/implementations.py."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.core.implementations import (
    HuggingFaceModelLoader,
    ThroughputMetricsCollector,
)
from llenergymeasure.domain.metrics import InferenceMetrics
from llenergymeasure.protocols import MetricsCollector, ModelLoader

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """Create a sample experiment config for testing."""
    return ExperimentConfig(
        config_name="test_config",
        model_name="test/model",
        gpus=[0],
        num_processes=1,
    )


@pytest.fixture
def mock_accelerator() -> MagicMock:
    """Create a mock Accelerator."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.is_main_process = True
    return accelerator


@pytest.fixture
def mock_inference_result() -> MagicMock:
    """Create a mock InferenceResult."""
    result = MagicMock()
    result.metrics = InferenceMetrics(
        total_tokens=100,
        input_tokens=20,
        output_tokens=80,
        inference_time_sec=1.0,
        tokens_per_second=80.0,
        latency_per_token_ms=12.5,
    )
    result.input_ids = torch.zeros((2, 10), dtype=torch.long)
    return result


# ============================================================
# HuggingFaceModelLoader Tests
# ============================================================


class TestHuggingFaceModelLoader:
    """Tests for HuggingFaceModelLoader class."""

    def test_implements_model_loader_protocol(self):
        """Verify class implements ModelLoader protocol."""
        loader = HuggingFaceModelLoader()
        assert isinstance(loader, ModelLoader)

    def test_has_load_method(self):
        """Verify loader has load method with correct signature."""
        loader = HuggingFaceModelLoader()
        assert hasattr(loader, "load")
        assert callable(loader.load)

    @patch("llenergymeasure.core.model_loader.load_model_tokenizer")
    def test_load_delegates_to_function(self, mock_load_fn, sample_config):
        """Verify load() delegates to load_model_tokenizer function."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_fn.return_value = (mock_model, mock_tokenizer)

        loader = HuggingFaceModelLoader()
        model, tokenizer = loader.load(sample_config)

        mock_load_fn.assert_called_once_with(sample_config)
        assert model is mock_model
        assert tokenizer is mock_tokenizer

    @patch("llenergymeasure.core.model_loader.load_model_tokenizer")
    def test_load_passes_config_unchanged(self, mock_load_fn, sample_config):
        """Verify the config object is passed through unchanged."""
        mock_load_fn.return_value = (MagicMock(), MagicMock())

        loader = HuggingFaceModelLoader()
        loader.load(sample_config)

        call_args = mock_load_fn.call_args[0]
        assert call_args[0] is sample_config


# ============================================================
# ThroughputMetricsCollector Tests
# ============================================================


class TestThroughputMetricsCollector:
    """Tests for ThroughputMetricsCollector class."""

    def test_implements_metrics_collector_protocol(self, mock_accelerator):
        """Verify class implements MetricsCollector protocol."""
        collector = ThroughputMetricsCollector(mock_accelerator)
        assert isinstance(collector, MetricsCollector)

    def test_stores_accelerator(self, mock_accelerator):
        """Verify accelerator is stored for later use."""
        collector = ThroughputMetricsCollector(mock_accelerator)
        assert collector._accelerator is mock_accelerator

    def test_has_collect_method(self, mock_accelerator):
        """Verify collector has collect method."""
        collector = ThroughputMetricsCollector(mock_accelerator)
        assert hasattr(collector, "collect")
        assert callable(collector.collect)

    @patch("llenergymeasure.core.compute_metrics.collect_compute_metrics")
    def test_collect_delegates_to_function(
        self,
        mock_collect_fn,
        mock_accelerator,
        mock_inference_result,
        sample_config,
    ):
        """Verify collect() delegates to collect_compute_metrics function."""
        from llenergymeasure.domain.metrics import ComputeMetrics

        mock_compute = ComputeMetrics(flops_total=1e12)
        mock_collect_fn.return_value = mock_compute

        mock_model = MagicMock()

        collector = ThroughputMetricsCollector(mock_accelerator)
        result = collector.collect(mock_model, mock_inference_result, sample_config)

        mock_collect_fn.assert_called_once_with(
            model=mock_model,
            device=mock_accelerator.device,
            input_ids=mock_inference_result.input_ids,
            accelerator=mock_accelerator,
            config=sample_config,
        )

        # Verify result is CombinedMetrics with correct structure
        assert result.inference is mock_inference_result.metrics
        assert result.compute is mock_compute

    @patch("llenergymeasure.core.compute_metrics.collect_compute_metrics")
    def test_collect_returns_placeholder_energy_metrics(
        self,
        mock_collect_fn,
        mock_accelerator,
        mock_inference_result,
        sample_config,
    ):
        """Verify collect() returns placeholder energy metrics."""
        from llenergymeasure.domain.metrics import ComputeMetrics

        mock_collect_fn.return_value = ComputeMetrics(flops_total=1e12)
        mock_model = MagicMock()

        collector = ThroughputMetricsCollector(mock_accelerator)
        result = collector.collect(mock_model, mock_inference_result, sample_config)

        # Energy metrics should be placeholders (zeros)
        assert result.energy.total_energy_j == 0.0
        assert result.energy.gpu_energy_j == 0.0
        assert result.energy.duration_sec == 0.0

    @patch("llenergymeasure.core.compute_metrics.collect_compute_metrics")
    def test_collect_uses_device_from_accelerator(
        self,
        mock_collect_fn,
        mock_accelerator,
        mock_inference_result,
        sample_config,
    ):
        """Verify collect() uses device from stored accelerator."""
        from llenergymeasure.domain.metrics import ComputeMetrics

        mock_collect_fn.return_value = ComputeMetrics(flops_total=1e12)

        # Set specific device
        test_device = torch.device("cuda:1")
        mock_accelerator.device = test_device

        collector = ThroughputMetricsCollector(mock_accelerator)
        collector.collect(MagicMock(), mock_inference_result, sample_config)

        call_kwargs = mock_collect_fn.call_args[1]
        assert call_kwargs["device"] is test_device


# ============================================================
# Integration-style Tests
# ============================================================


class TestImplementationsIntegration:
    """Integration tests verifying implementations work together."""

    def test_all_implementations_can_be_instantiated(self, mock_accelerator):
        """Verify all implementations can be created without error."""
        loader = HuggingFaceModelLoader()
        collector = ThroughputMetricsCollector(mock_accelerator)

        assert loader is not None
        assert collector is not None

    def test_implementations_satisfy_protocols_at_runtime(self, mock_accelerator):
        """Verify runtime_checkable protocols work with implementations."""
        loader = HuggingFaceModelLoader()
        collector = ThroughputMetricsCollector(mock_accelerator)

        # These should all pass due to @runtime_checkable
        assert isinstance(loader, ModelLoader)
        assert isinstance(collector, MetricsCollector)
