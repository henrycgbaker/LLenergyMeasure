"""Tests for experiment lifecycle management."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_energy_measure.orchestration.lifecycle import (
    cleanup_cuda,
    cleanup_distributed,
    ensure_clean_start,
    experiment_lifecycle,
    full_cleanup,
    warmup_model,
)


class TestCleanupCuda:
    """Tests for cleanup_cuda function."""

    @patch("llm_energy_measure.orchestration.lifecycle.torch.cuda.is_available")
    def test_cleans_up_when_cuda_available(self, mock_available):
        mock_available.return_value = True

        with (
            patch("torch.cuda.empty_cache") as mock_empty,
            patch("torch.cuda.reset_peak_memory_stats") as mock_reset,
            patch("torch.cuda.ipc_collect") as mock_ipc,
        ):
            cleanup_cuda()

            mock_empty.assert_called_once()
            mock_reset.assert_called_once()
            mock_ipc.assert_called_once()

    @patch("llm_energy_measure.orchestration.lifecycle.torch.cuda.is_available")
    def test_skips_when_cuda_not_available(self, mock_available):
        mock_available.return_value = False

        with (
            patch("torch.cuda.empty_cache") as mock_empty,
            patch("torch.cuda.reset_peak_memory_stats") as mock_reset,
        ):
            cleanup_cuda()

            mock_empty.assert_not_called()
            mock_reset.assert_not_called()

    @patch("llm_energy_measure.orchestration.lifecycle.torch.cuda.is_available")
    def test_handles_exception_gracefully(self, mock_available):
        mock_available.return_value = True

        with patch("torch.cuda.empty_cache", side_effect=RuntimeError("test error")):
            # Should not raise
            cleanup_cuda()


class TestCleanupDistributed:
    """Tests for cleanup_distributed function."""

    @patch("llm_energy_measure.orchestration.lifecycle.dist.is_available")
    @patch("llm_energy_measure.orchestration.lifecycle.dist.is_initialized")
    def test_destroys_process_group(self, mock_init, mock_avail):
        mock_avail.return_value = True
        mock_init.return_value = True

        with patch(
            "llm_energy_measure.orchestration.lifecycle.dist.destroy_process_group"
        ) as mock_destroy:
            cleanup_distributed()
            mock_destroy.assert_called_once()

    @patch("llm_energy_measure.orchestration.lifecycle.dist.is_available")
    @patch("llm_energy_measure.orchestration.lifecycle.dist.is_initialized")
    def test_skips_when_not_initialized(self, mock_init, mock_avail):
        mock_avail.return_value = True
        mock_init.return_value = False

        with patch(
            "llm_energy_measure.orchestration.lifecycle.dist.destroy_process_group"
        ) as mock_destroy:
            cleanup_distributed()
            mock_destroy.assert_not_called()


class TestFullCleanup:
    """Tests for full_cleanup function."""

    @patch("llm_energy_measure.orchestration.lifecycle.gc.collect")
    @patch("llm_energy_measure.orchestration.lifecycle.cleanup_cuda")
    @patch("llm_energy_measure.orchestration.lifecycle.cleanup_distributed")
    def test_calls_all_cleanup_functions(self, mock_dist, mock_cuda, mock_gc):
        full_cleanup()

        mock_dist.assert_called_once()
        mock_cuda.assert_called_once()
        mock_gc.assert_called_once()


class TestExperimentLifecycle:
    """Tests for experiment_lifecycle context manager."""

    @patch("llm_energy_measure.orchestration.lifecycle.full_cleanup")
    def test_cleans_up_on_normal_exit(self, mock_cleanup):
        with experiment_lifecycle():
            pass

        mock_cleanup.assert_called_once()

    @patch("llm_energy_measure.orchestration.lifecycle.full_cleanup")
    def test_cleans_up_on_exception(self, mock_cleanup):
        with pytest.raises(ValueError), experiment_lifecycle():
            raise ValueError("test error")

        mock_cleanup.assert_called_once()

    @patch("llm_energy_measure.orchestration.lifecycle.full_cleanup")
    def test_with_accelerator(self, mock_cleanup):
        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        with experiment_lifecycle(accelerator=mock_accelerator):
            pass

        mock_cleanup.assert_called_once()


class TestEnsureCleanStart:
    """Tests for ensure_clean_start function."""

    @patch("llm_energy_measure.orchestration.lifecycle.cleanup_cuda")
    @patch("llm_energy_measure.orchestration.lifecycle.dist.is_available")
    @patch("llm_energy_measure.orchestration.lifecycle.dist.is_initialized")
    def test_destroys_existing_process_group(self, mock_init, mock_avail, mock_cuda):
        mock_avail.return_value = True
        mock_init.return_value = True

        with patch(
            "llm_energy_measure.orchestration.lifecycle.dist.destroy_process_group"
        ) as mock_destroy:
            ensure_clean_start()
            mock_destroy.assert_called_once()
            mock_cuda.assert_called_once()


class TestWarmupModel:
    """Tests for warmup_model function."""

    def test_performs_warmup_runs(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.input_ids.to.return_value = torch.zeros(1, 10)

        warmup_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            num_runs=3,
        )

        # Model should be called num_runs times
        assert mock_model.call_count == 3

    def test_handles_failure_gracefully(self):
        mock_model = MagicMock(side_effect=RuntimeError("test error"))
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.input_ids.to.return_value = torch.zeros(1, 10)

        # Should not raise
        warmup_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
            num_runs=1,
        )
