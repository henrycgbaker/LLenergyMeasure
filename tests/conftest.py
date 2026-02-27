"""Pytest configuration and shared fixtures for v2.0 tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import AggregationMetadata, ExperimentResult

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_EPOCH_END = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def make_config(**overrides) -> ExperimentConfig:
    """Return a valid ExperimentConfig with sensible defaults.

    Tests override only what they care about.
    """
    defaults: dict = {
        "model": "gpt2",
        "backend": "pytorch",
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def make_result(**overrides) -> ExperimentResult:
    """Return a valid ExperimentResult with sensible defaults.

    Includes all required fields (measurement_config_hash,
    measurement_methodology, start_time, end_time) to prevent ValidationError.
    """
    defaults: dict = {
        "experiment_id": "test-001",
        "measurement_config_hash": "abc123def4567890",
        "measurement_methodology": "total",
        "aggregation": AggregationMetadata(num_processes=1),
        "total_tokens": 1000,
        "total_energy_j": 10.0,
        "total_inference_time_sec": 5.0,
        "avg_tokens_per_second": 200.0,
        "avg_energy_per_token_j": 0.01,
        "total_flops": 1e9,
        "start_time": _EPOCH,
        "end_time": _EPOCH_END,
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return make_config()


@pytest.fixture
def sample_result() -> ExperimentResult:
    return make_result()


@pytest.fixture
def tmp_results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d
