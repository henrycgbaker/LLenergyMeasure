"""Protocol conformance tests using protocol injection fakes.

Tests that FakeInferenceBackend, FakeEnergyBackend, and FakeResultsRepository
satisfy the runtime_checkable Protocol interfaces defined in llenergymeasure.

INF-10 compliance: no unittest.mock.patch on internal modules. Fakes are
injected via constructor args and satisfy isinstance() checks at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.core.backends.protocol import InferenceBackend
from llenergymeasure.protocols import EnergyBackend, ResultsRepository
from tests.conftest import make_config, make_result
from tests.fakes import FakeEnergyBackend, FakeInferenceBackend, FakeResultsRepository

# ---------------------------------------------------------------------------
# Protocol isinstance checks (structural conformance)
# ---------------------------------------------------------------------------


def test_fake_inference_backend_satisfies_protocol():
    """FakeInferenceBackend satisfies isinstance(InferenceBackend) at runtime."""
    fake = FakeInferenceBackend(result=make_result())
    assert isinstance(fake, InferenceBackend)


def test_fake_energy_backend_satisfies_protocol():
    """FakeEnergyBackend satisfies isinstance(EnergyBackend) at runtime."""
    fake = FakeEnergyBackend()
    assert isinstance(fake, EnergyBackend)


def test_fake_results_repository_satisfies_protocol():
    """FakeResultsRepository satisfies isinstance(ResultsRepository) at runtime."""
    fake = FakeResultsRepository()
    assert isinstance(fake, ResultsRepository)


# ---------------------------------------------------------------------------
# FakeInferenceBackend behaviour
# ---------------------------------------------------------------------------


def test_fake_inference_backend_returns_result():
    """FakeInferenceBackend.run(config) returns the injected ExperimentResult."""
    expected = make_result(experiment_id="injected-001")
    fake = FakeInferenceBackend(result=expected)
    config = make_config()

    result = fake.run(config)
    assert result is expected


def test_fake_inference_backend_records_calls():
    """FakeInferenceBackend.run(config) appends config to run_calls."""
    fake = FakeInferenceBackend(result=make_result())
    config = make_config()

    assert fake.run_calls == []
    fake.run(config)
    assert len(fake.run_calls) == 1
    assert fake.run_calls[0] is config


def test_fake_inference_backend_records_multiple_calls():
    """FakeInferenceBackend.run_calls records all calls in order."""
    result = make_result()
    fake = FakeInferenceBackend(result=result)
    config_a = make_config(model="gpt2")
    config_b = make_config(model="bert-base")

    fake.run(config_a)
    fake.run(config_b)

    assert len(fake.run_calls) == 2
    assert fake.run_calls[0] is config_a
    assert fake.run_calls[1] is config_b


def test_fake_inference_backend_raises_without_result():
    """FakeInferenceBackend raises ValueError when no result is injected."""
    fake = FakeInferenceBackend()
    with pytest.raises(ValueError, match="set result before calling run"):
        fake.run(make_config())


def test_fake_inference_backend_has_name_attribute():
    """FakeInferenceBackend.name attribute exists (protocol requirement)."""
    fake = FakeInferenceBackend()
    assert fake.name == "fake"


# ---------------------------------------------------------------------------
# FakeEnergyBackend behaviour
# ---------------------------------------------------------------------------


def test_fake_energy_backend_lifecycle():
    """start_tracking() then stop_tracking() returns an EnergyMeasurement."""
    from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement

    fake = FakeEnergyBackend(total_j=50.0, duration_sec=10.0)
    tracker = fake.start_tracking()
    measurement = fake.stop_tracking(tracker)

    assert isinstance(measurement, EnergyMeasurement)
    assert measurement.total_j == 50.0
    assert measurement.duration_sec == 10.0


def test_fake_energy_backend_is_available():
    """FakeEnergyBackend.is_available() always returns True."""
    fake = FakeEnergyBackend()
    assert fake.is_available() is True


def test_fake_energy_backend_start_tracking_returns_handle():
    """start_tracking() returns a non-None tracker handle."""
    fake = FakeEnergyBackend()
    tracker = fake.start_tracking()
    assert tracker is not None


# ---------------------------------------------------------------------------
# FakeResultsRepository behaviour
# ---------------------------------------------------------------------------


def test_fake_results_repository_save_load_roundtrip(tmp_path):
    """save() then load() returns the same ExperimentResult."""
    fake = FakeResultsRepository()
    result = make_result(experiment_id="repo-test")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    saved_path = fake.save(result, output_dir)
    loaded = fake.load(saved_path)

    assert loaded is result


def test_fake_results_repository_save_returns_path(tmp_path):
    """save() returns a Path object pointing to result.json."""
    fake = FakeResultsRepository()
    result = make_result()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    path = fake.save(result, output_dir)
    assert isinstance(path, Path)
    assert path.name == "result.json"


def test_fake_results_repository_load_raises_without_saves(tmp_path):
    """load() raises FileNotFoundError when no results have been saved."""
    fake = FakeResultsRepository()
    with pytest.raises(FileNotFoundError):
        fake.load(tmp_path / "nonexistent.json")


def test_fake_results_repository_records_saves(tmp_path):
    """save() appends (result, output_dir) to the saved list."""
    fake = FakeResultsRepository()
    result = make_result()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    fake.save(result, output_dir)
    assert len(fake.saved) == 1
    assert fake.saved[0][0] is result
    assert fake.saved[0][1] is output_dir
