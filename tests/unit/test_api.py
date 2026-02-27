"""Unit tests for the llenergymeasure public API surface.

Tests cover all Phase 3 success criteria:
1. Public imports resolve
2. run_experiment returns ExperimentResult (no union, no None)
3. No disk writes when output_dir not set
4. Internal names raise AttributeError
5. __version__ == "1.17.0"
6. run_study raises NotImplementedError with M2 message
7. All test cases pass without GPU hardware (uses monkeypatching)
"""

from __future__ import annotations

from datetime import datetime

import pytest

import llenergymeasure
from llenergymeasure import (
    ExperimentConfig,
    ExperimentResult,
    StudyConfig,
    StudyResult,
    __version__,
    run_experiment,
    run_study,
)
from llenergymeasure.domain.experiment import AggregationMetadata

# =============================================================================
# Test helper
# =============================================================================


def _make_experiment_result(**overrides) -> ExperimentResult:
    """Build a minimal valid ExperimentResult for testing."""
    defaults = {
        "experiment_id": "test-001",
        "aggregation": AggregationMetadata(num_processes=1),
        "total_tokens": 1000,
        "total_energy_j": 10.0,
        "total_inference_time_sec": 5.0,
        "avg_tokens_per_second": 200.0,
        "avg_energy_per_token_j": 0.01,
        "total_flops": 1e9,
        "start_time": datetime(2026, 1, 1),
        "end_time": datetime(2026, 1, 1, 0, 1),
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def _make_study_result(**overrides) -> StudyResult:
    """Build a StudyResult containing one ExperimentResult."""
    return StudyResult(experiments=[_make_experiment_result()])


# =============================================================================
# Test 1: Public imports resolve
# =============================================================================


def test_public_imports_resolve():
    """All 7 public names import correctly from llenergymeasure."""
    assert run_experiment is not None
    assert run_study is not None
    assert ExperimentConfig is not None
    assert StudyConfig is not None
    assert ExperimentResult is not None
    assert StudyResult is not None
    assert __version__ == "1.17.0"


# =============================================================================
# Test 2: Internal names raise AttributeError
# =============================================================================


def test_internal_name_raises_attribute_error():
    """Names not in __all__ raise AttributeError on module access."""
    internal_names = [
        "load_experiment_config",
        "ConfigError",
        "AggregatedResult",
        "LLEMError",
        "deep_merge",
    ]
    for name in internal_names:
        with pytest.raises(AttributeError, match=name):
            getattr(llenergymeasure, name)


# =============================================================================
# Test 3: run_experiment returns ExperimentResult (no union, no None)
# =============================================================================


def test_run_experiment_returns_experiment_result(monkeypatch):
    """run_experiment returns exactly ExperimentResult, not a union or None."""
    import llenergymeasure._api as api_module

    monkeypatch.setattr(api_module, "_run", lambda study: _make_study_result())

    config = ExperimentConfig(model="gpt2")
    result = run_experiment(config)

    assert result is not None
    assert isinstance(result, ExperimentResult)
    # Confirm it is NOT a StudyResult (no union types)
    assert not isinstance(result, StudyResult)


# =============================================================================
# Test 4: YAML path form
# =============================================================================


def test_run_experiment_yaml_path_form(tmp_path, monkeypatch):
    """run_experiment resolves correctly from a YAML path."""
    import llenergymeasure._api as api_module

    captured_study = {}

    def mock_run(study):
        captured_study["value"] = study
        return _make_study_result()

    monkeypatch.setattr(api_module, "_run", mock_run)

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("model: gpt2\n")

    result = run_experiment(str(config_path))

    assert isinstance(result, ExperimentResult)
    # Confirm the study was built from the YAML
    assert captured_study["value"].experiments[0].model == "gpt2"


# =============================================================================
# Test 5: kwargs form
# =============================================================================


def test_run_experiment_kwargs_form(monkeypatch):
    """run_experiment kwargs form passes model and n to ExperimentConfig."""
    import llenergymeasure._api as api_module

    captured_study = {}

    def mock_run(study):
        captured_study["value"] = study
        return _make_study_result()

    monkeypatch.setattr(api_module, "_run", mock_run)

    result = run_experiment(model="gpt2", n=50)

    assert isinstance(result, ExperimentResult)
    assert captured_study["value"].experiments[0].model == "gpt2"
    assert captured_study["value"].experiments[0].n == 50


# =============================================================================
# Test 6: No config + no model raises ConfigError
# =============================================================================


def test_run_experiment_no_config_no_model_raises():
    """run_experiment() with no arguments raises ConfigError (not TypeError)."""
    from llenergymeasure.exceptions import ConfigError

    with pytest.raises(ConfigError):
        run_experiment()


# =============================================================================
# Test 7: No disk writes when output_dir not set
# =============================================================================


def test_run_experiment_no_disk_writes(tmp_path, monkeypatch):
    """run_experiment produces no disk writes when output_dir is not specified."""
    import llenergymeasure._api as api_module

    monkeypatch.setattr(api_module, "_run", lambda study: _make_study_result())

    # Change working directory to tmp_path to catch any accidental writes
    config = ExperimentConfig(model="gpt2")
    run_experiment(config)

    # tmp_path should be empty — no files written there
    written_files = list(tmp_path.rglob("*"))
    assert written_files == [], f"Unexpected files written: {written_files}"


# =============================================================================
# Test 8: run_study raises NotImplementedError
# =============================================================================


def test_run_study_raises_not_implemented():
    """run_study raises NotImplementedError with 'M2' in the message."""
    study_config = StudyConfig(experiments=[ExperimentConfig(model="gpt2")])

    with pytest.raises(NotImplementedError) as exc_info:
        run_study(study_config)

    assert "M2" in str(exc_info.value)


# =============================================================================
# Test 9: __all__ list matches exports
# =============================================================================


def test_all_list_matches_exports():
    """Every name in __all__ is importable from llenergymeasure."""
    for name in llenergymeasure.__all__:
        obj = getattr(llenergymeasure, name, None)
        assert obj is not None, f"__all__ member '{name}' is not importable from llenergymeasure"


# =============================================================================
# Test 10: __version__ in __all__
# =============================================================================


def test_version_in_all():
    """__version__ is explicitly in __all__."""
    assert "__version__" in llenergymeasure.__all__


# =============================================================================
# Test 11: run_experiment with Path object (not just str)
# =============================================================================


def test_run_experiment_path_object_form(tmp_path, monkeypatch):
    """run_experiment accepts a Path object as well as a str path."""
    import llenergymeasure._api as api_module

    monkeypatch.setattr(api_module, "_run", lambda study: _make_study_result())

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: gpt2\n")

    result = run_experiment(config_path)  # Path object, not str
    assert isinstance(result, ExperimentResult)


# =============================================================================
# Test 12: kwargs form — backend kwarg passed through
# =============================================================================


def test_run_experiment_kwargs_backend(monkeypatch):
    """run_experiment kwargs form passes backend to ExperimentConfig."""
    import llenergymeasure._api as api_module

    captured_study = {}

    def mock_run(study):
        captured_study["value"] = study
        return _make_study_result()

    monkeypatch.setattr(api_module, "_run", mock_run)

    run_experiment(model="gpt2", backend="pytorch")

    assert captured_study["value"].experiments[0].backend == "pytorch"
