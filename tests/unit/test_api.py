"""Unit tests for the llenergymeasure public API surface.

Tests cover the public-API success criteria:
1. Public imports resolve
2. run_experiment returns ExperimentResult (no union, no None)
3. No disk writes when output_dir not set
4. Internal names raise AttributeError
5. __version__ matches pyproject.toml
6. run_study raises NotImplementedError with a clear message
7. _run() calls run_preflight once per experiment config
8. _run() calls get_engine with correct engine name
9. _run() returns StudyResult with experiment results
10. _run() propagates PreFlightError and EngineError unchanged
11. run_experiment end-to-end with mocked engine returns ExperimentResult
12. All test cases pass without GPU hardware (uses monkeypatching)
"""

from __future__ import annotations

import pytest

import llenergymeasure
import llenergymeasure.utils
from llenergymeasure import (
    ExperimentConfig,
    ExperimentResult,
    StudyConfig,
    StudyResult,
    __version__,
    run_experiment,
    run_study,
)
from llenergymeasure.domain.experiment import StudySummary
from llenergymeasure.utils.exceptions import EngineError, PreFlightError
from tests.conftest import (
    make_config,
    make_resolved_study,
    make_result,
    make_study_result,
    make_user_config,
)

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
    assert __version__ == llenergymeasure.__version__


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
    import llenergymeasure.study.orchestration as api_module

    monkeypatch.setattr(api_module, "orchestrate_study", lambda study, **kw: make_study_result())

    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
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
    import llenergymeasure.study.orchestration as api_module

    captured_study = {}

    def mock_run(study, **kw):
        captured_study["value"] = study
        return make_study_result()

    monkeypatch.setattr(api_module, "orchestrate_study", mock_run)

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("serving_mode: offline\ntask:\n  model: gpt2\n")

    result = run_experiment(str(config_path))

    assert isinstance(result, ExperimentResult)
    # Confirm the study was built from the YAML
    assert captured_study["value"].experiments[0].task.model == "gpt2"


# =============================================================================
# Test 5: kwargs form
# =============================================================================


def test_run_experiment_kwargs_form(monkeypatch):
    """run_experiment kwargs form passes model and dataset to ExperimentConfig."""
    import llenergymeasure.study.orchestration as api_module

    captured_study = {}

    def mock_run(study, **kw):
        captured_study["value"] = study
        return make_study_result()

    monkeypatch.setattr(api_module, "orchestrate_study", mock_run)

    result = run_experiment(model="gpt2", n_prompts=50)

    assert isinstance(result, ExperimentResult)
    assert captured_study["value"].experiments[0].task.model == "gpt2"
    assert captured_study["value"].experiments[0].task.dataset.n_prompts == 50


# =============================================================================
# Test 6: No config + no model raises ConfigError
# =============================================================================


def test_run_experiment_no_config_no_model_raises():
    """run_experiment() with no arguments raises ConfigError (not TypeError)."""
    from llenergymeasure.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        run_experiment()  # type: ignore[call-overload]  # asserts no-arg call raises ConfigError


# =============================================================================
# Test 7: No disk writes when output_dir not set
# =============================================================================


def test_run_experiment_no_disk_writes(tmp_path, monkeypatch):
    """run_experiment produces no disk writes when output_dir is not specified."""
    import llenergymeasure.study.orchestration as api_module

    monkeypatch.setattr(api_module, "orchestrate_study", lambda study, **kw: make_study_result())

    # Change working directory to tmp_path to catch any accidental writes
    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    run_experiment(config)

    # tmp_path should be empty - no files written there
    written_files = list(tmp_path.rglob("*"))
    assert written_files == [], f"Unexpected files written: {written_files}"


# =============================================================================
# Test 8: run_study is implemented
# =============================================================================


def test_run_study_invalid_type_raises_config_error():
    """run_study(42) raises ConfigError, not NotImplementedError."""
    from llenergymeasure.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        run_study(42)  # type: ignore[arg-type]


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
    import llenergymeasure.study.orchestration as api_module

    monkeypatch.setattr(api_module, "orchestrate_study", lambda study, **kw: make_study_result())

    config_path = tmp_path / "config.yaml"
    config_path.write_text("serving_mode: offline\ntask:\n  model: gpt2\n")

    result = run_experiment(config_path)  # Path object, not str
    assert isinstance(result, ExperimentResult)


# =============================================================================
# Test 12: kwargs form - engine kwarg passed through
# =============================================================================


def test_run_experiment_kwargs_engine(monkeypatch):
    """run_experiment kwargs form passes engine to ExperimentConfig."""
    import llenergymeasure.study.orchestration as api_module

    captured_study = {}

    def mock_run(study, **kw):
        captured_study["value"] = study
        return make_study_result()

    monkeypatch.setattr(api_module, "orchestrate_study", mock_run)

    run_experiment(model="gpt2", engine="transformers")

    assert captured_study["value"].experiments[0].engine == "transformers"


# =============================================================================
# _run() wiring tests
# =============================================================================


class _MockBackend:
    """Minimal EnginePlugin for _run() tests.

    Implements the 4-method EnginePlugin protocol. Tests that use this mock
    also patch MeasurementHarness.run to return the pre-built result directly,
    so only load_model/warmup/run_inference/cleanup stubs are needed here.
    """

    def __init__(self, result: ExperimentResult) -> None:
        self._result = result
        self.run_inference_calls: list[ExperimentConfig] = []

    @property
    def name(self) -> str:
        return "transformers"

    def load_model(self, config: ExperimentConfig, **kwargs):
        return object()  # Opaque model object

    def warmup(self, config: ExperimentConfig, model, prompts: list[str] | None = None):
        from llenergymeasure.domain.metrics import WarmupResult

        return WarmupResult(
            converged=True, final_cv=0.0, iterations_completed=0, target_cv=0.01, max_prompts=1
        )

    def run_inference(self, config: ExperimentConfig, model, prompts: list[str] | None = None):
        from llenergymeasure.engines.protocol import InferenceOutput

        self.run_inference_calls.append(config)
        return InferenceOutput(
            elapsed_time_sec=1.0,
            input_tokens=10,
            output_tokens=20,
            peak_memory_mb=0.0,
            model_memory_mb=0.0,
        )

    def cleanup(self, model) -> None:
        pass


def _mock_preflight_return(study, **kw):
    """Mock preflight that returns (runner_specs, system_overrides) tuple."""
    from llenergymeasure.config.runner_spec import RunnerSpec

    engines = {exp.engine for exp in study.experiments}
    specs = {b: RunnerSpec(mode="process", image=None, source="test") for b in engines}
    return specs, {}


def _patch_harness(monkeypatch, result: ExperimentResult) -> None:
    """Patch MeasurementHarness.run to return a pre-built result.

    Used by tests that verify _api.py wiring (preflight, get_engine) without
    running the actual measurement lifecycle.
    """
    import llenergymeasure.harness as harness_module

    monkeypatch.setattr(
        harness_module.MeasurementHarness, "run", lambda self, engine, config, **kw: result
    )


def test_run_calls_preflight_once_per_config(monkeypatch, tmp_path):
    """_run() calls run_preflight once for the single in-process experiment."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    preflight_calls: list = []

    def mock_preflight(config):
        preflight_calls.append(config)

    mock_result = make_result()
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", mock_preflight)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config1 = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    study = make_resolved_study([config1])

    api_module.orchestrate_study(study)

    assert len(preflight_calls) == 1, f"Expected 1 preflight call, got {len(preflight_calls)}"
    assert preflight_calls[0].task.model == "gpt2"


@pytest.mark.parametrize("abort_status", ["timed_out", "circuit_breaker"])
def test_run_preserves_runner_abort_status(monkeypatch, tmp_path, abort_status):
    """ST1: _run() must not overwrite a terminal abort status the runner set.

    On wall-clock timeout / circuit-breaker abort the StudyRunner marks the study
    'timed_out'/'circuit_breaker' and returns; _run() previously called
    mark_study_completed() unconditionally, clobbering that status and leaving the
    aborted study looking completed (and therefore non-resumable).
    """
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    captured: dict = {}

    def fake_run_via_runner(study, manifest, study_dir, **kw):
        if abort_status == "timed_out":
            manifest.mark_study_timed_out()
        else:
            manifest.mark_study_circuit_breaker()
        captured["manifest"] = manifest
        return [], [None, None], []

    monkeypatch.setattr(api_module, "_run_via_runner", fake_run_via_runner)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )

    # Two experiments -> multi-experiment path -> dispatches via _run_via_runner.
    study = make_resolved_study(
        [
            ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline"),
            ExperimentConfig(task={"model": "distilgpt2"}, serving_mode="offline"),
        ]
    )
    api_module.orchestrate_study(study)

    assert captured["manifest"].status == abort_status


def test_run_skips_preflight_when_preresolved_supplied(monkeypatch, tmp_path):
    """_run() does NOT re-run study preflight when preresolved is provided."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module
    from llenergymeasure.config.runner_spec import RunnerSpec

    study_preflight_calls: list = []

    def _counting_preflight(study, **kw):
        study_preflight_calls.append(study)
        return _mock_preflight_return(study, **kw)

    mock_result = make_result()
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _counting_preflight)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline")
    study = make_resolved_study([config])

    preresolved: tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]] = (
        {"transformers": RunnerSpec(mode="process", image=None, source="test")},
        {},
    )
    api_module.orchestrate_study(study, skip_preflight=True, preresolved=preresolved)

    assert study_preflight_calls == [], (
        "run_study_preflight must not be called inside _run when preresolved is supplied"
    )


# =============================================================================
# run_study output_dir routing: -o overrides the results dir for fresh runs
# =============================================================================


def _capture_results_base(monkeypatch, tmp_path) -> dict:
    """Patch _run's dependencies so a study 'runs' in-process, capturing the
    results-dir base passed to create_study_dir.

    Returns a dict populated with ``base`` (the Path handed to create_study_dir).
    """
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.preflight as study_pf_module

    captured: dict = {}
    mock_result = make_result()
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)

    def _capture_create(name, output_dir):
        captured["base"] = output_dir
        return tmp_path

    monkeypatch.setattr("llenergymeasure.study.manifest.create_study_dir", _capture_create)
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )
    return captured


def test_run_study_fresh_honors_output_dir(monkeypatch, tmp_path):
    """Fresh run: output_dir (CLI -o) overrides the results-dir base."""
    captured = _capture_results_base(monkeypatch, tmp_path)

    study = StudyConfig(experiments=[make_config()])
    custom = tmp_path / "custom-out"
    run_study(study, skip_preflight=True, output_dir=custom)

    assert captured["base"] == custom


def test_run_study_fresh_without_output_dir_uses_yaml_results_dir(monkeypatch, tmp_path):
    """Fresh run without output_dir falls back to YAML output.results_dir."""
    from llenergymeasure.config.models import OutputConfig

    captured = _capture_results_base(monkeypatch, tmp_path)

    yaml_dir = tmp_path / "yaml-results"
    study = StudyConfig(
        experiments=[make_config()],
        output=OutputConfig(results_dir=str(yaml_dir)),
    )
    run_study(study, skip_preflight=True)

    assert captured["base"] == yaml_dir


def test_run_study_resume_uses_output_dir_as_search_base(monkeypatch, tmp_path):
    """Resume: output_dir stays the resumable-study search base, not a results override."""
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.resume as resume_module

    search_bases: list = []
    fake_resume_dir = tmp_path / "study_2026"

    def _fake_find(base):
        search_bases.append(base)
        return fake_resume_dir

    monkeypatch.setattr(resume_module, "find_resumable_study", _fake_find)
    monkeypatch.setattr(resume_module, "load_resume_state", lambda d: ({}, set()))
    monkeypatch.setattr(resume_module, "validate_config_drift", lambda old, study: None)
    monkeypatch.setattr(resume_module, "validate_resolved_config_drift", lambda old, study: None)
    monkeypatch.setattr(resume_module, "prepare_resume_manifest", lambda d, old: None)

    captured: dict = {}

    def _capture_run(study, **kw):
        captured["study"] = study
        captured.update(kw)
        return make_study_result()

    monkeypatch.setattr(api_module, "orchestrate_study", _capture_run)

    study = StudyConfig(experiments=[make_config()])
    search_base = tmp_path / "search-here"
    run_study(study, resume=True, output_dir=search_base)

    assert search_bases == [search_base]
    assert captured["resume_dir"] == fake_resume_dir
    # output_dir was consumed as the search base, not as a results-dir override:
    # the resolved study keeps its own results_dir.
    assert captured["study"].output.results_dir != str(search_base)


def test_run_preresolved_without_skip_preflight_raises():
    """Passing preresolved with skip_preflight=False is rejected, not silently honoured."""
    import llenergymeasure.study.orchestration as api_module
    from llenergymeasure.config.runner_spec import RunnerSpec

    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline")
    study = make_resolved_study([config])

    preresolved: tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]] = (
        {"transformers": RunnerSpec(mode="process", image=None, source="test")},
        {},
    )

    with pytest.raises(ValueError, match="preresolved requires skip_preflight=True"):
        api_module.orchestrate_study(study, skip_preflight=False, preresolved=preresolved)


def test_run_calls_get_engine_with_correct_name(monkeypatch, tmp_path):
    """_run() calls get_engine with the experiment's engine name."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    mock_result = make_result()
    mock_engine = _MockBackend(mock_result)

    engine_calls: list[str] = []

    def mock_get_engine(name: str):
        engine_calls.append(name)
        return mock_engine

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", mock_get_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline")
    study = make_resolved_study([config])

    api_module.orchestrate_study(study)

    assert len(engine_calls) == 1
    assert engine_calls[0] == "transformers"


def test_run_returns_study_result(monkeypatch, tmp_path):
    """_run() returns a StudyResult containing the experiment results."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    mock_result = make_result(experiment_id="wired-001")
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    study = make_resolved_study([config], study_name="my-study")

    study_result = api_module.orchestrate_study(study)

    assert isinstance(study_result, StudyResult)
    assert study_result.study_name == "my-study"
    assert len(study_result.experiments) == 1
    assert study_result.experiments[0].experiment_id == "wired-001"


def test_run_propagates_preflight_error(monkeypatch, tmp_path):
    """_run() propagates PreFlightError without catching it."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    def failing_preflight(config):
        raise PreFlightError(["CUDA not available"])

    monkeypatch.setattr(pf_module, "run_preflight", failing_preflight)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    mock_result = make_result()
    monkeypatch.setattr(engines_module, "get_engine", lambda name: _MockBackend(mock_result))
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )

    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    study = make_resolved_study([config])

    with pytest.raises(PreFlightError):
        api_module.orchestrate_study(study)


def test_run_propagates_engine_error(monkeypatch, tmp_path):
    """_run() propagates EngineError without catching it."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness as harness_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    def _failing_harness_run(self, engine, config, **kw):
        raise EngineError("GPU out of memory")

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(engines_module, "get_engine", lambda name: _MockBackend(make_result()))
    monkeypatch.setattr(harness_module.MeasurementHarness, "run", _failing_harness_run)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )

    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    study = make_resolved_study([config])

    with pytest.raises(EngineError, match="GPU out of memory"):
        api_module.orchestrate_study(study)


def test_run_experiment_end_to_end_mocked(monkeypatch, tmp_path):
    """run_experiment() flows through the real _run() pipeline (mocked engine) and returns ExperimentResult."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.preflight as study_pf_module

    expected_result = make_result(experiment_id="e2e-test")
    mock_engine = _MockBackend(expected_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, expected_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    result = run_experiment(model="gpt2")

    assert isinstance(result, ExperimentResult)
    assert not isinstance(result, StudyResult)
    assert result.experiment_id == "e2e-test"


# =============================================================================
# Plan 02: run_study() and _run() dispatcher tests
# =============================================================================


def test_run_study_accepts_study_config(monkeypatch, tmp_path):
    """run_study(StudyConfig) returns StudyResult with populated summary."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.preflight as study_pf_module

    mock_result = make_result(experiment_id="study-test")
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    # Avoid real disk writes by patching create_study_dir and save_result
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    study = StudyConfig(
        experiments=[ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")]
    )
    result = run_study(study)

    assert isinstance(result, StudyResult)
    assert result.summary.completed == 1
    assert result.summary.failed == 0


def test_run_study_accepts_path(tmp_path, monkeypatch):
    """run_study(str path) loads YAML and returns StudyResult."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.preflight as study_pf_module

    yaml_content = "experiments:\n  - task:\n      model: gpt2\n    serving_mode: offline\n"
    yaml_path = tmp_path / "study.yaml"
    yaml_path.write_text(yaml_content)

    mock_result = make_result(experiment_id="path-test")
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    result = run_study(str(yaml_path))

    assert isinstance(result, StudyResult)


def test_run_dispatches_single_in_process(monkeypatch, tmp_path):
    """Single experiment + n_cycles=1 bypasses StudyRunner (in-process path)."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module
    from llenergymeasure.study.runner import StudyRunner

    mock_result = make_result(experiment_id="inproc-test")
    mock_engine = _MockBackend(mock_result)

    runner_created = []
    original_runner_init = StudyRunner.__init__

    def mock_runner_init(self, *args, **kwargs):
        runner_created.append(True)
        original_runner_init(self, *args, **kwargs)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )
    monkeypatch.setattr(StudyRunner, "__init__", mock_runner_init)

    study = make_resolved_study(
        [ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")],
        study_execution={"n_cycles": 1, "experiment_order": "sequential"},
    )
    api_module.orchestrate_study(study)

    # Single experiment + n_cycles=1 should NOT create StudyRunner
    assert runner_created == [], "StudyRunner was created for single in-process path"


def test_run_study_returns_study_result_type():
    """run_study return annotation is StudyResult (not a union)."""
    import typing

    import llenergymeasure.api._impl as api_module

    hints = typing.get_type_hints(api_module.run_study)
    assert hints.get("return") is StudyResult, (
        "run_study must have -> StudyResult return annotation"
    )


# =============================================================================
# Runner resolution wiring in _run()
# =============================================================================


def test_run_resolves_runners_and_passes_to_study_runner(monkeypatch, tmp_path):
    """_run() resolves runners via resolve_study_runners and passes runner_specs to StudyRunner."""
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module
    from llenergymeasure.config.runner_spec import RunnerSpec

    mock_result = make_result(experiment_id="runner-wired")

    resolved_specs = {
        "transformers": RunnerSpec(mode="process", image=None, source="default"),
    }

    # Capture what runner_specs was passed to _run_via_runner
    captured_runner_specs: list = []
    original_run_via_runner = api_module._run_via_runner

    def mock_run_via_runner(
        study,
        manifest,
        study_dir,
        runner_specs=None,
        progress=None,
        skip_set=None,
        no_lock=False,
        resolution_logs=None,
    ):
        captured_runner_specs.append(runner_specs)
        return original_run_via_runner(study, manifest, study_dir, runner_specs=runner_specs)

    monkeypatch.setattr(
        study_pf_module, "run_study_preflight", lambda study, **kw: (resolved_specs, {})
    )
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: make_user_config(),
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )
    monkeypatch.setattr(api_module, "_run_via_runner", mock_run_via_runner)

    # Use a 2-experiment study to force _run_via_runner path (not run_single_experiment)
    import llenergymeasure.engines as engines_module

    mock_engine = _MockBackend(mock_result)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)

    # Mock StudyRunner.run() to avoid real subprocess spawning
    from llenergymeasure.study.runner import StudyRunner

    monkeypatch.setattr(StudyRunner, "run", lambda self: [mock_result])

    study = make_resolved_study(
        [
            ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline"),
            ExperimentConfig(
                task={"model": "gpt2-medium"}, engine="transformers", serving_mode="offline"
            ),
        ]
    )

    api_module.orchestrate_study(study)

    assert len(captured_runner_specs) == 1, "_run_via_runner not called or called multiple times"
    assert captured_runner_specs[0] == resolved_specs


def test_run_mixed_runner_warning_logged(monkeypatch, tmp_path, caplog):
    """_run() logs a warning when runner_specs has mixed local/docker modes."""
    import logging

    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module
    from llenergymeasure.config.runner_spec import RunnerSpec

    mixed_specs = {
        "transformers": RunnerSpec(mode="process", image=None, source="default"),
        "vllm": RunnerSpec(mode="container", image=None, source="yaml"),
    }

    mock_result = make_result()
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(
        study_pf_module, "run_study_preflight", lambda study, **kw: (mixed_specs, {})
    )
    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: make_user_config(),
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    study = make_resolved_study([ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")])

    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.orchestration"):
        api_module.orchestrate_study(study)

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mixed" in m.lower() for m in warning_messages), (
        f"Expected mixed runner warning, got: {warning_messages}"
    )


# =============================================================================
# B3 fix: study experiment count no double-multiply
# =============================================================================


def test_study_summary_total_experiments_no_double_multiply(monkeypatch, tmp_path):
    """total_experiments == len(study.experiments) - no double-multiply by n_cycles.

    The study passed to _run() has experiments already cycle-expanded (as resolve_study
    does). With 2 unique configs and n_cycles=3, study.experiments has 6 entries.
    total_experiments must be 6, not 18 (the pre-fix bug: 6 * 3).
    unique_configurations must be 2 (6 / 3).
    """
    import llenergymeasure.study.orchestration as api_module
    import llenergymeasure.study.preflight as study_pf_module

    # Build 6 mock results (2 configs x 3 cycles, already cycle-expanded)
    mock_results = [make_result(experiment_id=f"b3-{i}") for i in range(6)]

    # Mock _run_via_runner to return pre-built results (bypasses real subprocess)
    def mock_run_via_runner(
        study,
        manifest,
        study_dir,
        runner_specs=None,
        progress=None,
        skip_set=None,
        no_lock=False,
        resolution_logs=None,
    ):
        result_files = [str(tmp_path / f"result-{i}.json") for i in range(6)]
        return result_files, mock_results, []

    monkeypatch.setattr(study_pf_module, "run_study_preflight", _mock_preflight_return)
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: make_user_config(),
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(api_module, "_run_via_runner", mock_run_via_runner)

    # resolve_study cycle-expands the declared configs, so hand it the 2 unique
    # ones and let it produce the 6-entry execution sequence.
    unique_experiments = [
        ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline"),
        ExperimentConfig(task={"model": "gpt2-medium"}, serving_mode="offline"),
    ]
    study = make_resolved_study(
        unique_experiments,
        study_execution={"n_cycles": 3, "experiment_order": "sequential"},
    )
    # Resolution expanded the 2 unique configs into 6 cycle entries
    assert len(study.experiments) == 6

    study_result = api_module.orchestrate_study(study)

    assert study_result.summary.total_experiments == 6, (
        f"Expected 6 (cycle-expanded count), got {study_result.summary.total_experiments} "
        f"(pre-fix bug would give 18 = 6 x 3)"
    )
    assert study_result.summary.unique_configurations == 2, (
        f"Expected 2 unique configurations (6 / 3), got {study_result.summary.unique_configurations}"
    )


# =============================================================================
# Quick Task 2: _resolve_gpu_indices unit tests
# =============================================================================


class TestResolveGpuIndices:
    """Unit tests for _resolve_gpu_indices(). No real GPU required - NVML is monkeypatched."""

    def _make_pytorch_config(self, device_map: str | None = None) -> ExperimentConfig:
        """Build a minimal PyTorch ExperimentConfig."""
        if device_map is not None:
            tfm_cfg: dict | None = {"engine_params": {"device_map": device_map}}
        else:
            tfm_cfg = None
        return ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=tfm_cfg,
            serving_mode="offline",
        )

    def _make_mock_pynvml(self, device_count: int):
        """Build a minimal pynvml mock with nvmlInit, nvmlDeviceGetCount, nvmlShutdown."""
        import types

        mod = types.ModuleType("pynvml")
        mod.nvmlInit = lambda: None
        mod.nvmlDeviceGetCount = lambda: device_count
        mod.nvmlShutdown = lambda: None
        return mod

    def test_pytorch_no_device_map_returns_zero(self):
        """PyTorch engine with device_map=None always returns [0]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = self._make_pytorch_config(device_map=None)
        assert _resolve_gpu_indices(config) == [0]

    def test_pytorch_device_map_auto_four_gpus(self, monkeypatch):
        """PyTorch with device_map='auto' and 4 visible GPUs returns [0, 1, 2, 3]."""
        import sys

        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        mock_pynvml = self._make_mock_pynvml(device_count=4)
        monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)

        config = self._make_pytorch_config(device_map="auto")
        assert _resolve_gpu_indices(config) == [0, 1, 2, 3]

    def test_pytorch_device_map_auto_one_gpu_returns_zero(self, monkeypatch):
        """PyTorch with device_map='auto' and only 1 GPU returns [0] (no-op multi-GPU)."""
        import sys

        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        mock_pynvml = self._make_mock_pynvml(device_count=1)
        monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)

        config = self._make_pytorch_config(device_map="auto")
        assert _resolve_gpu_indices(config) == [0]

    def test_pytorch_device_map_auto_pynvml_absent_returns_zero(self, monkeypatch):
        """PyTorch with device_map='auto' but pynvml absent falls through to [0]."""
        import sys

        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        # Remove pynvml from sys.modules so the local import raises ImportError
        monkeypatch.setitem(sys.modules, "pynvml", None)  # type: ignore[arg-type]

        config = self._make_pytorch_config(device_map="auto")
        assert _resolve_gpu_indices(config) == [0]

    def test_non_pytorch_non_vllm_engine_returns_zero(self):
        """Unknown engines return [0]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = ExperimentConfig.model_construct(task={"model": "gpt2"}, engine="tensorrt")
        assert _resolve_gpu_indices(config) == [0]

    def test_pytorch_engine_no_pytorch_block_returns_zero(self):
        """PyTorch engine with transformers=None (no pytorch block) returns [0]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = ExperimentConfig.model_construct(
            task={"model": "gpt2"}, engine="transformers", transformers=None
        )
        assert _resolve_gpu_indices(config) == [0]

    # ── vLLM engine tests ──

    def _make_vllm_config(self, tp: int | None = None, pp: int | None = None) -> ExperimentConfig:
        """Build a minimal vLLM ExperimentConfig with TP/PP settings."""
        engine_params: dict = {}
        if tp is not None:
            engine_params["tensor_parallel_size"] = tp
        if pp is not None:
            engine_params["pipeline_parallel_size"] = pp
        return ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            vllm={"engine_params": engine_params},
            serving_mode="offline",
        )

    def test_vllm_tp2_returns_two_gpus(self):
        """vLLM with tensor_parallel_size=2 returns [0, 1]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = self._make_vllm_config(tp=2)
        assert _resolve_gpu_indices(config) == [0, 1]

    def test_vllm_tp4_returns_four_gpus(self):
        """vLLM with tensor_parallel_size=4 returns [0, 1, 2, 3]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = self._make_vllm_config(tp=4)
        assert _resolve_gpu_indices(config) == [0, 1, 2, 3]

    def test_vllm_tp2_pp2_returns_four_gpus(self):
        """vLLM with tp=2, pp=2 returns [0, 1, 2, 3]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = self._make_vllm_config(tp=2, pp=2)
        assert _resolve_gpu_indices(config) == [0, 1, 2, 3]

    def test_vllm_tp1_returns_single_gpu(self):
        """vLLM with tensor_parallel_size=1 (default) returns [0]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = self._make_vllm_config(tp=1)
        assert _resolve_gpu_indices(config) == [0]

    def test_vllm_no_engine_block_returns_single_gpu(self):
        """vLLM with no engine config returns [0]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = ExperimentConfig(
            task={"model": "gpt2"}, engine="vllm", vllm={}, serving_mode="offline"
        )
        assert _resolve_gpu_indices(config) == [0]

    def test_vllm_no_vllm_block_returns_single_gpu(self):
        """vLLM engine with vllm=None returns [0]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="offline")
        assert _resolve_gpu_indices(config) == [0]


# ---------------------------------------------------------------------------
# Energy scope is self-documenting through data
# ---------------------------------------------------------------------------
# The per_gpu_j data flow is already wired:
#   NVMLSampler.stop_tracking() -> EnergyMeasurement.per_gpu_j
#   build_result() (harness.result_assembly) -> ExperimentResult.energy_per_device_j + multi_gpu
# With _resolve_gpu_indices returning correct indices for TRT-LLM,
# multi-GPU energy is automatically summed across all TP ranks.
# No methodology_notes string needed - data is self-documenting:
#   effective_config.tensorrt.tensor_parallel_size + multi_gpu.num_gpus + multi_gpu.energy_per_gpu_j


class TestResolveGpuIndicesTensorrt:
    """Unit tests for _resolve_gpu_indices() tensorrt branch."""

    def test_tensorrt_tp1_returns_single_index(self):
        """tensor_parallel_size=1 -> [0] (single GPU)."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
        )
        assert _resolve_gpu_indices(config) == [0]

    def test_tensorrt_tp2_returns_two_indices(self):
        """tensor_parallel_size=2 -> [0, 1] (two GPUs for energy monitoring)."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 2}}
        )
        assert _resolve_gpu_indices(config) == [0, 1]

    def test_tensorrt_tp4_returns_four_indices(self):
        """tensor_parallel_size=4 -> [0, 1, 2, 3]."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 4}}
        )
        assert _resolve_gpu_indices(config) == [0, 1, 2, 3]

    def test_tensorrt_tp_none_returns_single_index(self):
        """tensor_parallel_size=None (default) -> [0] (single GPU)."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = make_config(engine="tensorrt", tensorrt={})
        assert _resolve_gpu_indices(config) == [0]

    def test_tensorrt_no_config_returns_single_index(self):
        """engine=tensorrt but tensorrt=None -> [0] (fallback)."""
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        config = make_config(engine="tensorrt")
        assert _resolve_gpu_indices(config) == [0]


# =============================================================================
# Quick Task 9: run_experiment raises ExperimentError when experiments list is empty
# =============================================================================


def test_run_experiment_raises_experiment_error_on_empty_results(monkeypatch):
    """run_experiment raises ExperimentError (not IndexError) when _run returns empty experiments."""
    import llenergymeasure.study.orchestration as api_module
    from llenergymeasure.utils.exceptions import ExperimentError

    empty_study_result = StudyResult(
        experiments=[],
        summary=StudySummary(
            total_experiments=1,
            completed=0,
            failed=1,
            total_wall_time_s=0.1,
            total_energy_j=0.0,
            unique_configurations=1,
            warnings=["Docker container failed: image not found"],
        ),
    )

    monkeypatch.setattr(api_module, "orchestrate_study", lambda study, **kw: empty_study_result)

    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    with pytest.raises(ExperimentError) as exc_info:
        run_experiment(config)

    assert "Docker container failed: image not found" in str(exc_info.value)


def test_run_experiment_raises_experiment_error_no_warnings(monkeypatch):
    """run_experiment raises ExperimentError with fallback message when warnings list is empty."""
    import llenergymeasure.study.orchestration as api_module
    from llenergymeasure.utils.exceptions import ExperimentError

    empty_study_result = StudyResult(
        experiments=[],
        summary=StudySummary(
            total_experiments=1,
            completed=0,
            failed=1,
            total_wall_time_s=0.1,
            total_energy_j=0.0,
            unique_configurations=1,
        ),
    )

    monkeypatch.setattr(api_module, "orchestrate_study", lambda study, **kw: empty_study_result)

    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    with pytest.raises(ExperimentError, match="Experiment produced no results"):
        run_experiment(config)


def test_run_study_partial_failure_returns_partial_results(monkeypatch):
    """run_study returns StudyResult with partial results when some experiments fail.

    Simulates a study where one Docker experiment succeeds and another fails.
    The study should NOT raise - it returns a StudyResult with the successful
    experiments and a summary showing the failure count.
    """
    import llenergymeasure.study.orchestration as api_module

    successful_result = make_result(experiment_id="partial-ok")

    partial_study_result = StudyResult(
        experiments=[successful_result],  # 1 succeeded, 1 was filtered (None)
        summary=StudySummary(
            total_experiments=2,
            completed=1,
            failed=1,
            total_wall_time_s=5.0,
            total_energy_j=100.0,
            unique_configurations=2,
            warnings=["Docker container failed for experiment 2"],
        ),
    )

    monkeypatch.setattr(api_module, "orchestrate_study", lambda study, **kw: partial_study_result)

    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline"),
            ExperimentConfig(task={"model": "gpt2-medium"}, serving_mode="offline"),
        ]
    )
    result = run_study(study)

    # Study should return partial results, NOT raise
    assert isinstance(result, StudyResult)
    assert len(result.experiments) == 1
    assert result.experiments[0].experiment_id == "partial-ok"
    assert result.summary.completed == 1
    assert result.summary.failed == 1


def test_run_study_resolved_study_honors_output_dir(monkeypatch, tmp_path):
    """output_dir applies to an ALREADY-RESOLVED study (load_study output) too.

    The resolved branch of _resolve_objects must not silently drop the override
    (docs/reference/library/run_study.md documents output_dir as winning for any
    fresh run). It is applied directly and recorded as call_site provenance.
    """
    from llenergymeasure.api import load_study

    captured = _capture_results_base(monkeypatch, tmp_path)

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "study_name: resolved-override\n"
        "experiments:\n"
        "  - task:\n"
        "      model: gpt2\n"
        "    engine: transformers\n"
        "    serving_mode: offline\n"
    )
    resolved = load_study(study_yaml)
    assert resolved.study_design_hash is not None

    custom = tmp_path / "resolved-out"
    run_study(resolved, skip_preflight=True, output_dir=custom)

    assert captured["base"] == custom


def test_run_study_resolved_study_rejects_other_overrides(monkeypatch, tmp_path):
    """Non-results-dir overrides on an already-resolved study raise, never drop."""
    from llenergymeasure.api import load_study
    from llenergymeasure.api._impl import _resolve_objects
    from llenergymeasure.utils.exceptions import ConfigError

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "study_name: resolved-reject\n"
        "experiments:\n"
        "  - task:\n"
        "      model: gpt2\n"
        "    engine: transformers\n"
        "    serving_mode: offline\n"
    )
    resolved = load_study(study_yaml)

    with pytest.raises(ConfigError, match=r"already resolved.*study_execution\.n_cycles"):
        _resolve_objects(resolved, overrides={"study_execution": {"n_cycles": 5}})
