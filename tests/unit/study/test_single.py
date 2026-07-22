"""Unit tests for the single-experiment in-process execution seam.

run_single_experiment() is the shared body for single-experiment / n_cycles=1
studies. These tests verify it without GPU hardware (uses monkeypatching).
"""

from __future__ import annotations

import pytest

from llenergymeasure import ExperimentConfig, ExperimentResult, StudyConfig
from llenergymeasure.study.single import run_single_experiment
from tests.conftest import make_result


class _MockBackend:
    """Minimal EnginePlugin for run_single_experiment() tests.

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


def _patch_harness(monkeypatch, result: ExperimentResult) -> None:
    """Patch MeasurementHarness.run to return a pre-built result.

    Used by tests that verify execution wiring (preflight, get_engine) without
    running the actual measurement lifecycle.
    """
    import llenergymeasure.harness as harness_module

    monkeypatch.setattr(
        harness_module.MeasurementHarness, "run", lambda self, engine, config, **kw: result
    )


def test_run_single_experiment_calls_gpu_memory_check(monkeypatch, tmp_path):
    """run_single_experiment() calls check_gpu_memory_residual before running the experiment."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module

    gpu_check_calls: list[int] = []

    def mock_gpu_check(device_index=0, threshold_mb=1024.0):
        gpu_check_calls.append(device_index)

    mock_result = make_result(experiment_id="gpu-check-test")
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual",
        mock_gpu_check,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    from unittest.mock import MagicMock

    from llenergymeasure.study.manifest import ManifestWriter

    mock_manifest = MagicMock(spec=ManifestWriter)

    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config])

    run_single_experiment(study, mock_manifest, tmp_path, runner_specs=None)

    assert len(gpu_check_calls) == 1, (
        f"Expected check_gpu_memory_residual to be called once, got {len(gpu_check_calls)}"
    )


def test_single_experiment_passes_resolved_config_hash(monkeypatch, tmp_path):
    """ST3: the single-experiment path passes resolved_config_hash to _save_and_record.

    The multi-experiment runner path includes it; the single path previously omitted it,
    so single-experiment config.json sidecars lacked resolved_config_hash.
    """
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module
    import llenergymeasure.study.runner as runner_module
    from llenergymeasure.study.hashing import build_resolved_view, hash_config

    mock_result = make_result(experiment_id="resolved-hash-test")
    mock_engine = _MockBackend(mock_result)

    captured: dict = {}

    def fake_save_and_record(result, study_dir, manifest, config_hash, cycle, result_files, **kw):
        captured.update(kw)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_module, "_save_and_record", fake_save_and_record)

    from unittest.mock import MagicMock

    from llenergymeasure.study.manifest import ManifestWriter

    mock_manifest = MagicMock(spec=ManifestWriter)
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config])

    run_single_experiment(study, mock_manifest, tmp_path, runner_specs=None)

    assert captured.get("resolved_config_hash") == hash_config(build_resolved_view(config))
    assert captured["resolved_config_hash"]


def test_single_experiment_writes_runtime_observations(monkeypatch, tmp_path):
    """ST4: the single-experiment local path emits runtime_observations.jsonl.

    The multi-experiment worker wraps execution in capture_runtime_observations; the
    single path previously did not, so report-gaps found nothing for single-exp studies.
    """
    import json

    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module

    mock_result = make_result(experiment_id="runtime-obs-test")
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    from unittest.mock import MagicMock

    from llenergymeasure.study.manifest import ManifestWriter

    mock_manifest = MagicMock(spec=ManifestWriter)
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config])

    run_single_experiment(study, mock_manifest, tmp_path, runner_specs=None)

    obs_file = tmp_path / "runtime_observations.jsonl"
    assert obs_file.exists(), "runtime_observations.jsonl was not written"
    lines = [ln for ln in obs_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["config_hash"]
    assert record["outcome"] == "success"


def test_config_sidecar_materialises_without_timeseries(monkeypatch, tmp_path):
    """Regression: config.json + provenance still materialise when save_timeseries is off.

    config.json is the sole home of provenance and engine/model/methodology
    identity. Previously the runner passed output_dir=None to the harness when
    save_timeseries was False, so the harness (which writes config.json only
    when it has an output dir) wrote nothing and no provenance landed. The
    staging dir is now created regardless of save_timeseries.
    """
    import json as _json
    from pathlib import Path
    from unittest.mock import MagicMock

    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness as harness_module
    import llenergymeasure.harness.preflight as pf_module
    from llenergymeasure.domain.experiment import compute_declared_config_hash
    from llenergymeasure.study.manifest import ManifestWriter

    mock_result = make_result(experiment_id="no-ts-config-test")
    mock_engine = _MockBackend(mock_result)

    captured_output_dirs: list[str | None] = []

    def fake_run(self, engine, config, **kw):
        # Mirror the real harness: write config.json whenever it has an output
        # dir, independent of save_timeseries.
        output_dir = kw.get("output_dir")
        captured_output_dirs.append(output_dir)
        if output_dir is not None:
            (Path(output_dir) / "config.json").write_text(
                _json.dumps({"bundle_version": "2.0", "engine": "transformers"}),
                encoding="utf-8",
            )
        return mock_result

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(harness_module.MeasurementHarness, "run", fake_run)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual", lambda *a, **k: None
    )

    mock_manifest = MagicMock(spec=ManifestWriter)
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config], output={"save_timeseries": False})
    resolution_log = {"task.model": {"effective": "gpt2", "source": "yaml"}}
    resolution_logs = {compute_declared_config_hash(config): resolution_log}

    result_files, _results, _warnings = run_single_experiment(
        study, mock_manifest, tmp_path, runner_specs=None, resolution_logs=resolution_logs
    )

    # save_timeseries is off, yet the harness still received a staging dir.
    assert captured_output_dirs and captured_output_dirs[0] is not None, (
        "harness must receive an output dir for the config.json sidecar even when "
        "save_timeseries is off"
    )
    # config.json materialised in the experiment dir with provenance folded in.
    assert len(result_files) == 1
    dest_config = Path(result_files[0]).parent / "config.json"
    assert dest_config.exists(), "config.json must materialise even without timeseries"
    payload = _json.loads(dest_config.read_text())
    assert payload["provenance"] == resolution_log


def test_local_single_failure_persists_traceback(monkeypatch, tmp_path):
    """A local single-experiment failure persists its traceback into failed-runs/.

    The local branch previously just rmtree'd the staging dir and re-raised, so
    the real failure was never marked in the manifest nor kept on disk. It now
    mirrors the Docker single-experiment branch (persist traceback + mark_failed)
    while still re-raising the original exception so the real cause reaches the CLI.
    """
    from unittest.mock import MagicMock

    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness as harness_module
    import llenergymeasure.harness.preflight as pf_module
    from llenergymeasure.domain.experiment import compute_declared_config_hash
    from llenergymeasure.study.manifest import ManifestWriter

    mock_engine = _MockBackend(make_result(experiment_id="local-fail-test"))

    def _raise(self, engine, config, **kw):
        raise RuntimeError("engine blew up inside the measurement window")

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(harness_module.MeasurementHarness, "run", _raise)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual", lambda *a, **k: None
    )

    mock_manifest = MagicMock(spec=ManifestWriter)
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config])
    config_hash = compute_declared_config_hash(config)

    # The original exception is re-raised so the real cause still reaches the CLI.
    with pytest.raises(RuntimeError, match="engine blew up"):
        run_single_experiment(study, mock_manifest, tmp_path, runner_specs=None)

    # Traceback persisted under failed-runs/ with the expected name.
    tb_file = tmp_path / "failed-runs" / f"{config_hash}_cycle1_traceback.txt"
    assert tb_file.exists(), "local single-experiment traceback was not persisted"
    assert "engine blew up inside the measurement window" in tb_file.read_text()

    # Manifest marked failed with a log_file pointer to the traceback.
    mock_manifest.mark_failed.assert_called_once()
    log_file = mock_manifest.mark_failed.call_args.kwargs.get("log_file")
    assert log_file == "failed-runs/" + tb_file.name


def test_docker_single_failure_marks_log_file(monkeypatch, tmp_path):
    """A Docker single-experiment failure points the manifest at the persisted log.

    persist_failure_artefacts copies container.log/error JSON into failed-runs/ and
    records failure["log_file"]; the docker branch must forward that to mark_failed
    so single-mode docker does not persist artefacts the manifest never references.
    """
    from unittest.mock import MagicMock

    from llenergymeasure.config.runner_spec import RunnerSpec
    from llenergymeasure.study.manifest import ManifestWriter
    from llenergymeasure.utils.exceptions import DockerError

    # Fake exchange dir with a container.log so persist_failure_artefacts has
    # something to copy and therefore sets failure["log_file"].
    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    (exchange_dir / "container.log").write_text("boom", encoding="utf-8")

    def _raise_docker(self, config, **kw):
        exc = DockerError("container failed to start")
        exc.exchange_dir = str(exchange_dir)
        raise exc

    monkeypatch.setattr("llenergymeasure.infra.docker_runner.DockerRunner.run", _raise_docker)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual", lambda *a, **k: None
    )

    mock_manifest = MagicMock(spec=ManifestWriter)
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config])
    spec = RunnerSpec(mode="docker", image="img:test", source="yaml")

    _files, results, _warnings = run_single_experiment(
        study, mock_manifest, tmp_path, runner_specs={"transformers": spec}
    )

    assert results == [None]
    mock_manifest.mark_failed.assert_called_once()
    log_file = mock_manifest.mark_failed.call_args.kwargs.get("log_file")
    assert log_file is not None, "docker single failure must forward log_file to mark_failed"
    assert log_file.startswith("failed-runs/")
    assert list((tmp_path / "failed-runs").glob("*_container.log")), (
        "container.log was not persisted into failed-runs/"
    )
