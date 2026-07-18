"""Unit tests for _save_and_record timeseries sidecar handling.

Tests cover:
- Parquet sidecar is copied into experiment result subdirectory when present
- Stale source file is cleaned up after copy
- No regression when timeseries is None (no sidecar, no crash)
- Graceful handling when source parquet file is missing from disk
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from llenergymeasure.domain.environment import RunnerEnvironment
from llenergymeasure.domain.experiment import ExperimentResult, RunnerProvenance
from llenergymeasure.infra.runner_resolution import RunnerSpec
from llenergymeasure.study.runner import (
    _provenance_from_spec,
    _runner_environment,
    _save_and_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    with_timeseries: bool = True,
) -> ExperimentResult:
    """Construct a minimal ExperimentResult for testing _save_and_record."""
    return ExperimentResult(
        experiment_id="test-save-record-001",
        measurement_config_hash="aabb1122ccdd3344",
        input_tokens=192,
        output_tokens=64,
        total_tokens=256,
        total_energy_j=10.0,
        total_inference_time_sec=2.0,
        avg_tokens_per_second=128.0,
        avg_energy_per_token_j=0.039,
        total_flops=5e10,
        timeseries="timeseries.parquet" if with_timeseries else None,
        start_time=datetime(2026, 3, 25, 10, 0, 0),
        end_time=datetime(2026, 3, 25, 10, 0, 2),
    )


def _create_parquet(path: Path) -> None:
    """Write a minimal parquet file at the given path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"timestamp_s": [0.0, 1.0], "gpu_power_w": [100.0, 105.0]})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_and_record_copies_timeseries_sidecar(tmp_path: Path) -> None:
    """Parquet sidecar is copied into the experiment result subdirectory.

    The stale flat file written by MeasurementHarness in output_dir is also
    removed after the copy.
    """
    # Create a real parquet file at the location MeasurementHarness would write it
    source_parquet = tmp_path / "timeseries.parquet"
    _create_parquet(source_parquet)
    assert source_parquet.exists(), "Pre-condition: source parquet must exist"

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=True)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,
    )

    # result.json should have been saved into a subdirectory under study_dir
    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    assert result_json_path.exists()
    assert result_json_path.name == "result.json"

    # timeseries.parquet should be next to result.json in the subdirectory
    sidecar_dest = result_json_path.parent / "timeseries.parquet"
    assert sidecar_dest.exists(), "Parquet sidecar must be copied into experiment subdirectory"

    # Stale flat file must be cleaned up
    assert not source_parquet.exists(), "Stale source parquet must be removed after copy"

    # Manifest must be updated with a non-empty result path
    manifest.mark_completed.assert_called_once()
    call_args = manifest.mark_completed.call_args
    rel_path = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("result_file", "")
    assert rel_path, "mark_completed must be called with a non-empty result_file"


def test_save_and_record_no_timeseries(tmp_path: Path) -> None:
    """When result.timeseries is None, no sidecar is copied and no crash occurs."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    assert result.timeseries is None

    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "ccdd5566",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
    )

    # result.json still saved
    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    assert result_json_path.exists()

    # No timeseries sidecar in the subdirectory
    sidecar_dest = result_json_path.parent / "timeseries.parquet"
    assert not sidecar_dest.exists(), "No sidecar should be created when timeseries is None"

    # Manifest updated normally
    manifest.mark_completed.assert_called_once()


def test_save_and_record_missing_source_file(tmp_path: Path) -> None:
    """When timeseries field is set but the parquet file is not on disk, no crash.

    result.json is still saved and manifest.mark_completed is still called.
    The missing file is simply skipped (save_result handles this with a warning log).
    """
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Result claims timeseries exists but we deliberately do NOT create the file
    result = _make_result(with_timeseries=True)
    assert result.timeseries == "timeseries.parquet"
    source_parquet = tmp_path / "timeseries.parquet"
    assert not source_parquet.exists(), "Pre-condition: source file must NOT exist"

    manifest = MagicMock()
    result_files: list[str] = []

    # Must not raise
    _save_and_record(
        result,
        study_dir,
        manifest,
        "eeff7788",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,
    )

    # result.json still saved
    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    assert result_json_path.exists()

    # No timeseries.parquet in the subdirectory (nothing to copy)
    sidecar_dest = result_json_path.parent / "timeseries.parquet"
    assert not sidecar_dest.exists()

    # Manifest updated normally (non-empty path)
    manifest.mark_completed.assert_called_once()
    call_args = manifest.mark_completed.call_args
    rel_path = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("result_file", "")
    assert rel_path, "mark_completed must be called with a non-empty result_file"


def test_save_and_record_writes_resolved_config_hash(tmp_path: Path) -> None:
    """resolved_config_hash must be written into config.json sidecar when provided.

    Regression test for Bug 2: _save_and_record had a resolved_config_hash
    parameter that was never passed from the call site, leaving the sidecar
    branch unreachable.  This test verifies the end-to-end write-and-read path.
    """
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Write a minimal config.json in the ts_source_dir (simulates harness output)
    config_sidecar_src = tmp_path / "config.json"
    config_sidecar_src.write_text(
        json.dumps(
            {
                "experiment_id": "test-resolved-001",
                "config_hash": "aabb1122ccdd3344",
                "engine": "transformers",
                "engine_version": "4.50.0",
                "observed_config_hash": "sha256_h3_stub",
            }
        )
    )

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,
        resolved_config_hash="resolved_sha256_h1_value",
    )

    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    dest_config = result_json_path.parent / "config.json"
    assert dest_config.exists(), "config.json sidecar must be moved to result dir"

    payload = json.loads(dest_config.read_text())
    assert payload.get("resolved_config_hash") == "resolved_sha256_h1_value", (
        "resolved_config_hash must be patched into config.json by _save_and_record"
    )
    # Source sidecar must be cleaned up
    assert not config_sidecar_src.exists()


def test_save_and_record_folds_provenance_into_config(tmp_path: Path) -> None:
    """A resolution_log is folded into config.json as its provenance section.

    The retired _resolution.json sidecar must not be written; the per-field
    provenance now rides in the config.json sidecar the harness produced.
    """
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Minimal config.json in the ts_source_dir (simulates harness output).
    config_sidecar_src = tmp_path / "config.json"
    config_sidecar_src.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "experiment_id": "test-prov-001",
                "measurement_config_hash": "aabb1122ccdd3344",
                "engine": "transformers",
                "engine_version": "4.50.0",
            }
        )
    )

    resolution_log = {
        "task.model": {"effective": "gpt2", "source": "yaml"},
        "batching.batch_size": {"effective": 8, "source": "cli_flag", "default": 1},
    }

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,
        resolution_log=resolution_log,
    )

    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    dest_config = result_json_path.parent / "config.json"
    assert dest_config.exists(), "config.json sidecar must be moved to result dir"

    payload = json.loads(dest_config.read_text())
    assert payload.get("provenance") == resolution_log, (
        "resolution_log must be folded into config.json as its provenance section"
    )
    # schema_version from the harness sidecar survives the fold.
    assert payload["schema_version"] == "2.0"
    # The retired standalone sidecar must not appear.
    assert not (result_json_path.parent / "_resolution.json").exists()
    assert not config_sidecar_src.exists()


def test_provenance_from_spec_docker() -> None:
    """A docker RunnerSpec maps onto a docker RunnerProvenance."""
    spec = RunnerSpec(mode="docker", image="img:1.0", source="yaml", image_source="registry")
    provenance = _provenance_from_spec(spec)
    assert provenance == RunnerProvenance(
        mode="docker", image="img:1.0", source="yaml", image_source="registry"
    )


# ---------------------------------------------------------------------------
# Environment sidecar rescue (docker dispatch)
# ---------------------------------------------------------------------------


def _make_host_snapshot():
    """Host-collected EnvironmentSnapshot with distinctly HOST values.

    Mirrors the audit's observed host bleed-through (python 3.12.12, cuda null,
    container not detected) so a test can prove the container values win.
    """
    from datetime import datetime

    from llenergymeasure.domain.environment import (
        CPUEnvironment,
        CUDAEnvironment,
        EnvironmentMetadata,
        EnvironmentSnapshot,
        GPUEnvironment,
    )

    hardware = EnvironmentMetadata(
        gpu=GPUEnvironment(name="HOST-GPU", vram_total_mb=1.0),
        cuda=CUDAEnvironment(version="unknown", driver_version="unknown"),
        cpu=CPUEnvironment(platform="Linux"),
        collected_at=datetime(2026, 1, 1, 0, 0, 0),
    )
    return EnvironmentSnapshot(
        hardware=hardware,
        python_version="3.12.12",
        tool_version="0.11.0",
        cuda_version=None,
        cuda_version_source=None,
    )


def _write_container_environment_sidecar(path: Path) -> dict:
    """Write a rescued in-container environment.json (distinct CONTAINER values)."""
    import json

    payload = {
        "experiment_id": "test-save-record-001",
        "measurement_config_hash": "aabb1122ccdd3344",
        "hardware": {
            "gpu": {"name": "NVIDIA A100-SXM4-80GB", "vram_total_mb": 81920.0},
            "cuda": {"version": "12.4", "driver_version": "535.104"},
            "cpu": {"platform": "Linux"},
            "collected_at": "2026-01-02T00:00:00",
        },
        "python_version": "3.10.14",
        "tool_version": "0.11.0",
        "cuda_version": "12.4",
        "cuda_version_source": "torch",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_docker_rescued_environment_overrides_host(tmp_path: Path) -> None:
    """Under docker dispatch, the rescued in-container environment.json is
    preferred over the host snapshot for the persisted environment.json."""
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Container rescued environment.json lives in the artefact (ts_source) dir.
    _write_container_environment_sidecar(tmp_path / "environment.json")

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,
        environment_snapshot=_make_host_snapshot(),
        runner_provenance=RunnerProvenance(
            mode="docker", image="img:1.0", source="yaml", image_source="registry"
        ),
    )

    assert len(result_files) == 1
    env_dest = Path(result_files[0]).parent / "environment.json"
    assert env_dest.exists()
    payload = json.loads(env_dest.read_text())
    # Container values must win over the host snapshot written first.
    assert payload["python_version"] == "3.10.14", "container python must win, not host 3.12.12"
    assert payload["cuda_version"] == "12.4", "container cuda must win, not host null"
    assert payload["hardware"]["gpu"]["name"] == "NVIDIA A100-SXM4-80GB"
    # The rescued file in the staging dir must be consumed (moved).
    assert not (tmp_path / "environment.json").exists()


def test_docker_without_rescued_environment_warns(tmp_path: Path, caplog) -> None:
    """A docker run that lands without a rescued snapshot logs a loud warning
    (the persisted environment.json would carry host, not container, values)."""
    import logging

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.runner"):
        _save_and_record(
            result,
            study_dir,
            manifest,
            "aabb1122",
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
            ts_source_dir=tmp_path,  # no environment.json inside
            environment_snapshot=_make_host_snapshot(),
            runner_provenance=RunnerProvenance(
                mode="docker", image="img:1.0", source="yaml", image_source="registry"
            ),
        )

    assert any(
        "No in-container environment.json rescued" in rec.message for rec in caplog.records
    ), "docker dispatch without a rescued snapshot must warn loudly"


def test_local_run_uses_host_snapshot_without_warning(tmp_path: Path, caplog) -> None:
    """Local dispatch is unchanged: host snapshot is written, no rescue, no warning."""
    import json
    import logging

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.runner"):
        _save_and_record(
            result,
            study_dir,
            manifest,
            "aabb1122",
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
            ts_source_dir=tmp_path,  # local temp dir never holds environment.json
            environment_snapshot=_make_host_snapshot(),
            runner_provenance=RunnerProvenance(
                mode="local", image=None, source="local", image_source=None
            ),
        )

    env_dest = Path(result_files[0]).parent / "environment.json"
    assert env_dest.exists()
    payload = json.loads(env_dest.read_text())
    # Local: the host snapshot is authoritative and persisted as-is.
    assert payload["python_version"] == "3.12.12"
    assert not any(
        "No in-container environment.json rescued" in rec.message for rec in caplog.records
    ), "local dispatch must not emit the docker rescue warning"


def test_config_sidecar_rescue_permission_error_warns(tmp_path: Path, monkeypatch, caplog) -> None:
    """A config.json sidecar the host cannot read (0600, root-owned) warns loudly.

    Regression: under docker dispatch the container runs as root and wrote the
    sidecar 0600, so the non-root host's load_json raised PermissionError. That
    error was swallowed at debug level and the sidecar silently vanished. The
    unreadable file is simulated by monkeypatching load_json to raise
    PermissionError (a non-root test cannot create a root-owned 0600 file).
    """
    import logging

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Sidecar is present in the staging dir but "unreadable" (rescued 0600 root).
    config_sidecar_src = tmp_path / "config.json"
    config_sidecar_src.write_text("{}", encoding="utf-8")

    def _raise_permission(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("llenergymeasure.study.runner.load_json", _raise_permission)

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.runner"):
        _save_and_record(
            result,
            study_dir,
            manifest,
            "aabb1122",
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
            ts_source_dir=tmp_path,
        )

    warnings = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any(
        "Failed to move config.json sidecar" in m and "Permission denied" in m for m in warnings
    ), f"unreadable config.json must warn loudly with the reason; got {warnings}"
    # The path must be named in the warning for the user to act on it.
    assert any(str(config_sidecar_src) in m for m in warnings), (
        "the rescue-failure warning must name the offending path"
    )
    # result.json still lands (a sidecar failure must never lose the measurement).
    assert len(result_files) == 1
    assert Path(result_files[0]).exists()
    # The unreadable staging file is still cleaned up (finally block).
    assert not config_sidecar_src.exists()


def test_environment_sidecar_rescue_permission_error_warns(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """An environment.json sidecar the host cannot read warns loudly.

    Same root cause as the config.json path: a 0600 root-owned rescued sidecar
    raised PermissionError that was swallowed at debug. Simulated by patching
    load_json to raise.
    """
    import logging

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # A rescued environment.json is present but "unreadable".
    (tmp_path / "environment.json").write_text("{}", encoding="utf-8")

    def _raise_permission(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("llenergymeasure.study.runner.load_json", _raise_permission)

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.runner"):
        _save_and_record(
            result,
            study_dir,
            manifest,
            "aabb1122",
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
            ts_source_dir=tmp_path,
            environment_snapshot=_make_host_snapshot(),
            runner_provenance=RunnerProvenance(
                mode="docker", image="img:1.0", source="yaml", image_source="registry"
            ),
        )

    warnings = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any(
        "Failed to rescue in-container environment.json" in m and "Permission denied" in m
        for m in warnings
    ), f"unreadable environment.json must warn loudly with the reason; got {warnings}"
    # result.json still lands; the unreadable staging file is cleaned up.
    assert len(result_files) == 1
    assert not (tmp_path / "environment.json").exists()


def test_provenance_from_spec_none_defaults_to_local() -> None:
    """No spec (pure in-process local) records mode=local, source=local, no image."""
    provenance = _provenance_from_spec(None)
    assert provenance.mode == "local"
    assert provenance.image is None
    assert provenance.source == "local"
    assert provenance.image_source is None


def test_save_and_record_attaches_runner_provenance(tmp_path: Path) -> None:
    """runner_provenance passed to _save_and_record is written into result.json."""
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        runner_provenance=RunnerProvenance(mode="docker", image="img:2.0", source="env"),
    )

    assert len(result_files) == 1
    payload = json.loads(Path(result_files[0]).read_text())
    assert payload["runner_provenance"]["mode"] == "docker"
    assert payload["runner_provenance"]["image"] == "img:2.0"
    assert payload["runner_provenance"]["source"] == "env"


def test_save_and_record_calls_mark_failed_on_exception(tmp_path: Path) -> None:
    """When save_result raises, manifest.mark_failed is called (not mark_completed).

    Previously the except clause called mark_completed with result_file="" which
    silently recorded a failure as a success with no result path. This test
    verifies the corrected behaviour: mark_failed is called with a meaningful
    error type and message.
    """
    from unittest.mock import patch

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    with patch(
        "llenergymeasure.results.persistence.save_result",
        side_effect=OSError("disk full"),
    ):
        _save_and_record(
            result,
            study_dir,
            manifest,
            "aabb1122",
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
        )

    # mark_failed must be called - NOT mark_completed
    manifest.mark_failed.assert_called_once()
    call_args = manifest.mark_failed.call_args[0]
    assert call_args[0] == "aabb1122"  # config_hash
    assert call_args[1] == 1  # cycle
    assert "OSError" in call_args[2]  # error_type
    assert "disk full" in call_args[3]  # error_message

    manifest.mark_completed.assert_not_called()


# ---------------------------------------------------------------------------
# Runner block (environment.json)
# ---------------------------------------------------------------------------


def test_save_and_record_writes_local_runner_block(tmp_path: Path) -> None:
    """Local dispatch: the runner block is written into environment.json via the host snapshot."""
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,  # local temp dir never holds environment.json
        environment_snapshot=_make_host_snapshot(),
        runner_provenance=RunnerProvenance(
            mode="local", image=None, source="default", image_source=None
        ),
        runner_environment=RunnerEnvironment(
            type="local", image=None, image_digest=None, source="default"
        ),
    )

    env_dest = Path(result_files[0]).parent / "environment.json"
    payload = json.loads(env_dest.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["runner"] == {
        "type": "local",
        "image": None,
        "image_digest": None,
        "source": "default",
    }


def test_save_and_record_docker_rescue_patches_runner_block(tmp_path: Path) -> None:
    """Docker dispatch: the runner block (image + digest) is patched into the rescued env.json.

    The container writes environment.json without runner facts the host alone
    knows (image ref, registry digest, source), so the host patches them into
    the rescued snapshot while the container's hardware values still win.
    """
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    _write_container_environment_sidecar(tmp_path / "environment.json")

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,
        environment_snapshot=_make_host_snapshot(),
        runner_provenance=RunnerProvenance(
            mode="docker", image="ghcr.io/acme/vllm:1.0", source="yaml", image_source="registry"
        ),
        runner_environment=RunnerEnvironment(
            type="docker",
            image="ghcr.io/acme/vllm:1.0",
            image_digest="ghcr.io/acme/vllm@sha256:abc123",
            source="yaml",
        ),
    )

    env_dest = Path(result_files[0]).parent / "environment.json"
    payload = json.loads(env_dest.read_text())
    # Runner block (host-only facts) patched into the rescued snapshot.
    assert payload["runner"] == {
        "type": "docker",
        "image": "ghcr.io/acme/vllm:1.0",
        "image_digest": "ghcr.io/acme/vllm@sha256:abc123",
        "source": "yaml",
    }
    # schema_version stamped in when the (old-style) container payload omitted it.
    assert payload["schema_version"] == "1.0"
    # Container hardware/runtime values still win over the host snapshot.
    assert payload["python_version"] == "3.10.14"
    assert payload["hardware"]["gpu"]["name"] == "NVIDIA A100-SXM4-80GB"
    # Rescued staging file consumed.
    assert not (tmp_path / "environment.json").exists()


def test_save_and_record_docker_without_rescue_writes_runner_block(tmp_path: Path) -> None:
    """Docker dispatch without a rescued snapshot: runner block still lands via host snapshot."""
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        model_name="gpt2",
        engine="transformers",
        ts_source_dir=tmp_path,  # no environment.json rescued
        environment_snapshot=_make_host_snapshot(),
        runner_provenance=RunnerProvenance(
            mode="docker", image="ghcr.io/acme/vllm:1.0", source="yaml", image_source="registry"
        ),
        runner_environment=RunnerEnvironment(
            type="docker",
            image="ghcr.io/acme/vllm:1.0",
            image_digest=None,
            source="yaml",
        ),
    )

    env_dest = Path(result_files[0]).parent / "environment.json"
    payload = json.loads(env_dest.read_text())
    # Runner block present even in the degraded no-rescue case (host snapshot carries it).
    assert payload["runner"]["type"] == "docker"
    assert payload["runner"]["image"] == "ghcr.io/acme/vllm:1.0"
    assert payload["runner"]["image_digest"] is None


def test_runner_environment_local_and_none_spec() -> None:
    """_runner_environment maps local specs (and no spec) onto a local runner block."""
    local = _runner_environment(RunnerSpec(mode="local", image=None, source="user_config"))
    assert local.type == "local"
    assert local.image is None
    assert local.image_digest is None
    assert local.source == "user_config"

    no_spec = _runner_environment(None)
    assert no_spec.type == "local"
    assert no_spec.source == "local"


def test_runner_environment_docker_digest_failure_is_none() -> None:
    """A docker spec whose digest cannot be resolved records image_digest=None (never raises)."""
    from unittest.mock import patch

    with patch("llenergymeasure.infra.image_registry.resolve_image_digest", return_value=None):
        env = _runner_environment(
            RunnerSpec(mode="docker", image=None, source="auto_detected"),
            resolved_image="ghcr.io/acme/vllm:1.0",
        )
    assert env.type == "docker"
    assert env.image == "ghcr.io/acme/vllm:1.0"
    assert env.image_digest is None
    assert env.source == "auto_detected"
