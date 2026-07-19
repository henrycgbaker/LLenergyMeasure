"""Unit tests for BundleWriter - the per-experiment results-bundle owner.

These exercise the writer policies directly (not through study.runner._save_and_record,
which the tests in tests/unit/study/test_save_and_record.py cover end-to-end):

- bundle_version stamping across result.json / config.json / environment.json
- runner-provenance attach on result.json
- the environment rescue-preference policy + runner-block patch
- the config-sidecar move + patch
- the finalize loudness backstops (missing config, declared-but-missing timeseries)
- the artefact registry as the server-mode extension point (register + sweep)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from llenergymeasure.domain.bundle_artefacts import ARTEFACTS, BUNDLE_VERSION, ArtefactSpec
from llenergymeasure.domain.environment import (
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    EnvironmentSnapshot,
    GPUEnvironment,
    RunnerEnvironment,
)
from llenergymeasure.domain.experiment import ExperimentResult, RunnerProvenance
from llenergymeasure.results.bundle import BundleWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(*, with_timeseries: bool = False) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="bundle-test-001",
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


def _make_snapshot() -> EnvironmentSnapshot:
    hardware = EnvironmentMetadata(
        gpu=GPUEnvironment(name="HOST-GPU", vram_total_mb=1.0),
        cuda=CUDAEnvironment(version="unknown", driver_version="unknown"),
        cpu=CPUEnvironment(platform="Linux"),
        collected_at=datetime(2026, 1, 1, 0, 0, 0),
    )
    return EnvironmentSnapshot(
        hardware=hardware,
        python_version="3.12.12",
        tool_version="0.6.0",
        cuda_version=None,
        cuda_version_source=None,
    )


def _write_container_env(path: Path) -> None:
    """A rescued in-container environment.json with distinct CONTAINER values."""
    payload = {
        "experiment_id": "bundle-test-001",
        "hardware": {
            "gpu": {"name": "NVIDIA A100-SXM4-80GB", "vram_total_mb": 81920.0},
            "cuda": {"version": "12.4", "driver_version": "535.104"},
            "cpu": {"platform": "Linux"},
            "collected_at": "2026-01-02T00:00:00",
        },
        "python_version": "3.10.14",
        "tool_version": "0.6.0",
        "cuda_version": "12.4",
        "cuda_version_source": "torch",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _writer(study_dir: Path, *, ts_source_dir: Path | None = None) -> BundleWriter:
    return BundleWriter(
        study_dir,
        model_name="gpt2",
        engine="transformers",
        config_hash="aabb1122",
        cycle=1,
        ts_source_dir=ts_source_dir,
    )


# ---------------------------------------------------------------------------
# write_result
# ---------------------------------------------------------------------------


def test_write_result_stamps_bundle_version(tmp_path: Path) -> None:
    """result.json carries the single bundle_version stamp."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    result_path = writer.write_result(_make_result())
    payload = json.loads(result_path.read_text())
    assert payload["bundle_version"] == BUNDLE_VERSION == "1.0"


def test_write_result_attaches_runner_provenance(tmp_path: Path) -> None:
    """runner_provenance is folded into result.json before serialisation."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    result_path = writer.write_result(
        _make_result(),
        runner_provenance=RunnerProvenance(mode="docker", image="img:2.0", source="env"),
    )
    payload = json.loads(result_path.read_text())
    assert payload["runner_provenance"]["mode"] == "docker"
    assert payload["runner_provenance"]["image"] == "img:2.0"
    assert payload["runner_provenance"]["source"] == "env"


def test_bundle_dir_requires_write_result(tmp_path: Path) -> None:
    """Accessing bundle_dir before write_result raises (write order contract)."""
    import pytest

    writer = _writer(tmp_path)
    with pytest.raises(RuntimeError):
        _ = writer.bundle_dir


# ---------------------------------------------------------------------------
# write_environment - rescue preference + stamps
# ---------------------------------------------------------------------------


def test_write_environment_local_stamps_bundle_version(tmp_path: Path) -> None:
    """Local dispatch writes the host snapshot with a bundle_version stamp."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(_make_result())
    writer.write_environment(
        host_snapshot=_make_snapshot(),
        runner_environment=RunnerEnvironment(mode="local", source="default"),
        runner_provenance=RunnerProvenance(mode="local", source="local"),
    )
    payload = json.loads((writer.bundle_dir / "environment.json").read_text())
    assert payload["bundle_version"] == "1.0"
    assert payload["python_version"] == "3.12.12"
    assert payload["runner"] == {
        "mode": "local",
        "image": None,
        "image_digest": None,
        "source": "default",
    }


def test_write_environment_prefers_rescued_over_host(tmp_path: Path) -> None:
    """The rescued in-container environment.json wins over the host snapshot."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_container_env(staging / "environment.json")

    writer = _writer(study_dir, ts_source_dir=staging)
    writer.write_result(_make_result())
    writer.write_environment(
        host_snapshot=_make_snapshot(),
        runner_environment=RunnerEnvironment(
            mode="docker",
            image="ghcr.io/acme/vllm:1.0",
            image_digest="ghcr.io/acme/vllm@sha256:abc123",
            source="yaml",
        ),
        runner_provenance=RunnerProvenance(mode="docker", image="ghcr.io/acme/vllm:1.0"),
    )
    payload = json.loads((writer.bundle_dir / "environment.json").read_text())
    # Container hardware/runtime values win over the host snapshot.
    assert payload["python_version"] == "3.10.14"
    assert payload["hardware"]["gpu"]["name"] == "NVIDIA A100-SXM4-80GB"
    # Host-only runner block patched in, and the payload stamped.
    assert payload["runner"]["image_digest"] == "ghcr.io/acme/vllm@sha256:abc123"
    assert payload["bundle_version"] == "1.0"
    # Rescued staging file consumed.
    assert not (staging / "environment.json").exists()


def test_write_environment_docker_without_rescue_warns(tmp_path: Path, caplog) -> None:
    """A docker run with no rescued snapshot warns (host, not container, recorded)."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(_make_result())
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.write_environment(
            host_snapshot=_make_snapshot(),
            runner_environment=RunnerEnvironment(mode="docker", image="img:1.0", source="yaml"),
            runner_provenance=RunnerProvenance(mode="docker", image="img:1.0"),
        )
    assert any("No in-container environment.json rescued" in rec.message for rec in caplog.records)


def test_patch_runner_block_adds_block_and_stamps(tmp_path: Path) -> None:
    """_patch_runner_block injects the runner block and stamps bundle_version."""
    payload = BundleWriter._patch_runner_block(
        {"python_version": "3.10.14"},
        RunnerEnvironment(mode="docker", image="img:1.0", image_digest=None, source="yaml"),
    )
    assert payload["runner"]["mode"] == "docker"
    assert payload["bundle_version"] == "1.0"


def test_patch_runner_block_none_is_noop() -> None:
    """With no runner block the payload is returned unchanged (no stamp forced)."""
    payload = BundleWriter._patch_runner_block({"python_version": "3.10.14"}, None)
    assert payload == {"python_version": "3.10.14"}


# ---------------------------------------------------------------------------
# move_config_sidecar - patch + stamp
# ---------------------------------------------------------------------------


def test_move_config_sidecar_patches_and_stamps(tmp_path: Path) -> None:
    """The moved config.json gains resolved_config_hash, provenance, bundle_version."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "config.json").write_text(
        json.dumps({"experiment_id": "bundle-test-001", "engine": "transformers"}),
        encoding="utf-8",
    )

    writer = _writer(study_dir, ts_source_dir=staging)
    writer.write_result(_make_result())
    resolution_log = {"task.model": {"effective": "gpt2", "source": "yaml"}}
    writer.move_config_sidecar(resolved_config_hash="resolved_h1", resolution_log=resolution_log)

    payload = json.loads((writer.bundle_dir / "config.json").read_text())
    assert payload["resolved_config_hash"] == "resolved_h1"
    assert payload["provenance"] == resolution_log
    assert payload["bundle_version"] == "1.0"
    # Staged source consumed.
    assert not (staging / "config.json").exists()


# ---------------------------------------------------------------------------
# finalize - loudness backstops
# ---------------------------------------------------------------------------


def test_finalize_warns_missing_config(tmp_path: Path, caplog) -> None:
    """finalize warns when config.json (provenance/identity home) is absent."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)  # no ts_source_dir -> no config.json to move
    writer.write_result(_make_result())
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert any("config.json" in rec.message and "config" in rec.message for rec in caplog.records)


def test_finalize_warns_declared_but_missing_timeseries(tmp_path: Path, caplog) -> None:
    """finalize warns when the result declares a timeseries that did not land."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    # Result declares a timeseries but the staged parquet never existed.
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(_make_result(with_timeseries=True))
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert any("timeseries.parquet" in rec.message for rec in caplog.records)


def test_finalize_no_timeseries_warning_when_none_declared(tmp_path: Path, caplog) -> None:
    """No timeseries backstop fires when the result never declared one."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(_make_result(with_timeseries=False))
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert not any("timeseries.parquet" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# artefact registry - the server-mode extension point
# ---------------------------------------------------------------------------


def test_finalize_sweeps_newly_registered_artefact(tmp_path: Path, caplog, monkeypatch) -> None:
    """A future artefact (e.g. a server-mode series) is swept by finalize once
    registered - one registry entry is enough, no writer plumbing changes.

    Registering the entry plus a writer method that produces the file is all a
    v0.8.0 server-mode artefact would need; here we prove finalize picks it up.
    """
    monkeypatch.setitem(
        ARTEFACTS,
        "request_series",
        ArtefactSpec(
            "request_series.parquet", required=False, warn_if_missing=True, kind="parquet"
        ),
    )
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    writer.write_result(_make_result())
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert any(
        "request_series" in rec.message and "request_series.parquet" in rec.message
        for rec in caplog.records
    ), "finalize must sweep any registered warn_if_missing artefact"
