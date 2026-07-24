"""Unit tests for the per-experiment results-bundle owner (writer + reader).

The writer tests exercise the BundleWriter policies directly (not through
study.runner._save_and_record, which tests/unit/study/test_save_and_record.py
covers end-to-end):

- bundle_version stamping across result.json / config.json / system.json
- runner-provenance attach on result.json
- the system-snapshot rescue-preference policy + runner-block patch
- the config-sidecar move + patch
- the finalize loudness backstops (missing config, declared-but-missing timeseries)
- the artefact registry as the server-mode extension point (register + sweep)

The reader tests exercise BundleReader:

- happy-path read into a LoadedBundle (result + environment + config + paths)
- registry-driven discovery, including a newly-registered artefact surfacing
- the strict required-artefact contract and the legacy best-effort fallback
- the read_sidecar single-artefact accessor
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from llenergymeasure.domain.bundle_artefacts import ARTEFACTS, BUNDLE_VERSION, ArtefactSpec
from llenergymeasure.domain.environment import (
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    EnvironmentSnapshot,
    GPUEnvironment,
)
from llenergymeasure.domain.experiment import RunnerProvenance
from llenergymeasure.results.bundle import BundleReader, BundleWriter, LoadedBundle
from tests.conftest import make_result, write_container_system_sidecar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot() -> EnvironmentSnapshot:
    hardware = EnvironmentMetadata(
        gpu=GPUEnvironment(name="HOST-GPU", vram_total_mb=1.0),
        cuda=CUDAEnvironment(driver_supported_version="unknown", driver_version="unknown"),
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
    result_path = writer.write_result(make_result())
    payload = json.loads(result_path.read_text())
    assert payload["bundle_version"] == BUNDLE_VERSION == "2.0"


def test_write_result_attaches_runner_provenance(tmp_path: Path) -> None:
    """runner_provenance is folded into result.json before serialisation."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    result_path = writer.write_result(
        make_result(),
        runner_provenance=RunnerProvenance(mode="container", image="img:2.0", source="env"),
    )
    payload = json.loads(result_path.read_text())
    assert payload["runner_provenance"]["mode"] == "container"
    assert payload["runner_provenance"]["image"] == "img:2.0"
    assert payload["runner_provenance"]["source"] == "env"


def test_bundle_dir_requires_write_result(tmp_path: Path) -> None:
    """Accessing bundle_dir before write_result raises (write order contract)."""
    import pytest

    writer = _writer(tmp_path)
    with pytest.raises(RuntimeError):
        _ = writer.bundle_dir


# ---------------------------------------------------------------------------
# write_system - rescue preference + stamps
# ---------------------------------------------------------------------------


def test_write_system_local_stamps_bundle_version(tmp_path: Path) -> None:
    """Local dispatch writes the host snapshot with a bundle_version stamp."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(make_result())
    writer.write_system(
        host_snapshot=_make_snapshot(),
        runner=RunnerProvenance(mode="process", source="default"),
    )
    payload = json.loads((writer.bundle_dir / "system.json").read_text())
    assert payload["bundle_version"] == "2.0"
    assert payload["python_version"] == "3.12.12"
    assert payload["runner"] == {
        "mode": "process",
        "image": None,
        "source": "default",
        "image_source": None,
        "image_digest": None,
    }


def test_write_system_prefers_rescued_over_host(tmp_path: Path) -> None:
    """The rescued in-container system.json wins over the host snapshot."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    write_container_system_sidecar(staging / "system.json")

    writer = _writer(study_dir, ts_source_dir=staging)
    writer.write_result(make_result())
    writer.write_system(
        host_snapshot=_make_snapshot(),
        runner=RunnerProvenance(
            mode="container",
            image="ghcr.io/acme/vllm:1.0",
            image_digest="ghcr.io/acme/vllm@sha256:abc123",
            source="yaml",
        ),
    )
    payload = json.loads((writer.bundle_dir / "system.json").read_text())
    # Container hardware/runtime values win over the host snapshot.
    assert payload["python_version"] == "3.10.14"
    assert payload["hardware"]["gpu"]["name"] == "NVIDIA A100-SXM4-80GB"
    # Host-only runner block patched in, and the payload stamped.
    assert payload["runner"]["image_digest"] == "ghcr.io/acme/vllm@sha256:abc123"
    assert payload["bundle_version"] == "2.0"
    # Rescued staging file consumed.
    assert not (staging / "system.json").exists()


def test_write_system_docker_without_rescue_warns(tmp_path: Path, caplog) -> None:
    """A docker run with no rescued snapshot warns (host, not container, recorded)."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(make_result())
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.write_system(
            host_snapshot=_make_snapshot(),
            runner=RunnerProvenance(mode="container", image="img:1.0", source="yaml"),
        )
    assert any("No in-container system.json rescued" in rec.message for rec in caplog.records)


def test_patch_runner_block_adds_block_and_stamps(tmp_path: Path) -> None:
    """_patch_runner_block injects the runner block and stamps bundle_version."""
    payload = BundleWriter._patch_runner_block(
        {"python_version": "3.10.14"},
        RunnerProvenance(mode="container", image="img:1.0", image_digest=None, source="yaml"),
    )
    assert payload["runner"]["mode"] == "container"
    assert payload["bundle_version"] == "2.0"


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
    writer.write_result(make_result())
    resolution_log = {"task.model": {"effective": "gpt2", "source": "yaml"}}
    writer.move_config_sidecar(resolved_config_hash="resolved_h1", resolution_log=resolution_log)

    payload = json.loads((writer.bundle_dir / "config.json").read_text())
    assert payload["resolved_config_hash"] == "resolved_h1"
    assert payload["provenance"] == resolution_log
    assert payload["bundle_version"] == "2.0"
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
    writer.write_result(make_result())
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert any("config.json" in rec.message and "config" in rec.message for rec in caplog.records)


def test_finalize_warns_declared_but_missing_timeseries(tmp_path: Path, caplog) -> None:
    """finalize warns when the result declares a timeseries that did not land."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    # Result declares a timeseries but the staged parquet never existed.
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(make_result(timeseries="timeseries.parquet"))
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert any("timeseries.parquet" in rec.message for rec in caplog.records)


def test_finalize_no_timeseries_warning_when_none_declared(tmp_path: Path, caplog) -> None:
    """No timeseries backstop fires when the result never declared one."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir, ts_source_dir=tmp_path)
    writer.write_result(make_result())
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
    writer.write_result(make_result())
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.results.bundle"):
        writer.finalize()
    assert any(
        "request_series" in rec.message and "request_series.parquet" in rec.message
        for rec in caplog.records
    ), "finalize must sweep any registered warn_if_missing artefact"


# ---------------------------------------------------------------------------
# BundleReader - read side
# ---------------------------------------------------------------------------


def _write_full_bundle(tmp_path: Path) -> Path:
    """Write a complete bundle (result + environment + config) and return its dir."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": "test-001",
                "engine": "transformers",
                "provenance": {"task.model": {"effective": "gpt2", "source": "yaml"}},
            }
        ),
        encoding="utf-8",
    )

    writer = _writer(study_dir, ts_source_dir=staging)
    writer.write_result(make_result())
    writer.write_system(
        host_snapshot=_make_snapshot(),
        runner=RunnerProvenance(mode="process", source="default"),
    )
    writer.move_config_sidecar(resolved_config_hash="resolved_h1", resolution_log=None)
    writer.finalize()
    return writer.bundle_dir


def test_bundle_reader_happy_path(tmp_path: Path) -> None:
    """read() returns a LoadedBundle with result + environment + config + paths."""
    bundle_dir = _write_full_bundle(tmp_path)

    loaded = BundleReader.read(bundle_dir)

    assert isinstance(loaded, LoadedBundle)
    assert loaded.bundle_dir == bundle_dir
    assert loaded.result.experiment_id == "test-001"
    # Environment parsed and attached to the result (matching load_result).
    assert loaded.environment is not None
    assert loaded.result.environment is loaded.environment
    assert loaded.environment.python_version == "3.12.12"
    # Config payload surfaced as a raw dict.
    assert loaded.config is not None
    assert loaded.config["provenance"]["task.model"]["effective"] == "gpt2"
    # Registry-keyed paths for every present artefact.
    assert loaded.paths["result"].name == "result.json"
    assert loaded.paths["config"].name == "config.json"
    assert loaded.paths["system"].name == "system.json"


def test_bundle_reader_registry_driven_discovery(tmp_path: Path, monkeypatch) -> None:
    """A newly-registered artefact present in the dir surfaces in LoadedBundle.paths.

    Discovery is driven by the ARTEFACTS registry, so a future artefact (e.g. a
    server-mode per-request series) is picked up by read() with one registry
    entry - no reader plumbing changes.
    """
    monkeypatch.setitem(
        ARTEFACTS,
        "request_series",
        ArtefactSpec(
            "request_series.parquet", required=False, warn_if_missing=False, kind="parquet"
        ),
    )
    bundle_dir = _write_full_bundle(tmp_path)
    (bundle_dir / "request_series.parquet").write_bytes(b"PAR1")

    loaded = BundleReader.read(bundle_dir)
    assert loaded.paths["request_series"].name == "request_series.parquet"


def test_bundle_reader_missing_result_raises(tmp_path: Path) -> None:
    """A bundle dir without the required result.json raises (strict contract)."""
    empty = tmp_path / "empty-bundle"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        BundleReader.read(empty)


def test_bundle_reader_optional_sidecars_absent(tmp_path: Path) -> None:
    """A result-only bundle reads with environment and config as None."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    writer.write_result(make_result())

    loaded = BundleReader.read(writer.bundle_dir)
    assert loaded.environment is None
    assert loaded.config is None
    assert "config" not in loaded.paths
    assert "environment" not in loaded.paths


def test_bundle_reader_legacy_fallback_warns(tmp_path: Path) -> None:
    """A pre-bundle_version result.json reads best-effort with ONE UserWarning."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    result_path = writer.write_result(make_result())
    # Simulate a legacy bundle: retired per-artefact key, no bundle_version.
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw.pop("bundle_version", None)
    raw["schema_version"] = "5.0"
    result_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.warns(UserWarning, match="legacy results bundle"):
        loaded = BundleReader.read(writer.bundle_dir)
    assert loaded.result.experiment_id == "test-001"
    assert loaded.result.bundle_version == BUNDLE_VERSION


def test_bundle_reader_tolerates_1_0_bundle(tmp_path: Path) -> None:
    """A full bundle 1.0 (pre-unification shape) reads best-effort, ONE warning.

    Exercises the 2.0-break tolerances that REMAIN against a synthetic 1.0-shaped
    bundle written by hand: result.json carries the retired top-level
    baseline_power_w copy and a schema_version key with a runner_provenance block
    lacking image_digest; the system sidecar (system.json) carries the old separate
    RunnerEnvironment-shaped runner block (no image_source), the never-populated
    hardware fields (pcie_gen, mig_enabled, cudnn_version, fan_speed_pct), and the
    pre-rename cuda.version key. The reader drops/maps those legacy shapes rather
    than rejecting them, and emits exactly one bundle-level UserWarning.

    NOTE the clean breaks are exercised by their own tests, not tolerated here:
    the sidecar rename means the 1.0 content is placed under the CURRENT filename
    (system.json, not environment.json), and the runner blocks use the CURRENT
    vocabulary (``mode: "container"``) deliberately - the v0.7 runner-mode rename
    is a clean break, NOT a tolerated legacy shape, so a 1.0 (or 2.0-era) bundle
    whose runner block carries the pre-v0.7 ``docker``/``local`` mode fails
    validation on read. That exclusion is pinned by
    ``test_bundle_reader_1_0_bundle_with_legacy_runner_mode_raises``.
    """
    bundle_dir = tmp_path / "study" / "exp-legacy"
    bundle_dir.mkdir(parents=True)

    result_payload = {
        "bundle_version": "1.0",
        "schema_version": "5.0",  # retired per-artefact counter
        "experiment_id": "legacy-001",
        "measurement_config_hash": "deadbeef",
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "total_energy_j": 5.0,
        "total_inference_time_sec": 1.0,
        "avg_tokens_per_second": 30.0,
        "avg_energy_per_token_j": 0.25,
        "total_flops": 0.0,
        "baseline_power_w": 42.0,  # bundle 1.0 top-level copy (dropped in 2.0)
        "energy_breakdown": {"raw_j": 5.0, "baseline_power_w": 42.0},
        "start_time": "2026-01-01T00:00:00",
        "end_time": "2026-01-01T00:00:01",
        # runner_provenance in its 1.0 shape: no image_digest key.
        "runner_provenance": {
            "mode": "container",
            "image": "img:1.0",
            "source": "yaml",
            "image_source": "registry",
        },
    }
    (bundle_dir / "result.json").write_text(json.dumps(result_payload), encoding="utf-8")

    env_payload = {
        "bundle_version": "1.0",
        "hardware": {
            "gpu": {
                "name": "NVIDIA A100",
                "vram_total_mb": 81920.0,
                "compute_capability": "8.0",
                "pcie_gen": 4,  # dropped field
                "mig_enabled": False,  # dropped field
            },
            "cuda": {
                "version": "12.4",  # pre-rename key -> driver_supported_version
                "driver_version": "550.54",
                "cudnn_version": "9.1",  # dropped field
            },
            "thermal": {"temperature_c": 40.0, "fan_speed_pct": 30.0},  # fan_speed_pct dropped
            "cpu": {"platform": "Linux"},
            "container": {"detected": True, "runtime": "docker"},
            "collected_at": "2026-01-01T00:00:00",
        },
        "python_version": "3.10.14",
        "tool_version": "0.6.0",
        "cuda_version": "12.1",
        "cuda_version_source": "torch",
        # runner in its old RunnerEnvironment shape: no image_source key.
        "runner": {
            "mode": "container",
            "image": "img:1.0",
            "image_digest": "img@sha256:abc",
            "source": "yaml",
        },
    }
    (bundle_dir / "system.json").write_text(json.dumps(env_payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="legacy results bundle") as record:
        loaded = BundleReader.read(bundle_dir)

    # Exactly one bundle-level warning (not multiplied across artefacts).
    assert sum(issubclass(w.category, UserWarning) for w in record) == 1

    # result.json: legacy top-level keys dropped, model reads with 2.0 defaults.
    assert loaded.result.experiment_id == "legacy-001"
    assert loaded.result.serving_mode == "offline"  # new field defaults
    assert not hasattr(loaded.result, "baseline_power_w")  # dropped field
    # The single baseline home survives on the breakdown.
    assert loaded.result.energy_breakdown is not None
    assert loaded.result.energy_breakdown.baseline_power_w == 42.0
    # Old runner_provenance shape reads into the unified model (digest defaults None).
    assert loaded.result.runner_provenance is not None
    assert loaded.result.runner_provenance.image_source == "registry"
    assert loaded.result.runner_provenance.image_digest is None

    # system.json: dead fields ignored, cuda version mapped, runner unified.
    assert loaded.environment is not None
    assert loaded.environment.hardware.cuda.driver_supported_version == "12.4"
    assert loaded.environment.cuda_version == "12.1"
    assert loaded.environment.runner is not None
    assert loaded.environment.runner.image_digest == "img@sha256:abc"
    assert loaded.environment.runner.image_source is None


def test_bundle_reader_1_0_bundle_with_legacy_runner_mode_raises(tmp_path: Path) -> None:
    """A bundle whose runner block carries the pre-v0.7 mode fails validation on read.

    The runner-mode rename (``docker``/``local`` -> ``container``/``process``) is a
    clean break, NOT a tolerated legacy shape: ``RunnerProvenance.mode`` is a closed
    ``Literal``, so ``ExperimentResult.model_validate`` (via the strict read path)
    hard-crashes on a stale value rather than silently loading it. This crash is the
    INTENDED behavior and is pinned here - the companion
    ``test_bundle_reader_tolerates_1_0_bundle`` covers the tolerances that remain.
    """
    import pydantic

    bundle_dir = tmp_path / "study" / "exp-stale-runner"
    bundle_dir.mkdir(parents=True)

    result_payload = {
        "bundle_version": "1.0",
        "experiment_id": "legacy-002",
        "measurement_config_hash": "deadbeef",
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "total_energy_j": 5.0,
        "total_inference_time_sec": 1.0,
        "avg_tokens_per_second": 30.0,
        "avg_energy_per_token_j": 0.25,
        "total_flops": 0.0,
        "energy_breakdown": {"raw_j": 5.0},
        "start_time": "2026-01-01T00:00:00",
        "end_time": "2026-01-01T00:00:01",
        # Pre-v0.7 runner vocabulary - no longer accepted on read.
        "runner_provenance": {"mode": "docker", "image": "img:1.0", "source": "yaml"},
    }
    (bundle_dir / "result.json").write_text(json.dumps(result_payload), encoding="utf-8")

    with pytest.raises(pydantic.ValidationError, match=r"mode"):
        BundleReader.read(bundle_dir)


def test_bundle_reader_declared_but_missing_timeseries_warns(tmp_path: Path) -> None:
    """read() warns when the result references a parquet that did not land."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    writer.write_result(make_result(timeseries="timeseries.parquet"))

    with pytest.warns(UserWarning, match="Timeseries sidecar missing"):
        BundleReader.read(writer.bundle_dir)


def test_bundle_reader_corrupt_system_is_best_effort(tmp_path: Path) -> None:
    """A corrupt system.json yields environment=None, never an error."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    writer.write_result(make_result())
    (writer.bundle_dir / "system.json").write_text("{ not valid json", encoding="utf-8")

    loaded = BundleReader.read(writer.bundle_dir)
    assert loaded.environment is None
    assert loaded.result.experiment_id == "test-001"


def test_bundle_reader_ignores_pre_rename_environment_filename(tmp_path: Path) -> None:
    """Clean break: a sidecar under the pre-rename environment.json is NOT read.

    After the environment.json -> system.json rename the reader looks only for
    system.json. A bundle carrying the snapshot under the old name (and no
    system.json) loads with the system sidecar absent - no legacy fallback.
    """
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    writer = _writer(study_dir)
    writer.write_result(make_result())
    # Snapshot present ONLY under the pre-rename name; no system.json.
    write_container_system_sidecar(writer.bundle_dir / "environment.json")
    assert not (writer.bundle_dir / "system.json").exists()

    loaded = BundleReader.read(writer.bundle_dir)

    # The old-named file is invisible to the reader: no system artefact discovered.
    assert loaded.environment is None
    assert loaded.result.environment is None
    assert "system" not in loaded.paths


# ---------------------------------------------------------------------------
# BundleReader.read_sidecar - registry-driven single-artefact accessor
# ---------------------------------------------------------------------------


def test_read_sidecar_returns_config_payload(tmp_path: Path) -> None:
    """read_sidecar reads the config.json payload without loading result.json."""
    bundle_dir = _write_full_bundle(tmp_path)
    payload = BundleReader.read_sidecar(bundle_dir, "config")
    assert payload is not None
    assert payload["provenance"]["task.model"]["effective"] == "gpt2"


def test_read_sidecar_absent_returns_none(tmp_path: Path) -> None:
    """An absent sidecar returns None (not an error)."""
    empty = tmp_path / "bundle"
    empty.mkdir()
    assert BundleReader.read_sidecar(empty, "config") is None


def test_read_sidecar_corrupt_raises(tmp_path: Path) -> None:
    """A present-but-unparseable sidecar raises so callers can flag it a failure."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        BundleReader.read_sidecar(bundle, "config")


def test_read_sidecar_rejects_non_json_artefact(tmp_path: Path) -> None:
    """read_sidecar is JSON-only: a parquet artefact key is a programming error."""
    with pytest.raises(ValueError, match="not a JSON sidecar"):
        BundleReader.read_sidecar(tmp_path, "timeseries")
