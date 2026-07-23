"""Unit tests for v2.0 results persistence API.

Tests cover:
- Subdirectory creation and naming
- Atomic writes
- Collision handling
- Round-trip fidelity (datetime, tuple, nested models)
- Timeseries sidecar management
- Graceful degradation on missing sidecar
- Never-overwrite guarantee
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from llenergymeasure.domain.bundle_artefacts import BUNDLE_VERSION
from llenergymeasure.domain.environment import (
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    EnvironmentSnapshot,
    GPUEnvironment,
)
from llenergymeasure.domain.experiment import ExperimentResult, RunnerProvenance
from llenergymeasure.results.persistence import (
    load_result,
    save_environment,
)
from llenergymeasure.results.persistence import save_result as _persist_save_result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The authoritative home for model_name/engine is the config.json sidecar
# (result.json keeps convenience copies), so save_result takes them explicitly.
# This wrapper supplies gpt2/transformers defaults for the fixtures that do not
# exercise directory-slug specifics; the slug tests pass model_name explicitly.
_HF_MODEL = "meta-llama/Llama-3.1-8B"


def save_result(result, output_dir, *, model_name="gpt2", engine="transformers", **kwargs):
    return _persist_save_result(result, output_dir, model_name=model_name, engine=engine, **kwargs)


@pytest.fixture()
def minimal_result() -> ExperimentResult:
    """Minimal valid ExperimentResult."""
    return ExperimentResult(
        experiment_id="persist-test-001",
        measurement_config_hash="abcdef0123456789",
        input_tokens=384,
        output_tokens=128,
        total_tokens=512,
        total_energy_j=25.6,
        total_inference_time_sec=5.0,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=1e11,
        start_time=datetime(2026, 2, 26, 14, 0, 0),
        end_time=datetime(2026, 2, 26, 14, 0, 5),
    )


@pytest.fixture()
def hf_model_result() -> ExperimentResult:
    """ExperimentResult used with a HuggingFace-style model path containing /."""
    return ExperimentResult(
        experiment_id="persist-test-hf",
        measurement_config_hash="fedcba9876543210",
        input_tokens=768,
        output_tokens=256,
        total_tokens=1024,
        total_energy_j=51.2,
        total_inference_time_sec=10.0,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=2e11,
        start_time=datetime(2026, 2, 26, 14, 0, 0),
        end_time=datetime(2026, 2, 26, 14, 0, 10),
    )


@pytest.fixture()
def result_with_timeseries() -> ExperimentResult:
    """ExperimentResult that claims to have a timeseries sidecar."""
    return ExperimentResult(
        experiment_id="persist-test-ts",
        measurement_config_hash="1122334455667788",
        input_tokens=192,
        output_tokens=64,
        total_tokens=256,
        total_energy_j=12.8,
        total_inference_time_sec=2.5,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=5e10,
        timeseries="timeseries.parquet",
        start_time=datetime(2026, 2, 26, 15, 0, 0),
        end_time=datetime(2026, 2, 26, 15, 0, 2, 500000),
    )


def _create_temp_parquet(tmp_path: Path) -> Path:
    """Create a minimal parquet file for sidecar tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"timestamp_s": [0.0, 1.0], "gpu_power_w": [100.0, 105.0]})
    path = tmp_path / "temp_timeseries.parquet"
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# Directory creation and naming
# ---------------------------------------------------------------------------


def test_save_creates_subdirectory(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() creates a subdirectory containing result.json."""
    result_path = save_result(minimal_result, tmp_path)
    assert result_path.is_file()
    assert result_path.name == "result.json"
    # result.json is inside a subdirectory of tmp_path
    assert result_path.parent.parent == tmp_path


def test_save_directory_name_format(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Directory name matches c{cycle}_{model_short}-{engine}_{hash} pattern."""
    result_path = save_result(minimal_result, tmp_path)
    dir_name = result_path.parent.name
    # Pattern: c1_gpt2-pytorch_abcdef01 (c{cycle}_{model_short}-{engine}_{hash[:8]})
    pattern = re.compile(r"^c\d+_[\w\.\-]+-[a-z]+_[a-f0-9]{8,}(_\d+)?$")
    assert pattern.match(dir_name), f"Directory name '{dir_name}' does not match expected pattern"


def test_save_model_slug_normalisation(tmp_path: Path, hf_model_result: ExperimentResult) -> None:
    """HuggingFace model path strips org prefix, uses short model name."""
    result_path = save_result(hf_model_result, tmp_path, model_name=_HF_MODEL)
    dir_name = result_path.parent.name
    # meta-llama/Llama-3.1-8B -> Llama-3.1-8B (short name, no org prefix)
    assert "Llama-3.1-8B-transformers" in dir_name, f"Slug normalisation failed: {dir_name}"


def test_save_returns_result_json_path(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() return value ends with result.json."""
    result_path = save_result(minimal_result, tmp_path)
    assert result_path.name == "result.json"
    assert isinstance(result_path, Path)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_save_atomic_write(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """result.json is valid JSON after save_result()."""
    import json

    result_path = save_result(minimal_result, tmp_path)
    content = result_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["experiment_id"] == "persist-test-001"
    from tests.conftest import EXPERIMENT_BUNDLE_VERSION

    assert parsed["bundle_version"] == EXPERIMENT_BUNDLE_VERSION


# ---------------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------------


def test_collision_suffix_applied(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Second save to same base dir gets _1 suffix."""
    first = save_result(minimal_result, tmp_path)
    second = save_result(minimal_result, tmp_path)
    assert first != second
    assert "_1" in second.parent.name


def test_collision_multiple_suffixes(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Third save gets _2 (or _3 if first was _1 and _2 already exists)."""
    first = save_result(minimal_result, tmp_path)
    second = save_result(minimal_result, tmp_path)
    third = save_result(minimal_result, tmp_path)
    dirs = {first.parent, second.parent, third.parent}
    assert len(dirs) == 3, "Three saves should produce 3 distinct directories"


def test_never_overwrites(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """5 saves produce 5 distinct directories (never-overwrite guarantee)."""
    paths = [save_result(minimal_result, tmp_path) for _ in range(5)]
    dirs = [p.parent for p in paths]
    assert len(set(dirs)) == 5, f"Expected 5 distinct dirs, got {len(set(dirs))}: {dirs}"


# ---------------------------------------------------------------------------
# Load / round-trip
# ---------------------------------------------------------------------------


def test_from_json_loads_correctly(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """load_result() returns ExperimentResult with same field values."""
    result_path = save_result(minimal_result, tmp_path)
    loaded = load_result(result_path)

    assert loaded.experiment_id == minimal_result.experiment_id
    assert loaded.measurement_config_hash == minimal_result.measurement_config_hash
    from tests.conftest import EXPERIMENT_BUNDLE_VERSION

    assert loaded.bundle_version == EXPERIMENT_BUNDLE_VERSION
    assert loaded.input_tokens == minimal_result.input_tokens
    assert loaded.output_tokens == minimal_result.output_tokens
    assert loaded.total_tokens == minimal_result.total_tokens
    # Token split is consistent: total is the sum of the input/output halves.
    assert loaded.input_tokens + loaded.output_tokens == loaded.total_tokens
    assert loaded.total_energy_j == pytest.approx(minimal_result.total_energy_j)


def test_from_json_round_trip(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """model_dump_json() on loaded result matches original (datetime, tuple fidelity)."""
    result_path = save_result(minimal_result, tmp_path)
    loaded = load_result(result_path)

    original_json = minimal_result.model_dump_json(indent=2)
    loaded_json = loaded.model_dump_json(indent=2)
    assert original_json == loaded_json


def test_load_result_tolerates_legacy_schema_version(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """A pre-bundle_version result.json (carrying the retired ``schema_version``
    key instead of ``bundle_version``) still loads best-effort - the extra key is
    dropped rather than rejected by the forbid-extra model."""
    import json

    result_path = save_result(minimal_result, tmp_path)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    # Simulate a legacy bundle: retired per-artefact key, no bundle_version.
    raw.pop("bundle_version", None)
    raw["schema_version"] = "5.0"
    result_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_result(result_path)
    assert loaded.experiment_id == minimal_result.experiment_id
    # Falls back to the current default bundle_version (best-effort read).
    assert loaded.bundle_version == BUNDLE_VERSION


def test_convenience_identity_copies_round_trip(tmp_path: Path) -> None:
    """engine and model_name convenience copies survive save/load on result.json.

    Deliberate small duplication: the authoritative home is config.json, but
    result.json keeps the two fields so it is self-describing when separated
    from its directory.
    """
    result = ExperimentResult(
        experiment_id="persist-convenience-001",
        measurement_config_hash="abcdef0123456789",
        engine="vllm",
        model_name=_HF_MODEL,
        input_tokens=384,
        output_tokens=128,
        total_tokens=512,
        total_energy_j=25.6,
        total_inference_time_sec=5.0,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=1e12,
        start_time=datetime(2026, 2, 26, 14, 0, 0),
        end_time=datetime(2026, 2, 26, 14, 0, 5),
    )
    result_path = save_result(result, tmp_path, model_name=result.model_name, engine=result.engine)
    loaded = load_result(result_path)

    assert loaded.engine == "vllm"
    assert loaded.model_name == _HF_MODEL


# ---------------------------------------------------------------------------
# Runner provenance
# ---------------------------------------------------------------------------


def test_runner_provenance_docker_round_trips(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """A docker runner_provenance survives save/load and serialises into result.json."""
    import json

    result = minimal_result.model_copy(
        update={
            "runner_provenance": RunnerProvenance(
                mode="container",
                image="ghcr.io/example/transformers:1.0.0",
                source="yaml",
                image_source="registry",
            )
        }
    )
    result_path = save_result(result, tmp_path)

    # Persisted into result.json (NOT excluded, unlike environment).
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["runner_provenance"]["mode"] == "container"
    assert payload["runner_provenance"]["image"] == "ghcr.io/example/transformers:1.0.0"
    assert payload["runner_provenance"]["source"] == "yaml"

    loaded = load_result(result_path)
    assert loaded.runner_provenance is not None
    assert loaded.runner_provenance.mode == "container"
    assert loaded.runner_provenance.image == "ghcr.io/example/transformers:1.0.0"
    assert loaded.runner_provenance.source == "yaml"
    assert loaded.runner_provenance.image_source == "registry"


def test_runner_provenance_process_round_trips(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """A process runner_provenance survives save/load with no image."""
    result = minimal_result.model_copy(
        update={
            "runner_provenance": RunnerProvenance(mode="process", image=None, source="implicit")
        }
    )
    result_path = save_result(result, tmp_path)
    loaded = load_result(result_path)

    assert loaded.runner_provenance is not None
    assert loaded.runner_provenance.mode == "process"
    assert loaded.runner_provenance.image is None
    assert loaded.runner_provenance.source == "implicit"


# ---------------------------------------------------------------------------
# Timeseries sidecar
# ---------------------------------------------------------------------------


def test_save_with_timeseries_sidecar(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() with timeseries_source copies parquet file to output dir."""
    parquet_path = _create_temp_parquet(tmp_path)
    result_path = save_result(minimal_result, tmp_path, timeseries_source=parquet_path)
    sidecar = result_path.parent / "timeseries.parquet"
    assert sidecar.exists(), "timeseries.parquet should be copied to output dir"


def test_save_without_timeseries(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() with no timeseries_source produces no parquet file."""
    result_path = save_result(minimal_result, tmp_path)
    sidecar = result_path.parent / "timeseries.parquet"
    assert not sidecar.exists(), "No timeseries.parquet should exist when not provided"


def test_from_json_missing_sidecar_warns(
    tmp_path: Path,
    result_with_timeseries: ExperimentResult,
) -> None:
    """Loading result that claims sidecar but sidecar absent emits UserWarning."""
    # Save without providing the actual sidecar file
    result_path = save_result(result_with_timeseries, tmp_path)

    with pytest.warns(UserWarning, match="Timeseries sidecar missing"):
        loaded = load_result(result_path)

    # Result still loads
    assert loaded.experiment_id == result_with_timeseries.experiment_id


def test_from_json_missing_sidecar_loads_successfully(
    tmp_path: Path,
    result_with_timeseries: ExperimentResult,
) -> None:
    """Result loads with timeseries field preserved even when sidecar missing."""
    result_path = save_result(result_with_timeseries, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_result(result_path)

    # timeseries field is preserved (not set to None) - it's what was stored
    assert loaded.timeseries == "timeseries.parquet"
    # But the file itself is not there
    assert not (result_path.parent / "timeseries.parquet").exists()


# ---------------------------------------------------------------------------
# Cycle in directory name
# ---------------------------------------------------------------------------


def test_save_cycle_in_directory_name(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Directory name includes cycle number."""
    result_path = save_result(minimal_result, tmp_path, cycle=2)
    dir_name = result_path.parent.name
    assert dir_name.startswith("c2_"), f"Expected 'c2_' prefix in '{dir_name}'"


def test_save_experiment_index_in_directory_name(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """Directory name includes experiment index when provided."""
    result_path = save_result(minimal_result, tmp_path, experiment_index=5, cycle=3)
    dir_name = result_path.parent.name
    assert dir_name.startswith("005_c3_"), f"Expected '005_c3_' prefix in '{dir_name}'"


# ---------------------------------------------------------------------------
# Environment sidecar
# ---------------------------------------------------------------------------


@pytest.fixture()
def env_snapshot() -> EnvironmentSnapshot:
    """Minimal EnvironmentSnapshot for sidecar round-trip tests."""
    hardware = EnvironmentMetadata(
        gpu=GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920),
        cuda=CUDAEnvironment(driver_supported_version="12.4", driver_version="535.104"),
        cpu=CPUEnvironment(platform="Linux"),
        collected_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    return EnvironmentSnapshot(
        hardware=hardware,
        python_version="3.11.5",
        tool_version="0.9.0",
        cuda_version="12.4",
        cuda_version_source="torch",
    )


def test_load_result_attaches_environment_sidecar(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """load_result() reads back the environment.json sidecar onto result.environment."""
    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        env_snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )

    loaded = load_result(result_path)

    assert loaded.environment is not None
    assert loaded.environment.python_version == "3.11.5"
    assert loaded.environment.cuda_version == "12.4"
    assert loaded.environment.cuda_version_source == "torch"
    assert loaded.environment.hardware.gpu.name == "NVIDIA A100-SXM4-80GB"


def test_save_environment_includes_bundle_version(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """environment.json carries the single bundle_version stamp."""
    import json

    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        env_snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )
    payload = json.loads((result_path.parent / "environment.json").read_text())
    assert payload["bundle_version"] == BUNDLE_VERSION == "2.0"


def test_save_environment_writes_docker_runner_block_roundtrip(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """A docker runner block on the snapshot round-trips through save + load_result."""
    import json

    snapshot = env_snapshot.model_copy(
        update={
            "runner": RunnerProvenance(
                mode="container",
                image="ghcr.io/acme/vllm:1.19.0-cuda12",
                source="auto_detected",
                image_source="registry",
                image_digest="ghcr.io/acme/vllm@sha256:deadbeef",
            )
        }
    )
    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )

    # Raw JSON carries the runner block ...
    payload = json.loads((result_path.parent / "environment.json").read_text())
    assert payload["runner"] == {
        "mode": "container",
        "image": "ghcr.io/acme/vllm:1.19.0-cuda12",
        "source": "auto_detected",
        "image_source": "registry",
        "image_digest": "ghcr.io/acme/vllm@sha256:deadbeef",
    }

    # ... and load_result reconstructs it onto result.environment.runner.
    loaded = load_result(result_path)
    assert loaded.environment is not None
    assert loaded.environment.runner is not None
    assert loaded.environment.runner.mode == "container"
    assert loaded.environment.runner.image == "ghcr.io/acme/vllm:1.19.0-cuda12"
    assert loaded.environment.runner.image_digest == "ghcr.io/acme/vllm@sha256:deadbeef"
    assert loaded.environment.runner.source == "auto_detected"


def test_save_environment_writes_local_runner_block_roundtrip(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """A process runner block records mode=process with no image or digest."""
    snapshot = env_snapshot.model_copy(
        update={"runner": RunnerProvenance(mode="process", source="default")}
    )
    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )

    loaded = load_result(result_path)
    assert loaded.environment is not None
    assert loaded.environment.runner is not None
    assert loaded.environment.runner.mode == "process"
    assert loaded.environment.runner.image is None
    assert loaded.environment.runner.image_digest is None
    assert loaded.environment.runner.source == "default"


def test_load_result_maps_legacy_runner_vocabulary(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """A legacy 09ec455e-era 2.0 bundle (mode='docker', source='local') still reads.

    The runner-mode vocabulary was renamed local->process / docker->container, and
    the no-spec source sentinel local->implicit, within the SAME untagged
    bundle_version 2.0 stamp, so a reader keying on bundle_version alone cannot tell
    an old block apart. The RunnerProvenance read-path alias validator maps the
    legacy values onto the canonical ones so those bundles parse without a version
    bump.
    """
    import json

    snapshot = env_snapshot.model_copy(
        update={
            "runner": RunnerProvenance(
                mode="container", image="img:1.0", source="yaml", image_source="registry"
            )
        }
    )
    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )

    # Rewrite the on-disk runner block to the LEGACY vocabulary (mode="docker",
    # source="local"), simulating a bundle written before the SM2 rename.
    env_path = result_path.parent / "environment.json"
    payload = json.loads(env_path.read_text())
    payload["runner"]["mode"] = "docker"
    payload["runner"]["source"] = "local"
    env_path.write_text(json.dumps(payload))

    loaded = load_result(result_path)
    assert loaded.environment is not None
    assert loaded.environment.runner is not None
    # Legacy "docker" maps onto the canonical "container", and the legacy source
    # sentinel "local" maps onto "implicit", on read.
    assert loaded.environment.runner.mode == "container"
    assert loaded.environment.runner.source == "implicit"
    assert loaded.environment.runner.image == "img:1.0"


def test_runner_provenance_model_maps_legacy_vocabulary_on_read() -> None:
    """RunnerProvenance.model_validate maps legacy mode + source values to the canon."""
    assert RunnerProvenance.model_validate({"mode": "docker"}).mode == "container"
    assert RunnerProvenance.model_validate({"mode": "local"}).mode == "process"
    # The no-spec source sentinel "local" maps to "implicit".
    assert (
        RunnerProvenance.model_validate({"mode": "process", "source": "local"}).source == "implicit"
    )
    # Canonical values pass through unchanged (idempotent).
    assert RunnerProvenance.model_validate({"mode": "container"}).mode == "container"
    assert RunnerProvenance.model_validate({"mode": "process"}).mode == "process"
    assert (
        RunnerProvenance.model_validate({"mode": "process", "source": "implicit"}).source
        == "implicit"
    )
    # A real precedence source is never rewritten.
    assert (
        RunnerProvenance.model_validate({"mode": "process", "source": "default"}).source
        == "default"
    )


def test_save_environment_runner_absent_when_snapshot_has_none(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """A snapshot without a runner block writes runner: null and loads back as None."""
    import json

    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        env_snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )
    payload = json.loads((result_path.parent / "environment.json").read_text())
    assert payload["runner"] is None
    loaded = load_result(result_path)
    assert loaded.environment is not None
    assert loaded.environment.runner is None


def test_load_result_without_environment_sidecar(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """load_result() leaves environment as None when no sidecar is present (best-effort)."""
    result_path = save_result(minimal_result, tmp_path)
    loaded = load_result(result_path)
    assert loaded.environment is None


def test_load_result_corrupt_environment_sidecar(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """A corrupt environment.json must not break load_result (best-effort)."""
    result_path = save_result(minimal_result, tmp_path)
    (result_path.parent / "environment.json").write_text("{ not valid json", encoding="utf-8")

    loaded = load_result(result_path)

    assert loaded.environment is None
    assert loaded.experiment_id == minimal_result.experiment_id


def test_environment_field_excluded_from_result_json(
    tmp_path: Path,
    minimal_result: ExperimentResult,
    env_snapshot: EnvironmentSnapshot,
) -> None:
    """environment is loader-only - it never serialises back into result.json."""
    result_path = save_result(minimal_result, tmp_path)
    save_environment(
        env_snapshot,
        minimal_result.experiment_id,
        minimal_result.measurement_config_hash,
        result_path.parent,
    )

    loaded = load_result(result_path)
    assert loaded.environment is not None
    # model_dump_json drops the excluded field, so the re-serialised result
    # matches the on-disk result.json (no environment leakage).
    assert "environment" not in loaded.model_dump_json()


def test_save_config_sidecar_roundtrips_declared_config(tmp_path: Path) -> None:
    """The declared config survives a write/read round trip verbatim."""
    import json

    from llenergymeasure.results.persistence import save_config_sidecar

    declared = {
        "engine": "vllm",
        "task": {"model": "fake/model", "seed": 7},
        "vllm": {"engine_params": {"enforce_eager": True}},
    }
    path = save_config_sidecar(
        tmp_path,
        experiment_id="exp-1",
        config_hash="a" * 16,
        engine="vllm",
        engine_version="0.19.1",
        model_name="fake/model",
        measurement_methodology="total",
        observed_config_hash="b" * 64,
        declared_config=declared,
    )

    payload = json.loads(path.read_text())
    assert payload["declared_config"] == declared
    assert payload["engine_version"] == "0.19.1"
    assert payload["model_name"] == "fake/model"
    assert payload["measurement_methodology"] == "total"


def test_save_config_sidecar_omits_declared_config_when_absent(tmp_path: Path) -> None:
    """A call without declared_config omits the key (not written as null)."""
    import json

    from llenergymeasure.results.persistence import save_config_sidecar

    path = save_config_sidecar(
        tmp_path,
        experiment_id="exp-1",
        config_hash="a" * 16,
        engine="vllm",
        engine_version="0.19.1",
        model_name="fake/model",
        measurement_methodology="total",
        observed_config_hash="b" * 64,
    )

    payload = json.loads(path.read_text())
    assert "declared_config" not in payload
    # steady_state_window/discard_fraction omitted when None (total methodology).
    assert "steady_state_window" not in payload
    assert "measurement_window_discard_fraction" not in payload


def test_save_config_sidecar_includes_bundle_version(tmp_path: Path) -> None:
    """The config.json sidecar carries the single bundle_version stamp."""
    import json

    from llenergymeasure.results.persistence import save_config_sidecar

    path = save_config_sidecar(
        tmp_path,
        experiment_id="exp-1",
        config_hash="a" * 16,
        engine="vllm",
        engine_version="0.19.1",
        model_name="fake/model",
        measurement_methodology="total",
    )

    payload = json.loads(path.read_text())
    assert payload["bundle_version"] == BUNDLE_VERSION == "2.0"


def test_save_config_sidecar_carries_methodology_window(tmp_path: Path) -> None:
    """steady_state methodology writes the window + discard fraction + not-detected flag."""
    import json

    from llenergymeasure.results.persistence import save_config_sidecar

    path = save_config_sidecar(
        tmp_path,
        experiment_id="exp-1",
        config_hash="a" * 16,
        engine="transformers",
        engine_version="4.56.0",
        model_name="gpt2",
        measurement_methodology="steady_state",
        steady_state_window=(1.0, 4.0),
        measurement_window_discard_fraction=0.1,
        steady_state_not_detected=True,
    )

    payload = json.loads(path.read_text())
    assert payload["measurement_methodology"] == "steady_state"
    assert payload["steady_state_window"] == [1.0, 4.0]
    assert payload["measurement_window_discard_fraction"] == pytest.approx(0.1)
    assert payload["steady_state_not_detected"] is True


def test_save_result_does_not_write_resolution_sidecar(
    minimal_result: ExperimentResult, tmp_path: Path
) -> None:
    """The retired _resolution.json sidecar is never written by save_result.

    Per-field provenance now lives in the config.json sidecar's provenance
    section, folded in by the study layer.
    """
    result_path = save_result(minimal_result, tmp_path)
    assert not (result_path.parent / "_resolution.json").exists()
