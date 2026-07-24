"""Pytest configuration and shared fixtures for v2.0 tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    StudyResult,
)

_REPLAY_DIR = Path(__file__).parent / "fixtures" / "replay"

# ---------------------------------------------------------------------------
# Shared test constants - single source of truth for magic values
# ---------------------------------------------------------------------------

TEST_MODEL = "gpt2"
TEST_ENGINE = "transformers"
TEST_EXPERIMENT_ID = "test-001"
TEST_CONFIG_HASH = "deadbeef12345678"
TEST_DECLARED_CONFIG_HASH = "abc123def4567890"
TEST_POWER_MW = 200_000  # 200 W in milliwatts (pynvml convention)
TEST_POWER_W = 200.0

# Derived from model defaults - single source of truth for schema assertions
EXPERIMENT_BUNDLE_VERSION = ExperimentResult.model_fields["bundle_version"].default

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_EPOCH_END = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def make_config(**overrides) -> ExperimentConfig:
    """Return a valid ExperimentConfig with sensible defaults.

    Tests override only what they care about. Task-level fields (model, dataset,
    max_input_tokens, max_output_tokens, random_seed) are routed into task={}.
    A top-level dtype= kwarg is routed into the active engine's config section.
    All three engines use the generated nested Config, so dtype nests under
    engine_params.
    """
    _TASK_FIELDS = {"model", "dataset", "max_input_tokens", "max_output_tokens", "random_seed"}
    _MEASUREMENT_FIELDS = {"warmup", "baseline", "energy_sampler", "latency_profiling"}

    dtype = overrides.pop("dtype", None)

    task_defaults: dict = {"model": TEST_MODEL}
    ec_defaults: dict = {"engine": TEST_ENGINE}

    task_overrides: dict = {}
    measurement_overrides: dict = {}
    ec_overrides: dict = {}

    for key, value in overrides.items():
        if key in _TASK_FIELDS:
            task_overrides[key] = value
        elif key in _MEASUREMENT_FIELDS:
            measurement_overrides[key] = value
        else:
            ec_overrides[key] = value

    task = {**task_defaults, **task_overrides}
    ec = {**ec_defaults, **ec_overrides, "task": task}
    if measurement_overrides:
        ec["measurement"] = measurement_overrides

    if dtype is not None:
        engine_name = ec.get("engine", TEST_ENGINE)
        engine_key = engine_name.value if hasattr(engine_name, "value") else str(engine_name)
        # Nested generated Config (all engines): dtype lives on engine_params.
        existing = ec.get(engine_key)
        existing = existing if isinstance(existing, dict) else {}
        engine_params = {**existing.get("engine_params", {}), "dtype": dtype}
        ec[engine_key] = {**existing, "engine_params": engine_params}

    return ExperimentConfig(**ec)


def make_result(**overrides) -> ExperimentResult:
    """Return a valid ExperimentResult with sensible defaults.

    Includes all required fields (declared_config_hash, start_time, end_time)
    to prevent ValidationError.
    """
    defaults: dict = {
        "experiment_id": TEST_EXPERIMENT_ID,
        "declared_config_hash": TEST_DECLARED_CONFIG_HASH,
        "aggregation": AggregationMetadata(num_processes=1),
        "input_tokens": 800,
        "output_tokens": 200,
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


def make_study_result(**overrides) -> StudyResult:
    """Return a valid StudyResult with sensible defaults.

    Needed by CLI tests (Plan 03) and E2E tests (Plan 04).
    Tests override only what they care about.
    """
    from llenergymeasure.domain.experiment import StudySummary

    one_result = make_result()
    defaults: dict = {
        "study_name": "test-study",
        "experiments": [one_result],
        "summary": StudySummary(
            total_experiments=1,
            completed=1,
            failed=0,
            total_wall_time_s=5.0,
            total_energy_j=10.0,
        ),
        "result_files": [],
    }
    defaults.update(overrides)
    return StudyResult(**defaults)


def make_study(engines: list[str]) -> StudyConfig:
    """Build a minimal StudyConfig with one experiment per engine name.

    Shared by the pre-flight tests that only care about which engines a
    study spans (single vs multi-engine runner-elevation paths).
    """
    experiments = [ExperimentConfig(task={"model": f"model-{e}"}, engine=e) for e in engines]
    return StudyConfig(
        experiments=experiments,
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
    )


def make_user_config(**overrides):
    """Return a minimal mock UserConfig for tests that need load_user_config.

    Uses real Pydantic models to avoid fragile anonymous-type hacks.
    """
    from llenergymeasure.config.user_config import UserConfig

    defaults: dict = {}
    defaults.update(overrides)
    return UserConfig(**defaults)


def write_container_environment_sidecar(path: Path) -> dict:
    """Write a rescued in-container environment.json (distinct CONTAINER values).

    Shared by the docker-rescue tests: distinct hardware/runtime values so a test
    can prove the container snapshot wins over the dispatching-host snapshot.
    """
    import json

    payload = {
        "experiment_id": "test-save-record-001",
        "declared_config_hash": "aabb1122ccdd3344",
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


@pytest.fixture
def replay_results() -> list[ExperimentResult]:
    """Load GPU-produced ExperimentResult fixtures from tests/fixtures/replay/.

    Returns an empty list when no fixtures exist (safe for GPU-free CI).
    Uses model_validate_json directly (not from_json) because replay fixtures
    are standalone JSON files without timeseries sidecars.
    """
    results = []
    if _REPLAY_DIR.is_dir():
        for json_file in sorted(_REPLAY_DIR.glob("*.json")):
            content = json_file.read_text(encoding="utf-8")
            results.append(ExperimentResult.model_validate_json(content))
    return results


# ---------------------------------------------------------------------------
# Autouse fixtures: module-level singleton cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_gpu_locks(monkeypatch):
    """Prevent real GPU advisory locks from interfering with tests.

    Tests that specifically exercise gpu_locks can override this by
    importing and calling the real functions directly.
    """
    monkeypatch.setattr("llenergymeasure.study.gpu_locks.acquire_gpu_locks", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_locks.release_gpu_locks", lambda *_a, **_kw: None
    )


@pytest.fixture(autouse=True)
def clear_baseline_cache():
    """Clear _baseline_cache before and after each test.

    Prevents baseline measurement state from leaking between tests, which is
    especially important when pytest-randomly changes execution order.
    """
    from llenergymeasure.harness.baseline import _baseline_cache

    _baseline_cache.clear()
    yield
    _baseline_cache.clear()


@pytest.fixture(autouse=True)
def reset_lru_caches():
    """Clear lru_cache / functools.cache decorated functions between tests.

    Only clears caches that are known to produce order-dependent results
    (i.e. caches that depend on host environment state).
    """
    try:
        from llenergymeasure.infra.runner_resolution import is_docker_available

        if hasattr(is_docker_available, "cache_clear"):
            is_docker_available.cache_clear()
    except ImportError:
        pass

    yield

    try:
        from llenergymeasure.infra.runner_resolution import is_docker_available

        if hasattr(is_docker_available, "cache_clear"):
            is_docker_available.cache_clear()
    except ImportError:
        pass
