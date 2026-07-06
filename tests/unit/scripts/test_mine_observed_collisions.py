"""Tests for :mod:`scripts.mine_observed_collisions`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import mine_observed_collisions as miner  # noqa: E402


def _only_field(candidate: dict[str, Any]) -> str:
    """The single canonical field path a collision candidate constrains."""
    fields = list(candidate["match"]["fields"])
    assert len(fields) == 1
    return fields[0]


def _declared(eager: bool, seed: int | None = None) -> dict[str, Any]:
    """A small declared ExperimentConfig varying enforce_eager (and maybe seed)."""
    cfg: dict[str, Any] = {"engine": "vllm", "vllm": {"engine_params": {"enforce_eager": eager}}}
    if seed is not None:
        cfg["task"] = {"seed": seed}
    return cfg


def _write_sidecar(
    study_dir: Path,
    name: str,
    *,
    observed_hash: str | None = "a" * 64,
    declared: dict[str, Any] | None = None,
) -> None:
    """Fabricate one experiment directory with a config.json sidecar."""
    exp_dir = study_dir / name
    exp_dir.mkdir(parents=True)
    payload: dict[str, Any] = {"experiment_id": name, "engine": "vllm", "library_version": "0.19.1"}
    if observed_hash is not None:
        payload["observed_config_hash"] = observed_hash
    if declared is not None:
        payload["declared_config"] = declared
    (exp_dir / "config.json").write_text(json.dumps(payload, indent=2))


def _mine(study_dir: Path) -> tuple[list[dict[str, Any]], miner.ScanStats]:
    sidecars, stats = miner.load_sidecars(study_dir)
    return miner.mine_collisions(sidecars, stats), stats


def test_clean_two_way_collision_names_the_field(tmp_path: Path) -> None:
    """Declared differs on one field, observed identical -> that field is emitted."""
    _write_sidecar(tmp_path, "001_exp", declared=_declared(eager=True))
    _write_sidecar(tmp_path, "002_exp", declared=_declared(eager=False))

    candidates, stats = _mine(tmp_path)

    assert stats.collision_groups == 1
    assert len(candidates) == 1
    candidate = candidates[0]
    assert _only_field(candidate) == "vllm.engine_params.enforce_eager"
    assert candidate["match"]["fields"]["vllm.engine_params.enforce_eager"] == {"present": True}
    assert candidate["severity"] == "dormant"
    assert candidate["verdict"]["status"] == "unverified"
    prov = candidate["provenance"]
    assert prov["source"] == "observed_collision"
    assert prov["co_occurring_fields"] == []
    assert prov["colliding_experiments"] == ["001_exp", "002_exp"]
    assert sorted(prov["declared_values"], key=str) == [False, True]


def test_non_collision_emits_nothing(tmp_path: Path) -> None:
    """Different observed hashes never group, so no candidate is proposed."""
    _write_sidecar(tmp_path, "001_exp", observed_hash="a" * 64, declared=_declared(eager=True))
    _write_sidecar(tmp_path, "002_exp", observed_hash="b" * 64, declared=_declared(eager=False))

    candidates, stats = _mine(tmp_path)

    assert candidates == []
    assert stats.collision_groups == 0


def test_multi_field_diff_flags_co_occurrence(tmp_path: Path) -> None:
    """Two fields varying together yield one candidate each, cross-referenced."""
    _write_sidecar(tmp_path, "001_exp", declared=_declared(eager=True, seed=1))
    _write_sidecar(tmp_path, "002_exp", declared=_declared(eager=False, seed=2))

    candidates, stats = _mine(tmp_path)

    assert stats.collision_groups == 1
    by_field = {_only_field(c): c for c in candidates}
    assert set(by_field) == {"task.seed", "vllm.engine_params.enforce_eager"}
    assert by_field["task.seed"]["provenance"]["co_occurring_fields"] == [
        "vllm.engine_params.enforce_eager"
    ]
    assert by_field["vllm.engine_params.enforce_eager"]["provenance"]["co_occurring_fields"] == [
        "task.seed"
    ]


def test_identical_declared_configs_are_not_a_collision(tmp_path: Path) -> None:
    """Repeat runs (same declared config, same observed hash) emit nothing."""
    _write_sidecar(tmp_path, "001_c1", declared=_declared(eager=True))
    _write_sidecar(tmp_path, "001_c2", declared=_declared(eager=True))

    candidates, stats = _mine(tmp_path)

    assert candidates == []
    assert stats.collision_groups == 0


def test_legacy_sidecar_without_declared_config_is_skipped(tmp_path: Path) -> None:
    """A pre-field sidecar is counted and skipped, not crashed on."""
    _write_sidecar(tmp_path, "001_exp", declared=_declared(eager=True))
    _write_sidecar(tmp_path, "002_legacy", declared=None)

    candidates, stats = _mine(tmp_path)

    assert stats.experiments_scanned == 2
    assert stats.skipped_missing_declared == 1
    # Only one usable member remains in the group, so nothing is emitted.
    assert candidates == []


def test_group_of_three_with_single_varying_field(tmp_path: Path) -> None:
    """A three-member group over one differing field yields one candidate."""
    for idx, eager in enumerate([True, False, True], start=1):
        _write_sidecar(tmp_path, f"00{idx}_exp", declared=_declared(eager=eager))

    candidates, stats = _mine(tmp_path)

    assert stats.collision_groups == 1
    assert len(candidates) == 1
    assert _only_field(candidates[0]) == "vllm.engine_params.enforce_eager"
    assert len(candidates[0]["provenance"]["colliding_experiments"]) == 3


def test_write_candidates_targets_version_scoped_pool(tmp_path: Path) -> None:
    """Candidates land under <pool>/<engine>/v<version>/candidates/, never in src/."""
    _write_sidecar(tmp_path, "001_exp", declared=_declared(eager=True))
    _write_sidecar(tmp_path, "002_exp", declared=_declared(eager=False))
    candidates, _ = _mine(tmp_path)

    pool_root = tmp_path / "pool"
    written = miner.write_candidates(candidates, pool_root, generated_at="2026-07-06")

    expected = pool_root / "vllm" / "v0_19_1" / "candidates" / "observed_collisions.yaml"
    assert written == [expected]
    document = yaml.safe_load(expected.read_text())
    assert document["source"] == "observed_collision"
    assert document["engine"] == "vllm"
    assert document["engine_version"] == "0.19.1"
    assert len(document["candidates"]) == 1


def test_main_end_to_end(tmp_path: Path) -> None:
    """main() scans, writes to the pool, and returns 0."""
    study = tmp_path / "study"
    _write_sidecar(study, "001_exp", declared=_declared(eager=True))
    _write_sidecar(study, "002_exp", declared=_declared(eager=False))
    pool_root = tmp_path / "pool"

    assert miner.main([str(study), "--pool-root", str(pool_root)]) == 0
    assert (pool_root / "vllm" / "v0_19_1" / "candidates" / "observed_collisions.yaml").exists()


def test_main_missing_study_dir_returns_2(tmp_path: Path) -> None:
    """A nonexistent study directory is a usage error, not a crash."""
    assert miner.main([str(tmp_path / "nope")]) == 2
