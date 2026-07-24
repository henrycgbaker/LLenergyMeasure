"""Tests for the equivalence-groups sidecar writer + post-run observed-collision grouping."""

from __future__ import annotations

import json
from pathlib import Path

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.hashing import build_observed_view, hash_config
from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    ObservedCollisionGroup,
    PreRunGroup,
    find_observed_collisions,
    write_equivalence_groups,
)
from llenergymeasure.study.hashing import build_resolved_view


class TestWriteSerialisation:
    def test_write_emits_all_fields(self, tmp_path: Path):
        groups = EquivalenceGroups(
            study_id="study_abc",
            study_name="gpt2-sweep",
            dedup_mode="resolved",
            validated_rules_version="transformers:4.56.0@deadbee",
            groups=[
                PreRunGroup(
                    resolved_config_hash="sha256:abc",
                    canonical_config_excerpt={"engine": "transformers"},
                    member_experiment_ids=("exp_0001", "exp_0002"),
                    member_count=2,
                    representative_experiment_id="exp_0001",
                    would_dedup=True,
                    deduplicated=True,
                )
            ],
            observed_collision_groups=[
                ObservedCollisionGroup(
                    observed_config_hash="sha256:def",
                    engine="transformers",
                    engine_version="4.56.0",
                    member_resolved_config_hashes=("sha256:abc", "sha256:xyz"),
                    member_experiment_ids=("exp_0001", "exp_0003"),
                    gap_detected=True,
                )
            ],
        )
        path = tmp_path / "equivalence_groups.json"
        write_equivalence_groups(groups, path)
        loaded = json.loads(path.read_text())

        assert loaded["study_id"] == "study_abc"
        assert loaded["study_name"] == "gpt2-sweep"
        assert loaded["dedup_mode"] == "resolved"
        assert loaded["validated_rules_version"].startswith("transformers:")
        assert len(loaded["groups"]) == 1
        assert loaded["groups"][0]["member_count"] == 2
        assert loaded["groups"][0]["would_dedup"] is True
        assert len(loaded["observed_collision_groups"]) == 1
        assert loaded["observed_collision_groups"][0]["gap_detected"] is True


class TestFindObservedCollisions:
    def test_flags_gap_when_same_observed_distinct_resolved(self):
        sidecars = [
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_b",  # Distinct resolved - gap!
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_observed_collisions(sidecars)
        assert len(groups) == 1
        assert groups[0].gap_detected is True

    def test_no_flag_when_resolved_same(self):
        # Same resolved-config collapsing on the library side is not a gap - the
        # library-resolution mechanism already saw them as equivalent.
        sidecars = [
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_observed_collisions(sidecars)
        assert len(groups) == 1
        assert groups[0].gap_detected is False

    def test_distinct_observed_no_group(self):
        sidecars = [
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_a",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_b",
                "observed_config_hash": "observed_b",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_observed_collisions(sidecars)
        assert groups == []

    def test_different_versions_do_not_cross_groups(self):
        sidecars = [
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "engine_version": "4.57.0",  # Different version
                "resolved_config_hash": "resolved_b",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_observed_collisions(sidecars)
        # Version mismatch - grouped separately, each with len=1, so no gap.
        assert groups == []

    def test_sidecar_missing_observed_skipped(self):
        sidecars = [
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
            },
            {
                "engine": "transformers",
                "engine_version": "4.56.0",
                "resolved_config_hash": "resolved_b",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_b",
            },
        ]
        # Only one valid sidecar; no group formed.
        groups = find_observed_collisions(sidecars)
        assert groups == []

    def test_coerced_numeric_types_do_not_manufacture_false_gap(self):
        # Two sweep points that both mean cpu_offload_gb=0 - one left the int
        # literal default, one set 0.0 explicitly - and both produced the same
        # native engine object (float-coerced). After int/float unification
        # their resolved hashes match, so the shared observed hash is NOT
        # flagged as a library-resolution gap. Pre-fix the resolved hashes
        # differed (int 0 vs float 0.0) and this was a false-positive gap fed
        # into the rules-corpus mining loop.
        cfg_int = ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            vllm={"engine_params": {}},
            serving_mode="offline",
        )
        cfg_float = ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            vllm={"engine_params": {"cpu_offload_gb": 0.0}},
            serving_mode="offline",
        )
        resolved_int = hash_config(build_resolved_view(cfg_int))
        resolved_float = hash_config(build_resolved_view(cfg_float))
        assert resolved_int == resolved_float  # unified by the fix

        # A native engine object carrying the coerced float 0.0 hashes the same
        # as one carrying int 0 over an identical field set - coerced numeric
        # types collide with their declared counterparts.
        task = cfg_int.task.model_dump(mode="python")
        observed_from_float = hash_config(
            build_observed_view(
                engine="vllm",
                task=task,
                observed_engine_params={"cpu_offload_gb": 0.0},
                observed_sampling_params={},
            )
        )
        observed_from_int = hash_config(
            build_observed_view(
                engine="vllm",
                task=task,
                observed_engine_params={"cpu_offload_gb": 0},
                observed_sampling_params={},
            )
        )
        assert observed_from_float == observed_from_int

        sidecars = [
            {
                "engine": "vllm",
                "engine_version": "0.11.0",
                "resolved_config_hash": resolved_int,
                "observed_config_hash": observed_from_float,
                "experiment_id": "exp_int",
            },
            {
                "engine": "vllm",
                "engine_version": "0.11.0",
                "resolved_config_hash": resolved_float,
                "observed_config_hash": observed_from_float,
                "experiment_id": "exp_float",
            },
        ]
        groups = find_observed_collisions(sidecars)
        assert len(groups) == 1
        assert groups[0].gap_detected is False
