"""Tests for the equivalence-groups sidecar writer + post-run observed-collision grouping."""

from __future__ import annotations

import json
from pathlib import Path

from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    ObservedCollisionGroup,
    PreRunGroup,
    find_observed_collisions,
    write_equivalence_groups,
)


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
                    library_version="4.56.0",
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
                "library_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
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
                "library_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
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
                "library_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_a",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
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
                "library_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.57.0",  # Different version
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
                "library_version": "4.56.0",
                "resolved_config_hash": "resolved_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "resolved_config_hash": "resolved_b",
                "observed_config_hash": "observed_shared",
                "experiment_id": "exp_b",
            },
        ]
        # Only one valid sidecar; no group formed.
        groups = find_observed_collisions(sidecars)
        assert groups == []
