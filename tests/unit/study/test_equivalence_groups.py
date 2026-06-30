"""Tests for the equivalence-groups sidecar writer + post-run observed-collision grouping."""

from __future__ import annotations

from pathlib import Path

from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    ObservedCollisionGroup,
    PreRunGroup,
    _pre_from_dict,
    find_observed_collisions,
    load_equivalence_groups,
    write_equivalence_groups,
)


class TestRoundTripSerialisation:
    def test_write_then_load_preserves_fields(self, tmp_path: Path):
        groups = EquivalenceGroups(
            study_id="study_abc",
            dedup_mode="resolved",
            validated_invariants_version="transformers:4.56.0@deadbee",
            groups=[
                PreRunGroup(
                    resolved_config_hash="sha256:abc",
                    canonical_config_excerpt={"engine": "transformers"},
                    member_indices=(0, 2),
                    member_count=2,
                    representative_index=0,
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
                    proposed_invariant_id="candidate_invariant_1",
                )
            ],
        )
        path = tmp_path / "equivalence_groups.json"
        write_equivalence_groups(groups, path)
        loaded = load_equivalence_groups(path)

        assert loaded.study_id == "study_abc"
        assert loaded.dedup_mode == "resolved"
        assert loaded.validated_invariants_version.startswith("transformers:")
        assert len(loaded.groups) == 1
        assert loaded.groups[0].member_count == 2
        assert loaded.groups[0].member_indices == (0, 2)
        assert loaded.groups[0].representative_index == 0
        assert loaded.groups[0].would_dedup is True
        assert len(loaded.observed_collision_groups) == 1
        assert loaded.observed_collision_groups[0].gap_detected is True


class TestFinaliseStudyDictShape:
    def test_deserialises_finalise_study_serialisation(self):
        """The runner reconstructs ``PreRunGroup`` from the exact dict shape
        ``finalise_study`` writes into ``StudyConfig.pre_run_equivalence_groups``.

        Regression guard: the two sides keyed on different fields
        (``member_indices`` written vs ``member_experiment_ids`` read), so every
        reconstructed group silently lost its members.
        """
        serialised = {
            "resolved_config_hash": "sha256:abc",
            "canonical_config_excerpt": {"engine": "transformers"},
            "member_indices": [0, 2, 5],
            "member_count": 3,
            "representative_index": 0,
            "would_dedup": True,
            "deduplicated": True,
        }
        group = _pre_from_dict(serialised)
        assert group.member_indices == (0, 2, 5)
        assert group.representative_index == 0
        assert group.member_count == 3


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
