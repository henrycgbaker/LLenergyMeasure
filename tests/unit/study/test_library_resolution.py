"""Tests for the sweep library-resolution mechanism - fixpoint iteration + dedup.

Idempotence + shuffle-stability are enforced by
``scripts/miners/_fixpoint_test.py`` (CI-time contract - any corpus PR that
violates them is rejected before this module runs). These tests focus on
the *runtime* behaviour: does the library-resolution mechanism reach fixpoint, does it
collapse measurement-equivalent configs, does it detect cycles, does it
populate equivalence-group metadata.
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.engine_rules.loader import Invariant
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.study.hashing import build_resolved_view, hash_config
from llenergymeasure.study.library_resolution import (
    LibraryResolutionCycleError,
    _apply_invariants_fixpoint,
    resolve_library_effective,
)


def _mk_rule(
    *,
    invariant_id: str,
    match_fields: dict,
    normalised_fields: list[str] | None = None,
    severity: str = "dormant",
    engine: str = "transformers",
) -> Invariant:
    """Construct a minimal ``Invariant`` for library-resolution mechanism tests."""
    return Invariant(
        id=invariant_id,
        engine=engine,
        library=engine,
        invariant_under_test="",
        severity=severity,
        native_type=f"{engine}.NativeConfig",
        match_engine=engine,
        match_fields=match_fields,
        kwargs_positive={},
        kwargs_negative={},
        expected_outcome={"normalised_fields": normalised_fields or []},
        message_template=None,
        miner_source={},
        references=(),
        added_by="test",
        added_at="2026-04-23",
    )


def _mk_config(**overrides):
    base = {"task": {"model": "gpt2"}, "engine": "transformers"}
    base.update(overrides)
    return ExperimentConfig(**base)


# ---------------------------------------------------------------------------
# _apply_invariants_fixpoint()
# ---------------------------------------------------------------------------


class TestCanonicalise:
    def test_empty_invariants_returns_deep_copy(self):
        cfg = _mk_config()
        result = _apply_invariants_fixpoint(cfg, [])
        assert result is not cfg
        assert result.model_dump() == cfg.model_dump()

    def test_no_match_no_change(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": True, "temperature": 0.5}})
        invariant = _mk_rule(
            invariant_id="never_fires",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_invariants_fixpoint(cfg, [invariant])
        # do_sample=True disqualifies the invariant; temperature should be untouched.
        assert result.transformers.sampling_params.temperature == 0.5

    def test_greedy_normalises_temperature_via_not_equal(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": False, "temperature": 0.5}})
        invariant = _mk_rule(
            invariant_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_invariants_fixpoint(cfg, [invariant])
        # Canonical form is the not_equal sentinel.
        assert result.transformers.sampling_params.temperature == 1.0

    def test_idempotence_on_already_canonical(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": False, "temperature": 1.0}})
        invariant = _mk_rule(
            invariant_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_invariants_fixpoint(cfg, [invariant])
        assert result.transformers.sampling_params.temperature == 1.0

    def test_chained_rules_converge(self):
        # Invariant A: if temperature > 0.8, strip top_p (simulating greedy-when-hot
        # semantics - imaginary for this test).
        # Invariant B: if top_p is None, strip top_k.
        # Input with temperature 0.9, top_p=0.95, top_k=50 must converge in 2 passes.
        cfg = _mk_config(
            transformers={
                "sampling_params": {
                    "do_sample": True,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 50,
                }
            }
        )
        invariant_a = _mk_rule(
            invariant_id="hot_strips_top_p",
            match_fields={
                "transformers.sampling_params.temperature": {">": 0.8},
                "transformers.sampling_params.top_p": {"present": True},
            },
        )
        invariant_b = _mk_rule(
            invariant_id="no_top_p_strips_top_k",
            match_fields={
                "transformers.sampling_params.top_p": {"absent": True},
                "transformers.sampling_params.top_k": {"present": True, "not_equal": 50},
            },
        )
        result = _apply_invariants_fixpoint(cfg, [invariant_a, invariant_b])
        assert result.transformers.sampling_params.top_p is None
        # top_k unchanged: invariant_b only fires when top_k != 50, but input is 50.
        assert result.transformers.sampling_params.top_k == 50

    def test_cycle_detection(self):
        # Synthetic cycle: two invariants that flip a field back and forth.
        # invariant_a: if top_k=50, strip it
        # invariant_b: if top_k absent, set to 50 (via not_equal on present)
        # The real invariant shape is constrained; simulate a cycle by assigning a
        # path that gets reset each iteration.
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": True, "top_k": 50}})
        invariant_a = _mk_rule(
            invariant_id="clear_top_k",
            match_fields={
                "transformers.sampling_params.top_k": {"present": True, "not_equal": 999},
            },
        )
        invariant_b = _mk_rule(
            invariant_id="restore_top_k",
            match_fields={
                "transformers.sampling_params.top_k": {"present": True, "not_equal": 50},
            },
        )
        with pytest.raises(LibraryResolutionCycleError):
            _apply_invariants_fixpoint(cfg, [invariant_a, invariant_b])

    def test_non_dormant_rules_ignored(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": True, "temperature": 0.5}})
        invariant = _mk_rule(
            invariant_id="not_dormant",
            severity="warn",
            match_fields={
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_invariants_fixpoint(cfg, [invariant])
        # warn severity must not trigger normalisation.
        assert result.transformers.sampling_params.temperature == 0.5

    def test_vllm_dormant_rule_strips_dotted_normalised_field(self):
        """A vllm dormant rule strips its (dotted) normalised field in the fixpoint.

        The corpus carries bare names; the loader rewrites them to dotted paths at
        load (residual a). This exercises the canonicaliser end of that contract:
        a dotted normalised_field on a vllm config resolves and strips.
        """
        cfg = _mk_config(engine="vllm", vllm={"engine_params": {"seed": -1}})
        invariant = _mk_rule(
            invariant_id="vllm_seed_dormant",
            engine="vllm",
            match_fields={"vllm.engine_params.seed": {"present": True, "not_equal": 0}},
            normalised_fields=["vllm.engine_params.seed"],
        )
        result = _apply_invariants_fixpoint(cfg, [invariant])
        # The dormant field is stripped to None (the universal "strip" sentinel).
        assert result.vllm.engine_params.seed is None

    def test_tensorrt_dormant_rule_strips_dotted_normalised_field(self):
        """A tensorrt dormant rule strips its dotted normalised field in the fixpoint."""
        cfg = _mk_config(engine="tensorrt", tensorrt={"sampling_params": {"top_k": 5}})
        invariant = _mk_rule(
            invariant_id="tensorrt_top_k_dormant",
            engine="tensorrt",
            match_fields={"tensorrt.sampling_params.top_k": {"present": True, "not_equal": 0}},
            normalised_fields=["tensorrt.sampling_params.top_k"],
        )
        result = _apply_invariants_fixpoint(cfg, [invariant])
        assert result.tensorrt.sampling_params.top_k is None


# ---------------------------------------------------------------------------
# resolve_library_effective()
# ---------------------------------------------------------------------------


class TestDedupSweep:
    def test_empty_sweep_returns_empty(self):
        result = resolve_library_effective([])
        assert result.canonical_configs == []
        assert result.groups == []

    def test_collapse_equivalent_configs(self):
        # Two configs differ only by a dormant field under greedy decoding; they
        # must collapse into one after canonicalisation.
        cfg_a = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 0.5}}
        )
        cfg_b = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 0.7}}
        )
        invariant = _mk_rule(
            invariant_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = resolve_library_effective([cfg_a, cfg_b], invariants=[invariant])
        assert len(result.canonical_configs) == 1
        assert len(result.groups) == 1
        assert result.groups[0].member_count == 2
        assert result.deduplicated is True
        assert result.would_dedup is True

    def test_distinct_configs_stay_distinct(self):
        cfg_a = _mk_config(
            transformers={"sampling_params": {"do_sample": True, "temperature": 0.5}}
        )
        cfg_b = _mk_config(
            transformers={"sampling_params": {"do_sample": True, "temperature": 0.7}}
        )
        invariant = _mk_rule(
            invariant_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = resolve_library_effective([cfg_a, cfg_b], invariants=[invariant])
        assert len(result.canonical_configs) == 2
        assert len(result.groups) == 2
        assert result.would_dedup is False
        assert result.deduplicated is False

    def test_dedup_disabled_preserves_all_but_groups_still_populated(self):
        cfg_a = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 0.5}}
        )
        cfg_b = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 0.7}}
        )
        invariant = _mk_rule(
            invariant_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = resolve_library_effective(
            [cfg_a, cfg_b], invariants=[invariant], deduplicate=False
        )
        # Every declared config still executes.
        assert len(result.canonical_configs) == 2
        # But the groups record the collapse that *would* have happened.
        assert len(result.groups) == 1
        assert result.groups[0].member_count == 2
        assert result.would_dedup is True
        assert result.deduplicated is False

    def test_declared_resolved_hashes_length_matches_input(self):
        cfg_a = _mk_config(transformers={"sampling_params": {"do_sample": True}})
        cfg_b = _mk_config(transformers={"sampling_params": {"do_sample": False}})
        result = resolve_library_effective([cfg_a, cfg_b], invariants=[])
        assert len(result.declared_resolved_hashes) == 2

    def test_integration_with_real_corpus(self):
        # End-to-end with the actual validated invariants - the original motivating
        # example: do_sample x temperature = [T,F] x [0.5, 1.0, 1.5] -> 6 configs,
        # library-resolution mechanism collapses to 4 (1 greedy canonical + 3 sampling variants).
        from llenergymeasure.config.engine_rules.loader import EngineRulesLoader

        loader = EngineRulesLoader()
        invariants = loader.load_rules("transformers").invariants

        configs = []
        for do_sample in (True, False):
            for temp in (0.5, 1.0, 1.5):
                configs.append(
                    _mk_config(
                        transformers={
                            "sampling_params": {"do_sample": do_sample, "temperature": temp}
                        }
                    )
                )
        result = resolve_library_effective(configs, invariants=invariants)
        # do_sample=True x temps=[0.5, 1.0, 1.5] -> 3 distinct sampling configs
        # do_sample=False x any temp -> 1 canonical greedy
        assert len(result.canonical_configs) == 4

    def test_hashes_stable_across_repeated_dedup(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": False, "temperature": 0.5}})
        invariant = _mk_rule(
            invariant_id="greedy",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        r1 = resolve_library_effective([cfg], invariants=[invariant])
        r2 = resolve_library_effective([cfg], invariants=[invariant])
        assert r1.declared_resolved_hashes == r2.declared_resolved_hashes
        assert r1.groups[0].resolved_config_hash == r2.groups[0].resolved_config_hash


# ---------------------------------------------------------------------------
# resolved_config_hashing symmetry
# ---------------------------------------------------------------------------


class TestH1HashSymmetry:
    def test_equivalent_canonical_forms_share_h1(self):
        cfg_a = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 0.5}}
        )
        cfg_b = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 0.7}}
        )
        invariant = _mk_rule(
            invariant_id="greedy",
            match_fields={
                "transformers.sampling_params.do_sample": False,
                "transformers.sampling_params.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        canon_a = _apply_invariants_fixpoint(cfg_a, [invariant])
        canon_b = _apply_invariants_fixpoint(cfg_b, [invariant])
        assert hash_config(build_resolved_view(canon_a)) == hash_config(
            build_resolved_view(canon_b)
        )
