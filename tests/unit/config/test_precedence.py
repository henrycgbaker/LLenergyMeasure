"""Tests for the R7 config-resolution core (config/precedence.py).

The UNSET-sentinel precedence chain: a distinct 'use the layer below' sentinel,
recursive pruning, ascending-precedence deep merge, and the ruled layer order
(call-site > env > study YAML > user config > pydantic defaults), plus the v0.7
warmup-protocol wiring.
"""

from __future__ import annotations

import copy

from llenergymeasure.config.precedence import (
    UNSET,
    PrecedenceChain,
    is_unset,
    prune_unset,
    resolve_layers,
    resolve_server_warmup,
)


class TestSentinel:
    def test_unset_distinct_from_none(self):
        assert is_unset(UNSET)
        assert not is_unset(None)
        assert UNSET is not None

    def test_unset_is_singleton_and_deepcopy_stable(self):
        # A pruned layer must never smuggle a COPIED sentinel past an identity prune.
        assert copy.deepcopy(UNSET) is UNSET
        assert copy.copy(UNSET) is UNSET
        assert type(UNSET)() is UNSET

    def test_unset_is_falsey_with_legible_repr(self):
        assert not UNSET
        assert repr(UNSET) == "UNSET"


class TestPruneUnset:
    def test_drops_unset_keeps_none(self):
        # None is a real overriding null and must survive; UNSET is dropped.
        assert prune_unset({"a": 1, "b": UNSET, "n": None}) == {"a": 1, "n": None}

    def test_recurses_into_nested_mappings(self):
        assert prune_unset({"x": {"y": UNSET, "z": 2}, "w": UNSET}) == {"x": {"z": 2}}

    def test_empty_when_all_unset(self):
        # "a" is UNSET (dropped); "b" is a nested mapping that prunes to empty.
        assert prune_unset({"a": UNSET, "b": {"c": UNSET}}) == {"b": {}}


class TestResolveLayers:
    def test_ascending_precedence(self):
        # First layer lowest, last highest.
        assert resolve_layers({"a": 1}, {"a": 2}, {"a": 3}) == {"a": 3}

    def test_deep_merge_of_disjoint_nested_keys(self):
        merged = resolve_layers({"g": {"p": 1, "q": 1}}, {"g": {"q": 2, "r": 3}})
        assert merged == {"g": {"p": 1, "q": 2, "r": 3}}

    def test_unset_defers_to_lower_layer(self):
        assert resolve_layers({"a": 1}, {"a": UNSET}) == {"a": 1}

    def test_no_mutation_of_inputs(self):
        base = {"g": {"p": 1}}
        resolve_layers(base, {"g": {"q": 2}})
        assert base == {"g": {"p": 1}}


class TestPrecedenceChain:
    def test_call_site_beats_env_beats_study_beats_user_beats_defaults(self):
        chain = PrecedenceChain(
            defaults={"x": 0, "y": 0, "z": 0, "w": 0, "v": 0},
            user_config={"y": 1},
            study_yaml={"z": 2},
            env={"w": 3},
            call_site={"v": 4},
        )
        assert chain.resolve() == {"x": 0, "y": 1, "z": 2, "w": 3, "v": 4}

    def test_highest_layer_wins_on_conflict(self):
        chain = PrecedenceChain(
            defaults={"k": "default"},
            user_config={"k": "user"},
            study_yaml={"k": "study"},
            env={"k": "env"},
            call_site={"k": "call"},
        )
        assert chain.resolve()["k"] == "call"

    def test_unset_layer_defers(self):
        chain = PrecedenceChain(
            defaults={"k": "default"},
            study_yaml={"k": UNSET},
        )
        assert chain.resolve()["k"] == "default"


class TestResolveServerWarmup:
    def test_all_unset_yields_defaults(self):
        w = resolve_server_warmup()
        assert w.mode == "composite"
        assert w.timeout_seconds == 900.0
        assert w.duration_seconds == 300.0

    def test_study_yaml_overrides_defaults(self):
        w = resolve_server_warmup(study_yaml={"mode": "fixed", "duration_seconds": 60})
        assert w.mode == "fixed"
        assert w.duration_seconds == 60.0
        # Unset field falls through to the default.
        assert w.timeout_seconds == 900.0

    def test_call_site_beats_study_and_env(self):
        w = resolve_server_warmup(
            study_yaml={"mode": "fixed", "duration_seconds": 60},
            env={"duration_seconds": 90},
            call_site={"duration_seconds": 120},
        )
        assert w.mode == "fixed"
        assert w.duration_seconds == 120.0

    def test_resolved_value_is_validated(self):
        # An illegal resolved value fails validation (identity discipline: the
        # realised protocol is a real ServerWarmupConfig).
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            resolve_server_warmup(call_site={"mode": "nonsense"})
