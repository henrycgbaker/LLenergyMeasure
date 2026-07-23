"""Unit tests for numeric sweep-axis idioms (linear span, log, pow2).

Covers expand_axis_idiom directly (values, typing, rounding, endpoints,
every validation error path) and end-to-end expansion through the public
expand_grid, plus regression checks that existing list/mapping sweep
semantics are unchanged.
"""

from __future__ import annotations

import pytest
import yaml

from llenergymeasure.config.grid import count_sweep_structure, expand_grid
from llenergymeasure.config.sweep_expansion import _expand_group_entry
from llenergymeasure.config.sweep_idioms import expand_axis_idiom
from llenergymeasure.utils.exceptions import ConfigError

# =============================================================================
# Linear span idiom: {min, max, num}
# =============================================================================


class TestLinearIdiom:
    def test_int_bounds_integral_step_emits_ints(self):
        values = expand_axis_idiom({"min": 0, "max": 8, "num": 5})
        assert values == [0, 2, 4, 6, 8]
        assert all(type(v) is int for v in values)

    def test_int_bounds_non_integral_step_emits_floats(self):
        values = expand_axis_idiom({"min": 0, "max": 10, "num": 5})
        assert values == [0.0, 2.5, 5.0, 7.5, 10.0]
        assert all(type(v) is float for v in values)

    def test_float_bounds_rounding_kills_binary_noise(self):
        # Naive float arithmetic gives 0.30000000000000004 for the midpoint.
        values = expand_axis_idiom({"min": 0.1, "max": 0.5, "num": 3})
        assert values == [0.1, 0.3, 0.5]

    def test_endpoints_inclusive_and_exact(self):
        values = expand_axis_idiom({"min": 0.7, "max": 1.3, "num": 4})
        assert values[0] == 0.7
        assert values[-1] == 1.3
        assert len(values) == 4

    def test_mixed_int_float_bounds_emit_floats(self):
        values = expand_axis_idiom({"min": 0, "max": 1.0, "num": 3})
        assert values == [0.0, 0.5, 1.0]
        assert all(type(v) is float for v in values)

    def test_num_two_gives_exact_bounds(self):
        assert expand_axis_idiom({"min": 3, "max": 7, "num": 2}) == [3, 7]

    def test_equal_bounds_collapse_to_single_value(self):
        assert expand_axis_idiom({"min": 5, "max": 5, "num": 3}) == [5]

    def test_negative_range(self):
        assert expand_axis_idiom({"min": -4, "max": 4, "num": 3}) == [-4, 0, 4]


# =============================================================================
# Log idiom: {log: {min, max, num}}
# =============================================================================


class TestLogIdiom:
    def test_int_decades_emit_ints(self):
        values = expand_axis_idiom({"log": {"min": 1, "max": 100, "num": 3}})
        assert values == [1, 10, 100]
        assert all(type(v) is int for v in values)

    def test_float_decades(self):
        values = expand_axis_idiom({"log": {"min": 0.001, "max": 1.0, "num": 4}})
        assert values == [0.001, 0.01, 0.1, 1.0]

    def test_non_integral_points_emit_floats(self):
        values = expand_axis_idiom({"log": {"min": 1, "max": 10, "num": 3}})
        assert values == [1.0, 3.16227766, 10.0]
        assert all(type(v) is float for v in values)

    def test_endpoints_inclusive_and_exact(self):
        values = expand_axis_idiom({"log": {"min": 2, "max": 3, "num": 5}})
        assert values[0] == 2.0
        assert values[-1] == 3.0
        assert len(values) == 5

    def test_rounding_collapse_dedupes_preserving_order(self):
        # The three interior points all round to 1.0 at 10 significant digits
        # and collapse into the min endpoint; endpoints stay verbatim.
        values = expand_axis_idiom({"log": {"min": 1.0, "max": 1.0000000000001, "num": 5}})
        assert values == [1.0, 1.0000000000001]


# =============================================================================
# pow2 idiom: {pow2: {min, max}}
# =============================================================================


class TestPow2Idiom:
    def test_bounds_on_powers(self):
        values = expand_axis_idiom({"pow2": {"min": 4, "max": 32}})
        assert values == [4, 8, 16, 32]
        assert all(type(v) is int for v in values)

    def test_bounds_between_powers(self):
        assert expand_axis_idiom({"pow2": {"min": 3, "max": 33}}) == [4, 8, 16, 32]

    def test_single_power(self):
        assert expand_axis_idiom({"pow2": {"min": 16, "max": 16}}) == [16]

    def test_float_bounds_integral_powers_emit_ints(self):
        values = expand_axis_idiom({"pow2": {"min": 3.5, "max": 40.0}})
        assert values == [4, 8, 16, 32]
        assert all(type(v) is int for v in values)

    def test_fractional_powers_emit_floats(self):
        values = expand_axis_idiom({"pow2": {"min": 0.25, "max": 2}})
        assert values == [0.25, 0.5, 1.0, 2.0]
        assert all(type(v) is float for v in values)


# =============================================================================
# Validation error paths
# =============================================================================


class TestIdiomErrors:
    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match=r"min .* must not exceed max"):
            expand_axis_idiom({"min": 5, "max": 1, "num": 3})

    def test_log_min_greater_than_max(self):
        with pytest.raises(ValueError, match=r"min .* must not exceed max"):
            expand_axis_idiom({"log": {"min": 10, "max": 1, "num": 3}})

    @pytest.mark.parametrize("num", [1, 0, -3])
    def test_num_below_two(self, num):
        with pytest.raises(ValueError, match="'num' must be an integer >= 2"):
            expand_axis_idiom({"min": 0, "max": 8, "num": num})

    @pytest.mark.parametrize("num", [2.5, "3", True, None])
    def test_num_not_an_integer(self, num):
        with pytest.raises(ValueError, match="'num' must be an integer >= 2"):
            expand_axis_idiom({"min": 0, "max": 8, "num": num})

    @pytest.mark.parametrize("bad", ["a", None, True, [1], float("inf"), float("nan")])
    def test_non_numeric_bound(self, bad):
        with pytest.raises(ValueError, match="'min' must be a finite number"):
            expand_axis_idiom({"min": bad, "max": 8, "num": 3})

    @pytest.mark.parametrize("lo", [0, -1, -0.5])
    def test_log_requires_positive_min(self, lo):
        with pytest.raises(ValueError, match="log range shorthand requires min > 0"):
            expand_axis_idiom({"log": {"min": lo, "max": 10, "num": 3}})

    @pytest.mark.parametrize("lo", [0, -4])
    def test_pow2_requires_positive_min(self, lo):
        with pytest.raises(ValueError, match="pow2 range shorthand requires min > 0"):
            expand_axis_idiom({"pow2": {"min": lo, "max": 8}})

    @pytest.mark.parametrize(("lo", "hi"), [(33, 63), (0.3, 0.4)])
    def test_pow2_no_power_in_range(self, lo, hi):
        with pytest.raises(ValueError, match="no power of two lies within"):
            expand_axis_idiom({"pow2": {"min": lo, "max": hi}})

    def test_unknown_key_mixed_into_linear(self):
        with pytest.raises(
            ValueError, match=r"not a recognised range shorthand.*valid range shorthand"
        ):
            expand_axis_idiom({"min": 0, "max": 8, "num": 3, "step": 2})

    def test_linear_missing_num(self):
        with pytest.raises(
            ValueError, match=r"not a recognised range shorthand.*valid range shorthand"
        ):
            expand_axis_idiom({"min": 0, "max": 8})

    def test_unknown_key_next_to_log(self):
        with pytest.raises(
            ValueError, match=r"not a recognised range shorthand.*valid range shorthand"
        ):
            expand_axis_idiom({"log": {"min": 1, "max": 10, "num": 3}, "extra": 1})

    def test_log_inner_missing_num(self):
        with pytest.raises(ValueError, match="log range shorthand requires a nested mapping"):
            expand_axis_idiom({"log": {"min": 1, "max": 10}})

    def test_log_inner_not_a_mapping(self):
        with pytest.raises(ValueError, match="log range shorthand requires a nested mapping"):
            expand_axis_idiom({"log": 5})

    def test_pow2_inner_unknown_key(self):
        with pytest.raises(ValueError, match="pow2 range shorthand requires a nested mapping"):
            expand_axis_idiom({"pow2": {"min": 4, "max": 32, "num": 3}})

    @pytest.mark.parametrize("mapping", [{}, {"foo": 1}, {"values": [1, 2, 3]}])
    def test_arbitrary_mapping_rejected_loudly(self, mapping):
        with pytest.raises(
            ValueError, match=r"not a recognised range shorthand.*valid range shorthand"
        ):
            expand_axis_idiom(mapping)


# =============================================================================
# End-to-end through expand_grid
# =============================================================================


class TestIdiomsThroughExpandGrid:
    def test_study_yaml_with_all_three_idioms(self):
        raw = yaml.safe_load(
            """
            task: {model: gpt2}
            engine: transformers
            sweep:
              task.dataset.n_prompts: {min: 10, max: 30, num: 3}
              transformers.sampling_params.temperature: {log: {min: 0.1, max: 1.0, num: 2}}
              transformers.llem_execution.batch_size: {pow2: {min: 4, max: 16}}
            """
        )
        valid, skipped = expand_grid(raw)
        assert len(skipped) == 0
        # 3 n_prompts x 2 temperatures x 3 batch sizes = 18
        assert len(valid) == 18
        assert {c.task.dataset.n_prompts for c in valid} == {10, 20, 30}
        assert {c.transformers.sampling_params.temperature for c in valid} == {0.1, 1.0}
        assert {c.transformers.llem_execution.batch_size for c in valid} == {4, 8, 16}

    def test_idiom_expands_identically_to_explicit_list(self):
        base = {"task": {"model": "gpt2"}, "engine": "transformers"}
        as_list = {**base, "sweep": {"task.dataset.n_prompts": [10, 20, 30]}}
        as_idiom = {**base, "sweep": {"task.dataset.n_prompts": {"min": 10, "max": 30, "num": 3}}}
        valid_list, _ = expand_grid(as_list)
        valid_idiom, _ = expand_grid(as_idiom)
        assert [c.model_dump() for c in valid_idiom] == [c.model_dump() for c in valid_list]

    def test_non_idiom_mapping_fails_loudly_with_axis_name(self):
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {"task.dataset.n_prompts": {"start": 1, "stop": 5}},
        }
        with pytest.raises(ConfigError, match=r"sweep axis 'task\.dataset\.n_prompts'"):
            expand_grid(raw)

    def test_invalid_idiom_fields_fail_loudly(self):
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {"task.dataset.n_prompts": {"min": 30, "max": 10, "num": 3}},
        }
        with pytest.raises(ConfigError, match=r"min .* must not exceed max"):
            expand_grid(raw)

    def test_idiom_counts_as_single_axis(self):
        sweep = {"task.dataset.n_prompts": {"min": 10, "max": 30, "num": 3}}
        assert count_sweep_structure(sweep) == (1, 0)


# =============================================================================
# Regression: existing sweep semantics unchanged
# =============================================================================


class TestExistingMappingSemanticsUnchanged:
    def test_explicit_list_axis_unchanged(self):
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {"task.dataset.n_prompts": [50, 100]},
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 2
        assert len(skipped) == 0
        assert {c.task.dataset.n_prompts for c in valid} == {50, 100}

    def test_group_list_of_dicts_still_a_group(self):
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "precision": [
                    {"transformers.engine_params.dtype": "float16"},
                    {"transformers.engine_params.dtype": "bfloat16"},
                ]
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 2
        assert len(skipped) == 0

    def test_group_entry_dict_value_still_literal(self):
        """Dict-valued fields inside group entries pass through as literal values."""
        entry = {"transformers.some_param": {"a": 1}}
        result = _expand_group_entry(entry)
        assert result == [{"transformers.some_param": {"a": 1}}]
