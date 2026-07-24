"""Tests for canonical serialisation + resolved/observed config hashing.

Normalisation invariants come from ``sweep-dedup.md`` §9.Q3. Each test pins one
invariant from the table so a invariant change surfaces as a targeted failure rather
than a diffuse hash-mismatch.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.hashing import (
    ConfigHashView,
    build_observed_view,
    canonical_serialise,
    hash_config,
)
from llenergymeasure.study.hashing import build_resolved_view


def _mk_config(**overrides):
    base = {"task": {"model": "gpt2"}, "engine": "transformers"}
    base.update(overrides)
    return ExperimentConfig(**base)


class TestCanonicalSerialise:
    def test_none_and_missing_differ(self):
        a = {"seed": None}
        b: dict = {}
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_int_and_integral_float_unify(self):
        # An int and an integral-valued float for the same field now serialise
        # identically. This closes the resolved-vs-observed gap: a float-typed
        # field left at an int-literal default (python int in a mode="python"
        # resolved dump) must hash the same as the genuine float the native
        # engine object carries. Superseded the previous "int != float" rule.
        a = {"top_k": 0}
        b = {"top_k": 0.0}
        assert canonical_serialise(a) == canonical_serialise(b)
        assert canonical_serialise({"n": 1}) == canonical_serialise({"n": 1.0})
        assert canonical_serialise({"n": -1}) == canonical_serialise({"n": -1.0})

    def test_non_integral_float_stays_distinct_from_int(self):
        # Unification only collapses integral values; a fractional float must
        # never collapse onto an int (dedup must not over-collapse).
        assert canonical_serialise({"x": 0.5}) != canonical_serialise({"x": 0})
        assert canonical_serialise({"x": 1.5}) != canonical_serialise({"x": 1})

    def test_large_distinct_ints_stay_bit_exact(self):
        # Folding is integral-float -> int (never int -> float), so two large
        # distinct ints that a float round-trip would collapse (both sit above
        # 2**53) stay distinct - no lossy dedup of genuine integer identity
        # fields (seeds, token counts).
        big1 = 9007199254740993  # 2**53 + 1
        big2 = 9007199254740992  # 2**53
        assert canonical_serialise({"seed": big1}) != canonical_serialise({"seed": big2})

    def test_bool_and_int_differ(self):
        a = {"do_sample": True}
        b = {"do_sample": 1}
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_tuple_normalises_to_list(self):
        a = {"stop": ("a", "b")}
        b = {"stop": ["a", "b"]}
        assert canonical_serialise(a) == canonical_serialise(b)

    def test_dict_keys_sorted(self):
        a = {"a": 1, "b": 2}
        b = {"b": 2, "a": 1}
        assert canonical_serialise(a) == canonical_serialise(b)

    def test_nan_is_stable_string(self):
        out = canonical_serialise({"x": math.nan})
        assert b'"NaN"' in out

    def test_infinity_is_stable(self):
        pos = canonical_serialise({"x": math.inf})
        neg = canonical_serialise({"x": -math.inf})
        assert b"Infinity" in pos and b"-Infinity" in neg
        assert pos != neg

    def test_float_rounding_stability(self):
        # Jitter in the 13th+ significant digit should collapse to the same hash.
        a = {"temp": 0.123456789012345}
        b = {"temp": 0.1234567890123459}  # 13+ digits differ
        assert canonical_serialise(a) == canonical_serialise(b)

    def test_float_distinct_values_stay_distinct(self):
        # Two values that differ at the 11th digit must not collapse.
        a = {"temp": 0.12345678901}
        b = {"temp": 0.12345678902}
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_nested_dict_recursive_normalisation(self):
        a = {"outer": {"inner": (1.0, 2.0)}}
        b = {"outer": {"inner": [1.0, 2.0]}}
        assert canonical_serialise(a) == canonical_serialise(b)


class TestHashConfig:
    def test_returns_hex_digest(self):
        view = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        h = hash_config(view)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # parses as hex

    def test_identical_views_same_hash(self):
        v1 = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        v2 = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        assert hash_config(v1) == hash_config(v2)

    def test_different_engine_different_hash(self):
        v1 = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        v2 = ConfigHashView(engine="vllm", task={"model": "gpt2"})
        assert hash_config(v1) != hash_config(v2)


class TestBuildResolvedView:
    def test_extracts_engine_and_task(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": False}})
        view = build_resolved_view(cfg)
        assert view.engine == "transformers"
        assert view.task["model"] == "gpt2"

    def test_sampling_lifted_into_sampling_bucket(self):
        cfg = _mk_config(transformers={"sampling_params": {"do_sample": True, "temperature": 0.7}})
        view = build_resolved_view(cfg)
        assert view.observed_sampling_params["do_sample"] is True
        assert view.observed_sampling_params["temperature"] == 0.7
        assert "sampling_params" not in view.observed_engine_params

    def test_both_sections_lift_into_matching_buckets(self):
        # Regression for the sampling -> sampling_params alias bug: the resolved
        # view used to read a legacy flat "sampling" attribute, so the generated
        # sections' content could miss its bucket and resolved hashes would not
        # line up with the observed pipeline (which populates the same
        # ConfigHashView buckets). Prove each section lands in its own bucket
        # and no section-wrapper key leaks through.
        cfg = _mk_config(
            transformers={
                "engine_params": {"num_beams": 2},
                "sampling_params": {"do_sample": True, "temperature": 0.7},
            }
        )
        view = build_resolved_view(cfg)
        assert view.observed_engine_params["num_beams"] == 2
        assert view.observed_sampling_params["temperature"] == 0.7
        for stale_key in ("sampling", "sampling_params", "engine_params"):
            assert stale_key not in view.observed_engine_params
            assert stale_key not in view.observed_sampling_params

    def test_passthrough_kwargs_propagated(self):
        cfg = _mk_config(passthrough_kwargs={"my_key": "my_val"})
        view = build_resolved_view(cfg)
        assert view.passthrough_kwargs == {"my_key": "my_val"}


class TestBuildObservedView:
    def test_carries_inputs_through(self):
        view = build_observed_view(
            engine="vllm",
            task={"model": "gpt2"},
            observed_engine_params={"dtype": "float16"},
            observed_sampling_params={"temperature": 1.0},
        )
        assert view.engine == "vllm"
        assert view.observed_engine_params["dtype"] == "float16"
        assert view.observed_sampling_params["temperature"] == 1.0

    def test_resolved_and_observed_match_on_same_inputs(self):
        # Symmetry: both views hashed through the same pipe produce the same hash
        # when the underlying fields match. This is what makes the observed-collision
        # invariant meaningful.
        task = {"model": "gpt2"}
        engine_params = {"dtype": "float16"}
        sampling_params = {"temperature": 1.0}

        resolved_view = ConfigHashView(
            engine="vllm",
            task=task,
            observed_engine_params=engine_params,
            observed_sampling_params=sampling_params,
        )
        observed_view = build_observed_view(
            engine="vllm",
            task=task,
            observed_engine_params=engine_params,
            observed_sampling_params=sampling_params,
        )
        assert hash_config(resolved_view) == hash_config(observed_view)


class TestServingModeIdentity:
    """serving_mode is a conditioning identity axis in both hash families."""

    def test_config_hash_view_defaults_serving_mode_offline(self):
        # The view slot defaults to "offline" so existing direct constructions
        # keep hashing as before (offline is the only universe today).
        view = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        assert view.serving_mode == "offline"

    def test_resolved_view_carries_config_serving_mode(self):
        assert build_resolved_view(_mk_config()).serving_mode == "offline"
        assert build_resolved_view(_mk_config(serving_mode="server")).serving_mode == "server"

    def test_resolved_offline_vs_server_hash_differ(self):
        offline = hash_config(build_resolved_view(_mk_config(serving_mode="offline")))
        server = hash_config(build_resolved_view(_mk_config(serving_mode="server")))
        assert offline != server

    def test_observed_offline_vs_server_hash_differ(self):
        common = {
            "engine": "vllm",
            "task": {"model": "gpt2"},
            "observed_engine_params": {"dtype": "float16"},
            "observed_sampling_params": {"temperature": 1.0},
        }
        offline = hash_config(build_observed_view(serving_mode="offline", **common))
        server = hash_config(build_observed_view(serving_mode="server", **common))
        assert offline != server

    def test_resolved_view_differs_only_by_serving_mode_slot(self):
        # An offline and an otherwise-identical server config project resolved
        # views that differ in exactly one slot: serving_mode. Nothing else in
        # the view shifted when the field was added.
        offline = asdict(build_resolved_view(_mk_config(serving_mode="offline")))
        server = asdict(build_resolved_view(_mk_config(serving_mode="server")))
        differing = {k for k in offline if offline[k] != server[k]}
        assert differing == {"serving_mode"}


class TestHashStability:
    @pytest.mark.parametrize("_", range(5))
    def test_hash_stable_across_repeat_calls(self, _):
        cfg = _mk_config(
            transformers={"sampling_params": {"do_sample": False, "temperature": 1.0}},
        )
        h1 = hash_config(build_resolved_view(cfg))
        h2 = hash_config(build_resolved_view(cfg))
        assert h1 == h2


class TestIntFloatCanonicalisation:
    """Regression guard for the resolved-vs-observed int/float canonicalisation gap.

    Analogous to ``test_config_hash_stable_across_json_round_trip`` (#822) but for
    the resolved/observed pipeline: a float-typed field left at an int-literal
    default is python ``int`` in the resolved view (``mode="python"`` skips
    default validation) yet a genuine ``float`` in the native engine object the
    observed pipeline captures. The two must hash identically.
    """

    def test_int_default_resolved_matches_float_observed(self):
        # vLLM ``cpu_offload_gb`` is ``Annotated[float | None, ...] = 0``: the
        # int literal default survives a python-mode dump as int 0. Populating
        # engine_params (even empty) materialises the field at its default.
        cfg = ExperimentConfig(task={"model": "gpt2"}, engine="vllm", vllm={"engine_params": {}})
        resolved_view = build_resolved_view(cfg)
        assert resolved_view.observed_engine_params["cpu_offload_gb"] == 0
        assert isinstance(resolved_view.observed_engine_params["cpu_offload_gb"], int)

        # The native engine object coerces the same field to float 0.0.
        observed_view = build_observed_view(
            engine="vllm",
            task=cfg.task.model_dump(mode="python"),
            observed_engine_params={
                **resolved_view.observed_engine_params,
                "cpu_offload_gb": 0.0,
            },
            observed_sampling_params=resolved_view.observed_sampling_params,
            llem_execution=resolved_view.llem_execution,
            measurement=resolved_view.measurement,
        )
        assert isinstance(observed_view.observed_engine_params["cpu_offload_gb"], float)

        # Same value, different python type -> identical hash after the fix.
        assert hash_config(resolved_view) == hash_config(observed_view)

    def test_distinct_offload_values_stay_distinct(self):
        # Dedup must not over-collapse: a genuinely different offload value keeps
        # a distinct resolved hash.
        cfg_zero = ExperimentConfig(
            task={"model": "gpt2"}, engine="vllm", vllm={"engine_params": {}}
        )
        cfg_two = ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            vllm={"engine_params": {"cpu_offload_gb": 2.0}},
        )
        assert hash_config(build_resolved_view(cfg_zero)) != hash_config(
            build_resolved_view(cfg_two)
        )
