"""Unit tests for sweep auto-expansion from the curated Pydantic surface.

Covers the Feature-1 resolver (per-field ``auto`` shorthand) and the Feature-2
skeleton generator. See ``llenergymeasure.config.sweep_autoexpand``.
"""

from __future__ import annotations

import pytest
import yaml

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.config.sweep_autoexpand import (
    build_study_skeleton,
    expand_auto_sweep,
    resolve_auto_values,
)
from llenergymeasure.utils.exceptions import ConfigError

# =============================================================================
# resolve_auto_values - per-type policy
# =============================================================================


def test_literal_field_yields_all_members_verbatim() -> None:
    """A Literal[...] field resolves to all members in declared order."""
    values = resolve_auto_values("vllm.engine_params.kv_cache_dtype")
    assert values == ["auto", "fp8", "fp8_e5m2", "fp8_e4m3"]


def test_literal_int_field_yields_all_members() -> None:
    """A Literal[int, ...] field resolves to its int members verbatim."""
    assert resolve_auto_values("vllm.engine_params.block_size") == [8, 16, 32]


def test_bool_field_yields_false_then_true() -> None:
    """A bool field (do_sample) resolves to [False, True]."""
    assert resolve_auto_values("transformers.sampling_params.do_sample") == [False, True]


def test_bounded_float_inclusive_buckets_min_mid_max() -> None:
    """transformers temperature (ge=0.0, le=2.0) -> [0.0, 1.0, 2.0]."""
    assert resolve_auto_values("transformers.sampling_params.temperature") == [0.0, 1.0, 2.0]


def test_bounded_float_exclusive_bounds_nudged_inward() -> None:
    """vllm gpu_memory_utilization (gt=0.0, lt=1.0): both ends nudged inward.

    The mined exclusiveMinimum retires the stale hand-enforced inclusive ge=0.0,
    so the field carries a single coherent bound per edge (gt/lt) and the
    resolver nudges both endpoints inward rather than using either verbatim.
    """
    values = resolve_auto_values("vllm.engine_params.gpu_memory_utilization")
    assert values == [0.001, 0.5, 0.999]
    assert all(0.0 < v < 1.0 for v in values)


def test_unbounded_numeric_raises_config_error() -> None:
    """A numeric field with only one bound has no finite range -> ConfigError."""
    with pytest.raises(ConfigError, match="lacks both a lower and an upper"):
        resolve_auto_values("transformers.sampling_params.top_k")  # ge=0 only


def test_unknown_type_field_raises_config_error() -> None:
    """An Any-typed curated field has no natural value set -> ConfigError."""
    with pytest.raises(ConfigError, match="no natural value set"):
        resolve_auto_values("transformers.engine_params.dtype")  # Any | None


def test_non_curated_field_raises_config_error() -> None:
    """A field absent from the curated model -> ConfigError naming the FQN."""
    with pytest.raises(ConfigError, match="not a curated field"):
        resolve_auto_values("transformers.sampling_params.does_not_exist")


def test_malformed_fqn_raises_config_error() -> None:
    """A key that is not <engine>.<section>.<field> -> ConfigError."""
    with pytest.raises(ConfigError, match="not an engine-field key"):
        resolve_auto_values("transformers.temperature")


def test_unknown_engine_prefix_raises_config_error() -> None:
    """An unknown engine prefix -> ConfigError."""
    with pytest.raises(ConfigError, match="Unknown engine"):
        resolve_auto_values("bogus.sampling_params.temperature")


# =============================================================================
# resolve_auto_values - nested typed sub-model leaves (depth > 3)
# =============================================================================


def test_nested_literal_enum_leaf_yields_all_members() -> None:
    """A Literal leaf inside a typed nested config resolves to its members.

    vLLM's compilation_config is projected as a typed sub-model; its ``mode``
    leaf is ``Literal[0, 1, 2, 3] | None``.
    """
    assert resolve_auto_values("vllm.engine_params.compilation_config.mode") == [0, 1, 2, 3]


def test_nested_literal_str_leaf_yields_all_members() -> None:
    """A string-Literal nested leaf resolves to its members verbatim."""
    values = resolve_auto_values("vllm.engine_params.compilation_config.compile_cache_save_format")
    assert values == ["binary", "unpacked"]


def test_nested_bool_leaf_yields_false_then_true() -> None:
    """A bool leaf inside a typed nested config resolves to [False, True]."""
    assert resolve_auto_values("vllm.engine_params.compilation_config.compile_mm_encoder") == [
        False,
        True,
    ]


def test_nested_unbounded_numeric_leaf_raises_cleanly() -> None:
    """An unbounded numeric nested leaf has no finite range -> ConfigError, no crash."""
    with pytest.raises(ConfigError, match="lacks both a lower and an upper"):
        resolve_auto_values("vllm.engine_params.compilation_config.cudagraph_num_of_warmups")


def test_nested_leaf_under_any_blob_intermediate_raises_cleanly() -> None:
    """Descending through an Any|None intermediate (not a typed sub-model) errors cleanly."""
    with pytest.raises(ConfigError, match="not a typed nested config"):
        resolve_auto_values("vllm.engine_params.compilation_config.cudagraph_mode.foo")


def test_unknown_nested_leaf_raises_config_error() -> None:
    """An unknown leaf on a typed nested sub-model -> ConfigError naming the FQN."""
    with pytest.raises(ConfigError, match="not a curated field"):
        resolve_auto_values("vllm.engine_params.compilation_config.does_not_exist")


def test_expand_auto_sweep_resolves_nested_axis() -> None:
    """An ``auto`` axis on a nested-config leaf is replaced by its resolved list."""
    out = expand_auto_sweep({"vllm.engine_params.compilation_config.mode": "auto"})
    assert out == {"vllm.engine_params.compilation_config.mode": [0, 1, 2, 3]}


# =============================================================================
# expand_auto_sweep - block rewriting
# =============================================================================


def test_expand_auto_sweep_resolves_independent_axis() -> None:
    """An ``auto`` axis is replaced by its resolved list; lists pass through."""
    sweep = {
        "transformers.sampling_params.temperature": "auto",
        "transformers.sampling_params.do_sample": [True, False],
    }
    out = expand_auto_sweep(sweep)
    assert out["transformers.sampling_params.temperature"] == [0.0, 1.0, 2.0]
    assert out["transformers.sampling_params.do_sample"] == [True, False]


def test_expand_auto_sweep_resolves_auto_inside_group() -> None:
    """An ``auto`` field inside a dependent-group variant expands that variant."""
    sweep = {
        "transformers.sampling": [
            {"transformers.sampling_params.temperature": "auto"},
            {"transformers.sampling_params.do_sample": False},
        ]
    }
    out = expand_auto_sweep(sweep)
    group = out["transformers.sampling"]
    # First variant expands to three (one per resolved temperature); second stays.
    temps = [
        v["transformers.sampling_params.temperature"]
        for v in group
        if "transformers.sampling_params.temperature" in v
    ]
    assert temps == [0.0, 1.0, 2.0]
    assert {"transformers.sampling_params.do_sample": False} in group


def test_expand_auto_sweep_empty_is_noop() -> None:
    assert expand_auto_sweep({}) == {}


# =============================================================================
# build_study_skeleton - skeleton generator
# =============================================================================


@pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
def test_skeleton_contains_curated_fields_and_scaffold(engine: str) -> None:
    """Skeleton names curated fields and carries a valid scaffold + provenance."""
    text = build_study_skeleton(engine)
    assert "TRIM ME" in text
    assert f"engine: {engine}" in text
    assert "task:" in text and "model:" in text and "dataset:" in text
    assert "sweep:" in text
    # At least one auto-expanded sampling knob appears uncommented.
    assert f"{engine}.sampling_params.temperature:" in text


@pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
def test_skeleton_roundtrips_with_zero_invalid(engine: str, tmp_path) -> None:
    """The auto-expanded skeleton parses + round-trips with zero invalid configs."""
    parsed = yaml.safe_load(build_study_skeleton(engine))
    # Fill the task placeholders so the configs can validate.
    parsed["task"]["model"] = "gpt2"
    parsed["task"]["dataset"]["source"] = "arc"
    path = tmp_path / "skeleton.yaml"
    path.write_text(yaml.safe_dump(parsed))

    loaded = load_study_config(path)
    assert len(loaded.valid_experiments) > 0
    assert loaded.skipped == []


def test_skeleton_emits_valid_subset_of_partly_rejected_field() -> None:
    """A field whose natural set is partly rejected emits its valid subset.

    ``transformers.engine_params.early_stopping`` resolves to ``[false, true]``;
    the engine cross-field rule rejects ``true`` on its own (needs num_beams>1)
    but accepts ``false``. The skeleton must emit ``[false]`` as a normal
    (uncommented) sweep line, with an adjacent comment naming the omitted value -
    not comment out the whole field.
    """
    text = build_study_skeleton("transformers")
    early_stopping_lines = [line for line in text.splitlines() if "early_stopping" in line]
    assert len(early_stopping_lines) == 1
    line = early_stopping_lines[0]
    # Emitted uncommented (the sweep key is not behind a leading '#').
    assert line.lstrip().startswith("transformers.engine_params.early_stopping:")
    assert "[false]" in line
    # The rejected value is named in an adjacent comment, not emitted as a value.
    assert "omitted [true]" in line


def test_skeleton_emits_nested_config_leaves_commented() -> None:
    """vLLM's typed nested config interiors are emitted COMMENTED (opt-in).

    A wholesale-forwarded blob (compilation_config / speculative_config) carries
    dozens of auto-expandable leaves; emitting them all as live sweep axes would
    blow the skeleton's Cartesian product into the billions. So each nested leaf
    is commented out for the researcher to uncomment and trim.
    """
    text = build_study_skeleton("vllm")
    assert "nested config interiors" in text
    # A nested Literal leaf appears with its resolved set, behind a leading '#'.
    nested = [line for line in text.splitlines() if "compilation_config.mode" in line]
    assert len(nested) == 1
    assert nested[0].lstrip().startswith("# vllm.engine_params.compilation_config.mode:")
    assert "[0, 1, 2, 3]" in nested[0]


def test_skeleton_nested_leaves_are_not_live_sweep_axes() -> None:
    """Nested-config leaves stay commented, so they never enter the live sweep dict."""
    parsed = yaml.safe_load(build_study_skeleton("vllm"))
    sweep = parsed.get("sweep") or {}
    assert all("compilation_config" not in key for key in sweep)
    assert all("speculative_config" not in key for key in sweep)


def test_skeleton_unknown_engine_raises() -> None:
    with pytest.raises(ConfigError, match="Unknown engine"):
        build_study_skeleton("bogus")
