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


def test_bounded_float_exclusive_upper_stays_in_range() -> None:
    """vllm gpu_memory_utilization (ge=0.0, lt=1.0) keeps every value in range."""
    values = resolve_auto_values("vllm.engine_params.gpu_memory_utilization")
    assert all(0.0 <= v < 1.0 for v in values)
    assert values[0] == 0.0  # inclusive lower used verbatim
    assert values[-1] < 1.0  # exclusive upper nudged inward


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


def test_skeleton_unknown_engine_raises() -> None:
    with pytest.raises(ConfigError, match="Unknown engine"):
        build_study_skeleton("bogus")
