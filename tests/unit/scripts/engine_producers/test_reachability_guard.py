"""Mine-time reachability guard tests for the section classifier.

The guard (``scripts.engine_producers._section_classifier``) consults a
per-engine plugin-routing model (``_REACHABILITY``) when re-keying mined
``match.fields`` onto runtime paths:

- a flat-hoisted leaf keeps its byte-identical ``{engine}.{section}.{leaf}`` path;
- a routed blob interior becomes ``{engine}.{section}.{container}.{leaf}``;
- a leaf the plugin never routes (the 8 confirmed-dead shapes) FAILS LOUD with
  :class:`UnreachableMatchPathError` (or is dropped under ``allow_unreachable="drop"``);
- an unknown native class fails loud with :class:`UnknownNativeClassError`.

A host-only structural net (:func:`assert_path_resolves`) independently proves
an emitted path resolves against the committed generated Config + discovered
schema - the durable backstop that kills the dead-path class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._section_classifier import (  # noqa: E402
    UnknownNativeClassError,
    UnreachableMatchPathError,
    UnresolvableMatchPathError,
    assert_path_resolves,
    load_curated_sections,
    relabel_match_fields,
    relabel_match_fields_with_drops,
)


@pytest.fixture(scope="module")
def curated() -> dict[str, dict[str, str]]:
    return {
        engine: load_curated_sections(engine) for engine in ("transformers", "vllm", "tensorrt")
    }


# The 8 confirmed-dead (silently-dead) shapes: native classes / leaves the
# plugin never routes the value to. Each is (engine, native_type, leaf).
_DEAD_SHAPES: tuple[tuple[str, str, str], ...] = (
    # tensorrt: LookaheadDecodingConfig / TorchCompileConfig are NOT handled by
    # the plugin's _build_llm_kwargs at all.
    ("tensorrt", "tensorrt_llm.LookaheadDecodingConfig", "max_ngram_size"),
    ("tensorrt", "tensorrt_llm.LookaheadDecodingConfig", "max_verification_set_size"),
    ("tensorrt", "tensorrt_llm.LookaheadDecodingConfig", "max_window_size"),
    ("tensorrt", "tensorrt_llm.TorchCompileConfig", "max_num_streams"),
    # transformers: BitsAndBytesConfig is built from a CLOSED flat set that
    # EXCLUDES the llm_int8_* knobs.
    ("transformers", "transformers.BitsAndBytesConfig", "llm_int8_has_fp16_weight"),
    ("transformers", "transformers.BitsAndBytesConfig", "llm_int8_threshold"),
    ("transformers", "transformers.BitsAndBytesConfig", "llm_int8_skip_modules"),
    ("transformers", "transformers.BitsAndBytesConfig", "llm_int8_enable_fp32_cpu_offload"),
)


@pytest.mark.parametrize(("engine", "native_type", "leaf"), _DEAD_SHAPES)
def test_dead_shapes_fail_loud(
    engine: str, native_type: str, leaf: str, curated: dict[str, dict[str, str]]
) -> None:
    with pytest.raises(UnreachableMatchPathError):
        relabel_match_fields(
            {f"ns.{leaf}": {"<": 0}},
            engine=engine,
            native_type=native_type,
            curated_sections=curated[engine],
        )


@pytest.mark.parametrize(("engine", "native_type", "leaf"), _DEAD_SHAPES)
def test_dead_shapes_drop_under_allow_unreachable(
    engine: str, native_type: str, leaf: str, curated: dict[str, dict[str, str]]
) -> None:
    relabelled, dropped = relabel_match_fields_with_drops(
        {f"ns.{leaf}": {"<": 0}},
        engine=engine,
        native_type=native_type,
        curated_sections=curated[engine],
        allow_unreachable="drop",
    )
    assert relabelled == {}
    assert len(dropped) == 1
    assert dropped[0][0] == f"ns.{leaf}"


def test_tensorrt_quant_blob_leaf_survives_nested(curated: dict[str, dict[str, str]]) -> None:
    # quant_config.quant_algo is in the plugin's read allowlist -> nested path.
    relabelled = relabel_match_fields(
        {"ns.quant_algo": {"type_is_not": "str"}},
        engine="tensorrt",
        native_type="tensorrt_llm.QuantConfig",
        curated_sections=curated["tensorrt"],
    )
    assert relabelled == {"tensorrt.engine_params.quant_config.quant_algo": {"type_is_not": "str"}}


def test_tensorrt_quant_blob_leaf_outside_allowlist_unreachable(
    curated: dict[str, dict[str, str]],
) -> None:
    # A QuantConfig interior leaf the plugin never reads is unreachable.
    with pytest.raises(UnreachableMatchPathError):
        relabel_match_fields(
            {"ns.exclude_modules": {"present": True}},
            engine="tensorrt",
            native_type="tensorrt_llm.QuantConfig",
            curated_sections=curated["tensorrt"],
        )


def test_tensorrt_scheduler_policy_blob_survives(curated: dict[str, dict[str, str]]) -> None:
    relabelled = relabel_match_fields(
        {"ns.capacity_scheduling_policy": {"present": True}},
        engine="tensorrt",
        native_type="tensorrt_llm.SchedulerConfig",
        curated_sections=curated["tensorrt"],
    )
    assert relabelled == {
        "tensorrt.engine_params.scheduler_config.capacity_scheduling_policy": {"present": True}
    }


def test_transformers_load_in_4bit_stays_flat(curated: dict[str, dict[str, str]]) -> None:
    # BnB routed keys are FLAT (the user sets them flat), NOT a nested blob.
    relabelled = relabel_match_fields(
        {"ns.load_in_4bit": {"present": True}},
        engine="transformers",
        native_type="transformers.BitsAndBytesConfig",
        curated_sections=curated["transformers"],
    )
    assert relabelled == {"transformers.engine_params.load_in_4bit": {"present": True}}


def test_vllm_speculative_interior_leaf_becomes_nested(
    curated: dict[str, dict[str, str]],
) -> None:
    # speculative_config is forwarded WHOLESALE -> interior leaf is blob-reachable.
    relabelled = relabel_match_fields(
        {"ns.num_speculative_tokens": {"<": 0}},
        engine="vllm",
        native_type="vllm.config.SpeculativeConfig",
        curated_sections=curated["vllm"],
    )
    assert relabelled == {"vllm.engine_params.speculative_config.num_speculative_tokens": {"<": 0}}


def test_vllm_flat_hoisted_field_byte_identical(curated: dict[str, dict[str, str]]) -> None:
    # max_num_batched_tokens is flat-hoisted by EngineArgs - the emitted path must
    # be byte-identical to the historical {engine}.{section}.{leaf} shape.
    relabelled = relabel_match_fields(
        {"vllm.config.SchedulerConfig.max_num_batched_tokens": {"<": 0}},
        engine="vllm",
        native_type="vllm.config.SchedulerConfig",
        curated_sections=curated["vllm"],
    )
    assert relabelled == {"vllm.engine_params.max_num_batched_tokens": {"<": 0}}


def test_unknown_native_class_fails_loud(curated: dict[str, dict[str, str]]) -> None:
    with pytest.raises(UnknownNativeClassError):
        relabel_match_fields(
            {"ns.whatever": {"<": 0}},
            engine="vllm",
            native_type="vllm.config.MysteryConfig",
            curated_sections=curated["vllm"],
        )


# ---------------------------------------------------------------------------
# Host-only structural RESOLVES net
# ---------------------------------------------------------------------------


def test_assert_path_resolves_flat() -> None:
    assert_path_resolves("vllm", "vllm.engine_params.max_num_batched_tokens")
    assert_path_resolves("transformers", "transformers.engine_params.load_in_4bit")


def test_assert_path_resolves_blob_interior() -> None:
    assert_path_resolves("vllm", "vllm.engine_params.speculative_config.num_speculative_tokens")
    assert_path_resolves("tensorrt", "tensorrt.engine_params.quant_config.quant_algo")


def test_assert_path_resolves_rejects_bogus_leaf() -> None:
    with pytest.raises(UnresolvableMatchPathError):
        assert_path_resolves("vllm", "vllm.engine_params.totally_not_a_field_xyz")


def test_assert_path_resolves_rejects_bad_section() -> None:
    with pytest.raises(UnresolvableMatchPathError):
        assert_path_resolves("vllm", "vllm.not_a_section.dtype")


def test_verify_resolves_wired_into_relabel(curated: dict[str, dict[str, str]]) -> None:
    # A reachable, structurally-valid leaf passes the RESOLVES net.
    relabelled = relabel_match_fields(
        {"vllm.config.SchedulerConfig.max_num_batched_tokens": {"<": 0}},
        engine="vllm",
        native_type="vllm.config.SchedulerConfig",
        curated_sections=curated["vllm"],
        verify_resolves=True,
    )
    assert relabelled == {"vllm.engine_params.max_num_batched_tokens": {"<": 0}}
