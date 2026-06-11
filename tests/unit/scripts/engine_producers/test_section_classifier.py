"""Per-engine section-classifier emission tests (D2).

Pins the ``{engine}.{section}.{field}`` rule-path the classifier assigns each
field, including the known tricky cases where curation overrides the native
section (transformers ``num_beams`` etc. -> engine_params) and where a
non-curated field's native class decides (vllm ``seed`` -> sampling_params;
transformers ``compile_config`` -> sampling_params; ``llm_int8_*`` ->
engine_params). The expected sections are cross-checked against the committed
runtime-proven corpus by the per-engine oracle test elsewhere; these tests pin
the classifier logic in isolation against each pin's real curated.yaml.
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
    classify_section,
    field_path,
    load_curated_sections,
    relabel_match_fields,
)

# (engine, native_type, field, expected_section). Each row mirrors a path the
# committed corpus carries (runtime-proven) or a documented tricky case.
_CASES: tuple[tuple[str, str, str, str], ...] = (
    # transformers: curation overrides GenerationConfig's sampling-side default.
    ("transformers", "transformers.GenerationConfig", "num_beams", "engine_params"),
    ("transformers", "transformers.GenerationConfig", "early_stopping", "engine_params"),
    ("transformers", "transformers.GenerationConfig", "use_cache", "engine_params"),
    ("transformers", "transformers.GenerationConfig", "cache_implementation", "engine_params"),
    ("transformers", "transformers.GenerationConfig", "length_penalty", "engine_params"),
    # transformers: non-curated GenerationConfig field -> sampling via native origin.
    ("transformers", "transformers.GenerationConfig", "compile_config", "sampling_params"),
    ("transformers", "transformers.GenerationConfig", "watermarking_config", "sampling_params"),
    ("transformers", "transformers.GenerationConfig", "epsilon_cutoff", "sampling_params"),
    # transformers: curated sampling fields stay sampling.
    ("transformers", "transformers.GenerationConfig", "temperature", "sampling_params"),
    ("transformers", "transformers.GenerationConfig", "top_k", "sampling_params"),
    # transformers: BitsAndBytesConfig -> engine, curated or not.
    ("transformers", "transformers.BitsAndBytesConfig", "load_in_4bit", "engine_params"),
    ("transformers", "transformers.BitsAndBytesConfig", "bnb_4bit_quant_type", "engine_params"),
    ("transformers", "transformers.BitsAndBytesConfig", "llm_int8_threshold", "engine_params"),
    ("transformers", "transformers.BitsAndBytesConfig", "llm_int8_skip_modules", "engine_params"),
    # vllm: non-curated SamplingParams field -> sampling via native origin.
    ("vllm", "vllm.SamplingParams", "seed", "sampling_params"),
    ("vllm", "vllm.SamplingParams", "best_of", "sampling_params"),
    ("vllm", "vllm.SamplingParams", "max_tokens", "sampling_params"),
    # vllm: curated sampling fields stay sampling.
    ("vllm", "vllm.SamplingParams", "temperature", "sampling_params"),
    ("vllm", "vllm.SamplingParams", "n", "sampling_params"),
    # vllm: SchedulerConfig -> engine, curated or not.
    ("vllm", "vllm.config.SchedulerConfig", "num_scheduler_steps", "engine_params"),
    ("vllm", "vllm.config.SchedulerConfig", "max_num_batched_tokens", "engine_params"),
    ("vllm", "vllm.config.SchedulerConfig", "num_lookahead_slots", "engine_params"),
    # tensorrt: llm_args-side natives -> engine.
    ("tensorrt", "tensorrt_llm.LookaheadDecodingConfig", "max_ngram_size", "engine_params"),
    (
        "tensorrt",
        "tensorrt_llm.LookaheadDecodingConfig",
        "max_verification_set_size",
        "engine_params",
    ),
    ("tensorrt", "tensorrt_llm.LookaheadDecodingConfig", "max_window_size", "engine_params"),
    ("tensorrt", "tensorrt_llm.BaseLlmArgs", "load_format", "engine_params"),
    ("tensorrt", "tensorrt_llm.BaseLlmArgs", "tokenizer_mode", "engine_params"),
)


@pytest.fixture(scope="module")
def curated() -> dict[str, dict[str, str]]:
    """Real curated.yaml section maps for all three pins."""
    return {
        engine: load_curated_sections(engine) for engine in ("transformers", "vllm", "tensorrt")
    }


@pytest.mark.parametrize(("engine", "native_type", "field", "expected"), _CASES)
def test_classify_section(
    engine: str,
    native_type: str,
    field: str,
    expected: str,
    curated: dict[str, dict[str, str]],
) -> None:
    assert classify_section(engine, native_type, field, curated[engine]) == expected


@pytest.mark.parametrize(("engine", "native_type", "field", "expected"), _CASES)
def test_field_path(
    engine: str,
    native_type: str,
    field: str,
    expected: str,
    curated: dict[str, dict[str, str]],
) -> None:
    assert field_path(engine, native_type, field, curated[engine]) == f"{engine}.{expected}.{field}"


def test_curated_wins_over_native_origin(curated: dict[str, dict[str, str]]) -> None:
    # num_beams is a GenerationConfig (sampling-side) native, but curation puts
    # it in engine_params - curation is the SSOT, so it wins.
    assert curated["transformers"]["num_beams"] == "engine_params"
    assert (
        classify_section(
            "transformers", "transformers.GenerationConfig", "num_beams", curated["transformers"]
        )
        == "engine_params"
    )


def test_unknown_native_class_fails_loud(curated: dict[str, dict[str, str]]) -> None:
    with pytest.raises(UnknownNativeClassError):
        classify_section(
            "vllm", "vllm.config.MysteryConfig", "not_a_curated_field", curated["vllm"]
        )


def test_relabel_match_fields_rekeys_and_preserves_refs(
    curated: dict[str, dict[str, str]],
) -> None:
    # A stale-namespace match dict with a bare @ref rhs and a multi-op spec.
    stale = {
        "transformers.sampling.num_beams": {"<": "@num_return_sequences"},
        "transformers.sampling.compile_config": {"type_is_not": "CompileConfig"},
    }
    relabelled = relabel_match_fields(
        stale,
        engine="transformers",
        native_type="transformers.GenerationConfig",
        curated_sections=curated["transformers"],
    )
    assert relabelled == {
        # curated -> engine_params; @ref rhs is preserved bare (same-section sibling).
        "transformers.engine_params.num_beams": {"<": "@num_return_sequences"},
        # non-curated GenerationConfig -> sampling_params.
        "transformers.sampling_params.compile_config": {"type_is_not": "CompileConfig"},
    }


def test_relabel_handles_nested_namespace(curated: dict[str, dict[str, str]]) -> None:
    # Watermarking sub-walk emits a nested namespace; the field name is the
    # rightmost component and the section is recomputed.
    stale = {"transformers.sampling.watermarking_config.greenlist_ratio": {">": 1.0}}
    relabelled = relabel_match_fields(
        stale,
        engine="transformers",
        native_type="transformers.WatermarkingConfig",
        curated_sections=curated["transformers"],
    )
    assert relabelled == {"transformers.sampling_params.greenlist_ratio": {">": 1.0}}
