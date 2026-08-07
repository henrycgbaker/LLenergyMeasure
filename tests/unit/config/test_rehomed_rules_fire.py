"""Firing tests for shipped rules.yaml rules against the nested config shape.

Covers the re-homed single-field bounds (item 6b of the pin-advance) and the
transformers cross-section ``@field_ref`` rules. Each rule must resolve at
``Rule.try_match`` against the generated nested config shape
(``<engine>.engine_params.<field>`` / ``<engine>.sampling_params.<field>``); a
path that does not resolve silently never fires. These tests construct a
violating config through the public ``ExperimentConfig`` path and assert the
rule rejects it, covering every engine and both sub-sections so the path shape
is proven for the whole set.
"""

from __future__ import annotations

from typing import Literal

import pytest

from llenergymeasure.config.engine_rules import EngineRulesLoader
from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.config.models import ExperimentConfig

# (engine, section, field, violating_value, rule_id_fragment)
_CASES = [
    # vLLM - cited sampling ranges
    ("vllm", "sampling_params", "presence_penalty", -3.0, "presence_penalty"),
    ("vllm", "sampling_params", "presence_penalty", 3.0, "presence_penalty"),
    ("vllm", "sampling_params", "frequency_penalty", -3.0, "frequency_penalty"),
    ("vllm", "sampling_params", "top_p", 1.5, "top_p"),
    ("vllm", "sampling_params", "top_p", 0.0, "top_p"),
    ("vllm", "sampling_params", "min_p", -0.1, "min_p"),
    # vLLM - engine positivity floors
    ("vllm", "engine_params", "tensor_parallel_size", 0, "tensor_parallel_size"),
    ("vllm", "engine_params", "pipeline_parallel_size", 0, "pipeline_parallel_size"),
    ("vllm", "engine_params", "kv_cache_memory_bytes", 0, "kv_cache_memory_bytes"),
    # tensorrt - engine floors
    ("tensorrt", "engine_params", "max_seq_len", 0, "max_seq_len"),
    ("tensorrt", "engine_params", "tensor_parallel_size", 0, "tensor_parallel_size"),
    ("tensorrt", "engine_params", "max_num_tokens", 0, "max_num_tokens"),
    # tensorrt - sampling floors/ranges
    ("tensorrt", "sampling_params", "temperature", -1.0, "temperature"),
    ("tensorrt", "sampling_params", "top_p", 1.5, "top_p"),
    ("tensorrt", "sampling_params", "n", 0, "n"),
    # transformers - engine floors
    ("transformers", "engine_params", "prompt_lookup_num_tokens", 0, "prompt_lookup_num_tokens"),
    # transformers - sampling floors/ranges
    ("transformers", "sampling_params", "top_p", 2.0, "top_p"),
    ("transformers", "sampling_params", "temperature", -0.5, "temperature"),
    ("transformers", "sampling_params", "top_h", 1.5, "top_h"),
    ("transformers", "sampling_params", "typical_p", 1.5, "typical_p"),
    ("transformers", "sampling_params", "min_length", -1, "min_length"),
]


@pytest.mark.parametrize("engine,section,field,value,fragment", _CASES)
def test_rehomed_rule_rejects_violating_config(engine, section, field, value, fragment):
    """A violating value on the canonical nested path is rejected with the rule id."""
    section_payload = {section: {field: value}}
    with pytest.raises(ValueError, match=fragment):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine=engine,
            serving_mode="offline",
            **{engine: section_payload},
        )


def test_in_range_value_is_accepted():
    """A value inside every bound constructs cleanly (rules do not over-fire)."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={
            "engine_params": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            "sampling_params": {"presence_penalty": 0.5, "top_p": 0.9, "min_p": 0.0},
        },
    )
    assert cfg.active_sampling_params().top_p == 0.9


# ---------------------------------------------------------------------------
# Cross-section @field_ref rules
#
# These rules compare a field in one section against a root-dotted reference
# into the other (engine_params.num_beams vs
# @transformers.sampling_params.num_return_sequences). A reference that fails
# to resolve silently never fires, so presence in the corpus proves nothing -
# each rule must be driven to fire through the public ExperimentConfig path.
# ---------------------------------------------------------------------------


def test_cross_section_num_beams_lt_num_return_sequences_fires():
    """num_beams below sampling_params.num_return_sequences is rejected."""
    with pytest.raises(
        ValueError, match="transformers_num_return_vs_beams_num_beams_lt_num_return_sequences"
    ):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={
                "engine_params": {"num_beams": 2},
                "sampling_params": {"num_return_sequences": 4},
            },
        )


def test_cross_section_num_return_sequences_gt_num_beams_fires():
    """num_return_sequences above engine_params.num_beams is rejected.

    The real upstream constraint is num_return_sequences <= num_beams (not a
    divisibility check), so a value strictly greater than num_beams must be
    rejected regardless of divisibility. The re-encoded rule is asserted to
    fire in isolation (a sibling `num_beams < @num_return_sequences` rule
    encodes the same constraint and may short-circuit at the ExperimentConfig
    layer, so the config-path assertion only pins rejection, not which id).
    """
    from llenergymeasure.config.engine_rules import EngineRulesLoader

    rules = EngineRulesLoader().load_rules("transformers").rules
    violating = {
        "transformers": {
            "engine_params": {"num_beams": 4},
            "sampling_params": {"num_return_sequences": 5},
        }
    }
    fired = {r.id for r in rules if r.try_match(violating)}
    assert "transformers_num_return_vs_beams_num_return_sequences_gt_num_beams" in fired

    with pytest.raises(ValueError, match="num_return_sequences"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={
                "engine_params": {"num_beams": 4},
                "sampling_params": {"num_return_sequences": 5},
            },
        )


def test_cross_section_non_divisor_return_count_accepted():
    """num_return_sequences <= num_beams but not a divisor constructs cleanly.

    Regression guard for the re-encode: 3 return sequences with 4 beams is not
    a divisor pair yet is valid upstream, so the old not_divisible_by rule was a
    false positive here.
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={
            "engine_params": {"num_beams": 4},
            "sampling_params": {"num_return_sequences": 3},
        },
    )
    assert cfg.active_engine_params().num_beams == 4
    assert cfg.active_sampling_params().num_return_sequences == 3


# ---------------------------------------------------------------------------
# Type-incomparable predicate values are no-match, not an uncaught crash
#
# ``compile_config`` naturally holds a dict / CompileConfig shape, not a number,
# so any ordering predicate (e.g. ``{">": 0}``) mined against it compares
# incomparable types. The loader must treat such a pair as no-match so the
# ordering rule silently declines, rather than letting a raw TypeError
# (``'>' not supported between instances of 'dict' and 'int'``) escape config
# construction. A separate, legitimate ``type_is_not: [CompileConfig]`` rule
# then cleanly rejects a non-CompileConfig value with a ValueError - a clean
# rule rejection, never a TypeError. The generated pydantic model stays the
# authority on type validity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compile_config",
    [{"backend": "inductor"}, "inductor", ["inductor"]],
)
def test_non_numeric_compile_config_no_uncaught_typeerror(compile_config):
    """A dict / str / list compile_config yields a clean rejection, not a TypeError."""
    with pytest.raises(ValueError, match="type_not_in_CompileConfig") as excinfo:
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"sampling_params": {"compile_config": compile_config}},
        )
    # The bug was an uncaught TypeError from the mined ``{">": 0}`` bound; the
    # surviving rejection must be the clean type rule, never a TypeError.
    assert not isinstance(excinfo.value, TypeError)


def test_numeric_bound_still_fires_after_incomparable_guard():
    """Control: guarding incomparable types must not suppress real numeric matches."""
    with pytest.raises(ValueError, match="min_length"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"sampling_params": {"min_length": -1}},
        )


# ---------------------------------------------------------------------------
# Purged false-positive rules: legitimate configs must now validate cleanly.
#
# Each of these values was rejected by a probe-confirmed false-positive corpus
# rule (an over-fit allowlist or a mis-mined type bound). After the purge they
# must construct through the public ExperimentConfig path without raising.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,field,value",
    [
        # early_stopping=True enables beam-search early stopping (was rejected
        # by the {'>': 0} bound, since True > 0 in Python).
        ("engine_params", "early_stopping", True),
        # max_new_tokens=256 (was rejected by an over-fit not_in [1, 16]).
        ("sampling_params", "max_new_tokens", 256),
        # cache_implementation="sliding_window" is a documented value (was
        # rejected by a not_in [dynamic, static] allowlist).
        ("engine_params", "cache_implementation", "sliding_window"),
    ],
)
def test_legitimate_transformers_value_validates_cleanly(section, field, value):
    """A documented, valid transformers value constructs without raising."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={section: {field: value}},
    )
    assert getattr(getattr(cfg, "active_" + section)(), field) == value


def test_bare_vllm_config_has_no_phantom_dormant_observations():
    """The three deleted vllm `absent: true` rules no longer fire on a bare config.

    Each rule matched a field that was simply never set and normalised it to
    absence - a guaranteed no-op that fired a phantom dormant observation on
    every config. A bare vllm config must now yield zero dormant observations
    from those rules.
    """
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="offline")
    purged = {
        "vllm_samplingparams_dormant_bad_words_unset_true",
        "vllm_samplingparams_dormant_skip_reading_prefix_cache_unset_true",
        "vllm_samplingparams_dormant_stop_token_ids_unset_true",
    }
    assert not (set(cfg._dormant_observations) & purged)
    assert cfg._dormant_observations == {}


# ---------------------------------------------------------------------------
# Nested-object rule paths (engine_params.compilation_config.*)
#
# The 2026-07-11 miner recall check surfaced a construction-confirmed vLLM
# bound on a nested CompilationConfig field (probe verdict: confirmed, both
# legs, vllm/vllm-openai:v0.19.1). Its rule is the first corpus entry whose
# match path descends through a nested sub-model, so these tests pin that the
# 4-segment path resolves through the public ExperimentConfig route in both
# directions (fires on the violating pair, silent when satisfied or unset).
# ---------------------------------------------------------------------------


def test_nested_compilation_config_bound_fires():
    """cudagraph_mm_encoder=True with a negative image budget is rejected."""
    with pytest.raises(ValueError, match="encoder_cudagraph_max_images_per_batch"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            serving_mode="offline",
            vllm={
                "engine_params": {
                    "compilation_config": {
                        "cudagraph_mm_encoder": True,
                        "encoder_cudagraph_max_images_per_batch": -1,
                    }
                }
            },
        )


@pytest.mark.parametrize(
    "compilation_config",
    [
        # Satisfying pair: precondition holds, budget is non-negative.
        {"cudagraph_mm_encoder": True, "encoder_cudagraph_max_images_per_batch": 0},
        # Precondition off: a negative budget alone does not raise upstream.
        {"cudagraph_mm_encoder": False, "encoder_cudagraph_max_images_per_batch": -1},
        # Nested object entirely absent.
        None,
    ],
)
def test_nested_compilation_config_bound_silent_when_not_violated(compilation_config):
    """The nested rule stays silent on satisfying, precondition-off, and unset shapes."""
    engine_params = (
        {"compilation_config": compilation_config} if compilation_config is not None else {}
    )
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": engine_params},
    )
    assert cfg.engine == "vllm"


# ---------------------------------------------------------------------------
# Runtime-literal discovery: transformers early_stopping accepts "never".
#
# The typed union widens to ``bool | Literal["never"] | None`` once the caller
# regenerates the shipped config.py from the runtime-literal-augmented schema.
# The four tests below that construct or introspect that widened surface
# (public-path acceptance, the loader path, the round-trip, and the annotation
# golden, plus the "silent on a constructed never config" check) WILL FAIL until
# that regeneration lands; they are written to pass afterwards. The dict-grain
# rule-firing test and the union-rejection test do not depend on regeneration.
# ---------------------------------------------------------------------------

_NEVER_RULE_IDS = {
    "transformers_early_stopping_type_early_stopping_not_in_allowlist",
    "transformers_raises_early_stopping_not_in_set",
}


def test_early_stopping_never_accepted_public_path():
    """early_stopping="never" constructs and is preserved through ExperimentConfig."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"early_stopping": "never"}},
    )
    assert cfg.transformers.engine_params.early_stopping == "never"


def test_early_stopping_never_accepted_via_loader():
    """The same value survives the CLI-override loader path."""
    cfg = load_experiment_config(
        path=None,
        cli_overrides={
            "task.model": "gpt2",
            "engine": "transformers",
            "serving_mode": "offline",
            "transformers.engine_params.early_stopping": "never",
        },
    )
    assert cfg.transformers.engine_params.early_stopping == "never"


def test_early_stopping_never_survives_roundtrip():
    """A model_dump / re-construct round-trip preserves the literal value."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"early_stopping": "never"}},
    )
    reloaded = ExperimentConfig(**cfg.model_dump(mode="json"))
    assert reloaded.transformers.engine_params.early_stopping == "never"


def test_early_stopping_generated_annotation_golden():
    """The generated EngineParams field annotation is the widened union."""
    from llenergymeasure.config.generated.transformers import EngineParams

    annotation = EngineParams.model_fields["early_stopping"].annotation
    assert annotation == (bool | Literal["never"] | None)


def test_early_stopping_allowlist_rules_fire_at_try_match_grain():
    """The early_stopping allowlist rules fire on out-of-allowlist / wrong-type values.

    Both surviving rules encode the allowlist as ``not_in`` a fixed set, so any
    non-member - a nonsense string or a wrong-type float - is rejected. Dict-grain
    firing does not depend on the regenerated config: it exercises the shipped
    rules corpus directly.
    """
    rules = EngineRulesLoader().load_rules("transformers").rules

    def fired(value: object) -> set[str]:
        cfg = {"transformers": {"engine_params": {"early_stopping": value}}}
        return {r.id for r in rules if r.try_match(cfg)}

    fired_on_string = fired("sometimes")
    assert "transformers_early_stopping_type_early_stopping_not_in_allowlist" in fired_on_string
    assert "transformers_raises_early_stopping_not_in_set" in fired_on_string

    fired_on_float = fired(1.5)
    assert "transformers_early_stopping_type_early_stopping_not_in_allowlist" in fired_on_float
    assert "transformers_raises_early_stopping_not_in_set" in fired_on_float


def test_early_stopping_rules_silent_on_constructed_never_config():
    """The allowlist member "never" is now reachable and clean: no rule fires on it."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"early_stopping": "never"}},
    )
    rules = EngineRulesLoader().load_rules("transformers").rules
    fired = {r.id for r in rules if r.try_match(cfg)}
    assert not (fired & _NEVER_RULE_IDS)


def test_pydantic_union_rejects_sometimes_public_path():
    """A non-member string is still rejected through the public path.

    The typed union (``bool | Literal["never"] | None``) is the first guard; the
    corpus error rules remain the dict-grain / passthrough backstop. Either path
    raising is a rejection - the point is that widening to accept "never" did NOT
    open the door to arbitrary strings.
    """
    with pytest.raises(ValueError):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"engine_params": {"early_stopping": "sometimes"}},
        )
