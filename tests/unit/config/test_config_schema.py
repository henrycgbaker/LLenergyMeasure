"""Unit tests for ExperimentConfig Pydantic validation.

Tests v2.0 field renames, extra=forbid, engine composition, cross-validators,
and schema-driven dtype validation using SSOT constants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig, OutputConfig
from llenergymeasure.config.ssot import DTYPE_SUPPORT
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Minimal valid config
# ---------------------------------------------------------------------------


def test_minimal_valid_config():
    """ExperimentConfig(task={'model': 'gpt2'}, engine='transformers') succeeds."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert config.task.model == "gpt2"
    assert config.engine == "transformers"


def test_model_only_uses_pytorch_default():
    """ExperimentConfig with only model= uses engine='transformers' default."""
    config = ExperimentConfig(task={"model": "gpt2"})
    assert config.engine == "transformers"


# ---------------------------------------------------------------------------
# extra=forbid
# ---------------------------------------------------------------------------


def test_extra_fields_forbidden():
    """Unknown top-level fields are rejected with ValidationError (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, engine="transformers", unknown_field="x")  # type: ignore[call-arg]  # asserts extra rejected


def test_multiple_extra_fields_all_rejected():
    """Multiple unknown fields are all rejected."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, engine="transformers", foo="a", bar="b")  # type: ignore[call-arg]  # asserts extra rejected


# ---------------------------------------------------------------------------
# v2.0 field renames
# ---------------------------------------------------------------------------


def test_field_name_model():
    """v2.0 'model' field (not 'model_name') is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"})
    assert config.task.model == "gpt2"


def test_field_name_dtype():
    """dtype is per-engine; transformers nests it under engine_params."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "float16"}},
    )
    assert config.transformers is not None
    assert config.transformers.engine_params.dtype == "float16"


def test_field_name_n():
    """v2.0 dataset.n_prompts field (not 'num_input_prompts') is accepted."""
    from llenergymeasure.config.models import DatasetConfig

    config = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)})
    assert config.task.dataset.n_prompts == 50


def test_v1x_field_model_name_rejected():
    """v1.x 'model_name' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, model_name="gpt2")  # type: ignore[call-arg]


def test_v1x_field_fp_precision_rejected():
    """v1.x 'fp_precision' field is NOT accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, fp_dtype="float16")  # type: ignore[call-arg]


def test_top_level_dtype_rejected():
    """Top-level dtype is rejected - dtype lives per-engine (extra='forbid')."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, dtype="float16")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Engine validation
# ---------------------------------------------------------------------------


def test_invalid_engine_raises_validation_error():
    """Unknown engine value raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, engine="invalid_backend")


def test_default_engine_is_pytorch():
    """Default engine is 'transformers' when not specified."""
    config = ExperimentConfig(task={"model": "gpt2"})
    assert config.engine == "transformers"


def test_vllm_engine_accepted():
    """engine='vllm' is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="vllm")
    assert config.engine == "vllm"


def test_tensorrt_engine_accepted():
    """engine='tensorrt' is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="tensorrt")
    assert config.engine == "tensorrt"


# ---------------------------------------------------------------------------
# Engine section composition
# ---------------------------------------------------------------------------


def test_pytorch_config_section_composition():
    """config with a transformers engine_params section is accepted.

    batch_size is now a HarnessConfig knob (see test_harness.py); the engine
    section carries engine-native fields under engine_params.
    """
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"device_map": "auto"}},
    )
    assert config.transformers is not None
    assert config.transformers.engine_params.device_map == "auto"


# ---------------------------------------------------------------------------
# Generated transformers Config: no num_processes field
# ---------------------------------------------------------------------------


def test_transformers_engine_params_has_no_num_processes_field():
    """The generated transformers EngineParams has no num_processes field."""
    from llenergymeasure.engines.transformers.config import EngineParams

    assert "num_processes" not in EngineParams.model_fields


def test_transformers_num_processes_not_a_declared_field():
    """num_processes is not a declared field on the generated EngineParams.

    The generated Config uses extra='allow' for HuggingFace passthrough, so an
    unknown kwarg is accepted into model_extra but is NOT a typed model field.
    """
    from llenergymeasure.engines.transformers.config import EngineParams

    assert "num_processes" not in EngineParams.model_fields
    params = EngineParams(num_processes=4)  # type: ignore[call-arg]
    assert "num_processes" not in type(params).model_fields


def test_pytorch_section_with_wrong_engine_rejected():
    """pytorch: section with engine='vllm' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"transformers.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            transformers={"batch_size": 4},
        )


def test_vllm_section_with_pytorch_engine_rejected():
    """vllm: section with engine='transformers' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"vllm.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            vllm={"engine": {"max_num_seqs": 16}},
        )


def test_tensorrt_section_with_wrong_engine_rejected():
    """tensorrt: section with engine='transformers' raises ValidationError."""
    with pytest.raises(ValidationError, match=r"tensorrt.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            tensorrt={"max_batch_size": 8},
        )


# ---------------------------------------------------------------------------
# Precision validation
# ---------------------------------------------------------------------------


def test_transformers_dtype_no_longer_literal_validated():
    """transformers engine_params.dtype is a curated Any passthrough after migration.

    The hand-written TransformersConfig typed dtype as a 3-value Literal that
    rejected 'fp16'; the generated engine_params.dtype is Any (HF **kwargs
    discovery debt), so any string is accepted and the engine validates at load.
    vllm/tensorrt keep their dtype Literals (see test_vllm_dtype_float32_rejected).
    """
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "fp16"}},
    )
    assert config.transformers.engine_params.dtype == "fp16"


@pytest.mark.parametrize("dt", ["float32", "float16", "bfloat16"])
def test_valid_dtype_values_accepted(dt):
    """Standard dtype strings are accepted on transformers engine_params."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": dt}},
    )
    assert config.transformers.engine_params.dtype == dt


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["transformers"])  # type: ignore[index]  # Engine is str-enum
def test_all_pytorch_dtypes_valid(dt):
    """Schema-driven: all SSOT DTYPE_SUPPORT['transformers'] values are valid."""
    config = make_config(dtype=dt)
    assert config.transformers.engine_params.dtype == dt


# ---------------------------------------------------------------------------
# passthrough_kwargs collision cross-validator
# ---------------------------------------------------------------------------


def test_passthrough_kwargs_accepted():
    """passthrough_kwargs with non-colliding keys are accepted."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        passthrough_kwargs={"custom_flag": True, "my_special_param": 42},
    )
    assert config.passthrough_kwargs is not None
    assert config.passthrough_kwargs["custom_flag"] is True


def test_passthrough_kwargs_collision_with_top_level_field_rejected():
    """passthrough_kwargs keys colliding with ExperimentConfig fields are rejected."""
    with pytest.raises(ValidationError, match=r"passthrough_kwargs.*collide"):
        ExperimentConfig(
            task={"model": "gpt2"},
            passthrough_kwargs={"task": "override"},  # 'task' is a top-level field
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def test_make_config_helper_returns_valid_config():
    """make_config() factory from conftest creates a valid ExperimentConfig."""
    config = make_config()
    assert isinstance(config, ExperimentConfig)
    assert config.task.model == "gpt2"
    assert config.engine == "transformers"


def test_make_config_override():
    """make_config(**overrides) applies overrides over defaults (dtype -> engine_params)."""
    config = make_config(model="bert-base", dtype="float32")
    assert config.task.model == "bert-base"
    assert config.transformers.engine_params.dtype == "float32"


# ---------------------------------------------------------------------------
# energy_sampler field tests (flat Literal on ExperimentConfig)
# ---------------------------------------------------------------------------


def test_energy_sampler_default() -> None:
    """energy_sampler defaults to 'auto'."""
    cfg = ExperimentConfig(task={"model": "gpt2"})
    assert cfg.measurement.energy_sampler == "auto"


def test_energy_sampler_null_disables() -> None:
    """energy_sampler=None disables energy measurement."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": None})
    assert cfg.measurement.energy_sampler is None


def test_energy_sampler_valid_engines() -> None:
    """All energy_sampler literal values are accepted."""
    for engine in ("auto", "nvml", "zeus", "codecarbon"):
        cfg = ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": engine})
        assert cfg.measurement.energy_sampler == engine


def test_energy_sampler_invalid_engine() -> None:
    """Unknown energy_sampler values raise ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": "unknown_backend"})


def test_energy_sampler_override() -> None:
    """ExperimentConfig allows overriding energy_sampler."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, measurement={"energy_sampler": "nvml"})
    assert cfg.measurement.energy_sampler == "nvml"


# ---------------------------------------------------------------------------
# save_timeseries field tests (boolean on OutputConfig)
# ---------------------------------------------------------------------------


def test_save_timeseries_default_true() -> None:
    """save_timeseries defaults to True."""
    cfg = OutputConfig()
    assert cfg.save_timeseries is True


def test_save_timeseries_false_accepted() -> None:
    """save_timeseries=False is accepted."""
    cfg = OutputConfig(save_timeseries=False)
    assert cfg.save_timeseries is False


# ---------------------------------------------------------------------------
# Transformers tensor parallelism (tp_plan, tp_size) on the generated config
# ---------------------------------------------------------------------------
#
# After migration these are curated Any-typed engine_params fields (no Literal /
# ge constraint - that typing was lost to discovery debt). The V5 rule
# (tp_plan XOR device_map) is a conscious drop: enforcement is from_pretrained-
# side (execution grain), where the engine raises at model load, so it is no
# longer a config-parse validator. A deferred mining target.


def test_tp_plan_and_tp_size_accepted_on_engine_params():
    """tp_plan / tp_size pass through on the generated engine_params."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"tp_plan": "auto", "tp_size": 4}},
    )
    assert cfg.transformers.engine_params.tp_plan == "auto"
    assert cfg.transformers.engine_params.tp_size == 4


def test_tp_plan_with_device_map_no_longer_rejected_at_parse():
    """V5 (tp_plan XOR device_map) is dropped: the engine raises at model load.

    Config parse accepts both; from_pretrained enforces the exclusivity.
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"tp_plan": "auto", "device_map": "auto"}},
    )
    assert cfg.transformers.engine_params.tp_plan == "auto"
    assert cfg.transformers.engine_params.device_map == "auto"


# ---------------------------------------------------------------------------
# Bug 1.1 - fp8 quantization + float32 dtype (vLLM)
# ---------------------------------------------------------------------------


def test_vllm_dtype_float32_parses():
    """vllm dtype is an Any passthrough now (no Literal), so float32 parses.

    The generated config does not restrict dtype; vLLM itself rejects fp32 at
    runtime. This asserts the nested config accepts the value rather than the
    deleted Literal validator.
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"dtype": "float32"}},
    )
    assert cfg.vllm.engine_params.dtype == "float32"


def test_vllm_gpu_memory_utilization_above_one_rejected():
    """gpu_memory_utilization=1.5 errors at parse via the backfilled lt=1.0 bound.

    The R3 overlay restores the pre-v0.10 ge=0.0, lt=1.0 bound (a GPU memory
    fraction cannot reach or exceed 1.0). The generated config now refuses the
    out-of-range value at ExperimentConfig construction.
    """
    with pytest.raises(ValidationError):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            vllm={"engine_params": {"gpu_memory_utilization": 1.5}},
        )


def test_vllm_gpu_memory_utilization_in_range_accepted():
    """A valid gpu_memory_utilization fraction still parses after the backfill."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"gpu_memory_utilization": 0.85}},
    )
    assert cfg.vllm.engine_params.gpu_memory_utilization == 0.85


def test_vllm_fp8_float16_accepted():
    """fp8 quantization with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"dtype": "float16", "quantization": "fp8"}},
    )
    assert cfg.vllm.engine_params.dtype == "float16"


def test_vllm_fp8_bfloat16_accepted():
    """fp8 quantization with dtype=bfloat16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"dtype": "bfloat16", "quantization": "fp8"}},
    )
    assert cfg.vllm.engine_params.dtype == "bfloat16"


def test_vllm_non_fp8_float16_accepted():
    """Non-fp8 quantization (awq) with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"dtype": "float16", "quantization": "awq"}},
    )
    assert cfg.vllm.engine_params.dtype == "float16"


def test_vllm_no_quantization_default_dtype_accepted():
    """No quantization set, no explicit dtype, is accepted (engine default applies)."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {}},
    )
    # The generated default for vllm dtype is 'auto' (non-None).
    assert cfg.vllm.engine_params.dtype == "auto"


# ---------------------------------------------------------------------------
# Bug 1.2 - max_num_batched_tokens / max_model_len (vLLM engine)
#
# The hand-written max_num_batched_tokens >= max_model_len cross-validator was
# dropped with the migration to the generated config (vLLM enforces this at
# runtime). These tests now assert the nested config accepts the values.
# ---------------------------------------------------------------------------


def test_vllm_batched_tokens_equal_model_len_accepted():
    """max_num_batched_tokens == max_model_len is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"max_num_batched_tokens": 1024, "max_model_len": 1024}},
    )
    ep = cfg.vllm.engine_params
    assert ep.max_num_batched_tokens == 1024
    assert ep.max_model_len == 1024


def test_vllm_batched_tokens_greater_accepted():
    """max_num_batched_tokens > max_model_len is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"max_num_batched_tokens": 2048, "max_model_len": 1024}},
    )
    assert cfg.vllm.engine_params.max_num_batched_tokens == 2048


def test_vllm_batched_tokens_one_none_accepted():
    """Only one of max_num_batched_tokens / max_model_len set is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"max_num_batched_tokens": 512}},
    )
    ep = cfg.vllm.engine_params
    assert ep.max_num_batched_tokens == 512
    assert ep.max_model_len is None


# ---------------------------------------------------------------------------
# Bug 1.3 - flash_attention_2/3 + float32 dtype (PyTorch)
# ---------------------------------------------------------------------------


def test_pytorch_flash_attn2_float32_rejected():
    """flash_attention_2 with dtype=float32 raises ValidationError at parse time."""
    with pytest.raises(ValidationError, match=r"flash_attention_2.*requires.*float16"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers={
                "engine_params": {"dtype": "float32", "attn_implementation": "flash_attention_2"}
            },
        )


def test_pytorch_flash_attn3_float32_rejected():
    """flash_attention_3 with dtype=float32 raises ValidationError at parse time."""
    with pytest.raises(ValidationError, match=r"flash_attention_3.*requires.*float16"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers={
                "engine_params": {"dtype": "float32", "attn_implementation": "flash_attention_3"}
            },
        )


def test_pytorch_flash_attn2_bfloat16_accepted():
    """flash_attention_2 with dtype=bfloat16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={
            "engine_params": {"dtype": "bfloat16", "attn_implementation": "flash_attention_2"}
        },
    )
    assert cfg.transformers.engine_params.dtype == "bfloat16"


def test_pytorch_eager_float32_accepted():
    """attn_implementation=eager with dtype=float32 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "float32", "attn_implementation": "eager"}},
    )
    assert cfg.transformers.engine_params.dtype == "float32"


def test_pytorch_no_attn_impl_float32_accepted():
    """No attn_implementation set with dtype=float32 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "float32"}},
    )
    assert cfg.transformers.engine_params.dtype == "float32"


# ---------------------------------------------------------------------------
# TRT FP8+float32 cross-validator
# ---------------------------------------------------------------------------


def test_trt_dtype_float32_rejected() -> None:
    """tensorrt dtype rejects float32 at parse via the backfilled Literal.

    The R3 overlay restores the pre-v0.10 dtype Literal['float16', 'bfloat16']
    (TRT-LLM is optimised for fp16/bf16; fp32 unsupported), so the generated
    config refuses float32 at parse rather than deferring to TRT-LLM's runtime
    rejection. float16/bfloat16 still parse (see test_trt_fp8_accepts_*).
    """
    with pytest.raises(ValidationError):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="tensorrt",
            tensorrt={"engine_params": {"dtype": "float32"}},
        )


def test_trt_fp8_accepts_float16() -> None:
    """FP8 quantization with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt={"engine_params": {"dtype": "float16", "quant_config": {"quant_algo": "FP8"}}},
    )
    assert cfg.tensorrt.engine_params.dtype == "float16"


def test_trt_fp8_accepts_bfloat16() -> None:
    """FP8 quantization with dtype=bfloat16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt={"engine_params": {"dtype": "bfloat16", "quant_config": {"quant_algo": "FP8"}}},
    )
    assert cfg.tensorrt.engine_params.dtype == "bfloat16"


def test_trt_non_fp8_accepts_float16() -> None:
    """Non-FP8 quantization (INT8) with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt={"engine_params": {"dtype": "float16", "quant_config": {"quant_algo": "INT8"}}},
    )
    assert cfg.tensorrt.engine_params.dtype == "float16"


# ---------------------------------------------------------------------------
# n_prompts default
# ---------------------------------------------------------------------------


def test_n_prompts_default_is_100() -> None:
    """DatasetConfig().n_prompts defaults to 100."""
    from llenergymeasure.config.models import DatasetConfig

    assert DatasetConfig().n_prompts == 100


# ---------------------------------------------------------------------------
# V-row parity: BNB cross-field rules now come from the mined corpus
# (V1 = mined error, V3 = mined dormant) instead of hand-written validators.
# ---------------------------------------------------------------------------


def test_v1_load_in_4bit_and_8bit_errors_at_parse() -> None:
    """V1: load_in_4bit + load_in_8bit both True raises at parse (mined error rule).

    The engine raises this in BitsAndBytesConfig construction; the rule was
    gate-confirmed in the transformers container and now fires at config parse
    via _apply_invariants - no hand-written validator needed.
    """
    with pytest.raises(ValidationError, match="load_in_4bit and load_in_8bit"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers={"engine_params": {"load_in_4bit": True, "load_in_8bit": True}},
        )


def test_v1_load_in_4bit_only_is_accepted() -> None:
    """V1 negative: 4-bit alone constructs cleanly (matches the gate's negative probe)."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"load_in_4bit": True, "load_in_8bit": False}},
    )
    assert cfg.transformers.engine_params.load_in_4bit is True


def test_v3_bnb_4bit_without_load_in_4bit_parses_without_dormancy_warning(recwarn) -> None:
    """V3 decayed at 5.7.0: bnb_4bit_* without load_in_4bit parses, no dormancy warning.

    The V3 dormant rule was gate-confirmed at 4.57.3, but transformers 5.7.0
    construction-confirmed (in-container probe during the C10 bump campaign) that
    BitsAndBytesConfig now accepts ``bnb_4bit_quant_type`` with ``load_in_4bit=False``
    silently - no warning emitted. The validation gate therefore quarantines the V3
    dormant seed and it no longer ships in the corpus, so no parse-time
    ConfigValidationWarning fires. The config still parses (the field is accepted via
    extra="allow"). Honest 5.7.0 behaviour: parses cleanly, no dormancy warning.
    """
    from llenergymeasure.config.warnings import ConfigValidationWarning

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"bnb_4bit_quant_type": "nf4", "load_in_4bit": False}},
    )
    assert cfg.transformers.engine_params.bnb_4bit_quant_type == "nf4"
    dormant_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, ConfigValidationWarning)
        and "bnb_4bit_quant_type" in str(w.message)
    ]
    assert not dormant_warnings, (
        "V3 dormancy decayed at 5.7.0 (no upstream warning; rule quarantined by the gate); "
        f"expected no dormant-field warning but got: {[str(w.message) for w in dormant_warnings]}"
    )


# ---------------------------------------------------------------------------
# V-row parity: vllm cross-field rules.
# V7 = mined error rule (enforced at parse via the proposed-fallback - vllm has
# no validated YAML, so the loader serves rules.proposed.yaml directly).
# V6 / V8 = conscious drops.
# ---------------------------------------------------------------------------


def test_v7_max_num_batched_tokens_lt_max_model_len_errors_at_parse() -> None:
    """V7: max_num_batched_tokens < max_model_len raises at parse (mined error rule).

    vllm's SchedulerConfig._verify_args (config.py:1575) raises this. The rule is
    statically mined into rules.proposed.yaml; vllm ships no rules.validated.yaml
    (the validation gate is GPU-blocked, #445), so the loader serves the proposed
    corpus and _apply_invariants enforces V7 at config parse - proving the
    proposed-fallback the design asserts.
    """
    with pytest.raises(ValidationError, match="smaller than"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            vllm={"engine_params": {"max_num_batched_tokens": 1024, "max_model_len": 2048}},
        )


def test_v7_max_num_batched_tokens_ge_max_model_len_is_accepted() -> None:
    """V7 negative: a batched-token budget >= max_model_len constructs cleanly."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"max_num_batched_tokens": 4096, "max_model_len": 2048}},
    )
    assert cfg.vllm.engine_params.max_num_batched_tokens == 4096


def test_v8_beam_search_and_sampling_coexist_at_parse() -> None:
    """V8 conscious drop: beam_search + sampling_params coexist without a parse error.

    The hand-written VLLMConfig.validate_beam_search_exclusive was deleted. The
    exclusivity is a multi-entry-point structural choice (vllm dispatches to
    BeamSearchParams vs SamplingParams from two different call sites and raises at
    dispatch), so it is not a construction-time corpus rule. Both nest as
    Any-typed engine_params / sampling_params and parse.
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={
            "engine_params": {"beam_search": {"beam_width": 4}},
            "sampling_params": {"temperature": 0.7},
        },
    )
    assert cfg.vllm.engine_params.beam_search == {"beam_width": 4}
    assert cfg.vllm.sampling_params.temperature == 0.7


def test_v6_kv_cache_memory_bytes_with_gpu_util_coexist_at_parse() -> None:
    """V6 conscious drop (version-honest): the field does not exist in vllm 0.7.3.

    The hand-written VLLMEngineConfig.validate_kv_cache_memory guarded
    kv_cache_memory_bytes XOR gpu_memory_utilization, but kv_cache_memory_bytes is
    absent from vllm 0.7.3 (a curated discovery-debt stub that lands in a later
    vllm). With no field and no constraint in the pinned source there is no honest
    static_miner source to cite, so V6 is dropped for this pin; the re-mine at the
    bump that introduces the field will surface the real constraint. Both keys
    parse today (kv_cache_memory_bytes via extra="allow").
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"kv_cache_memory_bytes": 1024, "gpu_memory_utilization": 0.8}},
    )
    assert cfg.vllm.engine_params.gpu_memory_utilization == 0.8
