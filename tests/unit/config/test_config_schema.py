"""Unit tests for ExperimentConfig Pydantic validation.

Tests v2.0 field renames, extra=forbid, engine composition, cross-validators,
and schema-driven dtype validation using SSOT constants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig, OutputConfig
from llenergymeasure.config.ssot import ENGINES
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Minimal valid config
# ---------------------------------------------------------------------------


def test_minimal_valid_config():
    """ExperimentConfig(task={'model': 'gpt2'}, engine='transformers') succeeds."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline")
    assert config.task.model == "gpt2"
    assert config.engine == "transformers"


def test_model_only_uses_pytorch_default():
    """ExperimentConfig with only model= uses engine='transformers' default."""
    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
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
    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    assert config.task.model == "gpt2"


def test_field_name_dtype():
    """dtype is per-engine; transformers nests it under engine_params."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "float16"}},
    )
    assert config.transformers is not None
    assert config.transformers.engine_params.dtype == "float16"


def test_field_name_n():
    """v2.0 dataset.n_prompts field (not 'num_input_prompts') is accepted."""
    from llenergymeasure.config.models import DatasetConfig

    config = ExperimentConfig(
        task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)}, serving_mode="offline"
    )
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
        ExperimentConfig(task={"model": "gpt2"}, engine="invalid_backend", serving_mode="offline")


def test_default_engine_is_pytorch():
    """Default engine is 'transformers' when not specified."""
    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    assert config.engine == "transformers"


# ---------------------------------------------------------------------------
# serving_mode
# ---------------------------------------------------------------------------


def test_serving_mode_required_no_default():
    """serving_mode is required with no default: omitting it fails loudly."""
    with pytest.raises((ValidationError, ValueError), match="serving_mode is required"):
        ExperimentConfig(task={"model": "gpt2"})  # type: ignore[call-arg]


def test_serving_mode_offline_explicit():
    """serving_mode='offline' is accepted explicitly."""
    config = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    assert config.serving_mode == "offline"


def test_serving_mode_server_accepted():
    """serving_mode='server' is a valid config value (data model admits it)."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        serving_mode="server",
        server={"traffic": {"rate": 10, "window_seconds": 60}},
    )
    assert config.serving_mode == "server"


def test_serving_mode_typo_rejected():
    """A serving_mode typo is rejected loudly (closed Literal, edge-validated)."""
    with pytest.raises(ValidationError):
        ExperimentConfig(task={"model": "gpt2"}, serving_mode="offlien")  # type: ignore[arg-type]


def test_server_section_under_offline_rejected():
    """A server: section is illegal under serving_mode=offline (mode-section match)."""
    with pytest.raises((ValidationError, ValueError), match="section provided but"):
        ExperimentConfig(
            task={"model": "gpt2"},
            serving_mode="offline",
            server={"traffic": {"rate": 10, "window_seconds": 60}},
        )


def test_vllm_engine_accepted():
    """engine='vllm' is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="offline")
    assert config.engine == "vllm"


def test_tensorrt_engine_accepted():
    """engine='tensorrt' is accepted."""
    config = ExperimentConfig(task={"model": "gpt2"}, engine="tensorrt", serving_mode="offline")
    assert config.engine == "tensorrt"


# ---------------------------------------------------------------------------
# Engine section composition
# ---------------------------------------------------------------------------


def test_pytorch_config_section_composition():
    """config with transformers engine_params section is accepted.

    batch_size moved to transformers.llem_execution (TransformersLlemExecution); the
    engine section carries engine-native fields under engine_params.
    """
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"device_map": "auto"}},
    )
    assert config.transformers is not None
    assert config.transformers.engine_params.device_map == "auto"


def test_transformers_llem_execution_is_a_typed_section_sibling():
    """llem_execution nests inside the transformers section as a typed sibling.

    It composes with engine_params (native passthrough) and validates strictly
    against TransformersLlemExecution; active_llem_execution() reads it back.
    """
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={
            "engine_params": {"dtype": "bfloat16"},
            "llem_execution": {"batch_size": 8, "torch_compile": True},
        },
    )
    execution = config.active_llem_execution()
    assert execution is not None
    assert execution.batch_size == 8
    assert execution.torch_compile is True
    assert config.capacity_batch_size() == 8


def test_transformers_llem_execution_torch_compile_options_validated():
    """The strict TransformersLlemExecution validator still fires under the new nesting."""
    with pytest.raises(ValidationError, match="requires torch_compile=True"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"llem_execution": {"torch_compile_mode": "reduce-overhead"}},
        )


def test_retired_top_level_harness_key_rejected_on_construction():
    """Validating a payload with the retired top-level harness: key fails helpfully."""
    payload = {
        "task": {"model": "gpt2"},
        "engine": "transformers",
        "serving_mode": "offline",
        "harness": {"transformers": {"batch_size": 4}},
    }
    with pytest.raises(ValidationError, match="llem_execution"):
        ExperimentConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Generated transformers Config: no num_processes field
# ---------------------------------------------------------------------------


def test_pytorch_config_has_no_num_processes_field():
    """The generated transformers EngineParams has no num_processes field."""
    from llenergymeasure.config.generated.transformers import EngineParams

    assert "num_processes" not in EngineParams.model_fields


def test_pytorch_config_num_processes_not_a_declared_field():
    """num_processes is not a declared field on the generated EngineParams.

    The generated Config uses extra='allow' for HuggingFace passthrough, so an
    unknown kwarg is accepted into model_extra but is NOT a typed model field.
    """
    from llenergymeasure.config.generated.transformers import EngineParams

    # Verify it is absent from the declared model fields
    assert "num_processes" not in EngineParams.model_fields
    # Extra kwargs are accepted (extra='allow') but go into __pydantic_extra__
    params = EngineParams(num_processes=4)  # type: ignore[call-arg]
    # Not a typed field - no attribute access by name on the typed model
    assert "num_processes" not in type(params).model_fields


def test_pytorch_section_with_wrong_engine_rejected():
    """pytorch: section with engine='vllm' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"transformers.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            serving_mode="offline",
            transformers={"batch_size": 4},
        )


def test_vllm_section_with_pytorch_engine_rejected():
    """vllm: section with engine='transformers' raises ValidationError (cross-validator)."""
    with pytest.raises(ValidationError, match=r"vllm.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            vllm={"engine": {"max_num_seqs": 16}},
        )


def test_tensorrt_section_with_wrong_engine_rejected():
    """tensorrt: section with engine='transformers' raises ValidationError."""
    with pytest.raises(ValidationError, match=r"tensorrt.*config section provided.*engine"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
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
    """
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "fp16"}},
    )
    assert config.transformers.engine_params.dtype == "fp16"


def test_valid_dtype_float32():
    """dtype='float32' is valid on the generated transformers engine_params."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "float32"}},
    )
    assert config.transformers.engine_params.dtype == "float32"


def test_valid_dtype_float16():
    """dtype='float16' is valid on the generated transformers engine_params."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "float16"}},
    )
    assert config.transformers.engine_params.dtype == "float16"


def test_valid_dtype_bfloat16():
    """dtype='bfloat16' is valid on the generated transformers engine_params."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "bfloat16"}},
    )
    assert config.transformers.engine_params.dtype == "bfloat16"


@pytest.mark.parametrize("dt", ENGINES["transformers"].dtypes)  # type: ignore[index]  # Engine is str-enum
def test_all_pytorch_dtypes_valid(dt):
    """Schema-driven: all SSOT ENGINES['transformers'].dtypes values are valid."""
    config = make_config(dtype=dt)
    assert config.transformers.engine_params.dtype == dt


# ---------------------------------------------------------------------------
# passthrough_kwargs collision cross-validator
# ---------------------------------------------------------------------------


def test_passthrough_kwargs_accepted():
    """passthrough_kwargs with non-colliding keys are accepted."""
    config = ExperimentConfig(
        task={"model": "gpt2"},
        serving_mode="offline",
        passthrough_kwargs={"custom_flag": True, "my_special_param": 42},
    )
    assert config.passthrough_kwargs is not None
    assert config.passthrough_kwargs["custom_flag"] is True


def test_passthrough_kwargs_collision_with_top_level_field_rejected():
    """passthrough_kwargs keys colliding with ExperimentConfig fields are rejected."""
    with pytest.raises(ValidationError, match=r"passthrough_kwargs.*collide"):
        ExperimentConfig(
            task={"model": "gpt2"},
            serving_mode="offline",
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
    """make_config(**overrides) applies overrides over defaults (dtype -> engine section)."""
    config = make_config(model="bert-base", dtype="float32")
    assert config.task.model == "bert-base"
    assert config.transformers.engine_params.dtype == "float32"


# ---------------------------------------------------------------------------
# energy_sampler field tests (flat Literal on ExperimentConfig)
# ---------------------------------------------------------------------------


def test_energy_sampler_default() -> None:
    """energy_sampler defaults to 'auto'."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
    assert cfg.measurement.energy_sampler == "auto"


def test_energy_sampler_null_disables() -> None:
    """energy_sampler=None disables energy measurement."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"}, serving_mode="offline", measurement={"energy_sampler": None}
    )
    assert cfg.measurement.energy_sampler is None


def test_energy_sampler_valid_engines() -> None:
    """All energy_sampler literal values are accepted."""
    for engine in ("auto", "nvml", "zeus", "codecarbon"):
        cfg = ExperimentConfig(
            task={"model": "gpt2"},
            serving_mode="offline",
            measurement={"energy_sampler": engine},
        )
        assert cfg.measurement.energy_sampler == engine


def test_energy_sampler_invalid_engine() -> None:
    """Unknown energy_sampler values raise ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfig(
            task={"model": "gpt2"},
            serving_mode="offline",
            measurement={"energy_sampler": "unknown_backend"},
        )


def test_energy_sampler_override() -> None:
    """ExperimentConfig allows overriding energy_sampler."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"}, serving_mode="offline", measurement={"energy_sampler": "nvml"}
    )
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
# Transformers tensor parallelism fields (tp_plan, tp_size) on generated config
#
# After migration these are curated Any-typed engine_params fields (no Literal /
# ge constraint - that typing was lost to discovery debt). The V5 rule
# (tp_plan XOR device_map) is a conscious drop: enforcement is from_pretrained-
# side (execution grain), where the engine raises at model load, so it is no
# longer a config-parse validator.
# ---------------------------------------------------------------------------


def test_pytorch_config_tp_plan_accepts_auto():
    """tp_plan='auto' is accepted on the generated engine_params."""
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(tp_plan="auto")
    assert ep.tp_plan == "auto"


def test_pytorch_config_tp_plan_accepts_any_string():
    """tp_plan is a curated Any passthrough: arbitrary strings are accepted.

    The hand-written TransformersConfig typed tp_plan as Literal['auto'] and
    rejected other values; the generated engine_params.tp_plan is Any (discovery
    debt), so any value passes and the engine validates at load.
    """
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(tp_plan="custom")  # type: ignore[arg-type]
    assert ep.tp_plan == "custom"


def test_pytorch_config_tp_size_accepts_positive():
    """tp_plan='auto', tp_size=4 is accepted on the generated engine_params."""
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(tp_plan="auto", tp_size=4)
    assert ep.tp_plan == "auto"
    assert ep.tp_size == 4


def test_pytorch_config_tp_size_accepts_zero():
    """tp_size=0 is accepted on the generated engine_params (tp_size is Any, no ge=1).

    The hand-written TransformersConfig had ge=1 on tp_size; the generated
    engine_params.tp_size is Any (discovery debt), so 0 passes at parse and the
    engine validates at load.
    """
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(tp_size=0)
    assert ep.tp_size == 0


def test_pytorch_config_tp_plan_and_device_map_can_coexist():
    """tp_plan and device_map no longer raise at parse (V5 cross-validator dropped).

    The hand-written TransformersConfig raised with 'mutually exclusive'; the
    validator was from_pretrained-side enforcement (execution grain), not a
    config-parse invariant. Both values now parse; the engine raises at model load.
    """
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(tp_plan="auto", device_map="auto")
    assert ep.tp_plan == "auto"
    assert ep.device_map == "auto"


def test_pytorch_config_tp_plan_without_device_map_ok():
    """tp_plan='auto' without device_map succeeds."""
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(tp_plan="auto")
    assert ep.tp_plan == "auto"
    assert ep.device_map is None


def test_pytorch_config_device_map_without_tp_plan_ok():
    """device_map='auto' without tp_plan succeeds."""
    from llenergymeasure.config.generated.transformers import EngineParams

    ep = EngineParams(device_map="auto")
    assert ep.device_map == "auto"
    assert ep.tp_plan is None


# ---------------------------------------------------------------------------
# Bug 1.1 - fp8 quantization + float32 dtype (vLLM)
# ---------------------------------------------------------------------------


def test_vllm_dtype_float32_parses():
    """vllm dtype Literal widened to include float32; vLLM rejects fp32 at runtime.

    The generated vllm EngineParams dtype Literal now includes 'float32' (the
    mined enum is wider than the previous hand-written restriction). vLLM itself
    raises at runtime; float32 parses cleanly at config validation.
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"dtype": "float32"}},
    )
    assert cfg.vllm.engine_params.dtype == "float32"


def test_vllm_fp8_float16_accepted():
    """fp8 quantization with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"dtype": "float16", "quantization": "fp8"}},
    )
    assert cfg.vllm.engine_params.dtype == "float16"


def test_vllm_fp8_bfloat16_accepted():
    """fp8 quantization with dtype=bfloat16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"dtype": "bfloat16", "quantization": "fp8"}},
    )
    assert cfg.vllm.engine_params.dtype == "bfloat16"


def test_vllm_non_fp8_float16_accepted():
    """Non-fp8 quantization (awq) with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"dtype": "float16", "quantization": "awq"}},
    )
    assert cfg.vllm.engine_params.dtype == "float16"


def test_vllm_no_quantization_default_dtype_accepted():
    """No quantization set, no explicit dtype, is accepted (engine default applies)."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {}},
    )
    # dtype defaults to "auto" on the generated EngineParams
    assert cfg.vllm.engine_params.dtype == "auto"


# ---------------------------------------------------------------------------
# Bug 1.2 - max_num_batched_tokens < max_model_len (vLLM engine)
# ---------------------------------------------------------------------------


def test_vllm_batched_tokens_less_than_model_len_rejected():
    """max_num_batched_tokens < max_model_len raises ValidationError (mined error rule).

    The mined rule ``vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_model_len``
    fires at ExperimentConfig construction when max_num_batched_tokens < max_model_len
    and enable_chunked_prefill is absent.
    """
    with pytest.raises(ValidationError, match=r"smaller than"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            serving_mode="offline",
            vllm={"engine_params": {"max_num_batched_tokens": 512, "max_model_len": 1024}},
        )


def test_vllm_batched_tokens_equal_model_len_accepted():
    """max_num_batched_tokens == max_model_len is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"max_num_batched_tokens": 1024, "max_model_len": 1024}},
    )
    assert cfg.vllm.engine_params.max_num_batched_tokens == 1024
    assert cfg.vllm.engine_params.max_model_len == 1024


def test_vllm_batched_tokens_greater_accepted():
    """max_num_batched_tokens > max_model_len is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"max_num_batched_tokens": 2048, "max_model_len": 1024}},
    )
    assert cfg.vllm.engine_params.max_num_batched_tokens == 2048


def test_vllm_batched_tokens_one_none_accepted():
    """Only one of max_num_batched_tokens / max_model_len set is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="offline",
        vllm={"engine_params": {"max_num_batched_tokens": 512}},
    )
    assert cfg.vllm.engine_params.max_num_batched_tokens == 512
    assert cfg.vllm.engine_params.max_model_len is None


# ---------------------------------------------------------------------------
# Bug 1.3 - flash_attention_2/3 + float32 dtype (PyTorch)
# ---------------------------------------------------------------------------


def test_pytorch_flash_attn2_float32_rejected():
    """flash_attention_2 with dtype=float32 raises ValidationError at parse time."""
    with pytest.raises(ValidationError, match=r"flash_attention_2.*requires.*float16"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
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
            serving_mode="offline",
            transformers={
                "engine_params": {"dtype": "float32", "attn_implementation": "flash_attention_3"}
            },
        )


def test_pytorch_flash_attn2_bfloat16_accepted():
    """flash_attention_2 with dtype=bfloat16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
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
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "float32", "attn_implementation": "eager"}},
    )
    assert cfg.transformers.engine_params.dtype == "float32"


def test_pytorch_no_attn_impl_float32_accepted():
    """No attn_implementation set with dtype=float32 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={"engine_params": {"dtype": "float32"}},
    )
    assert cfg.transformers.engine_params.dtype == "float32"


# ---------------------------------------------------------------------------
# TRT dtype (plain str passthrough, no Literal enforcement)
# ---------------------------------------------------------------------------


def test_trt_dtype_float32_accepted() -> None:
    """tensorrt dtype is a plain str passthrough; float32 is accepted at parse.

    The hand-written TensorRTConfig had Literal['float16', 'bfloat16'] that
    rejected float32. The generated engine_params.dtype is a plain str (the mined
    schema did not surface TRT-LLM's internal dtype enum). TRT-LLM itself rejects
    fp32 at runtime; the constraint will be restored when a container re-mine
    surfaces the real enum.
    """
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        serving_mode="offline",
        tensorrt={"engine_params": {"dtype": "float32"}},
    )
    assert cfg.tensorrt.engine_params.dtype == "float32"


def test_trt_fp8_accepts_float16() -> None:
    """FP8 quantization with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        serving_mode="offline",
        tensorrt={
            "engine_params": {
                "backend": "trt",
                "dtype": "float16",
                "quant_config": {"quant_algo": "FP8"},
            }
        },
    )
    assert cfg.tensorrt.engine_params.dtype == "float16"


def test_trt_fp8_accepts_bfloat16() -> None:
    """FP8 quantization with dtype=bfloat16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        serving_mode="offline",
        tensorrt={
            "engine_params": {
                "backend": "trt",
                "dtype": "bfloat16",
                "quant_config": {"quant_algo": "FP8"},
            }
        },
    )
    assert cfg.tensorrt.engine_params.dtype == "bfloat16"


def test_trt_non_fp8_accepts_float16() -> None:
    """Non-FP8 quantization (INT8) with dtype=float16 is accepted."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        serving_mode="offline",
        tensorrt={
            "engine_params": {
                "backend": "trt",
                "dtype": "float16",
                "quant_config": {"quant_algo": "INT8"},
            }
        },
    )
    assert cfg.tensorrt.engine_params.dtype == "float16"


# ---------------------------------------------------------------------------
# n_prompts default
# ---------------------------------------------------------------------------


def test_n_prompts_default_is_100() -> None:
    """DatasetConfig().n_prompts defaults to 100."""
    from llenergymeasure.config.models import DatasetConfig

    assert DatasetConfig().n_prompts == 100
