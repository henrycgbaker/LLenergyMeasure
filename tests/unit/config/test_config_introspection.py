"""Unit tests for config introspection SSOT architecture (INF-11).

Tests that introspection functions return correct structure and that
test value generation is schema-driven (derived from Pydantic models,
not hard-coded lists).
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.introspection import (
    get_display_label,
    get_engine_capabilities,
    get_engine_params,
    get_experiment_config_schema,
    get_field_role,
    get_runtime_limitations,
    get_swept_field_paths,
    get_validation_rules,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import DTYPE_SUPPORT
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# get_engine_params
# ---------------------------------------------------------------------------


def test_get_engine_params_returns_pytorch_params():
    """get_engine_params('transformers') returns a dict with batch_size field."""
    params = get_engine_params("transformers")
    assert isinstance(params, dict)
    assert "transformers.batch_size" in params


def test_get_engine_params_pytorch_has_engine_support():
    """Each pytorch param has engine_support=['transformers']."""
    params = get_engine_params("transformers")
    for param_path, meta in params.items():
        assert "engine_support" in meta, f"Missing engine_support on {param_path}"
        assert "transformers" in meta["engine_support"]


def test_get_engine_params_vllm_returns_params():
    """get_engine_params('vllm') returns params including vllm.engine.max_num_seqs."""
    params = get_engine_params("vllm")
    assert isinstance(params, dict)
    assert "vllm.engine.max_num_seqs" in params


def test_get_engine_params_tensorrt_returns_params():
    """get_engine_params('tensorrt') returns params including nested sub-config paths."""
    params = get_engine_params("tensorrt")
    assert isinstance(params, dict)
    assert "tensorrt.max_batch_size" in params
    # Verify expanded nested sub-config params are registered
    assert "tensorrt.quant_config.quant_algo" in params
    assert "tensorrt.kv_cache_config.free_gpu_memory_fraction" in params
    assert "tensorrt.scheduler_config.capacity_scheduling_policy" in params
    # build_cache and calib sub-configs dropped (D1/D3); return_perf_metrics dropped (D1)
    assert "tensorrt.build_cache.max_records" not in params
    assert "tensorrt.sampling.return_perf_metrics" not in params
    # New fields from C.2
    assert "tensorrt.pipeline_parallel_size" in params
    assert "tensorrt.max_num_tokens" in params
    assert len(params) >= 10


def test_get_engine_params_unknown_engine_raises():
    """get_engine_params with unknown engine raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine_params("nonexistent_backend")


# ---------------------------------------------------------------------------
# get_experiment_config_schema
# ---------------------------------------------------------------------------


def test_get_experiment_config_schema_is_valid_json_schema():
    """get_experiment_config_schema() returns a dict with 'properties' key."""
    schema = get_experiment_config_schema()
    assert isinstance(schema, dict)
    # Pydantic v2 JSON schema always has 'properties' at top level
    assert "properties" in schema


def test_get_experiment_config_schema_contains_model_field():
    """Schema contains 'task' property with 'model' nested inside."""
    schema = get_experiment_config_schema()
    properties = schema.get("properties", {})
    assert "task" in properties


def test_get_experiment_config_schema_contains_engine_field():
    """Schema contains 'engine' property definition."""
    schema = get_experiment_config_schema()
    properties = schema.get("properties", {})
    assert "engine" in properties


# ---------------------------------------------------------------------------
# SSOT schema-driven test generation (INF-11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["transformers"])  # type: ignore[index]  # Engine is str-enum
def test_all_pytorch_dtype_values_produce_valid_config(dt):
    """Schema-driven: each SSOT DTYPE_SUPPORT['transformers'] value creates a valid config."""
    config = make_config(dtype=dt)
    assert config.transformers.dtype == dt


# ---------------------------------------------------------------------------
# get_validation_rules
# ---------------------------------------------------------------------------


def test_get_validation_rules_returns_list():
    """get_validation_rules() returns a list containing the engine mismatch invariant."""
    invariants = get_validation_rules()
    assert isinstance(invariants, list)
    combinations = [r["combination"] for r in invariants]
    assert any("mismatch" in c for c in combinations)


def test_get_validation_rules_each_has_required_keys():
    """Each validation invariant has engine, combination, reason, resolution keys."""
    invariants = get_validation_rules()
    for invariant in invariants:
        assert "engine" in invariant, f"Invariant missing 'engine': {invariant}"
        assert "combination" in invariant, f"Invariant missing 'combination': {invariant}"
        assert "reason" in invariant, f"Invariant missing 'reason': {invariant}"
        assert "resolution" in invariant, f"Invariant missing 'resolution': {invariant}"


def test_get_validation_rules_contains_engine_section_mismatch_rule():
    """Validation invariants include the engine section mismatch invariant."""
    invariants = get_validation_rules()
    combinations = [r["combination"] for r in invariants]
    assert any("mismatch" in c for c in combinations)


# ---------------------------------------------------------------------------
# Field metadata helpers (display_label / role)
# ---------------------------------------------------------------------------


def test_get_display_label_from_metadata():
    """get_display_label() returns 'Model' for TaskConfig.model field."""
    from llenergymeasure.config.models import TaskConfig

    fi = TaskConfig.model_fields["model"]
    label = get_display_label(fi, "model")
    assert label == "Model"


def test_get_display_label_fallback():
    """get_display_label() falls back to title-cased name for fields without metadata."""
    from llenergymeasure.config.models import TaskConfig

    fi = TaskConfig.model_fields["random_seed"]
    # random_seed has no json_schema_extra; expect title-cased fallback
    label = get_display_label(fi, "random_seed")
    assert label == "Random Seed"


def test_get_field_role_workload():
    """get_field_role() returns 'workload' for DatasetConfig.source field."""
    from llenergymeasure.config.models import DatasetConfig

    fi = DatasetConfig.model_fields["source"]
    assert get_field_role(fi) == "workload"


def test_get_field_role_none_for_unannotated():
    """get_field_role() returns None for fields without role metadata."""
    fi = ExperimentConfig.model_fields["engine"]
    assert get_field_role(fi) is None


# ---------------------------------------------------------------------------
# get_swept_field_paths
# ---------------------------------------------------------------------------


def test_get_swept_field_paths_single_experiment():
    """Single experiment yields an empty swept set."""
    exp = ExperimentConfig(task={"model": "gpt2"})
    result = get_swept_field_paths([exp])
    assert result == set()


def test_get_swept_field_paths_dtype_swept():
    """Two experiments with different engine dtypes sweep the engine subconfig path."""
    exp1 = ExperimentConfig(
        task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "float16"}
    )
    exp2 = ExperimentConfig(
        task={"model": "gpt2"}, engine="transformers", transformers={"dtype": "bfloat16"}
    )
    result = get_swept_field_paths([exp1, exp2])
    assert "transformers.dtype" in result


def test_get_swept_field_paths_nested_field():
    """Two experiments with different n_prompts yield task.dataset.n_prompts in swept."""
    from llenergymeasure.config.models import DatasetConfig

    exp1 = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=10)})
    exp2 = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)})
    result = get_swept_field_paths([exp1, exp2])
    assert "task.dataset.n_prompts" in result


def test_get_swept_field_paths_multi_engine_none_subconfigs():
    """Multi-engine study where optional sub-configs are None must not crash.

    In a multi-engine study, pytorch experiments have vllm=None and vice versa.
    get_swept_field_paths must handle None values in optional sub-config lists
    rather than raising AttributeError.
    """
    from llenergymeasure.config.engine_configs import TransformersConfig, VLLMConfig

    exp_pt = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="float16", batch_size=4),
    )
    exp_vllm = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(dtype="float16"),
    )
    # Must not raise AttributeError
    result = get_swept_field_paths([exp_pt, exp_vllm])
    # Engine itself varies
    assert "engine" in result
    # Optional sub-configs that are None on some experiments should be swept
    assert "transformers" in result
    assert "vllm" in result


# ---------------------------------------------------------------------------
# get_engine_capabilities (capability matrix must match real engine schema)
# ---------------------------------------------------------------------------


def test_capability_matrix_vllm_rejects_float32():
    """vLLM dtype is Literal[float16, bfloat16, auto]; float32 must be unsupported."""
    caps = get_engine_capabilities()
    assert caps["float32_precision"]["vllm"] is False


def test_capability_matrix_transformers_supports_tensor_parallel():
    """TransformersConfig exposes tp_plan/tp_size, so transformers TP must be supported."""
    caps = get_engine_capabilities()
    assert caps["tensor_parallel"]["transformers"] is True


def test_capability_matrix_cells_match_engine_dtype_support():
    """Each engine's float32 cell must agree with SSOT DTYPE_SUPPORT."""
    from llenergymeasure.config.ssot import DTYPE_SUPPORT, Engine

    caps = get_engine_capabilities()
    for engine, key in (
        (Engine.TRANSFORMERS, "transformers"),
        (Engine.VLLM, "vllm"),
        (Engine.TENSORRT, "tensorrt"),
    ):
        expected = "float32" in DTYPE_SUPPORT[engine]
        assert caps["float32_precision"][key] is expected


# ---------------------------------------------------------------------------
# get_runtime_limitations (parameter paths must reference real config fields)
# ---------------------------------------------------------------------------


def test_runtime_limitations_use_real_field_paths():
    """No stale/renamed paths: attention.engine, quantization_method, load_format are gone."""
    params = [limit["parameter"] for limit in get_runtime_limitations()]
    joined = "\n".join(params)
    assert "attention.engine" not in joined
    assert "quantization_method" not in joined
    assert "load_format" not in joined
    assert "tensorrt.quantization.method" not in joined


def test_runtime_limitations_reference_corrected_vllm_paths():
    """Corrected vLLM paths use the real nested engine prefix and backend field."""
    params = [limit["parameter"] for limit in get_runtime_limitations()]
    assert any("vllm.engine.attention.backend=" in p for p in params)
    assert any("vllm.engine.quantization=" in p for p in params)
    assert any("vllm.engine.kv_cache_dtype=" in p for p in params)


def test_streaming_constraints_removed():
    """get_streaming_constraints was removed because streaming is no longer a config field."""
    import llenergymeasure.config.introspection as introspection

    assert not hasattr(introspection, "get_streaming_constraints")
