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
    get_swept_field_paths,
    get_validation_rules,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import ENGINES
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
    """get_engine_params('vllm') returns params on the nested engine_params shape."""
    params = get_engine_params("vllm")
    assert isinstance(params, dict)
    assert "vllm.engine_params.max_num_seqs" in params


def test_get_engine_params_tensorrt_returns_params():
    """get_engine_params('tensorrt') returns params on the nested engine_params shape.

    quant_config / kv_cache_config / scheduler_config are Any-typed dicts on the
    generated config, so they register as single opaque paths (not expanded into
    their inner fields).
    """
    params = get_engine_params("tensorrt")
    assert isinstance(params, dict)
    assert "tensorrt.engine_params.max_batch_size" in params
    # Sub-config dicts register as single opaque engine_params paths (not recursed)
    assert "tensorrt.engine_params.quant_config" in params
    assert "tensorrt.engine_params.kv_cache_config" in params
    assert "tensorrt.engine_params.scheduler_config" in params
    # build_cache and calib sub-configs dropped (D1/D3); return_perf_metrics dropped (D1)
    assert "tensorrt.build_cache.max_records" not in params
    assert "tensorrt.sampling.return_perf_metrics" not in params
    # New fields from C.2
    assert "tensorrt.engine_params.pipeline_parallel_size" in params
    assert "tensorrt.engine_params.max_num_tokens" in params
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


@pytest.mark.parametrize("dt", ENGINES["transformers"].dtypes)  # type: ignore[index]  # Engine is str-enum
def test_all_pytorch_dtype_values_produce_valid_config(dt):
    """Schema-driven: each SSOT ENGINES['transformers'].dtypes value creates a valid config."""
    config = make_config(dtype=dt)
    assert config.transformers.engine_params.dtype == dt


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
    """Two experiments with different engine dtypes sweep the engine_params path."""
    exp1 = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "float16"}},
    )
    exp2 = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "bfloat16"}},
    )
    result = get_swept_field_paths([exp1, exp2])
    assert "transformers.engine_params.dtype" in result


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
    exp_pt = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers={"engine_params": {"dtype": "float16"}},
    )
    exp_vllm = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={"engine_params": {"dtype": "float16"}},
    )
    # Must not raise AttributeError
    result = get_swept_field_paths([exp_pt, exp_vllm])
    # Engine itself varies
    assert "engine" in result
    # Optional sub-configs that are None on some experiments should be swept
    assert "transformers" in result
    assert "vllm" in result


# ---------------------------------------------------------------------------
# get_engine_capabilities (capability matrix reports only the derivable subset)
# ---------------------------------------------------------------------------


def test_capability_matrix_transformers_supports_tensor_parallel():
    """TransformersConfig exposes tp_plan/tp_size, so transformers TP must be supported."""
    caps = get_engine_capabilities()
    assert caps["tensor_parallel"]["transformers"] is True


def test_capability_matrix_reports_only_derivable_rows():
    """Every row is field-presence-derivable; hand-authored prose rows stay retired."""
    caps = get_engine_capabilities()
    assert set(caps) == {
        "tensor_parallel",
        "bitsandbytes_4bit",
        "bitsandbytes_8bit",
        "prefix_caching",
        "torch_compile",
        "speculative_decoding",
        "static_kv_cache",
    }


def test_speculative_decoding_cells_track_engine_field_presence():
    """Each speculative_decoding cell is backed by the field it probes, not a claim."""
    from llenergymeasure.config.introspection import _engine_params_field_names

    caps = get_engine_capabilities()
    spec = caps["speculative_decoding"]

    # True cells must be backed by a concrete field in that engine's surface.
    assert "prompt_lookup_num_tokens" in _engine_params_field_names("transformers")
    assert spec["transformers"] is True
    assert "speculative_config" in _engine_params_field_names("vllm")
    assert spec["vllm"] is True

    # tensorrt's curated surface exposes no speculative field, so the cell is False.
    assert "speculative_config" not in _engine_params_field_names("tensorrt")
    assert spec["tensorrt"] is False


def test_static_kv_cache_cells_track_engine_field_presence():
    """static_kv_cache is True only where cache_implementation exists in the surface."""
    from llenergymeasure.config.introspection import _engine_params_field_names

    caps = get_engine_capabilities()
    kv = caps["static_kv_cache"]

    # transformers exposes cache_implementation (cache_implementation="static").
    assert "cache_implementation" in _engine_params_field_names("transformers")
    assert kv["transformers"] is True

    # vLLM (paged) and tensorrt expose no cache_implementation field, so both False.
    assert "cache_implementation" not in _engine_params_field_names("vllm")
    assert kv["vllm"] is False
    assert "cache_implementation" not in _engine_params_field_names("tensorrt")
    assert kv["tensorrt"] is False


def test_streaming_constraints_removed():
    """get_streaming_constraints was removed because streaming is no longer a config field."""
    import llenergymeasure.config.introspection as introspection

    assert not hasattr(introspection, "get_streaming_constraints")


def test_runtime_limitations_removed():
    """get_runtime_limitations was retired: its rows were 100% hand-authored prose."""
    import llenergymeasure.config.introspection as introspection

    assert not hasattr(introspection, "get_runtime_limitations")
