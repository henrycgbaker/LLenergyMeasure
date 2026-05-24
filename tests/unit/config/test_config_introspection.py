"""Unit tests for config introspection SSOT architecture (INF-11).

Tests that introspection functions return correct structure and that
test value generation is schema-driven (derived from Pydantic models,
not hard-coded lists).
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.introspection import (
    get_all_params,
    get_display_label,
    get_engine_params,
    get_experiment_config_schema,
    get_field_role,
    get_param_test_values,
    get_shared_params,
    get_swept_field_paths,
    get_validation_rules,
    list_all_param_paths,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import DTYPE_SUPPORT
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# get_engine_params
# ---------------------------------------------------------------------------


def test_get_engine_params_returns_pytorch_params():
    """get_engine_params('transformers') returns a dict including engine_params.dtype."""
    params = get_engine_params("transformers")
    assert isinstance(params, dict)
    assert "transformers.engine_params.dtype" in params


def test_get_engine_params_pytorch_has_engine_support():
    """Each pytorch param has engine_support=['transformers']."""
    params = get_engine_params("transformers")
    for param_path, meta in params.items():
        assert "engine_support" in meta, f"Missing engine_support on {param_path}"
        assert "transformers" in meta["engine_support"]


def test_get_engine_params_vllm_returns_params():
    """get_engine_params('vllm') returns nested engine_params.max_num_seqs path."""
    params = get_engine_params("vllm")
    assert isinstance(params, dict)
    assert "vllm.engine_params.max_num_seqs" in params


def test_get_engine_params_tensorrt_returns_params():
    """get_engine_params('tensorrt') returns expected nested engine_params/sampling_params paths.

    Nested sub-bundle inner fields (quant_config.*, kv_cache_config.*,
    scheduler_config.*) are currently dict[str, Any] terminal entries and
    will surface as inner paths once vllm/tensorrt producer cells re-run
    with the $defs propagation commit
    (see _spike/findings/phase3_audit_llem_fields.md NEEDS_DEFS_PROPAGATION).
    """
    params = get_engine_params("tensorrt")
    assert isinstance(params, dict)
    assert "tensorrt.engine_params.max_batch_size" in params
    assert "tensorrt.engine_params.pipeline_parallel_size" in params
    assert "tensorrt.engine_params.max_num_tokens" in params
    # Sub-bundles surface as terminal entries today (dict[str, Any])
    assert "tensorrt.engine_params.quant_config" in params
    assert "tensorrt.engine_params.kv_cache_config" in params
    assert "tensorrt.engine_params.scheduler_config" in params
    # Sampling params surface under sampling_params.*
    assert "tensorrt.sampling_params.temperature" in params
    assert len(params) >= 10


def test_get_engine_params_unknown_engine_raises():
    """get_engine_params with unknown engine raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine_params("nonexistent_backend")


# ---------------------------------------------------------------------------
# get_shared_params
# ---------------------------------------------------------------------------


def test_get_shared_params_returns_model_field():
    """get_shared_params() returns dataset and decoder params.

    dtype lives per-engine, not in the shared bucket. Shared now means
    decoder.* params and dataset.* params.
    """
    params = get_shared_params()
    assert isinstance(params, dict)
    # Dataset params are shared
    assert "dataset.n_prompts" in params


def test_get_shared_params_does_not_contain_dtype():
    """get_shared_params() does not expose 'dtype' - it's per-engine."""
    params = get_shared_params()
    assert "dtype" not in params


def test_get_shared_params_contains_n():
    """get_shared_params() contains 'dataset.n_prompts' (prompt count)."""
    params = get_shared_params()
    assert "dataset.n_prompts" in params


def test_get_shared_params_no_longer_contains_decoder():
    """Sampling params live per-engine - they're not in get_shared_params()."""
    params = get_shared_params()
    assert "decoder.temperature" not in params
    assert "decoder.top_k" not in params


def test_get_shared_params_all_have_engine_support():
    """Every shared param has engine_support list."""
    params = get_shared_params()
    for param_path, meta in params.items():
        assert "engine_support" in meta, f"Missing engine_support on {param_path}"
        assert isinstance(meta["engine_support"], list)


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
# get_param_test_values (INF-11: SSOT-driven test value generation)
# ---------------------------------------------------------------------------


def test_get_param_test_values_tensorrt_max_batch_size_returns_invalid_probe():
    """tensorrt.engine_params.max_batch_size has a hand-curated invalid probe [0]."""
    values = get_param_test_values("tensorrt.engine_params.max_batch_size")
    assert isinstance(values, list)
    assert values == [0]  # custom invalid probe; ge=1 violation


def test_get_param_test_values_sampling_temperature_returns_floats():
    """Per-engine sampling_params.temperature returns synthesized float test values."""
    values = get_param_test_values("transformers.sampling_params.temperature")
    assert isinstance(values, list)
    assert all(isinstance(v, int | float) for v in values)


def test_get_param_test_values_unknown_param_returns_empty():
    """get_param_test_values for unknown param path returns empty list."""
    values = get_param_test_values("nonexistent.param.path")
    assert values == []


# ---------------------------------------------------------------------------
# get_all_params
# ---------------------------------------------------------------------------


def test_get_all_params_covers_all_engines():
    """get_all_params() returns dict with 'transformers', 'vllm', 'tensorrt' keys."""
    all_params = get_all_params()
    assert "transformers" in all_params
    assert "vllm" in all_params
    assert "tensorrt" in all_params


def test_get_all_params_has_shared_key():
    """get_all_params() includes a 'shared' section."""
    all_params = get_all_params()
    assert "shared" in all_params


def test_get_all_params_pytorch_section_contains_engine_dtype():
    """get_all_params()['transformers'] contains the engine_params.dtype param."""
    all_params = get_all_params()
    assert "transformers.engine_params.dtype" in all_params["transformers"]


# ---------------------------------------------------------------------------
# list_all_param_paths
# ---------------------------------------------------------------------------


def test_list_all_param_paths_contains_expected_paths():
    """list_all_param_paths() returns a sorted list containing known nested paths."""
    paths = list_all_param_paths()
    assert isinstance(paths, list)
    assert "transformers.engine_params.dtype" in paths


def test_list_all_param_paths_contains_known_paths():
    """list_all_param_paths() contains expected well-known nested param paths."""
    paths = list_all_param_paths()
    # Sampling params live per-engine under sampling_params
    assert "transformers.sampling_params.temperature" in paths
    assert "vllm.sampling_params.temperature" in paths
    assert "tensorrt.sampling_params.temperature" in paths
    # dtype lives per-engine under engine_params
    assert "transformers.engine_params.dtype" in paths
    assert "vllm.engine_params.dtype" in paths
    assert "tensorrt.engine_params.dtype" in paths


def test_list_all_param_paths_filtered_by_engine():
    """list_all_param_paths(engine='transformers') returns only pytorch paths."""
    paths = list_all_param_paths(engine="transformers")
    assert all(p.startswith("transformers.") for p in paths)


def test_list_all_param_paths_unknown_engine_raises():
    """list_all_param_paths with unknown engine raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        list_all_param_paths(engine="nonexistent")


# ---------------------------------------------------------------------------
# SSOT schema-driven test generation (INF-11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["transformers"])
def test_all_pytorch_dtype_values_produce_valid_config(dt):
    """Each SSOT DTYPE_SUPPORT['transformers'] value creates a valid config.

    The generated engines.transformers.Config types dtype as ``str | None`` with
    extra='allow' (upstream mining did not surface a finite enum for dtype),
    so all string values are accepted. SSOT DTYPE_SUPPORT remains the
    authoritative llem-side curation of which values we test against.
    """
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
    """Two experiments with different engine dtypes sweep the engine subconfig path."""
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
        harness={"transformers": {"batch_size": 4}},
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
