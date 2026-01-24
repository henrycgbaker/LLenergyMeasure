"""Tests for configuration introspection module (SSOT architecture).

This module tests the Single Source of Truth (SSOT) introspection functions
that derive parameter metadata from Pydantic models.
"""

import pytest

from llenergymeasure.config.introspection import (
    get_all_params,
    get_backend_params,
    get_backend_specific_params,
    get_mutual_exclusions,
    get_param_options,
    get_param_skip_conditions,
    get_param_test_values,
    get_params_from_model,
    get_params_requiring_gpu_capability,
    get_shared_params,
    get_special_test_models,
    list_all_param_paths,
)


class TestGetParamsFromModel:
    """Tests for get_params_from_model function."""

    def test_extracts_simple_fields(self):
        """Should extract simple field types."""
        from llenergymeasure.config.backend_configs import VLLMConfig

        params = get_params_from_model(VLLMConfig, prefix="vllm")
        assert "vllm.max_num_seqs" in params
        assert params["vllm.max_num_seqs"]["type_str"] == "int"

    def test_extracts_literal_options(self):
        """Should extract Literal options."""
        from llenergymeasure.config.backend_configs import VLLMConfig

        params = get_params_from_model(VLLMConfig, prefix="vllm")
        assert "vllm.kv_cache_dtype" in params
        assert params["vllm.kv_cache_dtype"]["type_str"] == "literal"
        assert "auto" in params["vllm.kv_cache_dtype"]["options"]

    def test_extracts_nested_models(self):
        """Should recurse into nested Pydantic models."""
        from llenergymeasure.config.backend_configs import VLLMConfig

        params = get_params_from_model(VLLMConfig, prefix="vllm", include_nested=True)
        # VLLMAttentionConfig is nested under vllm.attention
        assert "vllm.attention.backend" in params

    def test_generates_test_values_for_bool(self):
        """Should generate [False, True] for boolean fields."""
        from llenergymeasure.config.backend_configs import VLLMConfig

        params = get_params_from_model(VLLMConfig, prefix="vllm")
        assert "vllm.enforce_eager" in params
        assert params["vllm.enforce_eager"]["test_values"] == [False, True]

    def test_generates_test_values_for_literal(self):
        """Should use all Literal values as test values."""
        from llenergymeasure.config.backend_configs import VLLMConfig

        params = get_params_from_model(VLLMConfig, prefix="vllm")
        kv_dtype = params["vllm.kv_cache_dtype"]
        # All options should be test values
        assert set(kv_dtype["test_values"]) == set(kv_dtype["options"])


class TestGetBackendParams:
    """Tests for get_backend_params function."""

    def test_pytorch_params(self):
        """Should return PyTorch backend parameters."""
        params = get_backend_params("pytorch")
        assert "pytorch.batch_size" in params
        assert "pytorch.attn_implementation" in params
        assert "pytorch.torch_compile" in params

    def test_vllm_params(self):
        """Should return vLLM backend parameters."""
        params = get_backend_params("vllm")
        assert "vllm.max_num_seqs" in params
        assert "vllm.gpu_memory_utilization" in params
        # Verify best_of is NOT present (removed in vLLM v1)
        assert "vllm.best_of" not in params

    def test_vllm_attention_backend_options(self):
        """Verify TORCH_SDPA is NOT in attention backend options (removed in vLLM v1)."""
        params = get_backend_params("vllm")
        assert "vllm.attention.backend" in params
        options = params["vllm.attention.backend"]["options"]
        assert "TORCH_SDPA" not in options
        assert "auto" in options
        assert "FLASH_ATTN" in options
        assert "FLASHINFER" in options

    def test_tensorrt_params(self):
        """Should return TensorRT backend parameters."""
        params = get_backend_params("tensorrt")
        assert "tensorrt.max_batch_size" in params
        assert "tensorrt.builder_opt_level" in params

    def test_invalid_backend_raises(self):
        """Should raise ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_params("invalid")


class TestGetSharedParams:
    """Tests for get_shared_params function."""

    def test_includes_decoder_params(self):
        """Should include decoder configuration params."""
        params = get_shared_params()
        assert "decoder.temperature" in params
        assert "decoder.top_p" in params
        assert "decoder.top_k" in params

    def test_includes_universal_params(self):
        """Should include universal top-level params."""
        params = get_shared_params()
        assert "fp_precision" in params
        assert "streaming" in params
        assert "max_input_tokens" in params
        assert "max_output_tokens" in params


class TestGetAllParams:
    """Tests for get_all_params function."""

    def test_returns_all_sections(self):
        """Should return params grouped by backend + shared."""
        all_params = get_all_params()
        assert "shared" in all_params
        assert "pytorch" in all_params
        assert "vllm" in all_params
        assert "tensorrt" in all_params

    def test_no_overlap_between_backends(self):
        """Backend-specific params should not overlap."""
        all_params = get_all_params()
        pytorch_keys = set(all_params["pytorch"].keys())
        vllm_keys = set(all_params["vllm"].keys())
        tensorrt_keys = set(all_params["tensorrt"].keys())

        # No overlap between backend params
        assert pytorch_keys.isdisjoint(vllm_keys)
        assert pytorch_keys.isdisjoint(tensorrt_keys)
        assert vllm_keys.isdisjoint(tensorrt_keys)


class TestGetParamTestValues:
    """Tests for get_param_test_values function."""

    def test_returns_values_for_known_param(self):
        """Should return test values for known parameters."""
        values = get_param_test_values("vllm.enforce_eager")
        assert values == [False, True]

    def test_returns_empty_for_unknown_param(self):
        """Should return empty list for unknown parameters."""
        values = get_param_test_values("nonexistent.param")
        assert values == []

    def test_decoder_params(self):
        """Should work for shared decoder params."""
        values = get_param_test_values("decoder.temperature")
        assert len(values) > 0


class TestGetParamOptions:
    """Tests for get_param_options function."""

    def test_returns_options_for_literal(self):
        """Should return options for Literal-typed params."""
        options = get_param_options("vllm.kv_cache_dtype")
        assert options is not None
        assert "auto" in options

    def test_returns_none_for_non_literal(self):
        """Should return None for non-Literal params."""
        options = get_param_options("vllm.max_num_seqs")
        assert options is None


class TestListAllParamPaths:
    """Tests for list_all_param_paths function."""

    def test_returns_sorted_list(self):
        """Should return sorted list of param paths."""
        paths = list_all_param_paths()
        assert paths == sorted(paths)

    def test_filters_by_backend(self):
        """Should filter to specific backend when requested."""
        pytorch_paths = list_all_param_paths("pytorch")
        assert all(p.startswith("pytorch.") for p in pytorch_paths)

    def test_invalid_backend_raises(self):
        """Should raise ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            list_all_param_paths("invalid")


# =============================================================================
# Tests for SSOT Constraint Metadata Functions
# =============================================================================


class TestGetMutualExclusions:
    """Tests for get_mutual_exclusions function."""

    def test_returns_dict(self):
        """Should return dict mapping params to exclusions."""
        exclusions = get_mutual_exclusions()
        assert isinstance(exclusions, dict)

    def test_pytorch_quantization_exclusions(self):
        """4-bit and 8-bit quantization should be mutually exclusive."""
        exclusions = get_mutual_exclusions()
        assert "pytorch.load_in_4bit" in exclusions
        assert "pytorch.load_in_8bit" in exclusions["pytorch.load_in_4bit"]
        assert "pytorch.load_in_4bit" in exclusions["pytorch.load_in_8bit"]

    def test_exclusions_are_symmetric(self):
        """If A excludes B, B should exclude A."""
        exclusions = get_mutual_exclusions()
        # Check 4bit <-> 8bit symmetry
        if "pytorch.load_in_8bit" in exclusions.get("pytorch.load_in_4bit", []):
            assert "pytorch.load_in_4bit" in exclusions.get("pytorch.load_in_8bit", [])


class TestGetBackendSpecificParams:
    """Tests for get_backend_specific_params function."""

    def test_returns_dict_of_lists(self):
        """Should return dict mapping backends to param lists."""
        params = get_backend_specific_params()
        assert isinstance(params, dict)
        assert "pytorch" in params
        assert "vllm" in params
        assert "tensorrt" in params
        assert all(isinstance(v, list) for v in params.values())

    def test_pytorch_params_correct(self):
        """PyTorch-specific params should be listed."""
        params = get_backend_specific_params()
        assert "pytorch.batch_size" in params["pytorch"]
        assert "pytorch.load_in_4bit" in params["pytorch"]

    def test_vllm_params_correct(self):
        """vLLM-specific params should be listed."""
        params = get_backend_specific_params()
        assert "vllm.max_num_seqs" in params["vllm"]
        assert "vllm.enable_prefix_caching" in params["vllm"]

    def test_tensorrt_params_correct(self):
        """TensorRT-specific params should be listed."""
        params = get_backend_specific_params()
        assert "tensorrt.max_batch_size" in params["tensorrt"]
        assert "tensorrt.builder_opt_level" in params["tensorrt"]


class TestGetSpecialTestModels:
    """Tests for get_special_test_models function."""

    def test_returns_dict(self):
        """Should return dict mapping param patterns to models."""
        models = get_special_test_models()
        assert isinstance(models, dict)

    def test_awq_model_defined(self):
        """AWQ quantization should have a test model."""
        models = get_special_test_models()
        assert "vllm.quantization=awq" in models
        assert "AWQ" in models["vllm.quantization=awq"]

    def test_gptq_model_defined(self):
        """GPTQ quantization should have a test model."""
        models = get_special_test_models()
        assert "vllm.quantization=gptq" in models
        assert "GPTQ" in models["vllm.quantization=gptq"]

    def test_tensorrt_quant_models(self):
        """TensorRT quantization methods should have test models."""
        models = get_special_test_models()
        assert "tensorrt.quantization=int4_awq" in models
        assert "tensorrt.quantization=int4_gptq" in models


class TestGetParamsRequiringGpuCapability:
    """Tests for get_params_requiring_gpu_capability function."""

    def test_returns_list(self):
        """Should return list of param patterns."""
        params = get_params_requiring_gpu_capability()
        assert isinstance(params, list)

    def test_ampere_params_included(self):
        """Should include Ampere-required params by default."""
        params = get_params_requiring_gpu_capability(min_compute_capability=8.0)
        assert any("fp8" in p for p in params)
        assert any("flash_attention_2" in p for p in params)

    def test_hopper_params_at_90(self):
        """Should include Hopper params when compute >= 9.0."""
        params = get_params_requiring_gpu_capability(min_compute_capability=9.0)
        assert any("flash_version=3" in p for p in params)


class TestGetParamSkipConditions:
    """Tests for get_param_skip_conditions function."""

    def test_returns_dict(self):
        """Should return dict mapping params to skip reasons."""
        conditions = get_param_skip_conditions()
        assert isinstance(conditions, dict)

    def test_multi_gpu_conditions(self):
        """Multi-GPU params should have skip conditions."""
        conditions = get_param_skip_conditions()
        assert any("tensor_parallel" in k for k in conditions)
        assert any("2+ GPUs" in v for v in conditions.values())

    def test_ray_backend_condition(self):
        """Ray backend should have skip condition."""
        conditions = get_param_skip_conditions()
        assert "vllm.distributed_backend=ray" in conditions
        assert "ray" in conditions["vllm.distributed_backend=ray"].lower()


# =============================================================================
# Integration Tests - SSOT Architecture Validation
# =============================================================================


class TestSSOTArchitectureIntegration:
    """Integration tests validating SSOT architecture consistency."""

    def test_all_backend_params_have_test_values(self):
        """Every backend param should have at least one test value."""
        # Some complex types don't generate test values cleanly
        # These are known exceptions
        known_exceptions = {
            "pytorch.torch_compile",  # bool | Literal[...] Union type
            "pytorch.torch_compile_backend",  # Optional Literal
            "pytorch.max_memory",  # dict[str, str] - device memory mapping
            "pytorch.extra",  # dict escape hatch
            "vllm.extra",  # dict escape hatch
            "vllm.logit_bias",  # dict[int, float]
            "tensorrt.extra_build_args",  # dict escape hatch
            "tensorrt.extra_runtime_args",  # dict escape hatch
        }

        for backend in ["pytorch", "vllm", "tensorrt"]:
            params = get_backend_params(backend)
            for path, meta in params.items():
                # Skip nested model references, strings, and known exceptions
                if meta["type_str"] not in ("unknown", "str") and path not in known_exceptions:
                    assert len(meta["test_values"]) > 0, f"{path} has no test values"

    def test_special_models_reference_valid_params(self):
        """Special test models should reference params that exist."""
        all_params = get_all_params()
        models = get_special_test_models()

        for param_pattern in models:
            # Parse "backend.param=value" format
            param_path = param_pattern.split("=")[0]
            # Find which backend section contains this param
            found = False
            for section in all_params.values():
                if param_path in section:
                    found = True
                    break
            assert found, f"Special model references unknown param: {param_path}"

    def test_mutual_exclusions_reference_valid_params(self):
        """Mutual exclusions should reference params that exist."""
        all_params = get_all_params()
        exclusions = get_mutual_exclusions()

        for param_path, excluded in exclusions.items():
            # Check the main param exists
            found = any(param_path in section for section in all_params.values())
            assert found, f"Exclusion references unknown param: {param_path}"

            # Check excluded params exist
            for exc_param in excluded:
                found = any(exc_param in section for section in all_params.values())
                assert found, f"Exclusion references unknown excluded param: {exc_param}"

    def test_vllm_best_of_removed_from_ssot(self):
        """Verify best_of is completely removed from SSOT architecture."""
        # Check backend params
        vllm_params = get_backend_params("vllm")
        assert "vllm.best_of" not in vllm_params

        # Check test values
        values = get_param_test_values("vllm.best_of")
        assert values == []

        # Check options
        options = get_param_options("vllm.best_of")
        assert options is None

    def test_vllm_attention_backend_torch_sdpa_removed(self):
        """Verify TORCH_SDPA is completely removed from attention backend."""
        vllm_params = get_backend_params("vllm")
        attention_backend = vllm_params.get("vllm.attention.backend", {})
        options = attention_backend.get("options", [])
        test_values = attention_backend.get("test_values", [])

        assert "TORCH_SDPA" not in options
        assert "TORCH_SDPA" not in test_values
