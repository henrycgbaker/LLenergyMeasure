"""Unit tests for parallelism constraint validation (Phase 3)."""

from __future__ import annotations

from llenergymeasure.config.backend_configs import PyTorchConfig, TensorRTConfig, VLLMConfig
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.validation import (
    ConfigWarning,
    validate_parallelism_constraints,
)


class TestVLLMTensorParallelValidation:
    """Tests for vLLM tensor_parallel_size validation against gpus."""

    def test_vllm_tensor_parallel_exceeds_gpus_returns_error(self) -> None:
        """tensor_parallel_size > len(gpus) should return error.

        Note: When TP exceeds GPUs, TP*PP also exceeds GPUs (since PP defaults to 1),
        so we get 2 errors: one for TP and one for TP*PP.
        """
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0],  # 1 GPU
            vllm=VLLMConfig(tensor_parallel_size=4),  # Needs 4 GPUs
        )
        warnings = validate_parallelism_constraints(config)

        # Both TP and TP*PP errors (TP exceeds -> TP*PP also exceeds)
        assert len(warnings) >= 1
        tp_errors = [w for w in warnings if "tensor_parallel_size" in w.field]
        assert len(tp_errors) == 1
        assert tp_errors[0].severity == "error"
        assert "tensor_parallel_size=4" in tp_errors[0].message

    def test_vllm_tensor_parallel_equals_gpus_passes(self) -> None:
        """tensor_parallel_size == len(gpus) should pass."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0, 1, 2, 3],  # 4 GPUs
            vllm=VLLMConfig(tensor_parallel_size=4),  # Needs 4 GPUs
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0

    def test_vllm_tensor_parallel_less_than_gpus_passes(self) -> None:
        """tensor_parallel_size < len(gpus) should pass."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0, 1, 2, 3],  # 4 GPUs
            vllm=VLLMConfig(tensor_parallel_size=2),  # Only needs 2
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0


class TestVLLMPipelineParallelValidation:
    """Tests for vLLM total parallelism (TP * PP) validation."""

    def test_vllm_total_parallelism_exceeds_gpus_returns_error(self) -> None:
        """tensor_parallel * pipeline_parallel > len(gpus) should return error."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0, 1],  # 2 GPUs
            vllm=VLLMConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,  # Total = 4, exceeds 2
            ),
        )
        warnings = validate_parallelism_constraints(config)

        # Should have error for TP*PP exceeding GPUs
        pipeline_errors = [w for w in warnings if "pipeline_parallel_size" in w.field]
        assert len(pipeline_errors) == 1
        assert pipeline_errors[0].severity == "error"
        assert "= 4" in pipeline_errors[0].message  # TP * PP = 4

    def test_vllm_total_parallelism_equals_gpus_passes(self) -> None:
        """tensor_parallel * pipeline_parallel == len(gpus) should pass."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0, 1, 2, 3],  # 4 GPUs
            vllm=VLLMConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,  # Total = 4
            ),
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0


class TestVLLMNoConfigPasses:
    """Tests for vLLM without explicit config (defaults)."""

    def test_vllm_no_config_passes(self) -> None:
        """vLLM backend without explicit vllm config uses defaults (tp=1)."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0],  # 1 GPU
            # No vllm config - uses VLLMConfig defaults (tensor_parallel_size=1)
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0


class TestTensorRTTpSizeValidation:
    """Tests for TensorRT tp_size validation against gpus."""

    def test_tensorrt_tp_size_exceeds_gpus_returns_error(self) -> None:
        """tp_size > len(gpus) should return error.

        Note: When TP exceeds GPUs, TP*PP also exceeds GPUs (since PP defaults to 1),
        so we get 2 errors: one for TP and one for TP*PP.
        """
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="tensorrt",
            gpus=[0, 1],  # 2 GPUs
            tensorrt=TensorRTConfig(tp_size=4),  # Needs 4 GPUs
        )
        warnings = validate_parallelism_constraints(config)

        # Both TP and TP*PP errors
        assert len(warnings) >= 1
        tp_errors = [w for w in warnings if w.field == "tensorrt.tp_size"]
        assert len(tp_errors) == 1
        assert tp_errors[0].severity == "error"
        assert "tp_size=4" in tp_errors[0].message

    def test_tensorrt_tp_size_equals_gpus_passes(self) -> None:
        """tp_size == len(gpus) should pass."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="tensorrt",
            gpus=[0, 1, 2, 3],  # 4 GPUs
            tensorrt=TensorRTConfig(tp_size=4),
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0

    def test_tensorrt_tp_pp_total_exceeds_gpus_returns_error(self) -> None:
        """tp_size * pp_size > len(gpus) should return error."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="tensorrt",
            gpus=[0, 1],  # 2 GPUs
            tensorrt=TensorRTConfig(tp_size=2, pp_size=2),  # Total = 4
        )
        warnings = validate_parallelism_constraints(config)

        # Should have error for TP*PP exceeding GPUs
        pp_errors = [w for w in warnings if "pp_size" in w.field]
        assert len(pp_errors) == 1
        assert pp_errors[0].severity == "error"


class TestPyTorchNumProcessesValidation:
    """Tests for PyTorch num_processes validation against gpus."""

    def test_pytorch_num_processes_exceeds_gpus_returns_error(self) -> None:
        """num_processes > len(gpus) should return error."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="pytorch",
            gpus=[0],  # 1 GPU
            pytorch=PyTorchConfig(num_processes=4),  # Needs 4 GPUs
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 1
        assert warnings[0].severity == "error"
        assert "num_processes=4" in warnings[0].message
        assert "pytorch.num_processes" in warnings[0].field

    def test_pytorch_num_processes_equals_gpus_passes(self) -> None:
        """num_processes == len(gpus) should pass."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="pytorch",
            gpus=[0, 1, 2, 3],  # 4 GPUs
            pytorch=PyTorchConfig(num_processes=4),
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0

    def test_pytorch_default_num_processes_passes(self) -> None:
        """PyTorch with default num_processes=1 should pass."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="pytorch",
            gpus=[0],  # 1 GPU
            # Default PyTorchConfig has num_processes=1
        )
        warnings = validate_parallelism_constraints(config)

        assert len(warnings) == 0


class TestEdgeCases:
    """Tests for edge cases in parallelism validation."""

    def test_empty_gpus_treated_as_single_gpu(self) -> None:
        """Empty gpus list should be treated as 1 GPU available."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[],  # Empty - treated as 1 GPU
            vllm=VLLMConfig(tensor_parallel_size=2),  # Needs 2
        )
        warnings = validate_parallelism_constraints(config)

        # Should have at least one error (TP exceeds available)
        assert len(warnings) >= 1
        tp_errors = [w for w in warnings if "tensor_parallel_size" in w.field]
        assert len(tp_errors) == 1
        assert tp_errors[0].severity == "error"

    def test_non_matching_backend_tensorrt_with_vllm_backend(self) -> None:
        """TensorRT config with vLLM backend should not trigger TensorRT validation.

        Note: ExperimentConfig validates that backend-specific config matches backend,
        so we can only test that TensorRT validation doesn't run for vLLM backend
        by checking the vLLM config doesn't trigger TensorRT errors.
        """
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0],
            vllm=VLLMConfig(tensor_parallel_size=1),  # Valid for 1 GPU
        )
        warnings = validate_parallelism_constraints(config)

        # Should not have tensorrt-related errors (wrong backend)
        tensorrt_errors = [w for w in warnings if "tensorrt" in w.field]
        assert len(tensorrt_errors) == 0

    def test_suggestion_included_in_error(self) -> None:
        """Error warnings should include a suggestion for fixing."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0],  # 1 GPU
            vllm=VLLMConfig(tensor_parallel_size=4),  # Needs 4
        )
        warnings = validate_parallelism_constraints(config)

        # At least one error should be present
        assert len(warnings) >= 1
        # First error (TP) should have suggestion
        tp_error = next(w for w in warnings if "tensor_parallel_size" in w.field)
        assert tp_error.suggestion is not None
        assert (
            "gpus" in tp_error.suggestion.lower()
            or "tensor_parallel" in tp_error.suggestion.lower()
        )


class TestConfigWarningDataclass:
    """Tests for ConfigWarning dataclass properties."""

    def test_config_warning_str_format(self) -> None:
        """ConfigWarning __str__ returns formatted string."""
        warning = ConfigWarning(
            field="test.field",
            message="Test message",
            severity="error",
        )
        result = str(warning)

        assert "[ERROR]" in result
        assert "test.field" in result
        assert "Test message" in result

    def test_config_warning_param_alias(self) -> None:
        """ConfigWarning.param is alias for field (backwards compat)."""
        warning = ConfigWarning(field="my_field", message="msg")

        assert warning.param == "my_field"
        assert warning.param == warning.field

    def test_config_warning_to_result_string(self) -> None:
        """ConfigWarning.to_result_string formats for embedding in results."""
        warning = ConfigWarning(
            field="test.field",
            message="Test message",
            severity="warning",
        )
        result = warning.to_result_string()

        assert "warning" in result
        assert "test.field" in result
        assert "Test message" in result
