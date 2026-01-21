"""CI-safe tests for config parsing validation.

These tests verify that Pydantic models correctly parse all parameter
values without requiring GPU hardware. Safe to run in CI.
"""

from __future__ import annotations

import pytest

from .backends.pytorch import register_pytorch_params
from .backends.tensorrt import register_tensorrt_params
from .backends.vllm import register_vllm_params
from .registry import ParamSpec, get_registry
from .shared import register_shared_params
from .verifiers import MockVerifier


def _ensure_registry_populated() -> None:
    """Ensure registry is populated (idempotent).

    Registration is idempotent - safe to call multiple times.
    """
    register_vllm_params()
    register_pytorch_params()
    register_tensorrt_params()
    register_shared_params()


def setup_module() -> None:
    """Register all param specs before tests."""
    _ensure_registry_populated()


def get_all_backend_params() -> list[tuple[str, ParamSpec, object]]:
    """Get all backend parameters for parametrization."""
    _ensure_registry_populated()

    params = []
    for spec in get_registry().all_specs:
        # Skip shared params (they're not in backend configs)
        if spec.backend == "shared":
            continue
        # Skip specs with skip_reason
        if spec.skip_reason:
            continue
        for test_value in spec.test_values:
            params.append((f"{spec.full_name}={test_value}", spec, test_value))

    return params


@pytest.mark.parametrize(
    "test_id,spec,test_value",
    get_all_backend_params(),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_config_parsing(test_id: str, spec: ParamSpec, test_value: object) -> None:
    """Test that each parameter value is correctly parsed by Pydantic.

    This test:
    1. Creates a config dict with the parameter value
    2. Parses it through the appropriate Pydantic model
    3. Verifies the value is correctly stored
    """
    verifier = MockVerifier()
    result = verifier.verify_config_parsing(spec, test_value)

    if result.failed:
        pytest.fail(f"{spec.full_name}={test_value}: {result.message}")


class TestVLLMConfigParsing:
    """Test vLLM config parsing specifically."""

    def test_max_num_seqs_bounds(self) -> None:
        """Test max_num_seqs respects bounds (1-1024)."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        # Valid values
        config = VLLMConfig(max_num_seqs=1)
        assert config.max_num_seqs == 1

        config = VLLMConfig(max_num_seqs=1024)
        assert config.max_num_seqs == 1024

        # Invalid values should raise
        with pytest.raises(ValueError):
            VLLMConfig(max_num_seqs=0)

        with pytest.raises(ValueError):
            VLLMConfig(max_num_seqs=1025)

    def test_gpu_memory_utilization_bounds(self) -> None:
        """Test gpu_memory_utilization respects bounds (0.5-0.99)."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        config = VLLMConfig(gpu_memory_utilization=0.5)
        assert config.gpu_memory_utilization == 0.5

        config = VLLMConfig(gpu_memory_utilization=0.99)
        assert config.gpu_memory_utilization == 0.99

        with pytest.raises(ValueError):
            VLLMConfig(gpu_memory_utilization=0.49)

        with pytest.raises(ValueError):
            VLLMConfig(gpu_memory_utilization=1.0)

    def test_block_size_literal(self) -> None:
        """Test block_size only accepts valid literal values."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        for valid in [8, 16, 32]:
            config = VLLMConfig(block_size=valid)
            assert config.block_size == valid

        with pytest.raises(ValueError):
            VLLMConfig(block_size=64)

    def test_kv_cache_dtype_literal(self) -> None:
        """Test kv_cache_dtype only accepts valid literal values."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        for valid in ["auto", "float16", "bfloat16", "fp8"]:
            config = VLLMConfig(kv_cache_dtype=valid)
            assert config.kv_cache_dtype == valid

        with pytest.raises(ValueError):
            VLLMConfig(kv_cache_dtype="invalid")

    def test_nested_lora_config(self) -> None:
        """Test nested LoRA config parsing."""
        from llm_energy_measure.config.backend_configs import VLLMConfig, VLLMLoRAConfig

        config = VLLMConfig(
            lora=VLLMLoRAConfig(
                enabled=True,
                max_loras=4,
                max_rank=32,
            )
        )

        assert config.lora is not None
        assert config.lora.enabled is True
        assert config.lora.max_loras == 4
        assert config.lora.max_rank == 32


class TestPyTorchConfigParsing:
    """Test PyTorch config parsing specifically."""

    def test_attn_implementation_literal(self) -> None:
        """Test attn_implementation only accepts valid values."""
        from llm_energy_measure.config.backend_configs import PyTorchConfig

        for valid in ["sdpa", "flash_attention_2", "eager"]:
            config = PyTorchConfig(attn_implementation=valid)
            assert config.attn_implementation == valid

        with pytest.raises(ValueError):
            PyTorchConfig(attn_implementation="invalid")

    def test_torch_compile_variants(self) -> None:
        """Test torch_compile accepts bool and string modes."""
        from llm_energy_measure.config.backend_configs import PyTorchConfig

        # Boolean
        config = PyTorchConfig(torch_compile=False)
        assert config.torch_compile is False

        config = PyTorchConfig(torch_compile=True)
        assert config.torch_compile is True

        # String modes
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            config = PyTorchConfig(torch_compile=mode)
            assert config.torch_compile == mode

    def test_cache_implementation_literal(self) -> None:
        """Test cache_implementation only accepts valid values."""
        from llm_energy_measure.config.backend_configs import PyTorchConfig

        for valid in ["dynamic", "static", "hybrid", "sliding_window"]:
            config = PyTorchConfig(cache_implementation=valid)
            assert config.cache_implementation == valid


class TestTensorRTConfigParsing:
    """Test TensorRT config parsing specifically."""

    def test_max_batch_size_bounds(self) -> None:
        """Test max_batch_size respects bounds (1-256)."""
        from llm_energy_measure.config.backend_configs import TensorRTConfig

        config = TensorRTConfig(max_batch_size=1)
        assert config.max_batch_size == 1

        config = TensorRTConfig(max_batch_size=256)
        assert config.max_batch_size == 256

        with pytest.raises(ValueError):
            TensorRTConfig(max_batch_size=0)

        with pytest.raises(ValueError):
            TensorRTConfig(max_batch_size=257)

    def test_builder_opt_level_bounds(self) -> None:
        """Test builder_opt_level respects bounds (0-5)."""
        from llm_energy_measure.config.backend_configs import TensorRTConfig

        for level in range(6):
            config = TensorRTConfig(builder_opt_level=level)
            assert config.builder_opt_level == level

        with pytest.raises(ValueError):
            TensorRTConfig(builder_opt_level=6)

    def test_kv_cache_type_literal(self) -> None:
        """Test kv_cache_type only accepts valid values."""
        from llm_energy_measure.config.backend_configs import TensorRTConfig

        for valid in ["paged", "continuous"]:
            config = TensorRTConfig(kv_cache_type=valid)
            assert config.kv_cache_type == valid

        with pytest.raises(ValueError):
            TensorRTConfig(kv_cache_type="invalid")

    def test_nested_quantization_config(self) -> None:
        """Test nested quantization config parsing."""
        from llm_energy_measure.config.backend_configs import (
            TensorRTConfig,
            TensorRTQuantizationConfig,
        )

        config = TensorRTConfig(quantization=TensorRTQuantizationConfig(method="fp8"))

        assert config.quantization.method == "fp8"


class TestConfigDiscovery:
    """Test the config discovery mechanism."""

    def test_discover_vllm_fields(self) -> None:
        """Test that we can discover all vLLM config fields."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        from .registry import discover_model_fields

        fields = discover_model_fields(VLLMConfig)

        # Check key fields exist
        assert "max_num_seqs" in fields
        assert "gpu_memory_utilization" in fields
        assert "enable_prefix_caching" in fields

        # Check field properties
        max_num_seqs = fields["max_num_seqs"]
        assert max_num_seqs.ge == 1
        assert max_num_seqs.le == 1024
        assert max_num_seqs.is_numeric is True

    def test_discover_pytorch_fields(self) -> None:
        """Test that we can discover all PyTorch config fields."""
        from llm_energy_measure.config.backend_configs import PyTorchConfig

        from .registry import discover_model_fields

        fields = discover_model_fields(PyTorchConfig)

        assert "attn_implementation" in fields
        assert "torch_compile" in fields
        assert "use_cache" in fields

    def test_discover_tensorrt_fields(self) -> None:
        """Test that we can discover all TensorRT config fields."""
        from llm_energy_measure.config.backend_configs import TensorRTConfig

        from .registry import discover_model_fields

        fields = discover_model_fields(TensorRTConfig)

        assert "max_batch_size" in fields
        assert "builder_opt_level" in fields
        assert "quantization" in fields

    def test_infer_test_values_from_literal(self) -> None:
        """Test that we infer correct test values from Literal types."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        from .registry import discover_model_fields, infer_test_values

        fields = discover_model_fields(VLLMConfig)

        # block_size is Literal[8, 16, 32]
        block_size = fields["block_size"]
        values = infer_test_values(block_size)
        assert set(values) == {8, 16, 32}

    def test_infer_test_values_from_bool(self) -> None:
        """Test that we infer True/False for boolean fields."""
        from llm_energy_measure.config.backend_configs import VLLMConfig

        from .registry import discover_model_fields, infer_test_values

        fields = discover_model_fields(VLLMConfig)

        # enable_prefix_caching is bool
        enable_prefix_caching = fields["enable_prefix_caching"]
        values = infer_test_values(enable_prefix_caching)
        assert set(values) == {True, False}
