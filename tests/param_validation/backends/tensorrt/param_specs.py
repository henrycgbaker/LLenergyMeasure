"""TensorRT-LLM backend parameter specifications.

Defines ParamSpecs for all TensorRT-LLM-specific parameters, covering
engine build, quantization, runtime, and parallelism configurations.
"""

from __future__ import annotations

from ...registry import (
    HardwareRequirement,
    ParamSpec,
    VerificationType,
    register_all,
)

# =============================================================================
# Build Configuration Parameters
# =============================================================================

BUILD_SPECS = [
    ParamSpec(
        name="max_batch_size",
        backend="tensorrt",
        config_path="tensorrt.max_batch_size",
        test_values=[1, 4, 8],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Maximum batch size for compiled engine",
        category="build",
        energy_impact=True,
    ),
    ParamSpec(
        name="max_input_len",
        backend="tensorrt",
        config_path="tensorrt.max_input_len",
        test_values=[128, 256, 512],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Maximum input sequence length",
        category="build",
    ),
    ParamSpec(
        name="max_output_len",
        backend="tensorrt",
        config_path="tensorrt.max_output_len",
        test_values=[32, 64, 128],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Maximum output tokens per request",
        category="build",
    ),
    ParamSpec(
        name="builder_opt_level",
        backend="tensorrt",
        config_path="tensorrt.builder_opt_level",
        test_values=[0, 3, 5],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="TensorRT builder optimization level (0-5)",
        category="build",
        energy_impact=True,
    ),
    ParamSpec(
        name="strongly_typed",
        backend="tensorrt",
        config_path="tensorrt.strongly_typed",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={
            HardwareRequirement.GPU,
            HardwareRequirement.TENSORRT,
            HardwareRequirement.HOPPER,
        },
        description="Enable strong typing for FP8 precision",
        category="build",
    ),
    ParamSpec(
        name="multiple_profiles",
        backend="tensorrt",
        config_path="tensorrt.multiple_profiles",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Build with multiple TensorRT profiles",
        category="build",
        energy_impact=True,
    ),
]

# =============================================================================
# Quantization Parameters
# =============================================================================

QUANTIZATION_SPECS = [
    ParamSpec(
        name="quantization.method",
        backend="tensorrt",
        config_path="tensorrt.quantization.method",
        test_values=["none"],  # FP8 tested separately in GPU tests with Hopper requirement
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Quantization method for TRT engine",
        category="quantization",
        energy_impact=True,
    ),
    ParamSpec(
        name="quantization.method_int8",
        backend="tensorrt",
        config_path="tensorrt.quantization.method",
        test_values=["int8_sq", "int8_weight_only"],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="INT8 quantization methods",
        category="quantization",
        energy_impact=True,
        skip_reason="INT8 quantization requires calibration data",
    ),
]

# =============================================================================
# Calibration Parameters (for INT8 quantization)
# =============================================================================

CALIBRATION_SPECS = [
    ParamSpec(
        name="quantization.calibration.dataset",
        backend="tensorrt",
        config_path="tensorrt.quantization.calibration.dataset",
        test_values=["wikitext"],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Calibration dataset",
        category="calibration",
        skip_reason="Calibration only used for INT8 quantization",
    ),
    ParamSpec(
        name="quantization.calibration.num_samples",
        backend="tensorrt",
        config_path="tensorrt.quantization.calibration.num_samples",
        test_values=[128, 512],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Number of calibration samples",
        category="calibration",
        skip_reason="Calibration only used for INT8 quantization",
    ),
]

# =============================================================================
# Parallelism Parameters (TensorRT-specific, supplements shared parallelism config)
# Note: Main parallelism settings (strategy, degree) are in shared/param_specs.py
# TensorRT uses parallelism.degree for tensor parallelism; pp_size is additional
# =============================================================================

PARALLELISM_SPECS = [
    ParamSpec(
        name="pp_size",
        backend="tensorrt",
        config_path="tensorrt.pp_size",
        test_values=[1, 2],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={
            HardwareRequirement.GPU,
            HardwareRequirement.TENSORRT,
            HardwareRequirement.MULTI_GPU,
        },
        description="TensorRT pipeline parallel size (additional to shared parallelism.degree for TP)",
        category="parallelism",
    ),
]

# =============================================================================
# Runtime Parameters
# =============================================================================

RUNTIME_SPECS = [
    ParamSpec(
        name="kv_cache_type",
        backend="tensorrt",
        config_path="tensorrt.kv_cache_type",
        test_values=["paged", "continuous"],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="KV cache management type",
        category="runtime",
        energy_impact=True,
    ),
    ParamSpec(
        name="enable_chunked_context",
        backend="tensorrt",
        config_path="tensorrt.enable_chunked_context",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Enable chunked context for long sequences",
        category="runtime",
    ),
    ParamSpec(
        name="max_num_tokens",
        backend="tensorrt",
        config_path="tensorrt.max_num_tokens",
        test_values=[512, 1024, 2048],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Maximum tokens per iteration",
        category="runtime",
    ),
    ParamSpec(
        name="gpu_memory_utilization",
        backend="tensorrt",
        config_path="tensorrt.gpu_memory_utilization",
        test_values=[0.5, 0.7, 0.9],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Fraction of GPU memory for KV cache",
        category="runtime",
        energy_impact=True,
    ),
    ParamSpec(
        name="enable_kv_cache_reuse",
        backend="tensorrt",
        config_path="tensorrt.enable_kv_cache_reuse",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Enable KV cache reuse for prefix caching",
        category="runtime",
        energy_impact=True,
    ),
]

# =============================================================================
# Cache Control Parameters
# =============================================================================

CACHE_CONTROL_SPECS = [
    ParamSpec(
        name="force_rebuild",
        backend="tensorrt",
        config_path="tensorrt.force_rebuild",
        test_values=[True, False],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Force engine rebuild even if cached",
        category="cache_control",
    ),
]

# =============================================================================
# Speculative Decoding Parameters
# =============================================================================

SPECULATIVE_SPECS = [
    ParamSpec(
        name="draft_model",
        backend="tensorrt",
        config_path="tensorrt.draft_model",
        test_values=["gpt2"],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Draft model for speculative decoding",
        category="speculative",
        skip_reason="Speculative decoding requires compatible model pairs",
    ),
    ParamSpec(
        name="num_draft_tokens",
        backend="tensorrt",
        config_path="tensorrt.num_draft_tokens",
        test_values=[3, 5],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.TENSORRT},
        description="Number of tokens to speculate per step",
        category="speculative",
        skip_reason="Speculative decoding requires compatible model pairs",
    ),
]


# =============================================================================
# Registry
# =============================================================================

TENSORRT_PARAM_SPECS = (
    BUILD_SPECS
    + QUANTIZATION_SPECS
    + CALIBRATION_SPECS
    + PARALLELISM_SPECS
    + RUNTIME_SPECS
    + CACHE_CONTROL_SPECS
    + SPECULATIVE_SPECS
)


def register_tensorrt_params() -> None:
    """Register all TensorRT parameter specs with the global registry."""
    register_all(TENSORRT_PARAM_SPECS)
