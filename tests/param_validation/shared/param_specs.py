"""Shared parameter specifications.

Defines ParamSpecs for parameters that apply across multiple backends,
such as decoder settings and batching configurations.
"""

from __future__ import annotations

from ..registry import (
    HardwareRequirement,
    ParamSpec,
    VerificationType,
    register_all,
)

# =============================================================================
# Decoder Parameters (shared across backends)
# =============================================================================

DECODER_SPECS = [
    ParamSpec(
        name="temperature",
        backend="shared",
        config_path="decoder.temperature",
        test_values=[0.0, 0.7, 1.0],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Sampling temperature (0=deterministic, higher=more random)",
        category="decoder",
        energy_impact=False,
    ),
    ParamSpec(
        name="top_p",
        backend="shared",
        config_path="decoder.top_p",
        test_values=[0.9, 0.95, 1.0],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Nucleus sampling probability threshold",
        category="decoder",
    ),
    ParamSpec(
        name="top_k",
        backend="shared",
        config_path="decoder.top_k",
        test_values=[0, 10, 50],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Top-k sampling (0=disabled)",
        category="decoder",
    ),
    ParamSpec(
        name="max_output_tokens",
        backend="shared",
        config_path="max_output_tokens",
        test_values=[16, 32, 64],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Maximum output tokens to generate",
        category="decoder",
    ),
    ParamSpec(
        name="repetition_penalty",
        backend="shared",
        config_path="decoder.repetition_penalty",
        test_values=[1.0, 1.1, 1.2],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Penalty for repeating tokens",
        category="decoder",
    ),
]

# =============================================================================
# Batching Parameters (shared across backends)
# =============================================================================

BATCHING_SPECS = [
    ParamSpec(
        name="batch_size",
        backend="shared",
        config_path="batching.batch_size",
        test_values=[1, 2, 4, 8],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Number of prompts to process in parallel",
        category="batching",
        energy_impact=True,
    ),
    ParamSpec(
        name="dynamic_batching",
        backend="shared",
        config_path="batching.dynamic_batching",
        test_values=[True, False],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Enable dynamic batching",
        category="batching",
        energy_impact=True,
    ),
]

# =============================================================================
# Precision Parameters (shared across backends)
# =============================================================================

PRECISION_SPECS = [
    ParamSpec(
        name="precision",
        backend="shared",
        config_path="precision",
        test_values=["float16", "bfloat16"],
        verification_type=VerificationType.INTROSPECTION,
        hardware_requirements={HardwareRequirement.GPU},
        description="Model precision (float16, bfloat16, float32)",
        category="precision",
        energy_impact=True,
    ),
    ParamSpec(
        name="precision_bf16",
        backend="shared",
        config_path="precision",
        test_values=["bfloat16"],
        verification_type=VerificationType.INTROSPECTION,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.BF16},
        description="BFloat16 precision (Ampere+ required)",
        category="precision",
        energy_impact=True,
    ),
]

# =============================================================================
# Seed/Determinism Parameters
# =============================================================================

DETERMINISM_SPECS = [
    ParamSpec(
        name="seed",
        backend="shared",
        config_path="decoder.seed",
        test_values=[42, 123],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Random seed for reproducibility",
        category="determinism",
    ),
]

# =============================================================================
# Parallelism Parameters (shared across all backends)
# =============================================================================

PARALLELISM_SPECS = [
    ParamSpec(
        name="parallelism.strategy",
        backend="shared",
        config_path="parallelism.strategy",
        test_values=["none", "tensor_parallel", "data_parallel"],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.MULTI_GPU},
        description="Parallelism strategy (none, tensor_parallel, pipeline_parallel, data_parallel)",
        category="parallelism",
        energy_impact=True,
    ),
    ParamSpec(
        name="parallelism.degree",
        backend="shared",
        config_path="parallelism.degree",
        test_values=[1, 2],  # Limited to 2 for dual-GPU setup
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.MULTI_GPU},
        description="Number of GPUs/workers for parallelism",
        category="parallelism",
        energy_impact=True,
    ),
]


# =============================================================================
# Registry
# =============================================================================

SHARED_PARAM_SPECS = (
    DECODER_SPECS + BATCHING_SPECS + PRECISION_SPECS + DETERMINISM_SPECS + PARALLELISM_SPECS
)


def register_shared_params() -> None:
    """Register all shared parameter specs with the global registry."""
    register_all(SHARED_PARAM_SPECS)
