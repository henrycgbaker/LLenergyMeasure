"""PyTorch backend parameter specifications.

Defines ParamSpecs for all PyTorch/Transformers-specific parameters,
covering attention, compilation, caching, and memory management.
"""

from __future__ import annotations

from ...registry import (
    HardwareRequirement,
    ParamSpec,
    VerificationType,
    register_all,
)

# =============================================================================
# Attention Parameters
# =============================================================================

ATTENTION_SPECS = [
    ParamSpec(
        name="attn_implementation",
        backend="pytorch",
        config_path="pytorch.attn_implementation",
        test_values=[
            "sdpa",
            "eager",
        ],  # flash_attention_2 tested separately in GPU tests with Flash Attn requirement
        verification_type=VerificationType.INTROSPECTION,
        hardware_requirements={HardwareRequirement.GPU},
        description="Attention implementation (sdpa, flash_attention_2, eager)",
        category="attention",
        energy_impact=True,
    ),
]

# =============================================================================
# Compilation Parameters
# =============================================================================

COMPILATION_SPECS = [
    ParamSpec(
        name="torch_compile",
        backend="pytorch",
        config_path="pytorch.torch_compile",
        test_values=[False, True, "default", "reduce-overhead"],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Enable torch.compile optimization",
        category="compilation",
        energy_impact=True,
    ),
]

# =============================================================================
# Legacy Optimization Parameters
# =============================================================================

LEGACY_SPECS = [
    ParamSpec(
        name="use_bettertransformer",
        backend="pytorch",
        config_path="pytorch.use_bettertransformer",
        test_values=[True, False],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Convert to BetterTransformer (pre-PyTorch 2.0)",
        category="legacy",
        skip_reason="BetterTransformer deprecated in PyTorch 2.0+",
    ),
]

# =============================================================================
# KV Cache Parameters
# =============================================================================

CACHE_SPECS = [
    ParamSpec(
        name="use_cache",
        backend="pytorch",
        config_path="pytorch.use_cache",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU},
        passthrough_path="generation_config.use_cache",
        description="Enable KV caching during generation",
        category="cache",
        energy_impact=True,
    ),
    ParamSpec(
        name="cache_implementation",
        backend="pytorch",
        config_path="pytorch.cache_implementation",
        test_values=["dynamic", "static"],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="KV cache implementation type",
        category="cache",
        energy_impact=True,
    ),
]

# =============================================================================
# Memory Management Parameters
# =============================================================================

MEMORY_SPECS = [
    ParamSpec(
        name="low_cpu_mem_usage",
        backend="pytorch",
        config_path="pytorch.low_cpu_mem_usage",
        test_values=[True, False],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Memory-efficient model loading",
        category="memory",
    ),
    ParamSpec(
        name="max_memory",
        backend="pytorch",
        config_path="pytorch.max_memory",
        test_values=[None, {"0": "10GiB"}],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Per-device memory limits",
        category="memory",
        skip_reason="max_memory verification requires multi-GPU or specific setup",
    ),
]

# =============================================================================
# Assisted Generation (Speculative Decoding) Parameters
# =============================================================================

ASSISTED_GENERATION_SPECS = [
    ParamSpec(
        name="assisted_generation.model",
        backend="pytorch",
        config_path="pytorch.assisted_generation.model",
        test_values=["facebook/opt-125m"],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Assistant/draft model for speculative decoding",
        category="assisted_generation",
        skip_reason="Assisted generation requires compatible model pairs",
    ),
    ParamSpec(
        name="assisted_generation.num_tokens",
        backend="pytorch",
        config_path="pytorch.assisted_generation.num_tokens",
        test_values=[3, 5],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Number of tokens to speculate per step",
        category="assisted_generation",
        skip_reason="Assisted generation requires compatible model pairs",
    ),
]

# =============================================================================
# Generation Output Parameters
# =============================================================================

OUTPUT_SPECS = [
    ParamSpec(
        name="output_scores",
        backend="pytorch",
        config_path="pytorch.output_scores",
        test_values=[True, False],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Return generation scores/logprobs",
        category="output",
    ),
    ParamSpec(
        name="return_dict_in_generate",
        backend="pytorch",
        config_path="pytorch.return_dict_in_generate",
        test_values=[True, False],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU},
        description="Return GenerateOutput dict instead of tensor",
        category="output",
    ),
]


# =============================================================================
# Registry
# =============================================================================

PYTORCH_PARAM_SPECS = (
    ATTENTION_SPECS
    + COMPILATION_SPECS
    + LEGACY_SPECS
    + CACHE_SPECS
    + MEMORY_SPECS
    + ASSISTED_GENERATION_SPECS
    + OUTPUT_SPECS
)


def register_pytorch_params() -> None:
    """Register all PyTorch parameter specs with the global registry."""
    register_all(PYTORCH_PARAM_SPECS)
