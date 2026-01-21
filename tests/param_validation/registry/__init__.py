"""Registry package for parameter validation framework.

Provides data models, hardware detection, Pydantic discovery, and the
central ParamSpec registry.
"""

from .discovery import (
    DiscoveredField,
    discover_all_backend_params,
    discover_model_fields,
    get_coverage_report,
    infer_test_values,
)
from .hardware_caps import (
    HardwareProfile,
    bf16_supported,
    check_requirements,
    detect_hardware,
    flash_attn_available,
    get_hardware_summary,
    gpu_available,
    is_ampere_or_newer,
    is_hopper_or_newer,
    multi_gpu_available,
    supports_fp8,
    tensorrt_available,
    vllm_available,
    vllm_version_at_least,
)
from .models import (
    BackendCapabilities,
    HardwareRequirement,
    ParamCoverage,
    ParamSpec,
    RunSummary,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from .param_registry import ParamRegistry, get_registry, register, register_all, registry

__all__ = [
    "BackendCapabilities",
    # Discovery
    "DiscoveredField",
    # Hardware
    "HardwareProfile",
    "HardwareRequirement",
    "ParamCoverage",
    # Registry
    "ParamRegistry",
    # Models
    "ParamSpec",
    "RunSummary",
    "VerificationResult",
    "VerificationStatus",
    "VerificationType",
    "bf16_supported",
    "check_requirements",
    "detect_hardware",
    "discover_all_backend_params",
    "discover_model_fields",
    "flash_attn_available",
    "get_coverage_report",
    "get_hardware_summary",
    "get_registry",
    "gpu_available",
    "infer_test_values",
    "is_ampere_or_newer",
    "is_hopper_or_newer",
    "multi_gpu_available",
    "register",
    "register_all",
    "registry",
    "supports_fp8",
    "tensorrt_available",
    "vllm_available",
    "vllm_version_at_least",
]
