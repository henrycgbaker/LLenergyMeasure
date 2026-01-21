"""Hardware and software capability detection.

Provides functions to detect available hardware (GPU, compute capability)
and software (vLLM, TensorRT-LLM, Flash Attention) for conditional test execution.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    """Detected hardware and software capabilities."""

    # GPU
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_name: str = ""
    compute_capability: tuple[int, int] | None = None
    cuda_version: str = ""
    bf16_supported: bool = False

    # Software
    vllm_available: bool = False
    vllm_version: tuple[int, int, int] | None = None
    tensorrt_available: bool = False
    tensorrt_version: tuple[int, int, int] | None = None
    flash_attn_available: bool = False
    flash_attn_version: str = ""

    # Derived capabilities
    is_hopper: bool = False  # SM 9.0+
    is_ampere: bool = False  # SM 8.0+
    supports_fp8: bool = False

    # Detection metadata
    detection_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive capabilities from base hardware info."""
        if self.compute_capability:
            major, _ = self.compute_capability
            self.is_hopper = major >= 9
            self.is_ampere = major >= 8
            self.supports_fp8 = self.is_hopper


@functools.lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Detect all hardware and software capabilities.

    Results are cached for the lifetime of the process.

    Returns:
        HardwareProfile with all detected capabilities.
    """
    profile = HardwareProfile()

    # GPU detection
    _detect_gpu(profile)

    # Software detection
    _detect_vllm(profile)
    _detect_tensorrt(profile)
    _detect_flash_attn(profile)

    return profile


def _detect_gpu(profile: HardwareProfile) -> None:
    """Detect GPU capabilities."""
    try:
        import torch

        profile.gpu_available = torch.cuda.is_available()
        if profile.gpu_available:
            profile.gpu_count = torch.cuda.device_count()
            profile.gpu_name = torch.cuda.get_device_name(0)
            profile.compute_capability = torch.cuda.get_device_capability(0)
            profile.cuda_version = torch.version.cuda or ""

            # BF16 detection with MIG fallback
            try:
                profile.bf16_supported = torch.cuda.is_bf16_supported()
            except Exception:
                # MIG GPUs may fail bf16 check, use compute capability fallback
                if profile.compute_capability:
                    profile.bf16_supported = profile.compute_capability[0] >= 8

    except ImportError:
        profile.detection_errors.append("PyTorch not installed")
    except Exception as e:
        profile.detection_errors.append(f"GPU detection error: {e}")


def _detect_vllm(profile: HardwareProfile) -> None:
    """Detect vLLM installation and version."""
    try:
        import vllm

        profile.vllm_available = True
        version_str = vllm.__version__
        parts = version_str.split(".")[:3]
        profile.vllm_version = tuple(int(p.split("+")[0]) for p in parts)  # type: ignore
    except ImportError:
        pass
    except Exception as e:
        profile.detection_errors.append(f"vLLM version detection error: {e}")
        profile.vllm_available = True  # Installed but version parsing failed


def _detect_tensorrt(profile: HardwareProfile) -> None:
    """Detect TensorRT-LLM installation and version."""
    try:
        import tensorrt_llm

        profile.tensorrt_available = True
        if hasattr(tensorrt_llm, "__version__"):
            version_str = tensorrt_llm.__version__
            parts = version_str.split(".")[:3]
            profile.tensorrt_version = tuple(int(p) for p in parts)  # type: ignore
    except ImportError:
        pass
    except Exception as e:
        profile.detection_errors.append(f"TensorRT-LLM detection error: {e}")


def _detect_flash_attn(profile: HardwareProfile) -> None:
    """Detect Flash Attention installation."""
    try:
        import flash_attn

        profile.flash_attn_available = True
        if hasattr(flash_attn, "__version__"):
            profile.flash_attn_version = flash_attn.__version__
    except ImportError:
        pass
    except Exception as e:
        profile.detection_errors.append(f"Flash Attention detection error: {e}")


# Convenience functions for quick checks


def gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    return detect_hardware().gpu_available


def vllm_available() -> bool:
    """Check if vLLM is installed."""
    return detect_hardware().vllm_available


def tensorrt_available() -> bool:
    """Check if TensorRT-LLM is installed."""
    return detect_hardware().tensorrt_available


def flash_attn_available() -> bool:
    """Check if Flash Attention is installed."""
    return detect_hardware().flash_attn_available


def is_hopper_or_newer() -> bool:
    """Check if GPU is Hopper (SM 9.0) or newer."""
    return detect_hardware().is_hopper


def is_ampere_or_newer() -> bool:
    """Check if GPU is Ampere (SM 8.0) or newer."""
    return detect_hardware().is_ampere


def bf16_supported() -> bool:
    """Check if GPU supports bfloat16."""
    return detect_hardware().bf16_supported


def supports_fp8() -> bool:
    """Check if GPU supports FP8 (Hopper+)."""
    return detect_hardware().supports_fp8


def multi_gpu_available() -> bool:
    """Check if multiple GPUs are available."""
    return detect_hardware().gpu_count > 1


def vllm_version_at_least(major: int, minor: int = 0, patch: int = 0) -> bool:
    """Check if vLLM version is at least the specified version."""
    profile = detect_hardware()
    if not profile.vllm_available or profile.vllm_version is None:
        return False
    return profile.vllm_version >= (major, minor, patch)


def get_hardware_summary() -> str:
    """Get a human-readable summary of detected hardware."""
    profile = detect_hardware()
    lines = []

    if profile.gpu_available:
        lines.append(f"GPU: {profile.gpu_name}")
        if profile.compute_capability:
            lines.append(
                f"Compute Capability: {profile.compute_capability[0]}.{profile.compute_capability[1]}"
            )
        lines.append(f"CUDA Version: {profile.cuda_version}")
        lines.append(f"GPU Count: {profile.gpu_count}")
        lines.append(f"BF16 Supported: {profile.bf16_supported}")
        lines.append(f"FP8 Supported: {profile.supports_fp8}")
    else:
        lines.append("GPU: Not available")

    lines.append("")
    lines.append(f"vLLM: {profile.vllm_version if profile.vllm_available else 'Not installed'}")
    lines.append(
        f"TensorRT-LLM: {profile.tensorrt_version if profile.tensorrt_available else 'Not installed'}"
    )
    lines.append(
        f"Flash Attention: {profile.flash_attn_version if profile.flash_attn_available else 'Not installed'}"
    )

    if profile.detection_errors:
        lines.append("")
        lines.append("Detection errors:")
        for err in profile.detection_errors:
            lines.append(f"  - {err}")

    return "\n".join(lines)


def check_requirements(
    requirements: set,
) -> tuple[bool, list[str]]:
    """Check if hardware requirements are met.

    Args:
        requirements: Set of HardwareRequirement enum values.

    Returns:
        Tuple of (all_met, list of unmet requirement descriptions).
    """
    from .models import HardwareRequirement

    profile = detect_hardware()
    unmet = []

    checks = {
        HardwareRequirement.GPU: (profile.gpu_available, "CUDA GPU not available"),
        HardwareRequirement.HOPPER: (profile.is_hopper, "Hopper (SM 9.0+) GPU required"),
        HardwareRequirement.AMPERE: (profile.is_ampere, "Ampere (SM 8.0+) GPU required"),
        HardwareRequirement.VLLM: (profile.vllm_available, "vLLM not installed"),
        HardwareRequirement.TENSORRT: (profile.tensorrt_available, "TensorRT-LLM not installed"),
        HardwareRequirement.FLASH_ATTN: (
            profile.flash_attn_available,
            "Flash Attention not installed",
        ),
        HardwareRequirement.BF16: (profile.bf16_supported, "BF16 not supported"),
        HardwareRequirement.MULTI_GPU: (
            profile.gpu_count > 1,
            "Multiple GPUs required",
        ),
    }

    for req in requirements:
        if req in checks:
            met, msg = checks[req]
            if not met:
                unmet.append(msg)

    return len(unmet) == 0, unmet
