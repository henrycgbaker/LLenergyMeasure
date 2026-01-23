"""Unified quantization configuration for cross-backend compatibility.

This module provides a method-based quantization configuration that backends
map to their specific implementations. This enables experiments to specify
quantization intent (e.g., "int8") without knowing backend-specific details.

Backend Mapping:
    | Intent | PyTorch          | vLLM          | TensorRT           |
    |--------|------------------|---------------|--------------------|
    | none   | FP16/BF16        | FP16/BF16     | FP16/BF16          |
    | int8   | BitsAndBytes 8b  | AWQ/SqueezeLLM| TRT INT8 (calib)   |
    | int4   | BitsAndBytes 4b  | GPTQ/AWQ      | TRT INT4 (calib)   |
    | fp8    | Not supported    | FP8 (Hopper+) | TRT FP8            |
    | auto   | Detect from model| Detect        | Detect from model  |
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CalibrationConfig(BaseModel):
    """Calibration configuration for post-training quantization.

    Required for TensorRT INT8/INT4 quantization and optional for other
    backends that support calibration-based quantization.
    """

    dataset: str | None = Field(
        default=None,
        description="HuggingFace dataset name or local path for calibration data",
    )
    split: str = Field(
        default="train",
        description="Dataset split to use for calibration",
    )
    num_samples: int = Field(
        default=512,
        ge=1,
        le=10000,
        description="Number of samples for calibration (512-1024 typical)",
    )
    max_length: int = Field(
        default=2048,
        ge=128,
        description="Maximum sequence length for calibration samples",
    )


class UnifiedQuantizationConfig(BaseModel):
    """Unified quantization configuration with method-based intent.

    Specifies quantization intent that backends map to their implementations.
    This enables switching backends without changing the core quantization request.

    Methods:
        none: No quantization, use native precision (FP16/BF16/FP32)
        int8: 8-bit integer quantization (weight-only or weight+activation)
        int4: 4-bit integer quantization (weight-only)
        fp8: 8-bit floating point (Hopper+ GPUs)
        auto: Auto-detect from model checkpoint (for pre-quantized models)

    Examples:
        # No quantization (default)
        quantization:
          method: none

        # 8-bit quantization (backend chooses implementation)
        quantization:
          method: int8

        # TensorRT INT8 with calibration
        quantization:
          method: int8
          weight_only: false
          calibration:
            dataset: wikitext
            num_samples: 512
    """

    method: Literal["none", "int8", "int4", "fp8", "auto"] = Field(
        default="none",
        description="Quantization method/intent (backend maps to implementation)",
    )
    weight_only: bool = Field(
        default=True,
        description="Weight-only quantization (vs weight+activation). "
        "Weight-only is more common and has less accuracy loss.",
    )

    # Calibration for PTQ (post-training quantization)
    calibration: CalibrationConfig | None = Field(
        default=None,
        description="Calibration config for PTQ (required for TRT INT8/INT4)",
    )

    # Backend-specific overrides (escape hatch)
    backend_method: str | None = Field(
        default=None,
        description="Override: explicit backend method (e.g., 'gptq', 'awq', 'bitsandbytes'). "
        "Takes precedence over 'method' for the selected backend.",
    )

    @property
    def enabled(self) -> bool:
        """True if any quantization is enabled."""
        return self.method != "none"

    @property
    def bits(self) -> int | None:
        """Number of bits for quantization, or None if not applicable."""
        if self.method == "int8" or self.method == "fp8":
            return 8
        elif self.method == "int4":
            return 4
        return None

    @model_validator(mode="after")
    def validate_calibration_requirements(self) -> UnifiedQuantizationConfig:
        """Warn if calibration might be needed but not provided."""
        # INT8/INT4 non-weight-only typically requires calibration
        # But we don't error here - backend will handle the warning
        return self
