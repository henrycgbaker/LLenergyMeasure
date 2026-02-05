"""Centralised validation types and utilities.

This module provides the Single Source of Truth (SSOT) for configuration
warnings and validation-related types used across the codebase.

All other modules should import ConfigWarning from here rather than
defining their own versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


@dataclass
class ConfigWarning:
    """Warning about configuration validity or compatibility.

    Used for:
    - Config validation during loading
    - Backend compatibility checks
    - Parameter semantic differences

    Severity levels:
    - error: Invalid config (blocks execution without --force)
    - warning: Problematic config that may cause unexpected behaviour
    - info: Suboptimal but valid configuration, or semantic differences

    Attributes:
        field: Parameter/field name that triggered the warning.
        message: Human-readable description of the issue.
        severity: One of 'error', 'warning', 'info'.
        suggestion: Optional suggestion for fixing the issue.
        migration_hint: Hint for deprecated/changed params.
    """

    field: str
    message: str
    severity: Literal["error", "warning", "info"] = "warning"
    suggestion: str | None = None
    migration_hint: str | None = None

    # Alias for backwards compatibility with protocols.py usage
    @property
    def param(self) -> str:
        """Alias for field (backwards compatibility)."""
        return self.field

    def __str__(self) -> str:
        """Format for logging/display."""
        return f"[{self.severity.upper()}] {self.field}: {self.message}"

    def to_result_string(self) -> str:
        """Format for embedding in results."""
        return f"{self.severity}: {self.field} - {self.message}"


# Type alias for validation function return type
ValidationWarnings = list[ConfigWarning]


def validate_parallelism_constraints(
    config: ExperimentConfig,
) -> ValidationWarnings:
    """Validate parallelism settings against available GPUs.

    Checks that tensor_parallel_size (vLLM) or tp_size (TensorRT) does not
    exceed the number of GPUs specified in config.gpus. This prevents cryptic
    backend initialisation errors.

    Args:
        config: The experiment configuration.

    Returns:
        List of ConfigWarning objects (severity="error" for violations).
    """
    warnings: ValidationWarnings = []
    gpu_count = len(config.gpus) if config.gpus else 1

    # vLLM tensor parallelism
    if config.backend == "vllm" and config.vllm:
        tp = config.vllm.tensor_parallel_size
        if tp > gpu_count:
            warnings.append(
                ConfigWarning(
                    field="vllm.tensor_parallel_size",
                    message=(
                        f"tensor_parallel_size={tp} exceeds available GPUs ({gpu_count}). "
                        f"vLLM requires tensor_parallel_size <= len(gpus)."
                    ),
                    severity="error",
                    suggestion=(
                        f"Either set gpus: [0, 1, ..., {tp - 1}] to provide {tp} GPUs, "
                        f"or reduce tensor_parallel_size to {gpu_count} or less."
                    ),
                )
            )
        # Also check pipeline parallelism
        pp = config.vllm.pipeline_parallel_size
        total_parallel = tp * pp
        if total_parallel > gpu_count:
            warnings.append(
                ConfigWarning(
                    field="vllm.pipeline_parallel_size",
                    message=(
                        f"tensor_parallel_size * pipeline_parallel_size = {total_parallel} "
                        f"exceeds available GPUs ({gpu_count})."
                    ),
                    severity="error",
                    suggestion=(
                        "Total parallelism (TP * PP) must not exceed GPU count. "
                        "Reduce parallelism or add more GPUs."
                    ),
                )
            )

    # TensorRT tensor parallelism
    if config.backend == "tensorrt" and config.tensorrt:
        tp = config.tensorrt.tp_size
        if tp > gpu_count:
            warnings.append(
                ConfigWarning(
                    field="tensorrt.tp_size",
                    message=(
                        f"tp_size={tp} exceeds available GPUs ({gpu_count}). "
                        f"TensorRT-LLM requires tp_size <= len(gpus)."
                    ),
                    severity="error",
                    suggestion=(
                        f"Either set gpus: [0, 1, ..., {tp - 1}] to provide {tp} GPUs, "
                        f"or reduce tp_size to {gpu_count} or less."
                    ),
                )
            )
        # Also check pipeline parallelism
        pp = config.tensorrt.pp_size
        total_parallel = tp * pp
        if total_parallel > gpu_count:
            warnings.append(
                ConfigWarning(
                    field="tensorrt.pp_size",
                    message=(
                        f"tp_size * pp_size = {total_parallel} exceeds available GPUs ({gpu_count})."
                    ),
                    severity="error",
                    suggestion=(
                        "Total parallelism (TP * PP) must not exceed GPU count. "
                        "Reduce parallelism or add more GPUs."
                    ),
                )
            )

    # PyTorch data parallelism
    if config.backend == "pytorch" and config.pytorch:
        num_procs = config.pytorch.num_processes
        if num_procs > gpu_count:
            warnings.append(
                ConfigWarning(
                    field="pytorch.num_processes",
                    message=(
                        f"num_processes={num_procs} exceeds available GPUs ({gpu_count}). "
                        f"Each process requires its own GPU for data parallelism."
                    ),
                    severity="error",
                    suggestion=(
                        f"Either set gpus: [0, 1, ..., {num_procs - 1}] to provide {num_procs} GPUs, "
                        f"or reduce num_processes to {gpu_count} or less."
                    ),
                )
            )

    return warnings
