"""Configuration models for LLM efficiency measurement experiments (v2.0 schema).

This module defines the Tier 1 (Universal) configuration that applies identically
across all backends. Backend-specific parameters live in backend_configs.py.

v2.0 field renames from v1.x:
    model_name         -> model
    fp_precision       -> precision
    num_input_prompts  -> n
    extra_metadata     -> passthrough_kwargs

v2.0 removals:
    config_name, schema_version, TrafficSimulation, ScheduleConfig, IOConfig,
    query_rate, streaming, streaming_warmup_requests, save_outputs,
    decode_token_to_text, extra_metadata, gpus, min_output_tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )


# Sampling presets aligned with industry best practices (vLLM, OpenAI, MLPerf)
SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "deterministic": {"temperature": 0.0, "do_sample": False},
    "standard": {"temperature": 1.0, "do_sample": True, "top_p": 0.95},
    "creative": {"temperature": 0.8, "do_sample": True, "top_p": 0.9, "repetition_penalty": 1.1},
    "factual": {"temperature": 0.3, "do_sample": True},
}


# =============================================================================
# Decoder Configuration
# =============================================================================


class DecoderConfig(BaseModel):
    """Universal decoder/generation configuration.

    Contains parameters with identical semantics across all backends.
    Backend-specific decoder params live in backend_configs.py.

    Presets:
        - deterministic: Greedy decoding (temp=0, do_sample=False)
        - standard: Balanced sampling (temp=1.0, top_p=0.95)
        - creative: Higher variance (temp=0.8, top_p=0.9, repetition_penalty=1.1)
        - factual: Lower variance (temp=0.3)
    """

    model_config = {"extra": "forbid"}

    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature (0=greedy)"
    )
    do_sample: bool = Field(default=True, description="Enable sampling (ignored if temp=0)")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling (0=disabled)")
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling (1.0=disabled)"
    )
    repetition_penalty: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Repetition penalty (1.0=no penalty)"
    )
    preset: Literal["deterministic", "standard", "creative", "factual"] | None = Field(
        default=None,
        description="Sampling preset (expands to preset values, overrides apply on top)",
    )

    @model_validator(mode="before")
    @classmethod
    def apply_preset(cls, data: Any) -> Any:
        """Expand preset, then apply explicit overrides on top."""
        if (
            isinstance(data, dict)
            and (preset_name := data.get("preset"))
            and preset_name in SAMPLING_PRESETS
        ):
            return {**SAMPLING_PRESETS[preset_name], **data}
        return data

    @property
    def is_deterministic(self) -> bool:
        """True if using greedy decoding (temp=0 or do_sample=False)."""
        return self.temperature == 0.0 or not self.do_sample


def _validate_sampling_presets() -> None:
    """Validate SAMPLING_PRESETS keys match DecoderConfig fields at import time."""
    valid_fields = set(DecoderConfig.model_fields.keys())
    for preset_name, values in SAMPLING_PRESETS.items():
        invalid_keys = set(values.keys()) - valid_fields
        if invalid_keys:
            raise ValueError(
                f"SAMPLING_PRESETS['{preset_name}'] has invalid keys: {invalid_keys}. "
                f"Valid keys are: {valid_fields}"
            )


_validate_sampling_presets()


# =============================================================================
# Warmup Configuration
# =============================================================================


class WarmupConfig(BaseModel):
    """Warmup configuration for the measurement phase.

    Controls the warmup phase before measurement begins. Uses fixed iteration count
    plus a thermal floor wait. CV-based convergence detection is a measurement
    concern, not a config concern — moved to Phase 5.
    """

    model_config = {"extra": "forbid"}

    n_warmup: int = Field(
        default=3,
        ge=1,
        description="Number of full-length warmup prompts before measurement",
    )
    thermal_floor_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Minimum seconds to wait after warmup before measuring (thermal stabilisation)",
    )


# =============================================================================
# Baseline Configuration
# =============================================================================


class BaselineConfig(BaseModel):
    """Baseline power measurement configuration.

    Controls whether and how idle GPU power is measured before experiments,
    enabling baseline-adjusted energy attribution.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable baseline power measurement")
    duration_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Baseline measurement duration in seconds",
    )


# =============================================================================
# Synthetic Dataset Configuration
# =============================================================================


class SyntheticDatasetConfig(BaseModel):
    """Synthetic dataset configuration for controlled experiments.

    Generates fixed-length synthetic prompts for reproducible benchmarking
    when real dataset variance is not desired.
    """

    model_config = {"extra": "forbid"}

    n: int = Field(ge=1, description="Number of synthetic prompts to generate")
    input_len: int = Field(default=512, ge=1, description="Synthetic input token length")
    output_len: int = Field(default=128, ge=1, description="Synthetic output token length")
    seed: int = Field(default=42, description="Random seed for synthetic data generation")


# =============================================================================
# LoRA Configuration
# =============================================================================


class LoRAConfig(BaseModel):
    """LoRA adapter configuration.

    Exactly one of adapter_id or adapter_path must be set.
    """

    model_config = {"extra": "forbid"}

    adapter_id: str | None = Field(default=None, description="HuggingFace Hub adapter ID")
    adapter_path: str | None = Field(default=None, description="Local path to adapter weights")
    merge_weights: bool = Field(
        default=False, description="Merge adapter weights into base model at load time"
    )

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> LoRAConfig:
        """Exactly one of adapter_id or adapter_path must be set."""
        has_id = self.adapter_id is not None
        has_path = self.adapter_path is not None
        if has_id == has_path:  # both set or neither set
            raise ValueError(
                "LoRAConfig requires exactly one of adapter_id or adapter_path. "
                f"Got: adapter_id={self.adapter_id!r}, adapter_path={self.adapter_path!r}"
            )
        return self


# =============================================================================
# Main Experiment Configuration (v2.0)
# =============================================================================


class ExperimentConfig(BaseModel):
    """v2.0 experiment configuration.

    Central configuration object controlling all aspects of a single LLM inference
    efficiency measurement. Backend-specific parameters live in nested sections
    (pytorch:, vllm:, tensorrt:).

    Field renames from v1.x:
        model_name -> model
        fp_precision -> precision
        num_input_prompts -> n
        extra_metadata -> passthrough_kwargs

    The backend section (pytorch:, vllm:, tensorrt:) must match the backend field.
    Providing a pytorch: section when backend=vllm is a configuration error.
    """

    model_config = {"extra": "forbid"}

    # Required
    model: str = Field(..., min_length=1, description="HuggingFace model ID or local path")

    # Backend selection
    backend: Literal["pytorch", "vllm", "tensorrt"] = Field(
        default="pytorch", description="Inference backend"
    )

    # Data
    n: int = Field(default=100, ge=1, description="Number of prompts from dataset")
    dataset: str | SyntheticDatasetConfig = Field(
        default="aienergyscore",
        description="Dataset name (built-in alias) or synthetic dataset config",
    )

    # Hardware
    precision: Literal["fp32", "fp16", "bf16"] = Field(
        default="bf16", description="Floating point precision"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # Token limits
    max_input_tokens: int = Field(default=512, ge=1, description="Max input tokens")
    max_output_tokens: int = Field(default=128, ge=1, description="Max output tokens")

    # Sub-configs
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig, description="Universal decoder/generation configuration"
    )
    warmup: WarmupConfig = Field(
        default_factory=WarmupConfig, description="Warmup phase configuration"
    )
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig, description="Baseline power measurement configuration"
    )

    # Backend sections (None = use backend's own defaults)
    # GPU detection and cpu-precision cross-validation is Phase 4 (pre-flight).
    # All current backends (pytorch, vllm, tensorrt) are GPU-only; cpu backend is future scope.
    pytorch: PyTorchConfig | None = Field(
        default=None,
        description="PyTorch-specific configuration (only used when backend=pytorch)",
    )
    vllm: VLLMConfig | None = Field(
        default=None,
        description="vLLM-specific configuration (only used when backend=vllm)",
    )
    tensorrt: TensorRTConfig | None = Field(
        default=None,
        description="TensorRT-LLM configuration (only used when backend=tensorrt)",
    )

    # LoRA adapter (optional)
    lora: LoRAConfig | None = Field(default=None, description="LoRA adapter configuration")

    # Escape hatch — explicitly declared for extra="forbid" compatibility
    passthrough_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra kwargs passed through to backend at execution time. "
        "Keys must not collide with ExperimentConfig top-level fields.",
    )

    # Output override
    output_dir: str | None = Field(
        default=None,
        description="Per-experiment output directory override",
    )

    # -------------------------------------------------------------------------
    # Cross-validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_backend_section_match(self) -> ExperimentConfig:
        """Backend section must match the backend field.

        A pytorch: section with backend=vllm is a configuration error — it indicates
        the researcher copied the wrong config block. Fail explicitly rather than
        silently ignoring the mismatched section.
        """
        if self.pytorch is not None and self.backend != "pytorch":
            raise ValueError(
                f"pytorch: config section provided but backend={self.backend!r}. "
                "Remove the pytorch: section or set backend: pytorch."
            )
        if self.vllm is not None and self.backend != "vllm":
            raise ValueError(
                f"vllm: config section provided but backend={self.backend!r}. "
                "Remove the vllm: section or set backend: vllm."
            )
        if self.tensorrt is not None and self.backend != "tensorrt":
            raise ValueError(
                f"tensorrt: config section provided but backend={self.backend!r}. "
                "Remove the tensorrt: section or set backend: tensorrt."
            )
        return self

    @model_validator(mode="after")
    def validate_passthrough_kwargs_no_collision(self) -> ExperimentConfig:
        """passthrough_kwargs keys must not collide with ExperimentConfig fields.

        If a researcher writes passthrough_kwargs: {model: gpt2}, they intended to
        set the model field directly. Collisions are always a misconfiguration.
        """
        if self.passthrough_kwargs:
            top_level_fields = set(ExperimentConfig.model_fields.keys())
            collisions = set(self.passthrough_kwargs.keys()) & top_level_fields
            if collisions:
                raise ValueError(
                    f"passthrough_kwargs keys collide with ExperimentConfig fields: "
                    f"{sorted(collisions)}. Use the named fields instead."
                )
        return self


# Rebuild to resolve forward references for backend configs
def _rebuild_experiment_config() -> None:
    """Rebuild ExperimentConfig to resolve forward references."""
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

    ExperimentConfig.model_rebuild(
        _types_namespace={
            "VLLMConfig": VLLMConfig,
            "PyTorchConfig": PyTorchConfig,
            "TensorRTConfig": TensorRTConfig,
        }
    )


_rebuild_experiment_config()


class StudyConfig(BaseModel):
    """Thin resolved container for a study (list of experiments + execution config).

    M1 stub: only experiments + name. ExecutionConfig and sweep grammar added in M2.
    """

    model_config = {"extra": "forbid"}

    experiments: list[ExperimentConfig] = Field(
        ..., min_length=1, description="Resolved list of experiments to run"
    )
    name: str | None = Field(
        default=None, description="Study name (used in output directory naming)"
    )
