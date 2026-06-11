"""Per-engine llem-orchestration knobs (the HarnessConfig residual).

These are the features llem *implements itself* because the engine exposes no
native API for them - prompt-batching in llem's own runner loop, PyTorch
backend globals, and torch.autocast context wrapping. They are NOT engine
config, so they live here (hand-written, tracked) rather than in the generated
``llenergymeasure.engines.<e>.Config`` classes.

Shape is uniform across engines (one sub-class per engine); contents differ by
the per-engine residual. For transformers the residual is the seven knobs that
used to live on the hand-written ``TransformersConfig``; vllm and tensorrt have
no residual today (their batching and precision are native engine APIs).

YAML shape mirrors the class shape::

    engine: transformers
    transformers:
      engine_params: { dtype: bfloat16 }
      sampling_params: { temperature: 0.7 }
    harness:
      transformers:
        batch_size: 4
        torch_compile: true
        torch_compile_mode: reduce-overhead
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TransformersHarness(BaseModel):
    """llem-orchestration knobs for the transformers engine.

    All fields default to None - None means "use llem's own default at
    execution". torch.compile, prompt-batching, the TF32 backend toggle, and
    autocast wrapping are llem-side orchestration: ``model.generate()`` has no
    ``batch_size`` kwarg, and the compile/TF32/autocast knobs drive PyTorch
    calls llem makes around the engine, not engine-native config.
    """

    model_config = {"extra": "forbid"}

    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Prompt-batching size for llem's runner loop (None -> 1).",
    )
    torch_compile: bool | None = Field(
        default=None,
        description="Enable torch.compile on the loaded model (None -> False).",
    )
    torch_compile_mode: str | None = Field(
        default=None,
        description="torch.compile mode: 'default', 'reduce-overhead', 'max-autotune' (None -> 'default').",
    )
    torch_compile_backend: str | None = Field(
        default=None,
        description="torch.compile backend (None -> 'inductor').",
    )
    allow_tf32: bool | None = Field(
        default=None,
        description="Allow TF32 on Ampere GPUs via torch.backends (None -> PyTorch default).",
    )
    autocast_enabled: bool | None = Field(
        default=None,
        description="Wrap generation in torch.autocast mixed precision (None -> False).",
    )
    autocast_dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description="torch.autocast dtype (None -> bfloat16 on Ampere).",
    )

    @model_validator(mode="after")
    def validate_torch_compile_options(self) -> TransformersHarness:
        """torch_compile_mode / torch_compile_backend require torch_compile=True.

        llem owns the torch.compile call, so this is llem-side orchestration
        policy (not an engine-mined rule): naming a mode/backend without
        enabling compilation is always a user mistake.
        """
        if (
            self.torch_compile_mode is not None or self.torch_compile_backend is not None
        ) and self.torch_compile is not True:
            raise ValueError("torch_compile_mode/torch_compile_backend requires torch_compile=True")
        return self


class VLLMHarness(BaseModel):
    """llem-orchestration knobs for vllm. No residual today (native batching)."""

    model_config = {"extra": "forbid"}


class TensorRTHarness(BaseModel):
    """llem-orchestration knobs for tensorrt. No residual today (native batching)."""

    model_config = {"extra": "forbid"}


class HarnessConfig(BaseModel):
    """Per-engine llem-orchestration knobs. Uniform shape, engine-specific contents."""

    model_config = {"extra": "forbid"}

    transformers: TransformersHarness | None = None
    vllm: VLLMHarness | None = None
    tensorrt: TensorRTHarness | None = None
