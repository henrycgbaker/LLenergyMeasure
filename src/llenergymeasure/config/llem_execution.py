"""Per-engine llem-owned execution knobs (the ``llem_execution`` block).

These are the features llem *implements itself* because the engine exposes no
native API for them: prompt-batching in llem's own runner loop, PyTorch backend
globals (TF32), and torch.autocast context wrapping. They are NOT engine config,
so they are hand-written and tracked here rather than mined into the generated
``llenergymeasure.config.generated.<engine>.Config`` classes.

They are exposed to the user as a per-engine ``llem_execution:`` sub-section,
sibling of the generated ``engine_params:`` / ``sampling_params:``, inside the
engine section. :class:`TransformersSection` composes the two: it subclasses the
generated (mined, byte-stable) transformers ``Config`` and adds the typed,
strictly-validated ``llem_execution`` field. Only transformers has an execution
residual today; vllm and tensorrt drive batching and precision through native
engine APIs, so their sections carry no ``llem_execution`` block.

YAML shape::

    engine: transformers
    transformers:
      engine_params: { dtype: bfloat16 }      # native passthrough (mined)
      sampling_params: { temperature: 0.7 }    # native passthrough (mined)
      llem_execution:                          # llem-owned execution knobs
        batch_size: 4
        torch_compile: true
        torch_compile_mode: reduce-overhead
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from llenergymeasure.config.generated.transformers import Config as _GeneratedTransformersConfig


class TransformersLlemExecution(BaseModel):
    """llem-owned execution knobs for the transformers engine.

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
    def validate_torch_compile_options(self) -> TransformersLlemExecution:
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


class TransformersSection(_GeneratedTransformersConfig):
    """The transformers engine config section: mined engine surface + llem knobs.

    Subclasses the generated (mined, byte-stable, ``extra="allow"``) transformers
    ``Config`` - so ``engine_params`` / ``sampling_params`` and their native
    passthrough are inherited untouched - and adds the hand-written, strictly
    validated ``llem_execution`` block as a third sibling. This keeps the
    llem-owned execution knobs out of the generated file while still exposing
    them, typed, at the config edge.
    """

    llem_execution: TransformersLlemExecution | None = Field(
        default=None,
        description="llem-owned execution knobs (prompt-batching, torch.compile, "
        "TF32, autocast) that have no engine-native API.",
    )
