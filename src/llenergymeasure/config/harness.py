"""Llem-side harness orchestration knobs.

This module holds the **small residual** of llem-implemented features that
do NOT have an engine-native API. Engine-knowledge fields live in the
generated ``llenergymeasure.engines.<e>.Config`` classes; this file
carries only the genuinely llem-side leftovers.

Reviewer rule: a new field belongs here only if direct introspection of
the target engine confirms no native API exists. If the engine exposes
the field, add it via ``engine_versions/<e>/v<safe>/outputs/overlay.yaml``
and have the plugin delegate to the engine's API. See
``.product/designs/engine-knowledge-as-data.md`` section on HarnessConfig.

Current residual (2026-05-24):

- **transformers**: ``batch_size`` (prompt chunking - HF.generate has no
  batch_size kwarg), ``allow_tf32`` (PyTorch backend global), ``autocast_*``
  (PyTorch context manager).
- **vllm**: empty. vllm has native ``max_num_seqs`` and runs its own
  scheduler.
- **tensorrt**: empty. TRT builds its engine offline; its runtime has
  native ``max_batch_size``.

All sub-classes use ``extra='allow'`` for forward-compat with new llem
features that maintainers may add before this design ratifies the
field-by-field migration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TransformersHarness(BaseModel):
    """Llem-orchestration residual for the transformers engine.

    Each field corresponds to a llem-side action with no equivalent HF
    transformers API (confirmed via direct introspection of HF
    ``PreTrainedModel.from_pretrained`` + ``GenerationConfig`` +
    ``CompileConfig``). Walker validation set entries in
    ``_spike/findings/walker_validation_set.md`` record the introspection
    evidence.
    """

    model_config = {"extra": "allow"}

    batch_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Prompt-chunking batch size for the llem runner loop. "
            "HF.generate() accepts a tensor batch but has no batch_size "
            "kwarg; llem chunks the prompt list in its own measurement "
            "loop. None -> 1."
        ),
    )
    allow_tf32: bool | None = Field(
        default=None,
        description=(
            "PyTorch backend global "
            "(torch.backends.cuda.matmul.allow_tf32). Not exposed by HF; "
            "llem sets it once before the measurement window. None -> "
            "PyTorch's own default."
        ),
    )
    autocast_enabled: bool | None = Field(
        default=None,
        description=(
            "Wrap model.generate() in torch.autocast(...). Not an HF "
            "kwarg; llem applies the context manager around inference. "
            "None -> False."
        ),
    )
    autocast_dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description=(
            "torch.autocast dtype. Only meaningful when autocast_enabled "
            "is True. None -> bfloat16 on Ampere+."
        ),
    )


class VLLMHarness(BaseModel):
    """Llem-orchestration residual for the vllm engine.

    Empty at the moment of writing. vllm exposes native batching via
    ``max_num_seqs`` (engine_params) and runs its own scheduler, so there
    is no prompt-chunking concept on the llem side. ``allow_tf32`` is
    handled internally by vllm and not surfaced. ``autocast_*`` does not
    apply because vllm has its own runtime separate from torch eager mode.
    """

    model_config = {"extra": "allow"}


class TensorRTHarness(BaseModel):
    """Llem-orchestration residual for the tensorrt engine.

    Empty at the moment of writing. TRT compiles its engine offline; its
    runtime has native ``max_batch_size``. PyTorch backend toggles don't
    apply because TRT-LLM bypasses the PyTorch runtime path.
    """

    model_config = {"extra": "allow"}


class HarnessConfig(BaseModel):
    """Per-engine llem-orchestration knobs.

    Uniform shape across engines (sibling sub-classes for each); each
    sub-class carries only the residual that genuinely needs llem-side
    handling. Reviewer enforces "engine API wins where it exists".
    """

    model_config = {"extra": "forbid"}

    transformers: TransformersHarness | None = None
    vllm: VLLMHarness | None = None
    tensorrt: TensorRTHarness | None = None
