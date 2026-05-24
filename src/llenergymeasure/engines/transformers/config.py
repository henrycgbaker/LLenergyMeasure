# DO NOT EDIT - regenerated from engine_versions/transformers/v4_57_3/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    dtype: Annotated[
        str | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
            }
        ),
    ] = None
    attn_implementation: Annotated[
        str | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
            }
        ),
    ] = None
    device_map: Annotated[
        str | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
            }
        ),
    ] = None
    max_memory: Annotated[
        dict[str, Any] | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
            }
        ),
    ] = None
    tp_plan: Annotated[
        str | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
            }
        ),
    ] = None
    tp_size: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
                'x-narrowing-applied': 'from_pretrained docstring (L167 of modeling_utils.py) claims\n`str`; actual code path at L326/L330 uses it as int via\ndevice_mesh.size() / torch.distributed.get_world_size().\nWalker faithfully reflects the docstring; this overlay corrects\nthe type to match runtime usage.',
            },
        ),
    ] = None
    load_in_4bit: Annotated[
        bool | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.BitsAndBytesConfig.__doc__',
            }
        ),
    ] = False
    load_in_8bit: Annotated[
        bool | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.BitsAndBytesConfig.__doc__',
            }
        ),
    ] = False
    bnb_4bit_compute_dtype: Annotated[
        Any | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.BitsAndBytesConfig.__doc__',
            }
        ),
    ] = 'torch.float32'
    """
    Upstream type: torch.dtype
    """
    bnb_4bit_quant_type: Annotated[
        str | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.BitsAndBytesConfig.__doc__',
            }
        ),
    ] = 'fp4'
    bnb_4bit_use_double_quant: Annotated[
        bool | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.BitsAndBytesConfig.__doc__',
            }
        ),
    ] = False


class CompileConfig(BaseModel):
    model_config = ConfigDict(
        use_attribute_docstrings=True,
    )
    mode: str | None = 'reduce-overhead'
    backend: str | None = 'inductor'
    fullgraph: bool | None = False
    dynamic: bool | None = None
    options: dict[str, Any] | None = None


class SamplingParams(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    cache_implementation: Annotated[
        Literal[
            'static',
            'offloaded_static',
            'sliding_window',
            'hybrid',
            'hybrid_chunked',
            'offloaded_hybrid',
            'offloaded_hybrid_chunked',
            'dynamic',
            'dynamic_full',
            'offloaded',
            'quantized',
        ]
        | None,
        Field(
            json_schema_extra={
                'x-source': 'module_validation_collection',
                'x-source-ref': 'transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS',
            }
        ),
    ] = None
    """
    runtime default was None; upstream has no type annotation
    """
    do_sample: bool | None = False
    early_stopping: bool | None = False
    length_penalty: float | None = 1.0
    min_new_tokens: Any | None = None
    """
    runtime default was None; upstream has no type annotation
    """
    min_p: Any | None = None
    """
    runtime default was None; upstream has no type annotation
    """
    no_repeat_ngram_size: int | None = 0
    num_beams: int | None = 1
    prompt_lookup_num_tokens: Any | None = None
    """
    runtime default was None; upstream has no type annotation
    """
    repetition_penalty: Annotated[
        float | None,
        Field(
            gt=0.0,
            json_schema_extra={
                'x-narrowing-applied': "Multiplicative penalty; <=0 breaks logits ranking. Engine\ndoesn't validate."
            },
        ),
    ] = 1.0
    temperature: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                'x-narrowing-applied': 'Engine accepts t<0 but produces NaN in softmax. Refuse at\nconfig-time rather than let inference produce garbage.'
            },
        ),
    ] = 1.0
    top_k: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                'x-narrowing-applied': 'Cardinality; negative k has no meaning. 0 means greedy (handled\nby HF).'
            },
        ),
    ] = 50
    top_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                'x-narrowing-applied': "Probability mass; <0 or >1 has no meaning. Engine doesn't\nvalidate; we refuse at config-time."
            },
            le=1.0,
        ),
    ] = 1.0
    use_cache: bool | None = True
    compile_config: Annotated[
        CompileConfig | None,
        Field(
            json_schema_extra={
                'x-source': 'engine_overlay',
                'x-completion-applied': "GenerationConfig.compile_config is typed as a CompileConfig\ndataclass; Move 1 walker depth doesn't traverse nested\ndataclasses yet. Bridges via this overlay completion until\nthe walker is deepened (validation set entry: compile_config).",
            }
        ),
    ] = None
    """
    torch.compile config (HF GenerationConfig.compile_config).
    Set on model.generation_config; HF compiles inside generate().

    """


class Config(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
