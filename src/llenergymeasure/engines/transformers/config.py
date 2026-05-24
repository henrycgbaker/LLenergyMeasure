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
        str | None,
        Field(
            json_schema_extra={
                'x-source': 'kwargs_docstring',
                'x-source-ref': 'transformers.PreTrainedModel.from_pretrained.__doc__',
            }
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
    repetition_penalty: float | None = 1.0
    temperature: float | None = 1.0
    top_k: int | None = 50
    top_p: float | None = 1.0
    use_cache: bool | None = True


class Config(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
