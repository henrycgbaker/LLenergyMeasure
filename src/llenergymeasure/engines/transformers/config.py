# DO NOT EDIT - regenerated from engine_versions/transformers/v5_12_0/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    dtype: Any | None = None
    attn_implementation: Any | None = None
    load_in_4bit: Any | None = None
    load_in_8bit: Any | None = None
    bnb_4bit_compute_dtype: Any | None = None
    bnb_4bit_quant_type: Any | None = None
    bnb_4bit_use_double_quant: Any | None = None
    use_cache: bool | None = None
    cache_implementation: str | None = None
    num_beams: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    early_stopping: bool | None = None
    length_penalty: float | None = None
    no_repeat_ngram_size: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    prompt_lookup_num_tokens: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    device_map: Any | None = None
    max_memory: Any | None = None
    low_cpu_mem_usage: Any | None = None
    tp_plan: Any | None = None
    tp_size: Any | None = None


class SamplingParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    temperature: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=2.0,
        ),
    ] = None
    do_sample: bool | None = None
    top_k: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1.0,
        ),
    ] = None
    repetition_penalty: Annotated[
        float | None,
        Field(
            ge=0.1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=10.0,
        ),
    ] = None
    min_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1.0,
        ),
    ] = None
    min_new_tokens: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
