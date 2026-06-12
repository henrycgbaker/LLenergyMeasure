# DO NOT EDIT - regenerated from engine_versions/tensorrt/v0_21_0/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    max_batch_size: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    """
    The maximum batch size.
    """
    tensor_parallel_size: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    """
    The tensor parallel size.
    """
    pipeline_parallel_size: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    """
    The pipeline parallel size.
    """
    max_input_len: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    """
    The maximum input length.
    """
    max_seq_len: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    """
    The maximum sequence length.
    """
    max_num_tokens: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    """
    The maximum number of tokens.
    """
    dtype: Annotated[
        Literal["float16", "bfloat16"] | None,
        Field(
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            }
        ),
    ] = "auto"
    """
    The data type to use for the model.
    """
    fast_build: Any | None = False
    """
    Enable fast build.
    """
    backend: Annotated[
        Literal["trt", "pytorch", "_autodeploy"] | None,
        Field(
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            }
        ),
    ] = None
    """
    The backend to use for this LLM instance.
    """
    quant_config: Any | None = None
    """
    Quantization config.
    """
    kv_cache_config: Any | None = None
    """
    KV cache config.
    """
    scheduler_config: Any | None = None
    """
    Scheduler config.
    """


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
    min_tokens: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    n: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    ignore_eos: bool | None = False


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
