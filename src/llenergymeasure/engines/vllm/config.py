# DO NOT EDIT - regenerated from engine_versions/vllm/v0_19_1/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    dtype: Any | None = "auto"
    gpu_memory_utilization: Annotated[
        float | None,
        Field(
            ge=0.0,
            gt=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1.0,
            lt=1.0,
        ),
    ] = 0.9
    swap_space: Annotated[
        Any | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    cpu_offload_gb: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 0
    block_size: Annotated[
        Literal[8, 16, 32] | None,
        Field(
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            }
        ),
    ] = None
    kv_cache_dtype: Annotated[
        Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"] | None,
        Field(
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            }
        ),
    ] = "auto"
    enforce_eager: bool | None = False
    enable_chunked_prefill: bool | None = None
    max_num_seqs: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    max_num_batched_tokens: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    max_model_len: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    num_scheduler_steps: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    tensor_parallel_size: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    pipeline_parallel_size: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    distributed_executor_backend: Annotated[
        Literal["mp", "ray"] | None,
        Field(
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            }
        ),
    ] = None
    enable_prefix_caching: bool | None = None
    quantization: Annotated[
        Literal["awq", "gptq", "fp8", "fp8_e5m2", "fp8_e4m3", "marlin", "bitsandbytes"] | None,
        Field(
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            }
        ),
    ] = None
    max_seq_len_to_capture: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    speculative_config: Any | None = None
    offload_group_size: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 0
    offload_num_in_group: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    offload_prefetch_step: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1
    offload_params: Any | None = []
    disable_custom_all_reduce: bool | None = False
    kv_cache_memory_bytes: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = None
    compilation_config: Any | None = (
        "{'mode': None, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': None, 'compile_ranges_endpoints': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': None, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, 'static_all_moe_layers': []}"
    )
    attention: Any | None = None
    beam_search: Any | None = None


class SamplingParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    temperature: Annotated[
        Any | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=2,
        ),
    ] = 1.0
    top_k: Any | None = 0
    top_p: Annotated[
        Any | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1,
        ),
    ] = 1.0
    repetition_penalty: Annotated[
        Any | None,
        Field(
            ge=0.1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=10,
        ),
    ] = 1.0
    min_p: Annotated[
        Any | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1,
        ),
    ] = 0.0
    min_tokens: Annotated[
        Any | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 0
    presence_penalty: Annotated[
        Any | None,
        Field(
            ge=-2,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=2,
        ),
    ] = 0.0
    frequency_penalty: Annotated[
        Any | None,
        Field(
            ge=-2,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=2,
        ),
    ] = 0.0
    ignore_eos: Any | None = False
    n: Annotated[
        Any | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
