# DO NOT EDIT - regenerated from engine_versions/vllm/v0_19_1/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CompilationConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    backend: str | None = None
    cache_dir: str | None = None
    compilation_time: float | None = None
    compile_cache_save_format: Literal["binary", "unpacked"] | None = None
    compile_mm_encoder: bool | None = None
    compile_ranges_endpoints: Any | None = None
    compile_sizes: Any | None = None
    cudagraph_capture_sizes: Any | None = None
    cudagraph_copy_inputs: bool | None = None
    cudagraph_mm_encoder: bool | None = None
    cudagraph_mode: Any | None = None
    cudagraph_num_of_warmups: int | None = None
    cudagraph_specialize_lora: bool | None = None
    custom_ops: Any | None = None
    debug_dump_path: Any | None = None
    disabled_custom_ops: Any | None = None
    dynamic_shapes_config: Any | None = None
    enabled_custom_ops: Any | None = None
    encoder_cudagraph_max_images_per_batch: int | None = None
    encoder_cudagraph_token_budgets: Any | None = None
    fast_moe_cold_start: Any | None = None
    inductor_compile_config: Any | None = None
    inductor_passes: Any | None = None
    local_cache_dir: str | None = None
    max_cudagraph_capture_size: int | None = None
    mode: Literal[0, 1, 2, 3] | None = None
    pass_config: Any | None = None
    splitting_ops: Any | None = None
    static_all_moe_layers: Any | None = None
    static_forward_context: Any | None = None
    traced_files: Any | None = None
    use_inductor_graph_partition: bool | None = None


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
    ] = 1.0
    top_k: int | None = 0
    top_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1.0,
        ),
    ] = 1.0
    repetition_penalty: Annotated[
        float | None,
        Field(
            ge=0.1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=10.0,
        ),
    ] = 1.0
    min_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=1.0,
        ),
    ] = 0.0
    min_tokens: Annotated[
        int | None,
        Field(
            ge=0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 0
    presence_penalty: Annotated[
        float | None,
        Field(
            ge=-2.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=2.0,
        ),
    ] = 0.0
    frequency_penalty: Annotated[
        float | None,
        Field(
            ge=-2.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            le=2.0,
        ),
    ] = 0.0
    ignore_eos: bool | None = False
    n: Annotated[
        int | None,
        Field(
            ge=1,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
        ),
    ] = 1


class SpeculativeConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    code_revision: str | None = None
    disable_padded_drafter_batch: bool | None = None
    draft_load_config: Any | None = None
    draft_model_config: Any | None = None
    draft_parallel_config: Any | None = None
    draft_tensor_parallel_size: int | None = None
    enforce_eager: bool | None = None
    max_model_len: int | None = None
    method: (
        Literal[
            "ngram",
            "medusa",
            "mlp_speculator",
            "draft_model",
            "suffix",
            "eagle",
            "eagle3",
            "extract_hidden_states",
            "deepseek_mtp",
            "mimo_mtp",
            "glm4_moe_mtp",
            "glm4_moe_lite_mtp",
            "glm_ocr_mtp",
            "ernie_mtp",
            "nemotron_h_mtp",
            "exaone_moe_mtp",
            "qwen3_next_mtp",
            "qwen3_5_mtp",
            "longcat_flash_mtp",
            "mtp",
            "pangu_ultra_moe_mtp",
            "step3p5_mtp",
            "ngram_gpu",
        ]
        | None
    ) = None
    model: str | None = None
    moe_backend: (
        Literal[
            "auto",
            "triton",
            "deep_gemm",
            "cutlass",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "flashinfer_cutedsl",
            "marlin",
            "aiter",
        ]
        | None
    ) = None
    num_speculative_tokens: int | None = None
    parallel_drafting: bool | None = None
    prompt_lookup_max: int | None = None
    prompt_lookup_min: int | None = None
    quantization: Any | None = None
    rejection_sample_method: Literal["strict", "probabilistic", "synthetic"] | None = None
    revision: str | None = None
    speculative_token_tree: str | None = None
    suffix_decoding_max_cached_requests: int | None = None
    suffix_decoding_max_spec_factor: float | None = None
    suffix_decoding_max_tree_depth: int | None = None
    suffix_decoding_min_token_prob: float | None = None
    synthetic_acceptance_rate: float | None = None
    target_model_config: Any | None = None
    target_parallel_config: Any | None = None
    tensor_parallel_size: int | None = None
    use_local_argmax_reduction: bool | None = None


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] | None = "auto"
    gpu_memory_utilization: Annotated[
        float | None,
        Field(
            gt=0.0,
            json_schema_extra={
                "x-narrowing-applied": "hand-enforced constraint carried from pre-v0.10 engine_configs.py; retires when declarative mining surfaces it at a newer pin"
            },
            lt=1.0,
        ),
    ] = 0.9
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
    speculative_config: SpeculativeConfig | None = None
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
    compilation_config: CompilationConfig | None = None
    attention: Any | None = None
    beam_search: Any | None = None


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
