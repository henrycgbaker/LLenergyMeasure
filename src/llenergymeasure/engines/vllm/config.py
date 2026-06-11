# DO NOT EDIT - regenerated from engine_versions/vllm/v0_7_3/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    dtype: str | None = "auto"
    gpu_memory_utilization: float | None = 0.9
    swap_space: float | None = 4
    cpu_offload_gb: float | None = 0
    block_size: int | None = None
    kv_cache_dtype: str | None = "auto"
    enforce_eager: bool | None = None
    enable_chunked_prefill: bool | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    num_scheduler_steps: int | None = 1
    tensor_parallel_size: int | None = 1
    pipeline_parallel_size: int | None = 1
    distributed_executor_backend: Any | None = None
    enable_prefix_caching: bool | None = None
    quantization: str | None = None
    max_seq_len_to_capture: int | None = 8192
    speculative_config: Any | None = None
    offload_group_size: Any | None = None
    offload_num_in_group: Any | None = None
    offload_prefetch_step: Any | None = None
    offload_params: Any | None = None
    disable_custom_all_reduce: bool | None = False
    kv_cache_memory_bytes: Any | None = None
    compilation_config: Any | None = None
    attention: Any | None = None
    beam_search: Any | None = None


class SamplingParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    temperature: Any | None = 1.0
    top_k: Any | None = -1
    top_p: Any | None = 1.0
    repetition_penalty: Any | None = 1.0
    min_p: Any | None = 0.0
    min_tokens: Any | None = 0
    presence_penalty: Any | None = 0.0
    frequency_penalty: Any | None = 0.0
    ignore_eos: Any | None = False
    n: Any | None = 1


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
