# DO NOT EDIT - regenerated from engine_versions/vllm/v0_7_3/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    gpu_memory_utilization: float | None = 0.9
    swap_space: float | None = 4
    cpu_offload_gb: float | None = 0
    block_size: int | None = None
    kv_cache_dtype: str | None = 'auto'
    enforce_eager: bool | None = None
    enable_chunked_prefill: bool | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    num_scheduler_steps: int | None = 1
    tensor_parallel_size: int | None = 1
    pipeline_parallel_size: int | None = 1
    distributed_executor_backend: str | dict[str, Any] | None = None
    enable_prefix_caching: bool | None = None
    dtype: str | None = 'auto'
    quantization: str | None = None
    speculative_model: str | None = None
    num_speculative_tokens: int | None = None


class SamplingParams(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    temperature: float | None = 1.0
    top_k: int | None = -1
    top_p: float | None = 1.0
    repetition_penalty: float | None = 1.0
    min_p: float | None = 0.0
    min_tokens: int | None = 0
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    ignore_eos: bool | None = False
    n: int | None = 1


class Config(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
