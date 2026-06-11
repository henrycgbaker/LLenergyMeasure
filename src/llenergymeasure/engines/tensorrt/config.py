# DO NOT EDIT - regenerated from engine_versions/tensorrt/v0_21_0/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class EngineParams(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    max_batch_size: Any | None = None
    """
    The maximum batch size.
    """
    tensor_parallel_size: Any | None = 1
    """
    The tensor parallel size.
    """
    pipeline_parallel_size: Any | None = 1
    """
    The pipeline parallel size.
    """
    max_input_len: Any | None = None
    """
    The maximum input length.
    """
    max_seq_len: Any | None = None
    """
    The maximum sequence length.
    """
    max_num_tokens: Any | None = None
    """
    The maximum number of tokens.
    """
    dtype: Any | None = "auto"
    """
    The data type to use for the model.
    """
    fast_build: Any | None = False
    """
    Enable fast build.
    """
    backend: Any | None = None
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
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    min_tokens: int | None = None
    n: int | None = 1
    ignore_eos: bool | None = False


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
