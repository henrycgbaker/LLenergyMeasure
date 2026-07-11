# DO NOT EDIT - regenerated from engine_versions/transformers/v5_7_0/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --engine transformers --version 5.7.0 --write`.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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
    num_beams: int | None = None
    early_stopping: bool | Literal["never"] | None = None
    length_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    prompt_lookup_num_tokens: int | None = None
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
    temperature: float | None = None
    do_sample: bool | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    min_new_tokens: int | None = None


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
