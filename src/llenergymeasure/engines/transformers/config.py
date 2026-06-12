# DO NOT EDIT - regenerated from engine_versions/transformers/v5_7_0/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Any

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
    use_cache: Any | None = None
    cache_implementation: Any | None = None
    num_beams: Any | None = None
    early_stopping: Any | None = None
    length_penalty: Any | None = None
    no_repeat_ngram_size: Any | None = None
    prompt_lookup_num_tokens: Any | None = None
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
    temperature: Any | None = None
    do_sample: Any | None = None
    top_k: Any | None = None
    top_p: Any | None = None
    repetition_penalty: Any | None = None
    min_p: Any | None = None
    min_new_tokens: Any | None = None


class Config(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
    )
    engine_params: EngineParams | None = None
    sampling_params: SamplingParams | None = None
