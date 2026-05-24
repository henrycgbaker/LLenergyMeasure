# DO NOT EDIT - regenerated from engine_versions/transformers/v4_57_3/outputs/{curated.yaml,schema.discovered.json}
# Edit those upstream and run `uv run python scripts/engine_producers/regen_engine_configs.py --write`.

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class SamplingParams(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_attribute_docstrings=True,
    )
    cache_implementation: Any | None = None
    """
    discovery emitted 'unknown' (untyped upstream)
    """
    do_sample: bool | None = False
    early_stopping: bool | None = False
    length_penalty: float | None = 1.0
    min_new_tokens: Any | None = None
    """
    discovery emitted 'unknown' (untyped upstream)
    """
    min_p: Any | None = None
    """
    discovery emitted 'unknown' (untyped upstream)
    """
    no_repeat_ngram_size: int | None = 0
    num_beams: int | None = 1
    prompt_lookup_num_tokens: Any | None = None
    """
    discovery emitted 'unknown' (untyped upstream)
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
    engine_params: Annotated[dict[str, Any] | None, Field(title='EngineParams')] = None
    sampling_params: Annotated[SamplingParams | None, Field(title='SamplingParams')] = (
        None
    )
