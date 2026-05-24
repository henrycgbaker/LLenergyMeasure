"""vLLM inference engine.

Public surface:

- ``Config`` / ``EngineParams`` / ``SamplingParams`` - generated Pydantic
  models from per-version mined schema (Phase 2-T pilot). DO NOT EDIT
  ``config.py`` directly; edit ``engine_versions/vllm/v<safe>/outputs/
  {curated.yaml,schema.discovered.json}`` and run
  ``regen_engine_configs.py --write``.
- ``VLLMEngine`` - the EnginePlugin runtime adapter (unchanged).
"""

from llenergymeasure.engines.vllm.config import (
    Config,
    EngineParams,
    SamplingParams,
)
from llenergymeasure.engines.vllm.plugin import VLLMEngine

__all__ = [
    "Config",
    "EngineParams",
    "SamplingParams",
    "VLLMEngine",
]
