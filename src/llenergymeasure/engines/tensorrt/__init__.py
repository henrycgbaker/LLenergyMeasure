"""TensorRT-LLM inference engine.

Public surface:

- ``Config`` / ``EngineParams`` / ``SamplingParams`` - generated Pydantic
  models from per-version mined schema (Phase 2-T pilot). DO NOT EDIT
  ``config.py`` directly; edit ``engine_versions/tensorrt/v<safe>/outputs/
  {curated.yaml,schema.discovered.json}`` and run
  ``regen_engine_configs.py --write``.
- ``TensorRTEngine`` - the EnginePlugin runtime adapter (unchanged).
"""

from llenergymeasure.engines.tensorrt.config import (
    Config,
    EngineParams,
    SamplingParams,
)
from llenergymeasure.engines.tensorrt.plugin import TensorRTEngine

__all__ = [
    "Config",
    "EngineParams",
    "SamplingParams",
    "TensorRTEngine",
]
