"""Transformers inference engine.

Public surface:

- ``Config`` / ``EngineParams`` / ``SamplingParams`` - generated Pydantic
  models from per-version mined schema (Phase 2-T pilot). DO NOT EDIT
  ``config.py`` directly; edit ``engine_versions/transformers/v<safe>/outputs/
  {curated.yaml,schema.discovered.json}`` and run
  ``regen_engine_configs.py --write``.
- ``TransformersEngine`` - the EnginePlugin runtime adapter (unchanged).
"""

from llenergymeasure.engines.transformers.config import (
    Config,
    EngineParams,
    SamplingParams,
)
from llenergymeasure.engines.transformers.plugin import TransformersEngine

__all__ = [
    "Config",
    "EngineParams",
    "SamplingParams",
    "TransformersEngine",
]
