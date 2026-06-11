"""Trimmed vllm-0.19-style ``config/cache.py`` excerpt for walker fixtures.

Reproduces the shapes the declarative-constraint walker and class-surface
enumeration must surface: pydantic ``Field(ge=/le=/...)`` bounds, a ``Literal``
field, a nested pydantic sub-config, and the per-concern ``config/*.py``
subpackage layout introduced in vllm 0.16. NOT importable vllm source - a
hand-trimmed structural excerpt only. No engine install required.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class PrefixCacheConfig:
    """Nested sub-config referenced by CacheConfig (exercises $defs surface)."""

    hash_algo: Literal["builtin", "sha256"] = "builtin"
    max_capacity: int = Field(default=1024, ge=1)


@dataclass
class CacheConfig:
    block_size: int = Field(default=16, ge=1, le=256, multiple_of=8)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    swap_space: int = Field(default=4, ge=0)
    # Optional-wrapped Literal must still surface its membership set.
    mamba_ssm_cache_dtype: Literal["auto", "float32"] | None = None
    prefix_cache: PrefixCacheConfig = Field(default_factory=PrefixCacheConfig)
    _private_internal: int = 0  # underscore field: must be ignored
