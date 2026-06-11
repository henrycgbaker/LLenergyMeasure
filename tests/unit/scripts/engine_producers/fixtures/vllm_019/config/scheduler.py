"""Trimmed vllm-0.19-style ``config/scheduler.py`` excerpt.

A second file in the ``config/*.py`` subpackage so the glob primitive has more
than one file to find, and a sibling config class the entry-point walk would
miss.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_num_seqs: int = Field(default=256, ge=1)
    max_num_batched_tokens: Annotated[int, Field(ge=1)] = 2048
    policy: str = "fcfs"
