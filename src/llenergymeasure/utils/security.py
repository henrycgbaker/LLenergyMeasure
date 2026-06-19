"""Security utilities for llenergymeasure."""

from typing import Final

from llenergymeasure.utils.env_config import parse_bool_env

ENV_TRUST_REMOTE_CODE: Final = "LLEM_TRUST_REMOTE_CODE"
"""Opt-in for HuggingFace `trust_remote_code=True`. Unset means HF default (False)."""


def trust_remote_code_enabled() -> bool:
    """Return whether HuggingFace `trust_remote_code` should be enabled.

    Reads ``LLEM_TRUST_REMOTE_CODE`` from the environment. Treats
    ``1``/``true``/``yes``/``on`` (case-insensitive) as True; anything else
    (including unset) as False - matching HuggingFace's own default.

    Setting True allows loading models that ship custom Python implementations
    (Qwen, DeepSeek, ChatGLM, etc.) at the cost of executing repo-supplied code.
    """
    return parse_bool_env(ENV_TRUST_REMOTE_CODE)
