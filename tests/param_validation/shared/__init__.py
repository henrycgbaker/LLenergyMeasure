"""Shared parameter specifications.

Parameters that apply across multiple backends (decoder, batching, etc.)
"""

from .param_specs import SHARED_PARAM_SPECS, register_shared_params

__all__ = ["SHARED_PARAM_SPECS", "register_shared_params"]
