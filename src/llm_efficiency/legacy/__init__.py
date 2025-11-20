"""
Backward compatibility layer for v1.0 code.

This module provides compatibility wrappers so v1.0 scripts can continue
to work with minimal modifications while using the new v2.0 infrastructure.

Example:
    # Old v1.0 code
    from experiment_core_utils.b_model_loader import load_model_tokenizer

    # Can now import from legacy
    from llm_efficiency.legacy import load_model_tokenizer
"""

import warnings

# Re-export v2.0 implementations with v1.0 compatible interfaces
from llm_efficiency.core.model_loader import load_model_tokenizer
from llm_efficiency.config import ExperimentConfig

__all__ = [
    "load_model_tokenizer",
    "ExperimentConfig",
]


# Show deprecation warning when this module is imported
warnings.warn(
    "llm_efficiency.legacy is for backward compatibility only. "
    "Please migrate to the new v2.0 API. See CHANGELOG.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)
