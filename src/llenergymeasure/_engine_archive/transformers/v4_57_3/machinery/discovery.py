"""Schema-introspector LANDMARKS for Transformers 4.57.3.

The introspector runs inside ``llenergymeasure:transformers-<tag>`` and
lifts engine parameter specs via ``inspect.signature(from_pretrained)``
and sampling parameter specs via ``GenerationConfig().to_dict()``.
"""

from __future__ import annotations

LANDMARKS: tuple[str, ...] = (
    "transformers.AutoModelForCausalLM",
    "transformers.AutoModelForCausalLM.from_pretrained",
    "transformers.PreTrainedModel",
    "transformers.PreTrainedModel.from_pretrained",
    "transformers.GenerationConfig",
    "transformers.GenerationConfig.to_dict",
)
