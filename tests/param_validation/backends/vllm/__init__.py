"""vLLM backend parameter specifications."""

from .param_specs import VLLM_PARAM_SPECS, register_vllm_params

__all__ = ["VLLM_PARAM_SPECS", "register_vllm_params"]
