"""TensorRT-LLM backend parameter specifications."""

from .param_specs import TENSORRT_PARAM_SPECS, register_tensorrt_params

__all__ = ["TENSORRT_PARAM_SPECS", "register_tensorrt_params"]
