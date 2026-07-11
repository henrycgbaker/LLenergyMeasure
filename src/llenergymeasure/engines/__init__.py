"""Inference engines for llenergymeasure."""

from llenergymeasure.config.ssot import Engine
from llenergymeasure.engines.protocol import EnginePlugin
from llenergymeasure.utils.exceptions import EngineError

__all__ = ["EnginePlugin", "get_engine"]


def get_engine(name: str) -> EnginePlugin:
    """Get an inference engine instance by name.

    Args:
        name: Engine name ('transformers', 'vllm', 'tensorrt').

    Returns:
        An EnginePlugin instance.

    Raises:
        EngineError: If the engine name is unknown.
    """
    if name == Engine.TRANSFORMERS:
        from llenergymeasure.engines.transformers import TransformersEngine

        return TransformersEngine()
    if name == Engine.VLLM:
        from llenergymeasure.engines.vllm import VLLMEngine

        return VLLMEngine()
    if name == Engine.TENSORRT:
        from llenergymeasure.engines.tensorrt import TensorRTEngine

        return TensorRTEngine()
    raise EngineError(f"Unknown engine: {name!r}. Available: {', '.join(Engine)}")
