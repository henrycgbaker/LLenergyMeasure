"""Inference engines for llenergymeasure."""

import importlib

from llenergymeasure.config.ssot import ENGINES, Engine
from llenergymeasure.engines.protocol import EnginePlugin
from llenergymeasure.utils.exceptions import EngineError

__all__ = ["EnginePlugin", "get_engine"]


def get_engine(name: str) -> EnginePlugin:
    """Get an inference engine instance by name.

    Dispatches through the ``ssot.ENGINES`` descriptor registry: the engine's
    plugin module is imported lazily (only the selected engine's deps load) and
    its ``EnginePlugin`` class instantiated.

    Args:
        name: Engine name ('transformers', 'vllm', 'tensorrt').

    Returns:
        An EnginePlugin instance.

    Raises:
        EngineError: If the engine name is unknown.
    """
    try:
        engine = Engine(name)
    except ValueError:
        raise EngineError(f"Unknown engine: {name!r}. Available: {', '.join(Engine)}") from None
    descriptor = ENGINES[engine]
    module = importlib.import_module(descriptor.plugin_module)
    plugin_cls: type[EnginePlugin] = getattr(module, descriptor.plugin_class)
    return plugin_cls()
