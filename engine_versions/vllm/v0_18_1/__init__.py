"""Frozen vLLM 0.18.1 machinery + outputs.

Vendored snapshot on the config-subpackage side of the refactor cliff:
0.7.3 shipped a flat single-file ``vllm.config`` module; by 0.18.1 it has
been split into the ``vllm/config/*.py`` per-concern subpackage (cache,
scheduler, parallel, model, ...) and the imperative ``_verify_*``
validators have largely migrated to declarative pydantic-dataclass
``Field(ge/le/...)`` bounds and ``Literal[...]`` membership sets;
``GuidedDecodingParams`` is renamed ``StructuredOutputsParams`` and
``DecodingConfig`` is removed. The static-miner LANDMARKS and the schema
introspector's source-constraint overlay are retargeted at that
subpackage; the walker / detector / kwargs-synth machinery is unchanged
from the 0.7.3 cut (it is engine-version-agnostic).

This is the rolling-window sweep baseline pin (the leg's N-5).
"""
