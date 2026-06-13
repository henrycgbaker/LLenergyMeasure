"""Frozen vLLM 0.19.1 machinery + outputs.

Vendored snapshot across the config-subpackage refactor cliff: 0.7.3
shipped a flat single-file ``vllm.config`` module; 0.19.1 splits it into
the ``vllm/config/*.py`` per-concern subpackage (cache, scheduler,
parallel, model, ...) and migrates the imperative ``_verify_*`` validators
to declarative pydantic-dataclass ``Field(ge/le/...)`` bounds and
``Literal[...]`` membership sets. The static-miner LANDMARKS and the
schema introspector's source-constraint overlay are retargeted at that
subpackage; the walker / detector / kwargs-synth machinery is unchanged
from the 0.7.3 cut (it is engine-version-agnostic).
"""
