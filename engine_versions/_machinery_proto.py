"""Protocol for per-version producer modules.

Each ``engine_versions.<engine>.<safe_version>.producers.<producer>``
module is expected to expose the attributes declared here. Optional
attributes are typed ``Any`` because per-engine shapes diverge:

- vllm static AST_TARGETS use the importable-attribute ``_ASTTarget`` dataclass.
- tensorrt static targets reference filesystem paths under ``/tmp/trt-llm-<V>/``.
- transformers AST work is split between an introspection-driven dynamic
  miner and a BNB-focused static miner, so its ``AST_TARGETS`` shape is
  not directly comparable.

``LANDMARKS`` (a ``tuple[str, ...]`` of dotted attribute paths) is the
introspector's resolution contract: the producer unit tests resolve each
path under the installed library so an upstream rename surfaces at test
time. The other attributes are consumed by the per-engine miners
themselves. Currently only LANDMARKS is vendored; AST_TARGETS /
WALKER_PATTERNS / INTROSPECTOR_TARGETS are deferred to a later scaffold.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Machinery(Protocol):
    """Per-(engine, version, producer) machinery namespace.

    Contract: the module MUST expose a ``LANDMARKS`` attribute that
    is a ``tuple[str, ...]`` of dotted attribute paths the probe will
    resolve under the installed library.

    Optional attributes (advisory; declared in the Protocol as required
    would make ``isinstance(m, Machinery)`` fail when a producer module
    does not need them, so they are documented here instead of typed):

    - ``AST_TARGETS``: engine-specific static-miner walk targets. vllm uses
      a ``_ASTTarget`` dataclass tuple; tensorrt uses filesystem-rooted
      ``(class_name, file_relative_path)`` tuples; transformers' AST work
      currently lives outside the per-version archive (split between a
      BNB-focused static miner and a combinatorial dynamic miner).
    - ``WALKER_PATTERNS``: per-version pattern matchers / detector tables.
    - ``INTROSPECTOR_TARGETS``: class / dataclass targets the introspector
      walks at runtime.

    Producers that don't need AST_TARGETS / WALKER_PATTERNS /
    INTROSPECTOR_TARGETS simply omit them. Callers that depend on a
    specific attribute MUST guard with ``hasattr`` or accept
    ``AttributeError``. The probe accesses ``LANDMARKS`` only.
    """

    LANDMARKS: tuple[str, ...]
    """Dotted attribute paths the probe checks for resolution at probe time."""
