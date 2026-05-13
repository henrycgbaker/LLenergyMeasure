"""Version-aware loader for per-engine producers.

Resolves the importable subpackage
``engine_versions.<engine>.<safe_version>.producers.<producer>``
and returns it. Producer modules in ``scripts/engine_producers/`` use this
dispatcher (via PEP 562 ``__getattr__`` on the module level) to delegate
``LANDMARKS`` exports to the version-pinned archive.

Producer kinds match the current.yaml ``miner_pins`` keys:

- ``static_invariant_miner``: validation-invariant miners (probe contract: ``invariants``)
- ``schema_introspector``: schema introspectors (probe contract: ``schemas``)
- ``dynamic_invariant_miner``: combinatorial miners (no probe contract today; reserved)

The version string is the dotted form from
``engine_versions/<engine>/current.yaml`` ``library.current_version`` (e.g.
``"0.7.3"``); the dispatcher converts it to a Python-identifier-safe form
(``v0_7_3``) via :func:`scripts.engine_producers._current.safe_version`.

No fallback. If the requested subpackage does not exist, the import
raises ``ModuleNotFoundError`` and the caller (or the probe) surfaces it
loud. The "no fallback" rule is load-bearing - it makes "missing
versioned producer" a visible CI failure, not silent stale-data drift.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

ProducerKind = Literal["static_invariant_miner", "dynamic_invariant_miner", "schema_introspector"]


def load_producer(*, engine: str, version: str, producer: ProducerKind) -> ModuleType:
    """Return the ``producers.<producer>`` submodule for ``(engine, version)``.

    Args:
        engine: Engine name (``"vllm"`` | ``"tensorrt"`` | ``"transformers"``).
        version: Dotted PEP-440 version string (e.g. ``"0.7.3"``).
        producer: One of ``"static_invariant_miner"``, ``"dynamic_invariant_miner"``,
            ``"schema_introspector"``.

    Returns:
        The imported ``engine_versions.<engine>.<safe>.producers.<producer>``
        module. The caller reads ``LANDMARKS`` (and optionally ``_CLASS_TARGETS``
        etc.) from it.

    Raises:
        ModuleNotFoundError: if the versioned subpackage does not exist. The
            error message names the exact file path to create - this is the
            primary signal a maintainer sees when a Renovate-driven version bump
            outpaces the per-version vendoring (the chunk PR hasn't landed yet).
    """
    # Late import: ``_current`` imports yaml/packaging which we'd rather not
    # pay at every ``__getattr__`` call site. Importing here keeps module-
    # load cheap.
    from scripts.engine_producers._current import safe_version

    safe = safe_version(version)
    qualified = f"engine_versions.{engine}.{safe}.producers.{producer}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        # The import may fail because the engine subpackage, the version
        # subpackage, or the producer module is missing. All three cases
        # share the same remediation: vendor this version's producers.
        raise ModuleNotFoundError(
            f"No archived producer for {engine}=={version} ({producer}). "
            f"Expected to import `{qualified}`. Create "
            f"engine_versions/{engine}/{safe}/producers/{producer}.py "
            f"(with sibling __init__.py files as needed) by copying the prior "
            f"version's producers and rewriting LANDMARKS for the new library "
            f"API. See the version-bump chunk pattern in the engine archive "
            f"package docstring."
        ) from exc
