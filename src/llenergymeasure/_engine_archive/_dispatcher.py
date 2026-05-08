"""Version-aware loader for per-engine machinery.

Resolves the importable subpackage
``llenergymeasure._engine_archive.<engine>.<safe_version>.machinery.<producer>``
and returns it. Producer modules in ``scripts/engine_miners/`` and
``scripts/engine_introspectors/`` use this dispatcher (via PEP 562
``__getattr__`` on the module level) to delegate ``LANDMARKS`` exports
to the version-pinned archive.

Producer kinds match the SSOT's ``miner_pins`` keys:

- ``static``: validation-invariant miners (probe contract: ``invariants``)
- ``discovery``: schema introspectors (probe contract: ``schemas``)
- ``dynamic``: combinatorial miners (no probe contract today; reserved)

The version string is the dotted form from
``engine_versions/<engine>.yaml`` ``library.current_version`` (e.g.
``"0.7.3"``); the dispatcher converts it to a Python-identifier-safe form
(``v0_7_3``) via :func:`scripts.engine_miners._ssot.safe_version`.

No fallback. If the requested subpackage does not exist, the import
raises ``ModuleNotFoundError`` and the caller (or the probe) surfaces it
loud. The "no fallback" rule is load-bearing — it makes "missing
versioned machinery" a visible CI failure, not silent stale-data drift.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

ProducerKind = Literal["static", "dynamic", "discovery"]


def load_machinery(*, engine: str, version: str, producer: ProducerKind) -> ModuleType:
    """Return the ``machinery.<producer>`` submodule for ``(engine, version)``.

    Args:
        engine: Engine name (``"vllm"`` | ``"tensorrt"`` | ``"transformers"``).
        version: Dotted PEP-440 version string (e.g. ``"0.7.3"``).
        producer: One of ``"static"``, ``"dynamic"``, ``"discovery"``.

    Returns:
        The imported ``llenergymeasure._engine_archive.<engine>.<safe>.machinery.<producer>``
        module. The caller reads ``LANDMARKS`` (and optionally ``AST_TARGETS``
        etc.) from it.

    Raises:
        ModuleNotFoundError: if the versioned subpackage does not exist. The
            error message names the exact file path to create — this is the
            primary signal a maintainer sees when a Renovate-driven SSOT bump
            outpaces the per-version vendoring (the chunk PR hasn't landed yet).
    """
    # Late import: ``_ssot`` imports yaml/packaging which we'd rather not
    # pay at every ``__getattr__`` call site. Importing here keeps module-
    # load cheap.
    from scripts.engine_miners._ssot import safe_version

    safe = safe_version(version)
    qualified = f"llenergymeasure._engine_archive.{engine}.{safe}.machinery.{producer}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        # The import may fail because the engine subpackage, the version
        # subpackage, or the producer module is missing. All three cases
        # share the same remediation: vendor this version's machinery.
        raise ModuleNotFoundError(
            f"No archived machinery for {engine}=={version} ({producer}). "
            f"Expected to import `{qualified}`. Create "
            f"src/llenergymeasure/_engine_archive/{engine}/{safe}/machinery/{producer}.py "
            f"(with sibling __init__.py files as needed) by copying the prior "
            f"version's machinery and rewriting LANDMARKS for the new library "
            f"API. See the version-bump chunk pattern in the engine archive "
            f"package docstring."
        ) from exc
