"""Per-engine, per-version archive of probe / mine / introspect machinery.

Importable, ships in the wheel. Holds frozen LANDMARKS tuples (and, in
later PRs, frozen AST_TARGETS / walker patterns / introspector targets)
keyed by ``(engine, version)``. Producer modules in
``scripts/engine_miners/`` and ``scripts/engine_introspectors/`` resolve
their version-specific data via :func:`._dispatcher.load_machinery`.

Layer rules: this package exists for CI-time tooling reproducibility.
Runtime engine plugins (under ``llenergymeasure.engines``) MUST NOT
import from here; they read their canonical invariants/schemas from
``src/llenergymeasure/engines/<engine>/`` as today. Enforced by an
import-linter ``forbidden`` contract in ``pyproject.toml``.
"""
