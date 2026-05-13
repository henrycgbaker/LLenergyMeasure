"""Per-engine, per-version archive of probe / mine / introspect machinery.

CI-time tooling only. Holds frozen LANDMARKS tuples (and, for later additions,
frozen AST_TARGETS / walker patterns / introspector targets) keyed by
``(engine, version)``. Producer modules in ``scripts/engine_producers/``
resolve their version-specific data via :func:`._dispatcher.load_producer`.

Not shipped in the wheel: this directory lives at the repo root (outside
``src/``), so hatchling never includes it. Importable from scripts and tests
via ``PYTHONPATH=/repo`` (the CI PYTHONPATH already includes the repo root).
"""
