"""Engine-producer machinery for the schema-discovery path.

Each ``scripts/engine_producers/{engine}_schema_introspector.py`` is a thin
version-dispatching shim (see :mod:`scripts.engine_producers._stub_factory`)
that forwards to the per-version archive under
``engine_versions/{engine}/v<version>/producers/schema_introspector.py``,
resolved from ``library.current_version`` in the engine's ``current.yaml``.
The shipped typed config models are code-generated from those discovered
schemas (:mod:`scripts.engine_producers.regen_engine_configs`).

Validation constraints (not schemas) are produced separately: the analyst cold
read and the deterministic ``scripts/cross_field_extractor.py`` propose
candidates that ``scripts/absorb.py`` gates into the shipped rules corpus. The
per-version invariant miners that once lived here were retired in favour of that
standing extractor.
"""
