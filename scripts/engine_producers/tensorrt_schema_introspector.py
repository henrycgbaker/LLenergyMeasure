"""Version-dispatching shim for the TensorRT-LLM schema introspector.

Forwards module-level attribute access to the per-version archive at
``engine_versions/tensorrt/v<safe>/producers/schema_introspector.py``,
resolved from ``library.current_version`` in ``engine_versions/tensorrt/current.yaml``.
See ``_stub_factory`` for the shared factory.
"""

from __future__ import annotations

from scripts.engine_producers._stub_factory import make_schema_stub

_resolve_producer, __getattr__, discover = make_schema_stub("tensorrt")
