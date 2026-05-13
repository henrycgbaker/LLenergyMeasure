"""Version-dispatching shim for the TensorRT-LLM static invariant miner.

Forwards module-level attribute access to the per-version archive at
``engine_versions/tensorrt/v<safe>/producers/static_invariant_miner.py``,
resolved from ``library.current_version`` in ``engine_versions/tensorrt/current.yaml``.
See ``_stub_factory`` for the shared factory.
"""

from __future__ import annotations

from scripts.engine_producers._stub_factory import make_static_stub

_resolve_producer, __getattr__, main = make_static_stub("tensorrt")


if __name__ == "__main__":
    raise SystemExit(main())
