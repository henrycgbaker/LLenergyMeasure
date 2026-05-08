"""vLLM per-version machinery archive.

Subpackages here are version-pinned snapshots, one per shipped pipeline
state. The active version is the one named in ``engine_versions/vllm.yaml``
``library.current_version``; the dispatcher maps that to the matching
subpackage via :func:`scripts.engine_miners._ssot.safe_version`.
"""
