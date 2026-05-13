"""TensorRT-LLM per-version machinery archive.

Subpackages here are version-pinned snapshots. The static miner walks an
extracted source tarball at ``/tmp/trt-llm-<V>/``; the schemas
introspector imports the live ``tensorrt_llm`` package inside the NGC
image. LANDMARKS are co-located here per version.
"""
