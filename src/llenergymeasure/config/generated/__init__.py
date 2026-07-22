"""Generated per-engine typed config models.

Each module here (``transformers``, ``vllm``, ``tensorrt``) holds the typed
Pydantic ``Config`` / ``EngineParams`` / ``SamplingParams`` classes that
``ExperimentConfig`` nests for one engine. They live in the config layer, beside
their only importers (``config.models`` and ``config.introspection``): engines
never import their own config model, so hosting the projection here keeps the
config-consumes-engine-shape edge inside a single layer.

DO NOT EDIT these modules by hand. They are regenerated from the mined
``engine_versions/<engine>/<version>/outputs/`` snapshot (``schema.discovered.json``
+ ``curated.yaml``) by
``scripts/engine_producers/regen_engine_configs.py --engine <engine> --version <version> --write``.
Each module carries its own regeneration header naming the exact snapshot and
command. To change the exposed surface, edit ``curated.yaml`` upstream and
regenerate; never hand-edit the projections.
"""
