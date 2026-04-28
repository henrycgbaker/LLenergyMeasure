"""Miners that extract validation rules from engine library source.

Each ``scripts/engine_miners/{engine}_miner.py`` is version-pinned to a
specific library release via the engine SSOT
(``engine_versions/{engine}.yaml`` ``miner_pins.{producer}``) and emits a
corpus-compatible YAML document of rule candidates.

Two extraction mechanisms are in scope:

- **Dynamic mining** — when the library exposes a structured validation method
  (e.g. HF transformers' ``GenerationConfig.validate(strict=True)``), the
  dynamic miner wraps the call and infers predicates from probe results.
- **Static mining** — when no such API exists (vLLM, TRT-LLM), the static
  miner parses the library source AST using the primitives in
  :mod:`scripts.engine_miners._base`.
"""
