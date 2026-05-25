"""Per-strategy executor modules for the mining-substrate empirical trial.

Each strategy implements a uniform contract: take a ``CellSpec``, emit
``(schema.json, invariants.proposed.yaml)`` into the cell's run directory,
return a list of structured observations (failure modes, adjacent notes).

Modules:
- ``llm_extractor`` - shared chunking + structured-output + retry infrastructure
  used by strategies (b), (c), and the LLM portion of (d-ab) / (d-ac).
- ``llm_b_oss`` - strategy (b) "pure OSS LLM" via container Ollama.
- ``claude_extractor`` - strategy (c) "pure Claude" via Anthropic SDK.
- ``hybrid_extractor`` - strategy (d-ab/d-ac) "hybrid".
- ``transformers_chunker`` - engine-specific source extraction + chunking
  (active version reads via ``inspect``; bumped versions read via venv).
"""
