"""Wave 2 strategy modules.

Wave 2 strategies extend the Wave 1 trial corpus along five axes:

- 2a: model sweep on the existing (b) pipeline (`b_modelsweep`).
- 2b: pure-LLM substrates without AST walkers (`b_stub`, `b_doc`, `b_rag`, `b_tree`).
- 2c: new hybrid chains (`h10_*` through `h17_*`).
- 2d: alternative deterministic substrates (`a_treesitter`, `a_pydantic_native`,
  `a_fuzz`, `a_runtime_trace`).
- 2e: external validity engines (vendored under engine_versions/sglang/,
  engine_versions/lmdeploy/; strategies dispatch via the same a/b/d-ab plumbing).

Each module exposes ``run_cell(engine, version, *, model=None, out_dir, **kwargs)``
returning a ``CellOutput`` dataclass (defined in `_base.py`). The Wave 2
runner (`scripts/wave2_runner.py`, written after scaffolding closes)
dispatches by strategy ID.

Protocol: see ``research/mining-substrate-trial/WAVE2_PROTOCOL.md``.
"""
