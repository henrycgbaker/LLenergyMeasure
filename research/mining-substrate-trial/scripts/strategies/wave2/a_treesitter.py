"""Wave 2d: tree-sitter-based universal deterministic substrate.

Hypothesis: the Wave 1 handwritten per-engine AST walkers (~1200 LoC for
transformers, similar for vllm + tensorrt) carry substantial accidental
complexity. A universal tree-sitter query-based walker should cover
~70-80% of the validator surface with engine-agnostic query patterns,
trading some recall for an order of magnitude less code per engine.

Approach:
1. Parse engine source via tree-sitter-python.
2. Run a fixed set of S-expression queries:
   - `(if_statement consequence: (raise_statement))` for raise-on-condition.
   - `(decorated_definition decorator: (decorator) ... )` for @field_validator + @model_validator.
   - `(if_statement consequence: (block (expression_statement (assignment ...))))` for normalisation patterns.
3. Cluster captures by enclosing class + method.
4. Emit invariants in the canonical envelope shape.

Tradeoff: tree-sitter sees syntax only, not semantics. A predicate like
`if self.foo and not self.bar.baz` parses fine but the walker can't
resolve `self.bar.baz` to a known field name without per-engine
configuration. Some loss of "secondary_field" identity is expected;
quantify it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strategies.wave2._base import CellOutput

# Universal queries shared across engines. Engine-specific overrides
# go in `_ENGINE_OVERRIDES`.
_QUERIES = {
    "raise_on_condition": """
        (if_statement
          condition: (_) @cond
          consequence: (block
            (raise_statement
              (call function: (_) @exc) @raise)))
    """,
    "field_validator_decorator": """
        (decorated_definition
          (decorator
            (call
              function: (identifier) @decorator_name
              (#match? @decorator_name "^(field_validator|validator|model_validator)$")))
          definition: (function_definition
            name: (identifier) @validator_name))
    """,
    "normalisation_pattern": """
        (if_statement
          condition: (comparison_operator) @cond
          consequence: (block
            (expression_statement
              (assignment
                left: (attribute
                  object: (identifier) @target
                  attribute: (identifier) @field)
                right: (_) @value))))
    """,
}


_ENGINE_OVERRIDES: dict[str, dict[str, str]] = {
    "transformers": {},
    "vllm": {},
    "tensorrt-llm": {},
    "sglang": {},
    "lmdeploy": {},
}


def _parse_engine_source(engine_source_root: Path) -> list[Any]:
    """Parse all .py files under engine_source_root via tree-sitter.

    Returns a list of (file_path, syntax_tree) tuples.
    """
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Wave 2d a_treesitter requires `tree-sitter` + `tree-sitter-python`. "
            "Install via `uv add tree-sitter tree-sitter-python` in the wave2 venv."
        ) from e

    # tree_sitter_python ships a precompiled grammar in v0.21+.
    # Older API: Language.build_library + Language(path, "python").
    raise NotImplementedError(
        "wave2 a_treesitter: parser instantiation depends on tree-sitter "
        "version installed; pin version in wave2_model_digests.toml. "
        "Scaffolded; not yet implemented."
    )


def run_cell(
    *,
    engine: str,
    version: str,
    engine_source_root: Path,
) -> CellOutput:
    """Execute one Wave 2d a-treesitter cell."""
    # TODO: resolve out_dir via standard_out_dir; start time.perf_counter();
    # parse via _parse_engine_source; run _QUERIES + per-engine overrides;
    # cluster captures into invariant envelope shape; write out.
    del engine, version, engine_source_root  # consumed once implemented
    raise NotImplementedError(
        "wave2 a_treesitter: dispatch + query execution + invariant "
        "clustering. Scaffolded; not yet implemented."
    )
