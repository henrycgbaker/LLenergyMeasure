"""Wave 2 Tier C single-cell informative benchmark: tree-sitter-AST substrate.

ONE cell: transformers active, llama3.1:70b q4 (Wave 1 baseline).

Question this cell answers: does feeding tree-sitter-parsed AST nodes
(instead of raw source text) shift the LLM extraction quality?

Decision rule:
- If recall is within +-5pp of (b) baseline: substrate shape is not the
  bottleneck (confirms Wave 1 H6 substrate-decomposition finding).
- If clearly higher (> +5pp): AST substrate is a Wave 3 candidate worth
  investigating across more engines.
- If clearly lower (< -5pp): AST substrate strips signal the LLM was
  using; the raw-text noise was actually useful information.

NOT a full matrix. Single cell deliberately. Per WAVE2_PROTOCOL section
2bis and section 3.4.
"""

from __future__ import annotations

import time
from pathlib import Path

from strategies.wave2._base import CellOutput, standard_out_dir


def _serialise_tree(engine_source_root: Path) -> str:
    """Parse engine source via tree-sitter; emit a structured text shape
    that exposes class definitions, function signatures, decorator
    chains, if/raise patterns. Skips function-body internals beyond
    those structural markers.
    """
    try:
        import tree_sitter_python  # noqa: F401
        from tree_sitter import Language, Parser  # type: ignore  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Wave 2 b_tree_bench requires `tree-sitter` + `tree-sitter-python`. "
            "Install via the wave2 venv."
        ) from e

    raise NotImplementedError(
        "wave2 b_tree_bench tree-sitter serialisation: emit "
        "class+function+decorator+if/raise skeleton as text for LLM "
        "consumption. Scaffolded; not yet implemented."
    )


def run_cell(
    *,
    engine_source_root: Path,
    ollama_host: str = "http://localhost:11435",
    model: str = "llama3.1:70b",
) -> CellOutput:
    """Execute the single Tier C tree-AST-substrate benchmark cell."""
    strategy_id = "w2-b-tree-bench"
    engine = "transformers"
    version = "v4_57_3"
    out_dir = standard_out_dir(strategy_id, engine, version)

    t0 = time.perf_counter()
    tree_text = _serialise_tree(engine_source_root)  # noqa: F841
    parse_sec = time.perf_counter() - t0

    raise NotImplementedError(
        "wave2 b_tree_bench LLM dispatch: feed serialised tree to "
        "llm_extractor with the (b) prompt. Scaffolded; not yet implemented."
    )
    # noqa: unreachable
    total_sec = time.perf_counter() - t0
    return CellOutput(
        strategy_id=strategy_id,
        engine=engine,
        engine_version=version,
        schema_path=out_dir / "schema.json",
        invariants_path=out_dir / "invariants.proposed.yaml",
        observations=[f"parse_sec={parse_sec:.1f}"],
        wall_sec={"parse": parse_sec, "total": total_sec},
    )
