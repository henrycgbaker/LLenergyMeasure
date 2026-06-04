"""Wave 2 Tier C single-cell informative benchmark: stub-substrate.

ONE cell: transformers active, llama3.1:70b q4 (Wave 1 baseline).

Question this cell answers: does the pyright type-stub surface alone
(no function bodies) carry the validator signal?

Decision rule:
- If recall on transformers active is < 30% under validated-union: stub
  substrate is dead. Logged in wave2_synthesis.md.
- If recall is >= (b) baseline (~62.5%): stub substrate is a real Wave 3
  candidate; investigation continues.
- Otherwise (30-62%): inconclusive; logged but no follow-up.

NOT a full matrix. Single cell deliberately. Per WAVE2_PROTOCOL section
2bis (systematic cell-selection) and section 3.4 (Tier C single-cell
informative benchmarks).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from strategies.wave2._base import CellOutput, standard_out_dir


def _generate_stubs(engine_source_root: Path, out_dir: Path) -> Path:
    """Run pyright stub generation against the engine source root."""
    stubs_dir = out_dir / "stubs"
    stubs_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pyright",
            "--createstub",
            engine_source_root.name,
            "-p",
            str(engine_source_root.parent),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return stubs_dir


def run_cell(
    *,
    engine_source_root: Path,
    ollama_host: str = "http://localhost:11435",
    model: str = "llama3.1:70b",
) -> CellOutput:
    """Execute the single Tier C stub-substrate benchmark cell.

    Hardcoded to transformers active per protocol; the function ignores
    engine/version kwargs and emits to a fixed output location.
    """
    strategy_id = "w2-b-stub-bench"
    engine = "transformers"
    version = "v4_57_3"
    out_dir = standard_out_dir(strategy_id, engine, version)

    t0 = time.perf_counter()
    stubs_dir = _generate_stubs(engine_source_root, out_dir)
    stub_gen_sec = time.perf_counter() - t0

    # TODO: feed stubs to LLM via the (b) prompt + chunker (chunker
    # adjusted to read .pyi files instead of .py). Extraction logic and
    # output format identical to llm_b_oss; only the substrate changes.
    raise NotImplementedError(
        "wave2 b_stub_bench LLM dispatch: feed stubs_dir to llm_extractor "
        "with stub-aware chunker. Scaffolded; not yet implemented."
    )
    # noqa: unreachable
    total_sec = time.perf_counter() - t0
    return CellOutput(
        strategy_id=strategy_id,
        engine=engine,
        engine_version=version,
        schema_path=out_dir / "schema.json",
        invariants_path=out_dir / "invariants.proposed.yaml",
        observations=[f"stubs_dir={stubs_dir}", f"stub_gen_sec={stub_gen_sec:.1f}"],
        wall_sec={"stub_gen": stub_gen_sec, "total": total_sec},
    )
