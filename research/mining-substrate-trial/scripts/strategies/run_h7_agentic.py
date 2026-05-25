"""H7 (agentic-loop with tool use) runner for Phase 3b.

Per ``research/mining-substrate-trial/findings/phase3b_hybrid_catalogue.md`` H7. The cheap-patterns
batch + H4 established a sharp LLM-role split at 70B-q4: diagnose-only
robust; validate-subtractive error-prone; extend-propose with runtime gate
modestly effective. H7 tests whether CLOSED-LOOP FEEDBACK shifts the
ceiling.

For each cell:

1. Construct an ``AgenticLoop`` with:
   - Ollama LLM transport (container @ port 11435, llama3.1:70b q4_K_M).
   - Sandbox over the engine source venv + the engine_versions tree.
   - Budget: 30 tool calls OR 30 min wall-clock.
2. Run the loop with task context: engine + version + reference path
   awareness. The LLM iterates: read source -> propose draft -> score
   against reference -> refine -> finalise.
3. Persist:
   - agent_trace.json: full tool-call log + LLM responses.
   - finalised_invariants.yaml: the LLM's finalise output (or empty).
   - score.json: final score vs the active reference + per-cell metadata.
   - observations.md: structured notes for the cross-cell summary.

Cells: transformers v4.57.3 + vllm v0.7.3 (per H7 spec).

Resilience: 529 retry x3 with 60s backoff inside the OllamaBackend
default path; persistent 529 -> ParseFailure -> caught here and
captured as observations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Project root sits four parents above research/mining-substrate-trial/scripts/strategies/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_TRIAL_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_TRIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_TRIAL_SCRIPTS_DIR))

from strategies.agentic_tool_harness import (  # noqa: E402
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_WALL_CLOCK_BUDGET_SEC,
    AgenticLoop,
    LoopResult,
    OllamaLLM,
    build_default_sandbox,
)
from strategies.llm_extractor import OllamaBackend  # noqa: E402
from trial_scoring import ScoringConfig, score_invariants  # noqa: E402

# ---------------------------------------------------------------------------
# Cell spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class H7Cell:
    engine: str
    version_slug: str
    venv_source_label: str
    """Human-readable label for the venv source root (for system prompt)."""


H7_CELLS: list[H7Cell] = [
    H7Cell(
        engine="transformers",
        version_slug="v4_57_3",
        venv_source_label="/tmp/trial_transformers_v4_57_6_venv/src/transformers",
    ),
    H7Cell(
        engine="vllm",
        version_slug="v0_7_3",
        venv_source_label="/tmp/trial_vllm_v0_7_3_venv/src/vllm",
    ),
]


# ---------------------------------------------------------------------------
# Per-cell task context
# ---------------------------------------------------------------------------


TASK_CONTEXT_TEMPLATE = """\
Task: extract validation invariants for engine `{engine}` at version
`{version_slug}`. The static-miner output for this cell lives at
`engine_versions/{engine}/{version_slug}/outputs/invariants.proposed.yaml`
- you can use `run_miner` to summarise what it covers (count + namespaces
+ IDs). You may NOT call `read_file` on that path; it's the scoring
reference.

The engine source tree is browsable through `read_file` rooted at
`{venv_source_label}`. Use relative paths from that root (e.g.
"config.py", "sampling_params.py", "engine/arg_utils.py").

Each invariant must use this match.fields identity (which is what
scoring compares):
  match:
    engine: {engine}
    fields:
      <namespace>.<field>: <expected_value_or_predicate_dict>

For transformers, namespaces are typically `transformers.sampling.*` or
`transformers.bitsandbytes.*`. For vllm, namespaces like `vllm.model.*`,
`vllm.cache.*`, `vllm.scheduler.*`, `vllm.sampling.*` etc. Mirror what
`run_miner` shows for the canonical conventions.

Predicate dicts:
  - `<value>` (plain) means exact-equality (predicate_kind="exact").
  - `{{not_in: [<list>]}}` means "not in list" (predicate_kind="not_in").
  - `{{not_equal: <val>}}` means "not equal" (predicate_kind="not_equal").
  - `{{">": <val>}}` / `{{"<": ...}}` / `{{">=": ...}}` / `{{"<=": ...}}`.
  - `{{type_is_not: <type>}}` for type-mismatch invariants.
  - `{{present: <bool>}}` for "field is set / not set" dormancies.

Each invariant's severity should be one of:
  - error: raises ValueError / RuntimeError at construction.
  - warning: emits a warning but doesn't raise.
  - dormant: silently normalises or ignores (e.g. `seed == -1` -> `seed = None`).
""".strip()


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------


@dataclass
class H7CellResult:
    engine: str
    version_slug: str
    finalised: bool
    stop_reason: str
    total_tool_calls: int
    total_wall_sec: float
    total_llm_sec: float
    tool_usage: dict[str, int]
    """Counts per tool name (read_file, list_validators, run_miner, score_against, finalise, parse_error)."""
    score: dict[str, Any]
    """Final score-against-reference dict (recall, precision, etc.) or
    error dict if finalise was never called / final YAML didn't parse."""
    out_dir: str


def _summarise_tool_usage(loop_result: LoopResult) -> dict[str, int]:
    """Tally tool calls by name + parse errors."""
    counts: Counter = Counter()
    for turn in loop_result.turns:
        if turn.tool_result is None:
            counts["parse_error"] += 1
            continue
        counts[turn.tool_result.tool] += 1
    return dict(counts)


def _score_final_yaml(
    *,
    final_yaml: str | None,
    engine: str,
    version_slug: str,
) -> dict[str, Any]:
    """Score the finalised YAML against the active reference."""
    import yaml as _yaml

    if not final_yaml or not final_yaml.strip():
        return {
            "ok": False,
            "error": "no final YAML (finalise never called)",
            "recall": 0.0,
            "precision": 0.0,
        }
    try:
        emitted_doc = _yaml.safe_load(final_yaml) or {}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"final YAML parse failure: {exc}",
            "recall": 0.0,
            "precision": 0.0,
        }
    if not isinstance(emitted_doc, dict):
        return {
            "ok": False,
            "error": f"final YAML root not a mapping ({type(emitted_doc).__name__})",
            "recall": 0.0,
            "precision": 0.0,
        }
    ref_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / engine
        / version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    ref_doc = _yaml.safe_load(ref_path.read_text()) or {}
    cfg = ScoringConfig()
    (
        recall,
        precision,
        severity_accuracy,
        failure_mode,
        ref_count,
        cell_count,
        int_count,
        recall_misses,
        precision_spurious,
        severity_mismatches,
    ) = score_invariants(
        reference_invariants=ref_doc,
        cell_invariants=emitted_doc,
        config=cfg,
    )
    return {
        "ok": True,
        "recall": recall,
        "precision": precision,
        "severity_accuracy": severity_accuracy,
        "failure_mode": failure_mode.value if hasattr(failure_mode, "value") else str(failure_mode),
        "ref_count": ref_count,
        "cell_count": cell_count,
        "intersection_count": int_count,
        "recall_misses_count": len(recall_misses),
        "precision_spurious_count": len(precision_spurious),
        "severity_mismatches_count": len(severity_mismatches),
    }


def run_h7_for_cell(
    cell: H7Cell,
    *,
    out_root: Path,
    backend: OllamaBackend,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    wall_clock_budget_sec: int = DEFAULT_WALL_CLOCK_BUDGET_SEC,
) -> H7CellResult:
    """Run the agentic loop on one cell + persist artefacts."""
    engine_out = out_root / cell.engine
    engine_out.mkdir(parents=True, exist_ok=True)

    # 1. Construct sandbox + LLM + loop
    sandbox = build_default_sandbox(
        project_root=_PROJECT_ROOT,
        engine=cell.engine,
        version_slug=cell.version_slug,
    )
    llm = OllamaLLM(backend=backend, timeout=600)
    loop = AgenticLoop(
        llm=llm,
        sandbox=sandbox,
        project_root=_PROJECT_ROOT,
        max_tool_calls=max_tool_calls,
        wall_clock_budget_sec=wall_clock_budget_sec,
    )

    # 2. Build additional context for the system prompt
    task_context = TASK_CONTEXT_TEMPLATE.format(
        engine=cell.engine,
        version_slug=cell.version_slug,
        venv_source_label=cell.venv_source_label,
    )

    # 3. Run the loop
    print(
        f"[H7 {cell.engine}] starting agentic loop; "
        f"budget={max_tool_calls} calls / {wall_clock_budget_sec / 60:.0f} min"
    )
    try:
        loop_result = loop.run(
            engine=cell.engine,
            version_slug=cell.version_slug,
            additional_context=task_context,
        )
    except Exception as exc:
        print(f"[H7 {cell.engine}] LOOP CRASH: {type(exc).__name__}: {exc}")
        loop_result = LoopResult(
            engine=cell.engine,
            version_slug=cell.version_slug,
            stop_reason="crash",
        )
        # Persist the partial trace if there is one.

    # 4. Persist artefacts
    (engine_out / "agent_trace.json").write_text(json.dumps(loop_result.to_dict(), indent=2))
    final_yaml = loop_result.final_yaml or ""
    (engine_out / "finalised_invariants.yaml").write_text(final_yaml)

    # 5. Score the final YAML
    score_dict = _score_final_yaml(
        final_yaml=final_yaml,
        engine=cell.engine,
        version_slug=cell.version_slug,
    )
    (engine_out / "score.json").write_text(json.dumps(score_dict, indent=2))

    # 6. Tool usage summary
    tool_usage = _summarise_tool_usage(loop_result)

    # 7. Per-cell observations.md
    obs_lines: list[str] = []
    obs_lines.append(f"# H7 observations: {cell.engine}")
    obs_lines.append("")
    obs_lines.append(f"- Stop reason: {loop_result.stop_reason}")
    obs_lines.append(f"- Total tool calls: {loop_result.total_tool_calls} / {max_tool_calls}")
    obs_lines.append(f"- Wall clock: {loop_result.total_wall_sec:.1f}s")
    obs_lines.append(f"- LLM time: {loop_result.total_llm_sec:.1f}s")
    obs_lines.append(f"- Finalised: {loop_result.finalised}")
    obs_lines.append("")
    obs_lines.append("## Tool usage")
    for tool, count in sorted(tool_usage.items(), key=lambda x: (-x[1], x[0])):
        obs_lines.append(f"- {tool}: {count}")
    obs_lines.append("")
    obs_lines.append("## Score vs active reference")
    if score_dict.get("ok"):
        obs_lines.append(
            f"- recall: {score_dict['recall']:.3f} ({score_dict['intersection_count']}/{score_dict['ref_count']})"
        )
        obs_lines.append(f"- precision: {score_dict['precision']:.3f}")
        obs_lines.append(f"- severity_accuracy: {score_dict['severity_accuracy']:.3f}")
        obs_lines.append(f"- ref_count: {score_dict['ref_count']}")
        obs_lines.append(f"- cell_count: {score_dict['cell_count']}")
    else:
        obs_lines.append(f"- error: {score_dict.get('error', 'unknown')}")
    obs_lines.append("")
    obs_lines.append("## Tool-call sequence")
    obs_lines.append("")
    for i, turn in enumerate(loop_result.turns):
        if turn.tool_result is None:
            obs_lines.append(f"{i + 1}. <parse_error>: {turn.parse_error[:120]}")
        else:
            r = turn.tool_result
            args_summary = json.dumps(r.args, default=str)[:120]
            status = "ok" if r.ok else f"FAIL ({r.error[:80]})"
            obs_lines.append(f"{i + 1}. {r.tool}({args_summary}) -> {status}")
    (engine_out / "observations.md").write_text("\n".join(obs_lines) + "\n")

    print(
        f"[H7 {cell.engine}] DONE: stop={loop_result.stop_reason}, "
        f"calls={loop_result.total_tool_calls}, wall={loop_result.total_wall_sec:.1f}s, "
        f"finalised={loop_result.finalised}, "
        f"recall={score_dict.get('recall', 0.0):.3f}, precision={score_dict.get('precision', 0.0):.3f}"
    )

    return H7CellResult(
        engine=cell.engine,
        version_slug=cell.version_slug,
        finalised=loop_result.finalised,
        stop_reason=loop_result.stop_reason,
        total_tool_calls=loop_result.total_tool_calls,
        total_wall_sec=loop_result.total_wall_sec,
        total_llm_sec=loop_result.total_llm_sec,
        tool_usage=tool_usage,
        score=score_dict,
        out_dir=str(engine_out),
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H7 (agentic-loop) experiment")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_PROJECT_ROOT / "research/mining-substrate-trial/findings/hybrid_experiments/h7_agentic",
        help="Output root directory.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["transformers", "vllm"],
        help="Subset of engines to run (transformers, vllm).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
    )
    parser.add_argument(
        "--wall-clock-budget-sec",
        type=int,
        default=DEFAULT_WALL_CLOCK_BUDGET_SEC,
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    backend = OllamaBackend()

    cells_to_run = [c for c in H7_CELLS if c.engine in args.engines]
    results: list[H7CellResult] = []
    for cell in cells_to_run:
        r = run_h7_for_cell(
            cell,
            out_root=args.out_root,
            backend=backend,
            max_tool_calls=args.max_tool_calls,
            wall_clock_budget_sec=args.wall_clock_budget_sec,
        )
        results.append(r)

    # Aggregate
    aggregate = {
        "pattern": "h7_agentic",
        "results": [
            {
                "engine": r.engine,
                "version_slug": r.version_slug,
                "finalised": r.finalised,
                "stop_reason": r.stop_reason,
                "total_tool_calls": r.total_tool_calls,
                "total_wall_sec": r.total_wall_sec,
                "total_llm_sec": r.total_llm_sec,
                "tool_usage": r.tool_usage,
                "score": r.score,
                "out_dir": r.out_dir,
            }
            for r in results
        ],
    }
    (args.out_root / "h7_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"H7 aggregate written: {args.out_root / 'h7_aggregate.json'}")


if __name__ == "__main__":
    main()
