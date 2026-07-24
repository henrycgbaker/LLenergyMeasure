"""Tests for the ``llem study plan`` funnel (``cli/_study_plan.py``).

These exercise the real machinery: a crafted study YAML is loaded through the
public ``load_study`` entry point (the same one ``llem run`` uses), so the
expansion, validation, and dedup are genuine, then the funnel is computed and
rendered.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import yaml

from llenergymeasure.api import load_study
from llenergymeasure.cli._study_defaults import study_cli_overrides_for_file
from llenergymeasure.cli._study_plan import (
    build_funnel,
    dormant_observation_lines,
    render_funnel,
)

MODEL = "Qwen/Qwen2.5-0.5B"

# A crafted vLLM study with a known shape:
#   sweep = top_p(2) x latency_profiling(2) = 4 declared grid points
#   top_p = 2.0 is rejected by a shipped engine rule (2 points)
#   the 2 valid points differ only in latency_profiling, a measurement dial that
#     joins the resolved-config hash (2026-07-11 ruling: sweeping methodology
#     creates distinct runs), so they stay 2 unique - dedup merges nothing
#   n_cycles = 3 -> 6 runs; gaps: 5 x 30s experiment + 2 x 120s cycle = 390s
_CRAFTED = f"""
study_name: crafted-demo
serving_mode: offline
task:
  model: {MODEL}
engine: vllm
study_execution:
  n_cycles: 3
  experiment_gap_seconds: 30
  cycle_gap_seconds: 120
sweep:
  vllm.sampling_params.top_p: [0.9, 2.0]
  measurement.latency_profiling: [false, true]
"""

_TOP_P_RULE = "vllm_samplingparams_raises_top_p_gt_1p0"


def _load(tmp_path: Path, text: str) -> object:
    path = tmp_path / "study.yaml"
    path.write_text(text)
    return load_study(path)


def _load_with_cli_defaults(tmp_path: Path, text: str) -> object:
    """Load exactly as ``llem run``/``llem study plan`` do - CLI defaults applied.

    Mirrors the command path: the CLI-layer effective defaults (n_cycles=3,
    shuffle) are injected before ``load_study`` unless the file sets them.
    """
    path = tmp_path / "study.yaml"
    path.write_text(text)
    return load_study(path, cli_overrides=study_cli_overrides_for_file(path))


def test_funnel_arithmetic_full(tmp_path: Path) -> None:
    """Declared / pruned / dedup / runs all reconcile on a crafted study."""
    funnel = build_funnel(_load(tmp_path, _CRAFTED))  # type: ignore[arg-type]

    assert funnel.declared_total == 4
    assert funnel.skipped == 2
    assert funnel.valid == 2
    assert funnel.dedup_enabled is True
    # latency_profiling now joins the identity hash, so the two valid points are
    # distinct: nothing merges away and both are unique.
    assert funnel.merged_away == 0
    assert funnel.unique == 2
    assert funnel.n_cycles == 3
    assert funnel.runs == 6


def test_funnel_rule_attribution(tmp_path: Path) -> None:
    """The rejecting engine-rule id is attributed with the right count."""
    funnel = build_funnel(_load(tmp_path, _CRAFTED))  # type: ignore[arg-type]

    by_rule = {a.rule_id: a for a in funnel.attributions}
    assert _TOP_P_RULE in by_rule
    assert by_rule[_TOP_P_RULE].count == 2
    # The rendered value survives; no leftover [rule_id] marker in the message.
    message = by_rule[_TOP_P_RULE].message
    assert "top_p" in message
    assert not message.startswith("[")
    assert _TOP_P_RULE not in message


def test_render_shows_rule_id_and_counts(tmp_path: Path) -> None:
    """The rendered preview names the rule id and the funnel counts."""
    out = render_funnel(build_funnel(_load(tmp_path, _CRAFTED)), "crafted-demo")  # type: ignore[arg-type]

    assert "Study plan: crafted-demo" in out
    assert _TOP_P_RULE in out
    assert "runs" in out
    # Gap-only wall-clock lower bound: 5*30 + 2*120 = 390s = 6m 30s.
    assert "6m 30s" in out


def test_non_rule_failures_grouped(tmp_path: Path) -> None:
    """Pydantic field failures (rule_id None) collapse to one 'other' line."""
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: vllm
sweep:
  vllm.engine_params.max_num_seqs: [4, 0]
"""
    funnel = build_funnel(_load(tmp_path, text))  # type: ignore[arg-type]
    assert funnel.skipped == 1
    none_lines = [a for a in funnel.attributions if a.rule_id is None]
    assert len(none_lines) == 1
    assert none_lines[0].message == "other validation errors"


def test_dedup_disabled_skips_stage(tmp_path: Path) -> None:
    """With deduplicate_equivalent false, the dedup stage is not applied.

    n_cycles is pinned to 1 to keep the focus on dedup - cycle semantics are
    covered by ``test_plan_previews_run_cycle_default``.
    """
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: vllm
study_execution:
  n_cycles: 1
  deduplicate_equivalent: false
sweep:
  measurement.latency_profiling: [false, true]
"""
    funnel = build_funnel(_load(tmp_path, text))  # type: ignore[arg-type]
    assert funnel.dedup_enabled is False
    assert funnel.merged_away == 0
    assert funnel.valid == 2
    assert funnel.runs == 2  # 2 declared configs x 1 cycle, dedup off
    out = render_funnel(funnel, "no-dedup")
    assert "dedup disabled" in out


def test_plan_previews_run_cycle_default(tmp_path: Path) -> None:
    """A file omitting n_cycles previews run's effective 3 cycles, not Pydantic 1.

    The plan mirrors run semantics: the CLI injects n_cycles=3 when the study
    file leaves it unset, so the funnel must show 3 cycles and 3x the unique
    experiment count.
    """
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: vllm
sweep:
  measurement.latency_profiling: [false, true]
"""
    funnel = build_funnel(_load_with_cli_defaults(tmp_path, text))  # type: ignore[arg-type]
    assert funnel.n_cycles == 3
    # Measurement dials join the identity hash (2026-07-11 ruling): the two
    # methodology variants stay distinct rather than deduping to one.
    assert funnel.unique == 2
    assert funnel.runs == 3 * funnel.unique


def test_plan_respects_pinned_cycles(tmp_path: Path) -> None:
    """An explicit n_cycles=1 is honoured - the CLI default does not override it."""
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: vllm
study_execution:
  n_cycles: 1
sweep:
  measurement.latency_profiling: [false, true]
"""
    funnel = build_funnel(_load_with_cli_defaults(tmp_path, text))  # type: ignore[arg-type]
    assert funnel.n_cycles == 1
    assert funnel.runs == funnel.unique


def test_dormant_observations_rendered_in_plan(tmp_path: Path) -> None:
    """A dormant-triggering field surfaces its rule id and field in the plan.

    ``epsilon_cutoff`` is a transformers pydantic-extra the engine silently
    strips; the dormant rule ``transformers_dormant_epsilon_cutoff_ne_0_0``
    fires and drives it to absent. That normalisation must be visible in
    ``llem study plan`` rather than mutating the executed config silently.
    """
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: transformers
transformers:
  sampling_params:
    do_sample: true
    epsilon_cutoff: 0.5
"""
    study = _load(tmp_path, text)
    lines = dormant_observation_lines(study)  # type: ignore[arg-type]
    joined = "\n".join(lines)
    assert "transformers_dormant_epsilon_cutoff_ne_0_0" in joined
    assert "epsilon_cutoff" in joined
    assert "transformers:" in joined


def test_dormant_observations_absent_when_none_fire(tmp_path: Path) -> None:
    """No dormant rule fires -> the section is empty (rendered only when non-empty)."""
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: transformers
transformers:
  sampling_params:
    do_sample: true
    temperature: 0.7
"""
    study = _load(tmp_path, text)
    assert dormant_observation_lines(study) == []  # type: ignore[arg-type]


def test_bounds_scaffold_round_trip(tmp_path: Path) -> None:
    """A bounds scaffold's declared count equals the product of its sweep lists."""
    from llenergymeasure.config.scaffold import render_study_bounds

    text = render_study_bounds(MODEL, ["vllm"])
    raw = yaml.safe_load(text)

    # The full bounds grid is large; keep two axes so the round-trip stays fast
    # while still asserting declared == the Cartesian product of the sweep lists.
    axes = list(raw["sweep"].items())[:2]
    raw["sweep"] = dict(axes)
    raw["serving_mode"] = "offline"
    expected = prod(len(values) for _key, values in axes)

    funnel = build_funnel(_load(tmp_path, yaml.safe_dump(raw)))  # type: ignore[arg-type]
    # Bounds scaffolds exclude engine-rejected values, so nothing is pruned.
    assert funnel.skipped == 0
    assert funnel.declared_total == expected
    assert funnel.valid == expected


def test_vocabulary_no_invariant(tmp_path: Path) -> None:
    """The user-facing preview never uses the word 'invariant'."""
    out = render_funnel(build_funnel(_load(tmp_path, _CRAFTED)), "crafted-demo")  # type: ignore[arg-type]
    assert "invariant" not in out.lower()
    assert "known-invalid" in out
    assert "engine rule" in out


def test_wall_clock_single_run(tmp_path: Path) -> None:
    """A one-run study reports no gaps rather than a bogus lower bound."""
    text = f"""
serving_mode: offline
task:
  model: {MODEL}
engine: vllm
"""
    out = render_funnel(build_funnel(_load(tmp_path, text)), "single")  # type: ignore[arg-type]
    assert "single run" in out
