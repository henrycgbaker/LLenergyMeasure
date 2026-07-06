"""Tests for the ``llem study init`` CLI command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llenergymeasure.cli import app

runner = CliRunner()

MODEL = "Qwen/Qwen2.5-0.5B"


def test_study_group_in_top_level_help() -> None:
    """The `study` group is registered on the top-level app."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "study" in result.output


def test_study_no_args_shows_help() -> None:
    """`llem study` with no subcommand shows its help and lists `init`."""
    result = runner.invoke(app, ["study"])
    assert result.exit_code != 0
    assert "init" in result.output


def test_init_single_engine_writes_file(tmp_path: Path) -> None:
    """`init -e vllm` writes a runnable single-engine study file."""
    out = tmp_path / "study.yaml"
    result = runner.invoke(app, ["study", "init", "-m", MODEL, "-e", "vllm", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert text.startswith("# llem study file - vllm")
    assert "engine: vllm" in text
    assert str(out) in result.output


def test_init_all_engines_when_engine_omitted(tmp_path: Path) -> None:
    """Omitting -e scaffolds one experiment per engine."""
    out = tmp_path / "study.yaml"
    result = runner.invoke(app, ["study", "init", "-m", MODEL, "-o", str(out)])
    assert result.exit_code == 0
    assert out.read_text().count("  - engine:") == 3


def test_init_defaults_uncomments_fields(tmp_path: Path) -> None:
    """--defaults produces live (uncommented) field lines."""
    bare = tmp_path / "bare.yaml"
    live = tmp_path / "live.yaml"
    runner.invoke(app, ["study", "init", "-m", MODEL, "-e", "vllm", "-o", str(bare)])
    runner.invoke(app, ["study", "init", "-m", MODEL, "-e", "vllm", "--defaults", "-o", str(live)])
    assert "        # dtype:" in bare.read_text()
    assert "        dtype:" in live.read_text()


def test_init_bounds_writes_sweep_file(tmp_path: Path) -> None:
    """--bounds emits a sweep: block of value lists instead of experiments:."""
    out = tmp_path / "study.yaml"
    result = runner.invoke(
        app, ["study", "init", "-m", MODEL, "-e", "vllm", "--bounds", "-o", str(out)]
    )
    assert result.exit_code == 0
    text = out.read_text()
    assert "sweep:" in text
    assert "experiments:" not in text
    assert "vllm.engine_params.dtype:" in text


def test_init_bounds_and_defaults_are_mutually_exclusive(tmp_path: Path) -> None:
    """Combining --bounds with --defaults is a usage error and writes nothing."""
    out = tmp_path / "study.yaml"
    result = runner.invoke(
        app, ["study", "init", "-m", MODEL, "--bounds", "--defaults", "-o", str(out)]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert not out.exists()


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    """A second write to the same path is refused with a non-zero exit."""
    out = tmp_path / "study.yaml"
    first = runner.invoke(app, ["study", "init", "-m", MODEL, "-e", "vllm", "-o", str(out)])
    assert first.exit_code == 0
    second = runner.invoke(app, ["study", "init", "-m", MODEL, "-e", "vllm", "-o", str(out)])
    assert second.exit_code == 1
    assert "Refusing to overwrite" in second.output


def test_init_rejects_unknown_engine(tmp_path: Path) -> None:
    """An unknown engine name is a usage error and writes nothing."""
    out = tmp_path / "study.yaml"
    result = runner.invoke(app, ["study", "init", "-m", MODEL, "-e", "bogus", "-o", str(out)])
    assert result.exit_code != 0
    assert "Unknown engine" in result.output
    assert not out.exists()


def test_init_rejects_empty_model(tmp_path: Path) -> None:
    """A blank model is a usage error."""
    out = tmp_path / "study.yaml"
    result = runner.invoke(app, ["study", "init", "-m", "   ", "-o", str(out)])
    assert result.exit_code != 0
    assert not out.exists()


def test_init_requires_model(tmp_path: Path) -> None:
    """The model option is required."""
    result = runner.invoke(app, ["study", "init", "-o", str(tmp_path / "study.yaml")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# llem study plan
# ---------------------------------------------------------------------------

_PLAN_STUDY = f"""
study_name: plan-smoke
task:
  model: {MODEL}
engine: vllm
study_execution:
  n_cycles: 3
sweep:
  vllm.sampling_params.top_p: [0.9, 2.0]
  measurement.latency_profiling: [false, true]
"""


def test_plan_help() -> None:
    """`llem study plan --help` documents the preview command."""
    result = runner.invoke(app, ["study", "plan", "--help"])
    assert result.exit_code == 0
    assert "plan" in result.output
    assert "preview" in result.output.lower()


def test_plan_happy_path(tmp_path: Path) -> None:
    """A valid study renders the funnel and exits 0, even with skips."""
    study = tmp_path / "study.yaml"
    study.write_text(_PLAN_STUDY)
    result = runner.invoke(app, ["study", "plan", str(study)])
    assert result.exit_code == 0
    assert "Study plan: plan-smoke" in result.output
    assert "declared" in result.output
    assert "runs" in result.output
    # The rejecting engine rule is attributed by id.
    assert "vllm_samplingparams_raises_top_p_gt_1p0" in result.output


def test_plan_previews_run_effective_cycles(tmp_path: Path) -> None:
    """A study omitting n_cycles previews run's effective 3 cycles, not Pydantic 1."""
    import re

    study = tmp_path / "study.yaml"
    study.write_text(
        f"""
study_name: default-cycles
task:
  model: {MODEL}
engine: vllm
sweep:
  measurement.latency_profiling: [false, true]
"""
    )
    result = runner.invoke(app, ["study", "plan", str(study)])
    assert result.exit_code == 0
    # 2 measurement-only configs dedup to 1 unique x 3 cycles = 3 runs.
    assert re.search(r"cycles\s+3", result.output)
    assert re.search(r"runs\s+3", result.output)


def test_plan_output_has_no_invariant_vocab(tmp_path: Path) -> None:
    """The command output never uses the word 'invariant'."""
    study = tmp_path / "study.yaml"
    study.write_text(_PLAN_STUDY)
    result = runner.invoke(app, ["study", "plan", str(study)])
    assert result.exit_code == 0
    assert "invariant" not in result.output.lower()


def test_plan_missing_file_exits_nonzero(tmp_path: Path) -> None:
    """A missing study file exits nonzero with a clean message (no traceback)."""
    result = runner.invoke(app, ["study", "plan", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


def test_plan_zero_experiments_exits_nonzero(tmp_path: Path) -> None:
    """A study that produces no experiments exits nonzero with a clean message."""
    study = tmp_path / "empty.yaml"
    study.write_text("study_name: empty\n")
    result = runner.invoke(app, ["study", "plan", str(study)])
    assert result.exit_code != 0
    assert "no experiments" in result.output.lower()
    assert "Traceback" not in result.output
