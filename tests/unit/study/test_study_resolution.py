"""Tests for the single study-resolution entry point (``study.loading``).

Every study - loaded from YAML or built from objects - passes through
``resolve_study``, so the two routes cannot drift apart. These tests pin:

- unset thermal gaps resolve to the machine-local defaults from the user config
  rather than collapsing to zero, and the runner waits for the resolved value;
- a study file's own gap values, including an explicit zero, win over them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml

from llenergymeasure.config.loader import LoadedStudyRaw, load_study_config
from llenergymeasure.config.models import (
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
)
from llenergymeasure.config.user_config import UserConfig, UserExecutionConfig
from llenergymeasure.study.loading import resolve_study

# The built-in machine-local thermal defaults an unset gap must fall back to.
_DEFAULT_EXPERIMENT_GAP = UserExecutionConfig.model_fields["experiment_gap_seconds"].default
_DEFAULT_CYCLE_GAP = UserExecutionConfig.model_fields["cycle_gap_seconds"].default


def _experiment(model: str = "gpt2") -> ExperimentConfig:
    return ExperimentConfig(task={"model": model}, engine="transformers", serving_mode="offline")


def _raw(
    experiments: list[ExperimentConfig] | None = None,
    *,
    execution: ExecutionConfig | None = None,
) -> LoadedStudyRaw:
    return LoadedStudyRaw(
        valid_experiments=experiments if experiments is not None else [_experiment()],
        skipped=[],
        study_name="gaps",
        output=OutputConfig(),
        execution=execution if execution is not None else ExecutionConfig(),
        runners=None,
        images=None,
    )


def _write_study(tmp_path: Path, study: dict) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    return path


# ---------------------------------------------------------------------------
# Thermal gaps: None means "use the machine default", not zero
# ---------------------------------------------------------------------------


def test_unset_gaps_resolve_to_the_machine_defaults() -> None:
    """A study that declares no gaps gets the user config's defaults, not zero."""
    resolved = resolve_study(_raw(), user_config=UserConfig())

    assert resolved.study_execution.experiment_gap_seconds == _DEFAULT_EXPERIMENT_GAP
    assert resolved.study_execution.cycle_gap_seconds == _DEFAULT_CYCLE_GAP
    # The values the defaults are documented to be - a regression here means the
    # thermal gaps a researcher relies on silently changed.
    assert resolved.study_execution.experiment_gap_seconds == 60.0
    assert resolved.study_execution.cycle_gap_seconds == 300.0


def test_unset_gaps_take_the_user_configs_own_values() -> None:
    """A user config with custom gaps supplies them to a study that declares none."""
    user = UserConfig(
        execution=UserExecutionConfig(experiment_gap_seconds=12.5, cycle_gap_seconds=90.0)
    )
    resolved = resolve_study(_raw(), user_config=user)

    assert resolved.study_execution.experiment_gap_seconds == 12.5
    assert resolved.study_execution.cycle_gap_seconds == 90.0


def test_declared_gaps_win_over_the_user_config() -> None:
    """Gaps the study declares are never overridden by the machine defaults."""
    execution = ExecutionConfig(experiment_gap_seconds=5.0, cycle_gap_seconds=7.0)
    resolved = resolve_study(_raw(execution=execution), user_config=UserConfig())

    assert resolved.study_execution.experiment_gap_seconds == 5.0
    assert resolved.study_execution.cycle_gap_seconds == 7.0


def test_explicit_zero_gap_is_honoured() -> None:
    """An explicit zero means "no gap" and is not treated as unset."""
    execution = ExecutionConfig(experiment_gap_seconds=0.0, cycle_gap_seconds=0.0)
    resolved = resolve_study(_raw(execution=execution), user_config=UserConfig())

    assert resolved.study_execution.experiment_gap_seconds == 0.0
    assert resolved.study_execution.cycle_gap_seconds == 0.0


def test_gaps_resolve_the_same_way_from_yaml(tmp_path: Path) -> None:
    """The YAML route resolves gaps identically to the object route."""
    path = _write_study(
        tmp_path,
        {
            "study_name": "gaps",
            "serving_mode": "offline",
            "engine": "transformers",
            "task": {"model": "gpt2"},
        },
    )
    resolved = resolve_study(load_study_config(path), user_config=UserConfig())

    assert resolved.study_execution.experiment_gap_seconds == _DEFAULT_EXPERIMENT_GAP
    assert resolved.study_execution.cycle_gap_seconds == _DEFAULT_CYCLE_GAP


def test_runner_waits_for_the_resolved_gaps(monkeypatch) -> None:
    """The runner runs the resolved gap instead of skipping it as a zero."""
    from llenergymeasure.study.runner import StudyRunner

    resolved = resolve_study(
        _raw(
            [_experiment("gpt2"), _experiment("distilgpt2")], execution=ExecutionConfig(n_cycles=2)
        ),
        user_config=UserConfig(),
    )
    runner = StudyRunner(resolved, MagicMock(), Path("/tmp/gap-test"))

    gaps: list[tuple[float, str]] = []
    monkeypatch.setattr(
        StudyRunner, "_run_gap", lambda self, seconds, label: gaps.append((seconds, label))
    )

    # index 0 opens the study (no experiment gap); index 2 is a cycle boundary
    # under the default sequential-per-config ordering used by the fixture.
    runner._run_inter_experiment_gaps(0, frozenset())
    runner._run_inter_experiment_gaps(1, frozenset())
    runner._run_inter_experiment_gaps(2, frozenset({2}))

    assert gaps == [
        (60.0, "Experiment gap"),
        (60.0, "Experiment gap"),
        (300.0, "Cycle gap"),
    ]


def test_runner_skips_a_zero_gap(monkeypatch) -> None:
    """A study that pins gaps to zero still runs back-to-back."""
    from llenergymeasure.study.runner import StudyRunner

    resolved = resolve_study(
        _raw(execution=ExecutionConfig(experiment_gap_seconds=0.0, cycle_gap_seconds=0.0)),
        user_config=UserConfig(),
    )
    runner = StudyRunner(resolved, MagicMock(), Path("/tmp/gap-test"))

    gaps: list[float] = []
    monkeypatch.setattr(StudyRunner, "_run_gap", lambda self, seconds, label: gaps.append(seconds))

    runner._run_inter_experiment_gaps(1, frozenset({1}))

    assert gaps == []


def test_hermetic_resolution_leaves_gaps_unset() -> None:
    """Without a user config, resolution invents no machine default."""
    resolved = resolve_study(_raw())

    assert resolved.study_execution.experiment_gap_seconds is None
    assert resolved.study_execution.cycle_gap_seconds is None
