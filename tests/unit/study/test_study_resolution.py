"""Tests for the single study-resolution entry point (``study.loading``).

Every study - loaded from YAML or built from objects - passes through
``resolve_study``, so the two routes cannot drift apart. These tests pin:

- a study file and the equivalent objects resolve to the same thing, down to the
  ``study_design_hash`` and the resolved experiment list;
- a study built from objects is deduplicated, hashed, cycle-expanded and given
  its equivalence groups, with no file involved;
- unset thermal gaps resolve to the machine-local defaults from the user config
  rather than collapsing to zero, and the runner waits for the resolved value;
- a study file's own gap values, including an explicit zero, win over them;
- caller-supplied execution defaults fill only what the file left unset;
- both routes reject a study that mixes serving_mode values, with one message;
- an unresolved study, or one carrying a hand-written identity hash, is refused
  rather than run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llenergymeasure.api import load_study, run_experiment, run_study
from llenergymeasure.config.loader import LoadedStudyRaw, load_study_config
from llenergymeasure.config.models import (
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
    StudyConfig,
)
from llenergymeasure.config.user_config import UserConfig, UserExecutionConfig
from llenergymeasure.study.loading import resolve_study
from llenergymeasure.study.orchestration import orchestrate_study
from llenergymeasure.utils.exceptions import ConfigError
from tests.conftest import make_study_result

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


def _capture_orchestrated(monkeypatch) -> list[StudyConfig]:
    """Capture the studies handed to the orchestrator instead of running them.

    Also pins the user config to the built-in defaults, so what a test asserts
    does not depend on the user config of the machine running it.
    """
    import llenergymeasure.study.orchestration as orchestration

    captured: list[StudyConfig] = []

    def _capture(study: StudyConfig, **_kwargs) -> object:
        captured.append(study)
        return make_study_result()

    monkeypatch.setattr(orchestration, "orchestrate_study", _capture)
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config", lambda **_kw: UserConfig()
    )
    return captured


# A study file and the objects that declare exactly the same thing.
_PARITY_STUDY = {
    "study_name": "parity",
    "serving_mode": "offline",
    "engine": "transformers",
    "study_execution": {"n_cycles": 2, "experiment_order": "sequential"},
    "experiments": [{"task": {"model": "gpt2"}}, {"task": {"model": "distilgpt2"}}],
}


def _parity_objects() -> StudyConfig:
    return StudyConfig(
        experiments=[_experiment("gpt2"), _experiment("distilgpt2")],
        study_name="parity",
        study_execution=ExecutionConfig(n_cycles=2, experiment_order="sequential"),
    )


# ---------------------------------------------------------------------------
# One resolution for both routes
# ---------------------------------------------------------------------------


def test_yaml_and_object_routes_resolve_identically(tmp_path: Path, monkeypatch) -> None:
    """The same study declared as a file and as objects resolves to one thing."""
    captured = _capture_orchestrated(monkeypatch)

    run_study(_write_study(tmp_path, _PARITY_STUDY))
    run_study(_parity_objects())

    from_yaml, from_objects = captured
    assert from_yaml.study_design_hash == from_objects.study_design_hash
    assert [exp.model_dump(mode="json") for exp in from_yaml.experiments] == [
        exp.model_dump(mode="json") for exp in from_objects.experiments
    ]
    assert from_yaml.dedup_mode == from_objects.dedup_mode
    assert from_yaml.pre_run_equivalence_groups == from_objects.pre_run_equivalence_groups
    assert from_yaml.declared_resolved_config_hashes == from_objects.declared_resolved_config_hashes
    # 2 unique configs x 2 cycles, on both routes.
    assert len(from_objects.experiments) == 4


def test_object_route_dedups_hashes_and_cycles(tmp_path: Path, monkeypatch) -> None:
    """A duplicate config built in memory is deduplicated, hashed and cycled.

    ``tmp_path`` is only here to prove the point: the programmatic route touches
    no file at all.
    """
    captured = _capture_orchestrated(monkeypatch)

    run_study(
        StudyConfig(
            experiments=[_experiment("gpt2"), _experiment("gpt2")],
            study_execution=ExecutionConfig(n_cycles=2, experiment_order="sequential"),
        )
    )

    resolved = captured[0]
    # 1 unique config x 2 cycles - the duplicate collapsed, and n_cycles is no
    # longer silently ignored on this route.
    assert len(resolved.experiments) == 2
    assert resolved.study_design_hash is not None
    assert resolved.dedup_mode == "resolved"
    assert len(resolved.declared_resolved_config_hashes) == 2
    groups = resolved.pre_run_equivalence_groups
    assert len(groups) == 1
    assert groups[0]["member_count"] == 2
    assert groups[0]["would_dedup"] is True
    assert groups[0]["deduplicated"] is True
    assert list(tmp_path.iterdir()) == []


def test_dedup_off_keeps_every_declared_config(monkeypatch) -> None:
    """deduplicate_equivalent=False runs both duplicates and records the mode."""
    captured = _capture_orchestrated(monkeypatch)

    run_study(
        StudyConfig(
            experiments=[_experiment("gpt2"), _experiment("gpt2")],
            study_execution=ExecutionConfig(deduplicate_equivalent=False),
        )
    )

    resolved = captured[0]
    assert resolved.dedup_mode == "off"
    assert len(resolved.experiments) == 2


def test_run_experiment_resolves_its_single_experiment_study(monkeypatch) -> None:
    """The one-experiment entry point resolves too - identity, not a bare config."""
    captured = _capture_orchestrated(monkeypatch)

    run_experiment(_experiment("gpt2"))

    resolved = captured[0]
    assert resolved.study_design_hash is not None
    assert len(resolved.pre_run_equivalence_groups) == 1


def test_an_already_resolved_study_is_not_resolved_twice(tmp_path: Path, monkeypatch) -> None:
    """Passing a load_study result to run_study must not re-expand the cycles."""
    captured = _capture_orchestrated(monkeypatch)

    study = load_study(_write_study(tmp_path, _PARITY_STUDY))
    assert len(study.experiments) == 4

    run_study(study)

    assert len(captured[0].experiments) == 4
    assert captured[0].study_design_hash == study.study_design_hash


def test_orchestration_refuses_an_unresolved_study() -> None:
    """A study that skipped resolution is rejected, not run undeduped."""
    with pytest.raises(ValueError, match="requires a resolved study"):
        orchestrate_study(StudyConfig(experiments=[_experiment()]))


def test_both_routes_reject_a_mixed_serving_mode_study(tmp_path: Path, monkeypatch) -> None:
    """The mixed-serving_mode gate fires identically on the file and object routes."""
    _capture_orchestrated(monkeypatch)

    mixed_yaml = _write_study(
        tmp_path,
        {
            "engine": "vllm",
            "experiments": [
                {"task": {"model": "gpt2"}, "serving_mode": "offline"},
                {
                    "task": {"model": "gpt2"},
                    "serving_mode": "server",
                    "server": {"traffic": {"rate": 5, "window_seconds": 60}},
                },
            ],
        },
    )
    mixed_objects = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="offline"),
            ExperimentConfig(
                task={"model": "gpt2"},
                engine="vllm",
                serving_mode="server",
                server={"traffic": {"rate": 5, "window_seconds": 60}},
            ),
        ]
    )

    with pytest.raises(ConfigError) as from_yaml:
        run_study(mixed_yaml)
    with pytest.raises(ConfigError) as from_objects:
        run_study(mixed_objects)

    assert str(from_yaml.value) == str(from_objects.value)
    assert "mixes serving_mode values" in str(from_objects.value)


def test_a_hand_set_design_hash_does_not_pass_for_resolution() -> None:
    """Writing study_design_hash by hand buys no free pass through the gate.

    A study carrying only the hash has none of resolution's other outputs - no
    equivalence records, no cycle expansion, no resolved thermal gaps - so
    accepting it would run a study whose recorded identity describes nothing that
    was actually resolved. run_study must refuse it rather than run it.
    """
    forged = StudyConfig(experiments=[_experiment()], study_design_hash="deadbeefdeadbeef")
    with pytest.raises(ValueError, match="requires a resolved study"):
        run_study(forged)


# ---------------------------------------------------------------------------
# Execution defaults sit beneath the study file
# ---------------------------------------------------------------------------


def test_execution_defaults_fill_only_what_the_study_left_unset() -> None:
    """A default applies to an omitted field and never overrides a declared one."""
    execution = ExecutionConfig(n_cycles=1)
    resolved = resolve_study(
        _raw(execution=execution),
        execution_defaults={"n_cycles": 3, "experiment_order": "shuffle"},
    )

    assert resolved.study_execution.n_cycles == 1  # declared wins
    assert resolved.study_execution.experiment_order == "shuffle"  # default fills


def test_execution_defaults_reach_the_yaml_route(tmp_path: Path) -> None:
    """load_study applies the defaults without re-reading the study file."""
    path = _write_study(
        tmp_path,
        {
            "serving_mode": "offline",
            "engine": "transformers",
            "task": {"model": "gpt2"},
            "study_execution": {"experiment_order": "sequential"},
        },
    )
    study = load_study(path, execution_defaults={"n_cycles": 3, "experiment_order": "shuffle"})

    assert study.study_execution.n_cycles == 3
    assert study.study_execution.experiment_order == "sequential"


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
