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


def _pin_default_user_config(monkeypatch) -> None:
    """Pin the user config to the built-in defaults.

    What a test asserts about a resolved study must not depend on the user config
    of the machine running it.
    """
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config", lambda **_kw: UserConfig()
    )


def _capture_orchestrated(monkeypatch) -> list[StudyConfig]:
    """Capture the studies handed to the orchestrator instead of running them."""
    import llenergymeasure.study.orchestration as orchestration

    captured: list[StudyConfig] = []

    def _capture(study: StudyConfig, **_kwargs) -> object:
        captured.append(study)
        return make_study_result()

    monkeypatch.setattr(orchestration, "orchestrate_study", _capture)
    _pin_default_user_config(monkeypatch)
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


# ---------------------------------------------------------------------------
# Which form of a dormant-field config actually gets dispatched
# ---------------------------------------------------------------------------

# ``do_sample: false`` makes decoding greedy, so transformers ignores
# ``temperature`` entirely. A shipped engine rule marks it dormant under greedy
# decoding and drives it back to absent.
_DORMANT_DECLARED = {
    "serving_mode": "offline",
    "engine": "transformers",
    "task": {"model": "gpt2"},
    "transformers": {"sampling_params": {"do_sample": False, "temperature": 0.9}},
}
_DORMANT_RULE_ID = "transformers_greedy_strips_temperature"


def _dormant_experiment() -> ExperimentConfig:
    return ExperimentConfig(**_DORMANT_DECLARED)


def test_the_canonicalised_form_is_dispatched_on_every_route(tmp_path: Path, monkeypatch) -> None:
    """Pin which object a dormant-field config actually runs as, on every route.

    Resolution rewrites a field the engine ignores back to absent, and it is the
    REWRITTEN config that is dispatched and recorded - so its
    ``declared_config_hash`` (the experiment id, and hence the artefact names)
    differs from the hash of the config as written. The equivalence-group record
    keeps the as-declared hash, which is how a run traces back to what was asked
    for. The study file, a study built from objects and the single-experiment
    entry point must agree on all of this, or the same declared config would
    produce differently named results depending on how it was submitted.
    """
    from llenergymeasure.domain.experiment import compute_declared_config_hash

    as_declared = _dormant_experiment()
    assert as_declared.transformers is not None
    assert as_declared.transformers.sampling_params.temperature == 0.9
    declared_hash = compute_declared_config_hash(as_declared)

    captured = _capture_orchestrated(monkeypatch)
    run_study(_write_study(tmp_path, dict(_DORMANT_DECLARED, study_name="dormant")))
    run_study(StudyConfig(experiments=[_dormant_experiment()]))
    run_experiment(_dormant_experiment())
    from_yaml, from_objects, from_run_experiment = captured
    routes = (
        ("yaml", from_yaml),
        ("objects", from_objects),
        ("run_experiment", from_run_experiment),
    )

    for route, resolved in routes:
        dispatched = resolved.experiments[0]
        assert dispatched.transformers is not None
        # The dispatched config is the canonicalised one: the ignored field is gone.
        assert dispatched.transformers.sampling_params.temperature is None, route
        # ...so its identity is NOT the identity of the config as written.
        dispatched_hash = compute_declared_config_hash(dispatched)
        assert dispatched_hash != declared_hash, route
        # The as-declared identity survives in the equivalence-group record.
        groups = resolved.pre_run_equivalence_groups
        assert [g["member_experiment_ids"] for g in groups] == [[declared_hash]], route
        # And the rewrite is surfaced rather than silent.
        assert [obs["rule_id"] for obs in resolved.dormant_observations] == [_DORMANT_RULE_ID], (
            route
        )

    # Every route agrees exactly on the dispatched form and its identity.
    assert len({compute_declared_config_hash(r.experiments[0]) for _, r in routes}) == 1
    assert len({r.study_design_hash for _, r in routes}) == 1


def test_resolution_does_not_rewrite_the_callers_own_config(monkeypatch) -> None:
    """Canonicalisation works on copies, so a caller's own object keeps its fields.

    The one thing resolution does write onto the caller's objects is the resolved
    server warmup protocol, which is attached as side-channel state rather than a
    declared field; see ``test_server_dispatch``.
    """
    captured = _capture_orchestrated(monkeypatch)
    mine = _dormant_experiment()

    run_study(StudyConfig(experiments=[mine]))

    assert mine.transformers is not None
    assert mine.transformers.sampling_params.temperature == 0.9
    dispatched = captured[0].experiments[0]
    assert dispatched is not mine


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


def test_bad_execution_defaults_raise_a_config_error() -> None:
    """A typo in a caller-supplied execution layer names the bad key, not a traceback."""
    with pytest.raises(ConfigError) as exc:
        resolve_study(_raw(), execution_defaults={"n_cyles": 3})

    msg = str(exc.value)
    assert "execution settings" in msg
    assert "n_cyles" in msg
    assert "n_cycles" in msg  # the valid field names are listed


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


def test_hermetic_resolution_applies_machine_default_gaps() -> None:
    """Without a user config, unset gaps resolve to the built-in machine defaults.

    The machine defaults are the chain's bottom layer, so no resolution path can
    leave a gap as None (which the runner would wait zero seconds on). The
    provenance says so: the gaps are labelled ``default``, not claimed by a user
    config that was never there.
    """
    resolved = resolve_study(_raw())

    assert resolved.study_execution.experiment_gap_seconds == _DEFAULT_EXPERIMENT_GAP
    assert resolved.study_execution.cycle_gap_seconds == _DEFAULT_CYCLE_GAP
    assert resolved.settings_provenance["study_execution.experiment_gap_seconds"] == "default"
    assert resolved.settings_provenance["study_execution.cycle_gap_seconds"] == "default"


# ---------------------------------------------------------------------------
# Per-experiment provenance is emitted by the merges, formatted at resolution
# ---------------------------------------------------------------------------


def test_provenance_logs_label_swept_and_overridden_fields(tmp_path, monkeypatch):
    """The sidecar provenance labels come from the merges that resolved the study.

    The sweep expansion emits the paths it varied and the CLI-override merge
    records the paths it overlaid; resolve_study formats them into per-experiment
    logs keyed by declared hash. No post-hoc diffing decides a label.
    """
    _pin_default_user_config(monkeypatch)
    path = _write_study(
        tmp_path,
        {
            "study_name": "prov",
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "serving_mode": "offline",
            "sweep": {"transformers.llem_execution.batch_size": [1, 8]},
        },
    )
    study = load_study(path, cli_overrides={"task": {"dataset": {"n_prompts": 7}}})

    assert len(study.provenance_logs) == 2
    for log in study.provenance_logs.values():
        assert log["transformers.llem_execution.batch_size"]["source"] == "sweep"
        assert log["task.dataset.n_prompts"]["source"] == "call_site"
        assert log["task.dataset.n_prompts"]["effective"] == 7
        assert log["task.model"]["source"] == "yaml"


def test_provenance_logs_keyed_by_declared_hash(tmp_path, monkeypatch):
    """Each unique declared config gets one log, keyed by its declared hash."""
    from llenergymeasure.domain.experiment import compute_declared_config_hash

    _pin_default_user_config(monkeypatch)
    path = _write_study(
        tmp_path,
        {
            "study_name": "prov-keys",
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "serving_mode": "offline",
        },
    )
    study = load_study(path)

    assert set(study.provenance_logs) == {
        compute_declared_config_hash(exp) for exp in study.experiments
    }


def test_object_path_studies_get_yaml_labelled_provenance():
    """A study built from objects has no sweep or CLI merges - fields label as yaml."""
    resolved = resolve_study(_raw([_experiment()]))
    (log,) = resolved.provenance_logs.values()
    assert log["task.model"] == {"effective": "gpt2", "source": "yaml"}


# ---------------------------------------------------------------------------
# Layer order and provenance truth for the study settings (FIX B, PR #937)
# ---------------------------------------------------------------------------


def test_execution_defaults_beat_user_config_gaps() -> None:
    """Caller execution defaults rank above the user config: file > defaults > user.

    A caller's execution_defaults gap wins over the user config's machine-local
    preference; a gap the caller's defaults leave unset still falls through to
    the user config.
    """
    user = UserConfig(
        execution=UserExecutionConfig(experiment_gap_seconds=123.0, cycle_gap_seconds=456.0)
    )
    resolved = resolve_study(
        _raw(),
        user_config=user,
        execution_defaults={"experiment_gap_seconds": 77.0},
    )

    assert resolved.study_execution.experiment_gap_seconds == 77.0
    assert resolved.study_execution.cycle_gap_seconds == 456.0
    assert (
        resolved.settings_provenance["study_execution.experiment_gap_seconds"]
        == "call_site_default"
    )
    assert resolved.settings_provenance["study_execution.cycle_gap_seconds"] == "user_config"


def test_silent_user_config_claims_no_provenance() -> None:
    """A user config whose file wrote nothing labels nothing as user_config.

    Resolved values are the built-in defaults either way; the label must say so.
    """
    resolved = resolve_study(_raw(), user_config=UserConfig())

    assert resolved.output.results_dir == "./results"
    assert resolved.settings_provenance["output.results_dir"] == "default"
    assert resolved.settings_provenance["study_execution.experiment_gap_seconds"] == "default"
    assert "user_config" not in set(resolved.settings_provenance.values())


def test_written_user_config_values_are_labelled_user_config() -> None:
    """Fields the user config file actually wrote carry the user_config label."""
    user = UserConfig.model_validate(
        {"output": {"results_dir": "/uc/results"}, "execution": {"cycle_gap_seconds": 200.0}}
    )
    resolved = resolve_study(_raw(), user_config=user)

    assert resolved.output.results_dir == "/uc/results"
    assert resolved.settings_provenance["output.results_dir"] == "user_config"
    assert resolved.settings_provenance["study_execution.cycle_gap_seconds"] == "user_config"
    # The gap the file did not write stays a built-in default.
    assert resolved.settings_provenance["study_execution.experiment_gap_seconds"] == "default"


def test_yaml_null_gap_falls_through_to_execution_defaults() -> None:
    """A study-file explicit null gap defers to the caller's execution defaults.

    Ratified corner (PR #937): an explicit ``null`` means "use the machine
    default", and the caller's effective defaults now sit directly below the file,
    so they catch it before the user config does.
    """
    raw = _raw(execution=ExecutionConfig(experiment_gap_seconds=None))
    user = UserConfig(execution=UserExecutionConfig(experiment_gap_seconds=123.0))
    resolved = resolve_study(
        raw,
        user_config=user,
        execution_defaults={"experiment_gap_seconds": 77.0},
    )

    assert resolved.study_execution.experiment_gap_seconds == 77.0


def test_provenance_keeps_labelled_fields_at_default_value(tmp_path, monkeypatch):
    """An override or swept axis equal to the pydantic default keeps its entry.

    The sidecar trim is presentation only; it must never erase a label the merges
    emitted. n_prompts is overridden to its own default here - the provenance
    still records the call_site override.
    """
    _pin_default_user_config(monkeypatch)
    path = _write_study(
        tmp_path,
        {
            "study_name": "prov-default-equal",
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "serving_mode": "offline",
        },
    )
    from llenergymeasure.config.models import DatasetConfig

    default_value = DatasetConfig().n_prompts
    study = load_study(path, cli_overrides={"task": {"dataset": {"n_prompts": default_value}}})

    (log,) = study.provenance_logs.values()
    entry = log["task.dataset.n_prompts"]
    assert entry["source"] == "call_site"
    assert entry["effective"] == default_value


# ---------------------------------------------------------------------------
# GPU allowlist: fill when the study is silent, constrain when it is not
# ---------------------------------------------------------------------------


def test_no_allowlist_leaves_gpu_indices_alone() -> None:
    """Without a machine allowlist, an undeclared selector stays "all GPUs"."""
    resolved = resolve_study(_raw(), user_config=UserConfig())

    assert resolved.study_execution.gpu_indices is None


def test_allowlist_fills_an_undeclared_study_selector() -> None:
    """A study that declares no GPUs inherits the machine's allowed set."""
    user = UserConfig(execution=UserExecutionConfig(gpu_indices=[2, 3]))
    resolved = resolve_study(_raw(), user_config=user)

    assert resolved.study_execution.gpu_indices == [2, 3]


def test_allowlist_fill_is_labelled_as_the_user_configs(monkeypatch) -> None:
    """The fill is an ordinary precedence layer, and the provenance says so."""
    user = UserConfig(execution=UserExecutionConfig(gpu_indices=[2, 3]))
    resolved = resolve_study(_raw(), user_config=user)

    assert resolved.settings_provenance["study_execution.gpu_indices"] == "user_config"


def test_explicit_null_gpu_indices_still_defers_to_the_allowlist() -> None:
    """`gpu_indices: null` means "every GPU I may use", not "escape the allowlist".

    The study block documents null as every visible GPU. On a machine that narrows
    what llem may use, that means the allowed set, so an explicit null defers to
    the allowlist exactly as an explicit null gap defers to the machine default.
    """
    user = UserConfig(execution=UserExecutionConfig(gpu_indices=[2, 3]))
    resolved = resolve_study(_raw(execution=ExecutionConfig(gpu_indices=None)), user_config=user)

    assert resolved.study_execution.gpu_indices == [2, 3]


def test_allowlist_admits_a_study_subset() -> None:
    """A study inside the allowed set keeps exactly what it declared."""
    user = UserConfig(execution=UserExecutionConfig(gpu_indices=[1, 2, 3]))
    resolved = resolve_study(_raw(execution=ExecutionConfig(gpu_indices=[2])), user_config=user)

    assert resolved.study_execution.gpu_indices == [2]


def test_allowlist_rejects_a_study_index_outside_it() -> None:
    """An out-of-set study index fails loudly, naming both sets - no silent narrowing."""
    from llenergymeasure.utils.exceptions import ConfigError

    user = UserConfig(execution=UserExecutionConfig(gpu_indices=[2, 3]))
    with pytest.raises(ConfigError) as exc:
        resolve_study(_raw(execution=ExecutionConfig(gpu_indices=[0, 3])), user_config=user)

    message = str(exc.value)
    assert "requests GPU [0]" in message
    assert "[0, 3]" in message
    assert "execution.gpu_indices=[2, 3]" in message


def test_allowlist_is_not_applied_without_a_user_config() -> None:
    """The hermetic route (no user config) resolves exactly what it was handed."""
    resolved = resolve_study(_raw(execution=ExecutionConfig(gpu_indices=[7])), user_config=None)

    assert resolved.study_execution.gpu_indices == [7]


def test_allowlist_fill_does_not_move_the_study_design_hash() -> None:
    """The filled selector is placement metadata: study identity must not shift.

    Same study, resolved once with no allowlist and once under a machine allowlist
    that fills gpu_indices. The design hash - what dedup grouping and resume drift
    checks key on - has to be byte-identical, or restricting llem to a subset of a
    host's GPUs would silently fork every study's identity.
    """
    unrestricted = resolve_study(_raw(), user_config=UserConfig())
    restricted = resolve_study(
        _raw(), user_config=UserConfig(execution=UserExecutionConfig(gpu_indices=[2, 3]))
    )

    assert restricted.study_execution.gpu_indices == [2, 3]
    assert unrestricted.study_execution.gpu_indices is None
    assert restricted.study_design_hash == unrestricted.study_design_hash
    assert restricted.declared_resolved_config_hashes == (
        unrestricted.declared_resolved_config_hashes
    )


def test_allowlist_fill_leaves_the_experiment_list_byte_identical() -> None:
    """The fill touches the execution block only, never a hashed experiment field."""
    unrestricted = resolve_study(_raw(), user_config=UserConfig())
    restricted = resolve_study(
        _raw(), user_config=UserConfig(execution=UserExecutionConfig(gpu_indices=[2, 3]))
    )

    assert [exp.model_dump(mode="json") for exp in restricted.experiments] == [
        exp.model_dump(mode="json") for exp in unrestricted.experiments
    ]
