"""End-to-end integration test for sweep canonicalisation + resolved-config dedup.

Exercises the full load path: a study YAML with measurement-equivalent
sweep configs goes through the config-loader parse step plus the study-layer
finalisation, and the resulting ``StudyConfig`` records the pre-run
equivalence groups + deduplicated canonical configs.

Run time: < 1s - no GPU involved, all operations are on Pydantic models
and the engine-invariants loader.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.config.models import StudyConfig
from llenergymeasure.study.loading import resolve_study


def _write_study(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "study.yaml"
    # serving_mode is required (no default); default these offline studies to it
    # while letting a caller's raw dict override.
    path.write_text(yaml.safe_dump({"serving_mode": "offline", **raw}))
    return path


def load_study_config_resolved(path: Path, **kwargs) -> StudyConfig:
    """Parse + resolve a study YAML into a runnable StudyConfig."""
    return resolve_study(load_study_config(path, **kwargs))


def test_greedy_temperature_sweep_collapses(tmp_path: Path) -> None:
    """Six-config sweep with dormant sampling fields collapses to four unique."""
    study = {
        "study_name": "dedup_test",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {
            "transformers.sampling_params.do_sample": [True, False],
            "transformers.sampling_params.temperature": [0.5, 1.0, 1.5],
        },
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    # Dedup mode default is resolved.
    assert study_config.dedup_mode == "resolved"
    # 6 declared x 1 cycle -> 4 unique x 1 cycle = 4 experiments.
    assert len(study_config.experiments) == 4
    # Declared count preserved via resolved-config-hash list.
    assert len(study_config.declared_resolved_config_hashes) == 6
    # At least one group has multiple members (the greedy-family collapse).
    group_sizes = sorted(g["member_count"] for g in study_config.pre_run_equivalence_groups)
    assert max(group_sizes) >= 2
    assert sum(group_sizes) == 6

    # ST2 regression: pre-run groups must carry real member ids. The producer previously
    # hand-rolled member_indices / representative_index (the wrong keys + raw ints), so the
    # runner's deserialiser - which reads member_experiment_ids / representative_experiment_id -
    # loaded every group with empty members.
    for g in study_config.pre_run_equivalence_groups:
        assert "member_indices" not in g and "representative_index" not in g
        assert len(g["member_experiment_ids"]) == g["member_count"]
        assert g["representative_experiment_id"]
        assert g["representative_experiment_id"] in g["member_experiment_ids"]


def test_no_dedup_preserves_all_configs(tmp_path: Path) -> None:
    """With ``deduplicate_equivalent: false`` every declared config runs."""
    study = {
        "study_name": "no_dedup",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {
            "transformers.sampling_params.do_sample": [True, False],
            "transformers.sampling_params.temperature": [0.5, 1.0, 1.5],
        },
        "study_execution": {"deduplicate_equivalent": False},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    assert study_config.dedup_mode == "off"
    # All 6 declared configs run - library-resolution mechanism still populated the groups.
    assert len(study_config.experiments) == 6
    # Groups still computed for the sidecar trail.
    assert sum(g["member_count"] for g in study_config.pre_run_equivalence_groups) == 6


def test_cli_override_no_dedup(tmp_path: Path) -> None:
    """CLI-equivalent override (``study_execution.deduplicate_equivalent: false``)."""
    study = {
        "study_name": "cli_no_dedup",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 5}},
        "sweep": {
            "transformers.sampling_params.do_sample": [True, False],
            "transformers.sampling_params.temperature": [0.5, 0.7],
        },
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(
        path,
        cli_overrides={"study_execution": {"deduplicate_equivalent": False}},
    )
    assert study_config.dedup_mode == "off"
    # 2 x 2 = 4 declared configs all run.
    assert len(study_config.experiments) == 4


def test_n_cycles_multiplies_unique_set(tmp_path: Path) -> None:
    """Dedup happens within a cycle; ``n_cycles`` multiplies the deduped set."""
    study = {
        "study_name": "cycles",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 5}},
        "sweep": {
            "transformers.sampling_params.do_sample": [True, False],
            "transformers.sampling_params.temperature": [0.5, 0.7],
        },
        "study_execution": {"n_cycles": 3},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    # 4 declared -> 3 unique (greedy-0.5 + greedy-0.7 collapse to greedy-1.0,
    # plus 2 sampling variants). 3 unique x 3 cycles = 9 runs.
    assert study_config.dedup_mode == "resolved"
    unique = {h for h in study_config.declared_resolved_config_hashes}
    # Two declared configs share the same resolved_config_hash (both greedy).
    assert len(unique) == 3
    assert len(study_config.experiments) == 9


def test_single_config_sweep_no_dedup(tmp_path: Path) -> None:
    """A sweep with one axis and no equivalence runs normally."""
    study = {
        "study_name": "single",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 5}},
        "sweep": {
            "transformers.sampling_params.temperature": [0.5, 0.7, 0.9],
        },
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    # Sampling is default-true; three temps should stay distinct.
    assert len(study_config.experiments) == 3
    group_sizes = sorted(g["member_count"] for g in study_config.pre_run_equivalence_groups)
    assert group_sizes == [1, 1, 1]


# ---------------------------------------------------------------------------
# Config-identity hash fixes (S9): llem_execution + measurement join the dedup
# hash; --no-dedup with a repeated declared config must not crash the runner.
# ---------------------------------------------------------------------------


def test_llem_execution_batch_size_sweep_not_deduped(tmp_path: Path) -> None:
    """A llem_execution.batch_size sweep must expand to distinct experiments.

    Regression for the config-identity bug: build_resolved_view omitted the
    execution knobs, so a batch_size sweep - which drives execution (the
    transformers plugin reads batch_size, torch_compile, allow_tf32, autocast) -
    collapsed to a single resolved_config_hash. Default dedup then ran ONE
    experiment and reported the other three as dedup-merged.
    """
    study = {
        "study_name": "batch_size_sweep",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {"transformers.llem_execution.batch_size": [1, 4, 8, 16]},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    assert study_config.dedup_mode == "resolved"
    # Four distinct batch sizes are four distinct runs (was 1 before the fix).
    assert len(study_config.experiments) == 4
    assert len(set(study_config.declared_resolved_config_hashes)) == 4


def test_measurement_warmup_sweep_not_deduped(tmp_path: Path) -> None:
    """An offline.warmup.* sweep must expand to distinct experiments.

    Measurement/mode-protocol fields join the identity
    hash - sweeping methodology creates distinct runs; dedup collapses only true
    duplicates. Warmup migrated to the offline: mode namespace, so it now enters the
    identity via the mode_section projection rather than the measurement block; a
    warmup sweep must still not collapse to a single run.
    """
    study = {
        "study_name": "warmup_sweep",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {"offline.warmup.n_prompts": [5, 10, 20]},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    assert len(study_config.experiments) == 3
    assert len(set(study_config.declared_resolved_config_hashes)) == 3


def test_no_dedup_repeated_config_runs_without_keyerror(tmp_path: Path) -> None:
    """--no-dedup with a sweep that collapses grid points must not crash the runner.

    The manifest builder created ``n_cycles`` entries per UNIQUE declared hash,
    but the runner advances a per-hash cycle counter on every OCCURRENCE in the
    ordered execution list. With dedup off, the greedy family collapses the three
    temperature points to one canonical config, so that declared hash occurs
    three times yet had only one manifest entry - the runner's second
    ``mark_running(hash, cycle=2)`` raised an uncaught KeyError, aborting the
    study.
    """
    from llenergymeasure.domain.experiment import compute_declared_config_hash
    from llenergymeasure.study.manifest import ManifestWriter

    study = {
        "study_name": "no_dedup_repeat",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {
            "transformers.sampling_params.do_sample": [True, False],
            "transformers.sampling_params.temperature": [0.5, 1.0, 1.5],
        },
        "study_execution": {"deduplicate_equivalent": False, "n_cycles": 1},
    }
    path = _write_study(tmp_path, study)
    study_config = load_study_config_resolved(path)

    # Precondition: at least one declared hash repeats in the ordered list
    # (the greedy family canonicalises to a single config).
    hashes = [compute_declared_config_hash(e) for e in study_config.experiments]
    assert len(hashes) > len(set(hashes)), "sweep must yield a repeated declared config"

    writer = ManifestWriter(study=study_config, study_dir=tmp_path)

    # Reproduce the runner's per-occurrence cycle assignment (StudyRunner._run_one):
    # each occurrence increments the per-hash counter and marks that entry running.
    counters: dict[str, int] = {}
    for cfg in study_config.experiments:
        h = compute_declared_config_hash(cfg)
        cycle = counters.get(h, 0) + 1
        counters[h] = cycle
        writer.mark_running(h, cycle)  # raised KeyError at cycle=2 before the fix

    # One manifest entry per occurrence - manifest and runner stay aligned.
    assert len(writer.manifest.experiments) == len(study_config.experiments)
