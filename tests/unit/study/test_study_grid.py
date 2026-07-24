"""Unit tests for study grid expansion, cycle ordering, hash, and invalid handling.

TDD RED phase: all expand_grid / compute_study_design_hash / apply_cycles tests
must fail until grid.py is implemented. ExecutionConfig and StudyConfig model
tests pass immediately from the models.py changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llenergymeasure.config.grid import (
    ExperimentOrder,
    SkippedConfig,
    apply_cycles,
    compute_study_design_hash,
    count_sweep_structure,
    cycle_boundary_indices,
    expand_grid,
)
from llenergymeasure.config.models import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    StudyConfig,
)
from llenergymeasure.utils.exceptions import ConfigError

# =============================================================================
# ExecutionConfig model tests
# =============================================================================


class TestExecutionConfig:
    def test_default_values(self):
        ec = ExecutionConfig()
        assert ec.n_cycles == 1
        assert ec.experiment_order == "sequential"
        assert ec.experiment_gap_seconds is None
        assert ec.cycle_gap_seconds is None
        assert ec.shuffle_seed is None

    def test_n_cycles_zero_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(n_cycles=0)

    def test_n_cycles_negative_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(n_cycles=-1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(unknown_field=42)  # type: ignore[call-arg]  # asserts extra rejected

    def test_valid_cycle_orders(self):
        for order in ("sequential", "interleave", "shuffle", "reverse", "latin_square"):
            ec = ExecutionConfig(experiment_order=order)
            assert ec.experiment_order == order

    def test_invalid_cycle_order_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(experiment_order="random")

    def test_gap_fields_non_negative(self):
        ec = ExecutionConfig(experiment_gap_seconds=0.0, cycle_gap_seconds=60.5)
        assert ec.experiment_gap_seconds == 0.0
        assert ec.cycle_gap_seconds == 60.5

    def test_gap_fields_negative_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(experiment_gap_seconds=-1.0)

    def test_shuffle_seed_explicit(self):
        ec = ExecutionConfig(shuffle_seed=12345)
        assert ec.shuffle_seed == 12345

    def test_experiment_timeout_default(self):
        assert ExecutionConfig().experiment_timeout_seconds == 600.0

    def test_experiment_timeout_zero_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(experiment_timeout_seconds=0.0)

    def test_experiment_timeout_negative_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(experiment_timeout_seconds=-1.0)

    def test_experiment_timeout_roundtrip_from_yaml(self):
        yaml_text = "n_cycles: 2\nexperiment_timeout_seconds: 1800.0\n"
        ec = ExecutionConfig(**yaml.safe_load(yaml_text))
        assert ec.experiment_timeout_seconds == 1800.0


# =============================================================================
# StudyConfig model tests
# =============================================================================


class TestStudyConfig:
    def test_accepts_all_fields(self):
        exp = ExperimentConfig(task={"model": "gpt2"})
        sc = StudyConfig(
            experiments=[exp],
            study_name="my-study",
            study_execution=ExecutionConfig(n_cycles=3),
            study_design_hash="abc123def456abcd",
            skipped_configs=[{"raw_config": {}, "reason": "test"}],
        )
        assert sc.study_name == "my-study"
        assert sc.study_execution.n_cycles == 3
        assert sc.study_design_hash == "abc123def456abcd"
        assert len(sc.skipped_configs) == 1

    def test_empty_experiments_raises(self):
        with pytest.raises(ValidationError):
            StudyConfig(experiments=[])

    def test_default_execution(self):
        exp = ExperimentConfig(task={"model": "gpt2"})
        sc = StudyConfig(experiments=[exp])
        assert sc.study_execution.n_cycles == 1
        assert sc.study_design_hash is None
        assert sc.skipped_configs == []

    def test_extra_fields_forbidden(self):
        exp = ExperimentConfig(task={"model": "gpt2"})
        with pytest.raises(ValidationError):
            StudyConfig(experiments=[exp], unknown_field="x")  # type: ignore[call-arg]  # asserts extra rejected


# =============================================================================
# expand_grid() - grid sweep mode
# =============================================================================


class TestExpandGridSweep:
    def test_universal_sweep_cartesian_product(self):
        """2 dtypes x 2 n values = 4 configs (dtype now engine-scoped)."""
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "transformers.engine_params.dtype": ["float16", "bfloat16"],
                "task.dataset.n_prompts": [50, 100],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 4
        assert len(skipped) == 0
        dtypes_set = {c.transformers.engine_params.dtype for c in valid}
        ns = {c.task.dataset.n_prompts for c in valid}
        assert dtypes_set == {"float16", "bfloat16"}
        assert ns == {50, 100}

    def test_latency_profiling_sweep_two_distinct_hashes(self):
        """Sweeping measurement.latency_profiling yields 2 configs with distinct hashes."""
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "measurement.latency_profiling": [False, True],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 2
        assert len(skipped) == 0
        flags = {c.measurement.latency_profiling for c in valid}
        assert flags == {False, True}
        hashes = {compute_declared_config_hash(c) for c in valid}
        assert len(hashes) == 2

    def test_engine_scoped_sweep_routes_to_section(self):
        """transformers.llem_execution.batch_size routes to the transformers section, not top-level."""
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "transformers.llem_execution.batch_size": [1, 8],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 2
        assert len(skipped) == 0
        batch_sizes = {c.transformers.llem_execution.batch_size for c in valid}
        assert batch_sizes == {1, 8}

    def test_multi_engine_scoped_sweep(self):
        """Multi-engine with per-engine dtype axes: independent grids per engine."""
        raw = {
            "task": {"model": "gpt2"},
            "engine": ["transformers", "vllm"],
            "sweep": {
                "transformers.engine_params.dtype": ["float16", "bfloat16"],
                "vllm.engine_params.dtype": ["float16", "bfloat16"],
                "transformers.llem_execution.batch_size": [1, 8],
                "vllm.engine_params.max_num_seqs": [64, 256],
            },
        }
        valid, skipped = expand_grid(raw)
        # transformers: 2 dtypes x 2 batch_sizes = 4
        # vllm: 2 dtypes x 2 max_num_seqs = 4
        # total = 8
        assert len(valid) == 8
        assert len(skipped) == 0
        pytorch_configs = [c for c in valid if c.engine == "transformers"]
        vllm_configs = [c for c in valid if c.engine == "vllm"]
        assert len(pytorch_configs) == 4
        assert len(vllm_configs) == 4
        # transformers configs must not have vllm section and vice versa
        for c in pytorch_configs:
            assert c.vllm is None
        for c in vllm_configs:
            assert c.transformers is None
            assert c.vllm is not None
            assert c.vllm.engine_params is not None
            assert c.vllm.engine_params.max_num_seqs in (64, 256)

    def test_fixed_engine_union_with_scoped_axis(self):
        """A study fixing engine: transformers while sweeping a vllm axis keeps both.

        Regression (PR-D change 3): the explicit fixed engine used to be silently
        dropped when scope-derived engines were computed from a differently-scoped
        sweep axis. It must be unioned in - transformers gets its baseline run and
        vllm gets its swept grid.
        """
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "vllm.engine_params.max_num_seqs": [64, 256],
            },
        }
        valid, skipped = expand_grid(raw)
        assert skipped == []
        engines = {c.engine for c in valid}
        assert engines == {"transformers", "vllm"}
        # transformers: one baseline (no transformers-scoped axis) ; vllm: 2 swept
        assert len([c for c in valid if c.engine == "transformers"]) == 1
        assert len([c for c in valid if c.engine == "vllm"]) == 2

    def test_default_engine_not_unioned_with_scoped_axis(self):
        """An unset engine (defaulting to transformers) is NOT spuriously added.

        Only an *explicitly* set engine is unioned into scope-derived engines;
        a sweep over only vllm axes with no engine: line stays vllm-only.
        """
        raw = {
            "task": {"model": "gpt2"},
            "sweep": {
                "vllm.engine_params.max_num_seqs": [64, 256],
            },
        }
        valid, skipped = expand_grid(raw)
        assert skipped == []
        assert {c.engine for c in valid} == {"vllm"}


# =============================================================================
# expand_grid() - explicit experiments mode
# =============================================================================


class TestExpandGridExplicit:
    def test_explicit_experiments_list(self):
        raw = {
            "experiments": [
                {"task": {"model": "gpt2"}, "engine": "transformers"},
                {"task": {"model": "gpt2"}, "engine": "vllm"},
            ]
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 2
        assert valid[0].engine == "transformers"
        assert valid[1].engine == "vllm"


# =============================================================================
# expand_grid() - combined mode
# =============================================================================


class TestExpandGridCombined:
    def test_sweep_plus_explicit(self):
        """Sweep configs come first, then explicit entries appended."""
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "transformers.engine_params.dtype": ["float16", "bfloat16"],
            },
            "experiments": [
                {"task": {"model": "gpt2-xl"}, "engine": "transformers"},
            ],
        }
        valid, _skipped = expand_grid(raw)
        # 2 sweep + 1 explicit = 3
        assert len(valid) == 3
        # Sweep configs first
        sweep_configs = valid[:2]
        explicit_config = valid[2]
        assert {c.transformers.engine_params.dtype for c in sweep_configs} == {
            "float16",
            "bfloat16",
        }
        assert explicit_config.task.model == "gpt2-xl"


# =============================================================================
# expand_grid() - inline-model baseline vs phantom baseline
# =============================================================================


class TestExpandGridInlineBaseline:
    def test_inline_model_form_yields_single_baseline(self):
        """No sweep, no experiments, top-level task.model -> exactly one baseline."""
        raw = {"study_name": "inline", "task": {"model": "gpt2"}}
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 1
        assert valid[0].engine == "transformers"
        assert valid[0].task.model == "gpt2"

    def test_experiments_with_shared_task_no_phantom_baseline(self):
        """A shared top-level task: plus an explicit experiments: list (no sweep)
        must yield exactly the user's entries, not an extra synthesized
        default-engine baseline.
        """
        raw = {
            "study_name": "shared-task",
            "task": {"model": "gpt2"},
            "experiments": [
                {"engine": "transformers"},
                {"engine": "vllm"},
            ],
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 2
        assert [c.engine for c in valid] == ["transformers", "vllm"]
        # None of them is a synthesized baseline: every entry carries the engine
        # the user wrote, and there is no duplicate transformers baseline.
        assert all(c.task.model == "gpt2" for c in valid)


# =============================================================================
# expand_grid() - base: resolution
# =============================================================================


class TestExpandGridBase:
    def test_base_loads_relative_to_study_yaml(self, tmp_path: Path):
        base_config = {
            "task": {"model": "gpt2", "dataset": {"n_prompts": 50}},
            "engine": "transformers",
        }
        base_file = tmp_path / "base_experiment.yaml"
        base_file.write_text(yaml.dump(base_config))

        raw = {
            "base": "base_experiment.yaml",
            "sweep": {
                "transformers.engine_params.dtype": ["float16", "bfloat16"],
            },
        }
        study_yaml = tmp_path / "study.yaml"
        valid, _skipped = expand_grid(raw, study_yaml_path=study_yaml)
        assert len(valid) == 2
        for c in valid:
            assert c.task.model == "gpt2"
            assert c.task.dataset.n_prompts == 50

    def test_base_strips_study_only_keys(self, tmp_path: Path):
        """Study-only keys in base file are stripped before merging."""
        base_config = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            # These should be stripped
            "sweep": {"transformers.engine_params.dtype": ["float32"]},
            "experiments": [{"model": "other"}],
            "study_execution": {"n_cycles": 5},
            "base": "another.yaml",
            "study_name": "should-be-stripped",
        }
        base_file = tmp_path / "base_experiment.yaml"
        base_file.write_text(yaml.dump(base_config))

        raw = {
            "base": "base_experiment.yaml",
            "sweep": {
                "transformers.engine_params.dtype": ["float16"],
            },
        }
        study_yaml = tmp_path / "study.yaml"
        valid, _skipped = expand_grid(raw, study_yaml_path=study_yaml)
        assert len(valid) == 1
        assert valid[0].transformers.engine_params.dtype == "float16"
        assert valid[0].task.model == "gpt2"

    def test_missing_base_file_raises(self, tmp_path: Path):
        raw = {
            "base": "nonexistent.yaml",
            "sweep": {"transformers.engine_params.dtype": ["float16"]},
        }
        study_yaml = tmp_path / "study.yaml"
        with pytest.raises(ConfigError, match="base"):
            expand_grid(raw, study_yaml_path=study_yaml)

    def test_top_level_images_key_is_study_level(self):
        """A top-level ``images:`` override must not leak into experiment configs.

        ``images:`` is study-level metadata (per-engine Docker image overrides).
        Before the fix it was absent from ``_STUDY_ONLY_KEYS``, so it flowed into
        every experiment dict and ``ExperimentConfig(extra="forbid")`` rejected
        all of them ("all generated configs are invalid").
        """
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "images": {"transformers": "ghcr.io/org/img:tag"},
            "sweep": {"task.dataset.n_prompts": [10, 20]},
        }
        valid, skipped = expand_grid(raw)
        assert len(skipped) == 0
        assert len(valid) == 2
        assert {c.task.dataset.n_prompts for c in valid} == {10, 20}


# =============================================================================
# expand_grid() - invalid combination handling
# =============================================================================


class TestExpandGridInvalidHandling:
    def test_invalid_configs_collected_as_skipped(self):
        """Invalid configs become SkippedConfig, valid ones are returned."""
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {
                "transformers.engine_params.dtype": ["float16", "bfloat16"],
            },
            "experiments": [
                # This will fail: vllm section + engine=transformers
                {"task": {"model": "gpt2"}, "engine": "transformers", "vllm": {"max_num_seqs": 64}},
            ],
        }
        valid, skipped = expand_grid(raw)
        # The two sweep configs are valid; the explicit one fails cross-validation
        assert len(valid) == 2
        assert len(skipped) == 1
        assert "vllm" in skipped[0].reason.lower() or "engine" in skipped[0].reason.lower()

    def test_all_invalid_raises_config_error(self):
        """All invalid configs raises ConfigError with count and reasons."""
        raw = {
            "experiments": [
                # Invalid: transformers section with vllm engine
                {"task": {"model": "gpt2"}, "engine": "vllm", "transformers": {"batch_size": 4}},
                # Invalid: vllm section with pytorch engine
                {"task": {"model": "gpt2"}, "engine": "transformers", "vllm": {"max_num_seqs": 64}},
            ]
        }
        with pytest.raises(ConfigError, match=r"nothing to run|all.*invalid|0.*valid"):
            expand_grid(raw)

    def test_skipped_config_short_label(self):
        sc = SkippedConfig(
            raw_config={
                "engine": "transformers",
                "transformers": {"engine_params": {"dtype": "float32"}},
            },
            reason="some validation error",
        )
        assert sc.short_label == "transformers, float32"

    def test_skipped_config_to_dict(self):
        sc = SkippedConfig(
            raw_config={"engine": "vllm", "vllm": {"engine_params": {"dtype": "float16"}}},
            reason="cross-validation error",
            errors=[{"loc": ["engine"], "msg": "test"}],
        )
        d = sc.to_dict()
        assert d["raw_config"] == {
            "engine": "vllm",
            "vllm": {"engine_params": {"dtype": "float16"}},
        }
        assert d["reason"] == "cross-validation error"
        assert d["short_label"] == "vllm, float16"
        assert len(d["errors"]) == 1

    def test_no_experiments_raises_config_error(self):
        """A sweep with no model and no experiments raises ConfigError."""
        raw = {"study_name": "empty-study"}
        with pytest.raises(ConfigError):
            expand_grid(raw)

    def test_experiments_yaml_null_treated_as_empty(self):
        """`experiments:` present-but-null must not TypeError.

        Regression (PR-D change 4): YAML `experiments:` with no value yields
        None, which used to blow up the iteration. It is treated as [] so the
        sweep (or inline baseline) still produces the study.
        """
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "experiments": None,
            "sweep": {"transformers.engine_params.dtype": ["float16", "bfloat16"]},
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 2
        assert skipped == []

    def test_skipped_config_errors_are_json_serialisable(self):
        """A rule-rejected config's stored error survives json.dumps.

        Regression (PR-D change 5): Pydantic error dicts carry a non-serialisable
        `ctx` (the raw exception object). `_extract_rule_id` reads it, then it is
        stripped so SkippedConfig.errors stays json.dumps-able.
        """
        import json

        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "transformers": {"engine_params": {"num_beams": 2}},
            "sweep": {"transformers.sampling_params.num_return_sequences": [1, 4]},
        }
        _valid, skipped = expand_grid(raw)
        assert len(skipped) == 1
        # rule_id was still extracted before ctx was dropped.
        assert skipped[0].rule_id is not None
        # errors carry no ctx and round-trip through json.dumps.
        for err in skipped[0].errors:
            assert "ctx" not in err
        json.dumps(skipped[0].to_dict())


class TestSkippedConfigRuleAttribution:
    """expand_grid records which engine rule rejected which config."""

    def test_rule_rejection_records_rule_id(self):
        """A sweep point that violates a corpus rule carries the rejecting rule id.

        num_beams=2 with num_return_sequences=4 fires the transformers
        cross-section rule; the num_return_sequences=1 point stays valid.
        """
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "transformers": {"engine_params": {"num_beams": 2}},
            "sweep": {"transformers.sampling_params.num_return_sequences": [1, 4]},
        }
        valid, skipped = expand_grid(raw)

        assert len(valid) == 1
        assert len(skipped) == 1
        assert (
            skipped[0].rule_id
            == "transformers_num_return_vs_beams_num_beams_lt_num_return_sequences"
        )

    def test_non_rule_failure_has_no_rule_id(self):
        """A non-rule rejection (wrong engine section) leaves rule_id None."""
        raw = {
            "task": {"model": "gpt2"},
            "engine": "transformers",
            "sweep": {"transformers.engine_params.dtype": ["float16"]},
            "experiments": [
                {"task": {"model": "gpt2"}, "engine": "transformers", "vllm": {"max_num_seqs": 64}},
            ],
        }
        valid, skipped = expand_grid(raw)

        assert len(valid) == 1
        assert len(skipped) == 1
        assert skipped[0].rule_id is None

    def test_to_dict_includes_rule_id(self):
        """rule_id round-trips through the serialised form."""
        sc = SkippedConfig(
            raw_config={"engine": "transformers", "transformers": {}},
            reason="[some_rule_id] boom",
            rule_id="some_rule_id",
        )
        assert sc.to_dict()["rule_id"] == "some_rule_id"

    def test_to_dict_rule_id_none_by_default(self):
        """A skipped config with no rule attribution serialises rule_id as None."""
        sc = SkippedConfig(
            raw_config={"engine": "vllm", "vllm": {}},
            reason="type error",
        )
        assert sc.to_dict()["rule_id"] is None


class TestMultiBackendSectionStripping:
    """Top-level engine sections are stripped for non-matching engines in multi-engine studies."""

    def test_sweep_strips_inherited_engine_sections(self):
        """A top-level tensorrt: section must not leak into pytorch/vllm sweep configs."""
        raw = {
            "task": {"model": "gpt2"},
            "tensorrt": {"engine_params": {"max_input_len": 1024}},
            "sweep": {
                "transformers.engine_params.dtype": ["bfloat16"],
                "tensorrt.engine_params.dtype": ["bfloat16"],
                "transformers.llem_execution.batch_size": [1],
                "tensorrt.engine_params.max_batch_size": [4],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(skipped) == 0, f"Expected 0 skipped, got: {[s.reason for s in skipped]}"
        # One pytorch config, one tensorrt config
        pytorch_configs = [c for c in valid if c.engine == "transformers"]
        tensorrt_configs = [c for c in valid if c.engine == "tensorrt"]
        assert len(pytorch_configs) == 1
        assert len(tensorrt_configs) == 1
        # Pytorch config must NOT have tensorrt section
        assert pytorch_configs[0].tensorrt is None
        # Tensorrt config inherits the top-level tensorrt section
        assert tensorrt_configs[0].tensorrt is not None
        assert tensorrt_configs[0].tensorrt.engine_params.max_input_len == 1024

    def test_explicit_experiment_strips_inherited_not_explicit(self):
        """Inherited engine sections are stripped; explicitly written ones still fail."""
        raw = {
            "tensorrt": {"engine_params": {"max_input_len": 1024}},
            "experiments": [
                # Inherited tensorrt: should be stripped for this pytorch experiment
                {"task": {"model": "gpt2"}, "engine": "transformers"},
                # Explicit vllm: section with engine=transformers is a user error - should fail
                {
                    "task": {"model": "gpt2"},
                    "engine": "transformers",
                    "vllm": {"engine_params": {"max_num_seqs": 64}},
                },
            ],
        }
        valid, skipped = expand_grid(raw)
        assert len(valid) == 1
        assert valid[0].engine == "transformers"
        assert valid[0].tensorrt is None
        assert len(skipped) == 1
        assert "vllm" in skipped[0].reason.lower()

    def test_sweep_with_all_three_engines(self):
        """Three-engine sweep with a shared tensorrt section produces valid configs for all."""
        raw = {
            "task": {"model": "gpt2"},
            "tensorrt": {"engine_params": {"max_input_len": 512}},
            "sweep": {
                "transformers.llem_execution.batch_size": [1],
                "vllm.engine_params.max_num_seqs": [64],
                "tensorrt.engine_params.max_batch_size": [4],
            },
        }
        valid, skipped = expand_grid(raw)
        assert len(skipped) == 0, f"Unexpected skips: {[s.reason for s in skipped]}"
        engines = sorted(c.engine for c in valid)
        assert engines == ["tensorrt", "transformers", "vllm"]


# =============================================================================
# compute_study_design_hash() tests
# =============================================================================


class TestComputeStudyDesignHash:
    def test_returns_16_char_hex(self):
        experiments = [ExperimentConfig(task={"model": "gpt2"})]
        h = compute_study_design_hash(experiments)
        assert len(h) == 16
        int(h, 16)  # must be valid hex

    def test_same_experiments_same_hash(self):
        exps1 = [
            ExperimentConfig(task={"model": "gpt2"}),
            ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)}),
        ]
        exps2 = [
            ExperimentConfig(task={"model": "gpt2"}),
            ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)}),
        ]
        assert compute_study_design_hash(exps1) == compute_study_design_hash(exps2)

    def test_different_experiments_different_hash(self):
        exps1 = [ExperimentConfig(task={"model": "gpt2"})]
        exps2 = [ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=25)})]
        assert compute_study_design_hash(exps1) != compute_study_design_hash(exps2)

    def test_stable_across_calls(self):
        experiments = [
            ExperimentConfig(task={"model": "gpt2"}),
            ExperimentConfig(task={"model": "gpt2-xl"}),
        ]
        h1 = compute_study_design_hash(experiments)
        h2 = compute_study_design_hash(experiments)
        assert h1 == h2

    def test_hash_excludes_order_sensitivity(self):
        """Same experiments in same order produce same hash (order matters for reproducibility)."""
        exps_a = [
            ExperimentConfig(task={"model": "gpt2"}),
            ExperimentConfig(
                task={"model": "gpt2"}, transformers={"engine_params": {"dtype": "float16"}}
            ),
        ]
        exps_b = [
            ExperimentConfig(task={"model": "gpt2"}),
            ExperimentConfig(
                task={"model": "gpt2"}, transformers={"engine_params": {"dtype": "float16"}}
            ),
        ]
        assert compute_study_design_hash(exps_a) == compute_study_design_hash(exps_b)


# =============================================================================
# apply_cycles() tests
# =============================================================================


class TestApplyCycles:
    @pytest.fixture
    def two_experiments(self):
        a = ExperimentConfig(task={"model": "gpt2"})
        b = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=25)})
        return [a, b]

    @pytest.fixture
    def study_hash(self, two_experiments):
        return compute_study_design_hash(two_experiments)

    def test_sequential_ordering(self, two_experiments, study_hash):
        """sequential with 3 cycles and [A, B] -> [A, A, A, B, B, B]."""
        result = apply_cycles(two_experiments, 3, ExperimentOrder.SEQUENTIAL, study_hash)
        assert len(result) == 6
        # First 3 should be A (gpt2, n_prompts=100 default)
        assert all(r.task.dataset.n_prompts == 100 for r in result[:3])
        # Last 3 should be B (gpt2, n_prompts=25)
        assert all(r.task.dataset.n_prompts == 25 for r in result[3:])

    def test_interleaved_ordering(self, two_experiments, study_hash):
        """interleave with 3 cycles and [A, B] -> [A, B, A, B, A, B]."""
        result = apply_cycles(two_experiments, 3, ExperimentOrder.INTERLEAVE, study_hash)
        assert len(result) == 6
        # Alternating: A, B, A, B, A, B
        for i in range(0, 6, 2):
            assert result[i].task.dataset.n_prompts == 100  # A
        for i in range(1, 6, 2):
            assert result[i].task.dataset.n_prompts == 25  # B

    def test_shuffled_with_explicit_seed_deterministic(self, two_experiments, study_hash):
        """Shuffle with explicit seed produces deterministic reproducible order."""
        result1 = apply_cycles(
            two_experiments, 3, ExperimentOrder.SHUFFLE, study_hash, shuffle_seed=42
        )
        result2 = apply_cycles(
            two_experiments, 3, ExperimentOrder.SHUFFLE, study_hash, shuffle_seed=42
        )
        assert [r.task.dataset.n_prompts for r in result1] == [
            r.task.dataset.n_prompts for r in result2
        ]

    def test_shuffled_with_same_hash_same_order(self, two_experiments, study_hash):
        """Same study_design_hash without explicit seed = same shuffle."""
        result1 = apply_cycles(two_experiments, 3, ExperimentOrder.SHUFFLE, study_hash)
        result2 = apply_cycles(two_experiments, 3, ExperimentOrder.SHUFFLE, study_hash)
        assert [r.task.dataset.n_prompts for r in result1] == [
            r.task.dataset.n_prompts for r in result2
        ]

    def test_shuffled_different_seeds_different_orders(self, study_hash):
        """Seeds 1 and 999 produce different orderings (verified deterministic)."""
        exps = [
            ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=i)})
            for i in range(1, 6)
        ]
        result1 = apply_cycles(exps, 2, ExperimentOrder.SHUFFLE, study_hash, shuffle_seed=1)
        result2 = apply_cycles(exps, 2, ExperimentOrder.SHUFFLE, study_hash, shuffle_seed=999)
        # Seeds 1 and 999 confirmed to produce distinct orderings for 5 experiments x 2 cycles
        # (seed 1 → [3,4,5,1,2,1,3,2,5,4], seed 999 → [3,5,2,4,1,2,4,5,1,3])
        assert [r.task.dataset.n_prompts for r in result1] != [
            r.task.dataset.n_prompts for r in result2
        ]

    def test_n_cycles_one_unchanged(self, two_experiments, study_hash):
        """n_cycles=1 returns the original list unchanged."""
        result = apply_cycles(two_experiments, 1, ExperimentOrder.SEQUENTIAL, study_hash)
        assert len(result) == 2
        assert result[0].task.dataset.n_prompts == two_experiments[0].task.dataset.n_prompts
        assert result[1].task.dataset.n_prompts == two_experiments[1].task.dataset.n_prompts

    def test_shuffled_contains_all_experiments_each_cycle(self, two_experiments, study_hash):
        """Each cycle in shuffle mode contains all experiments exactly once."""
        result = apply_cycles(two_experiments, 3, ExperimentOrder.SHUFFLE, study_hash)
        assert len(result) == 6
        # Check that each pair of 2 contains both experiments
        for i in range(0, 6, 2):
            pair_ns = {result[i].task.dataset.n_prompts, result[i + 1].task.dataset.n_prompts}
            assert pair_ns == {100, 25}

    # -- reverse mode --

    def test_reverse_ordering(self, two_experiments, study_hash):
        """reverse with 4 cycles and [A, B] -> [A, B, B, A, A, B, B, A]."""
        result = apply_cycles(two_experiments, 4, ExperimentOrder.REVERSE, study_hash)
        assert len(result) == 8
        ns = [r.task.dataset.n_prompts for r in result]
        assert ns == [100, 25, 25, 100, 100, 25, 25, 100]

    def test_reverse_single_cycle(self, two_experiments, study_hash):
        """reverse with 1 cycle = forward order (same as sequential for one cycle)."""
        result = apply_cycles(two_experiments, 1, ExperimentOrder.REVERSE, study_hash)
        assert [r.task.dataset.n_prompts for r in result] == [100, 25]

    def test_reverse_contains_all_experiments_each_cycle(self, two_experiments, study_hash):
        """Each cycle in reverse mode contains all experiments exactly once."""
        result = apply_cycles(two_experiments, 3, ExperimentOrder.REVERSE, study_hash)
        assert len(result) == 6
        for i in range(0, 6, 2):
            pair_ns = {result[i].task.dataset.n_prompts, result[i + 1].task.dataset.n_prompts}
            assert pair_ns == {100, 25}

    # -- latin_square mode --

    def test_latin_square_ordering(self, study_hash):
        """latin_square with 3 experiments x 3 cycles produces balanced rows."""
        exps = [
            ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=i)})
            for i in [1, 2, 3]
        ]
        result = apply_cycles(exps, 3, ExperimentOrder.LATIN_SQUARE, study_hash)
        assert len(result) == 9
        # Each cycle (row) contains all 3 experiments exactly once
        for i in range(0, 9, 3):
            row_ns = [r.task.dataset.n_prompts for r in result[i : i + 3]]
            assert sorted(row_ns) == [1, 2, 3]

    def test_latin_square_each_position_balanced(self, study_hash):
        """Each experiment appears in each position exactly once across k cycles."""
        exps = [
            ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=i)})
            for i in [1, 2, 3]
        ]
        result = apply_cycles(exps, 3, ExperimentOrder.LATIN_SQUARE, study_hash)
        # Column j should contain each experiment exactly once
        for col in range(3):
            col_ns = [result[row * 3 + col].task.dataset.n_prompts for row in range(3)]
            assert sorted(col_ns) == [1, 2, 3]

    def test_latin_square_cycles_exceed_k(self, study_hash):
        """When n_cycles > k, rows wrap around the square."""
        exps = [
            ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=i)})
            for i in [1, 2]
        ]
        result = apply_cycles(exps, 4, ExperimentOrder.LATIN_SQUARE, study_hash)
        assert len(result) == 8
        # Cycle 3 (idx 2) should equal cycle 1 (idx 0), cycle 4 = cycle 2
        row0 = [r.task.dataset.n_prompts for r in result[0:2]]
        row2 = [r.task.dataset.n_prompts for r in result[4:6]]
        assert row0 == row2

    def test_latin_square_single_experiment(self, study_hash):
        """latin_square with 1 experiment x 3 cycles = [A, A, A]."""
        exps = [ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=1)})]
        result = apply_cycles(exps, 3, ExperimentOrder.LATIN_SQUARE, study_hash)
        assert len(result) == 3
        assert all(r.task.dataset.n_prompts == 1 for r in result)

    def test_latin_square_empty(self, study_hash):
        """latin_square with 0 experiments returns empty list."""
        result = apply_cycles([], 3, ExperimentOrder.LATIN_SQUARE, study_hash)
        assert result == []


# =============================================================================
# cycle_boundary_indices() tests
# =============================================================================


class TestCycleBoundaryIndices:
    """Cycle-gap boundary positions per experiment_order.

    A cycle gap is the larger thermal-equalisation pause. Its position in the
    execution sequence depends on how apply_cycles() laid the sequence out, so
    these assert the exact indices for a small 2-config x 3-cycle matrix.
    """

    def test_sequential_2x3_gap_between_config_blocks(self):
        """Regression: sequential [A,A,A,B,B,B] gaps between blocks, at index 3 only.

        This is the confirmed-before-fix bug: the old positional modulo used
        ``index % n_unique`` (n_unique=2), which fired mid-repetition at indices
        {2, 4}. The correct sequential boundary is the config-block transition
        at index 3 (last A -> first B).
        """
        boundaries = cycle_boundary_indices(2, 3, ExperimentOrder.SEQUENTIAL)
        assert boundaries == {3}
        # Guard against the specific pre-fix behaviour ever returning.
        assert 2 not in boundaries
        assert 4 not in boundaries

    def test_interleave_2x3_gap_between_full_passes(self):
        """interleave [A,B,A,B,A,B] gaps between full passes, at indices 2 and 4."""
        boundaries = cycle_boundary_indices(2, 3, ExperimentOrder.INTERLEAVE)
        assert boundaries == {2, 4}

    def test_sequential_3x3_gaps_at_each_block_boundary(self):
        """sequential 3 configs x 3 cycles -> boundaries at 3 and 6 (not 3,6 by modulo)."""
        boundaries = cycle_boundary_indices(3, 3, ExperimentOrder.SEQUENTIAL)
        assert boundaries == {3, 6}

    @pytest.mark.parametrize(
        "order",
        [
            ExperimentOrder.INTERLEAVE,
            ExperimentOrder.REVERSE,
            ExperimentOrder.SHUFFLE,
            ExperimentOrder.LATIN_SQUARE,
        ],
    )
    def test_pass_structured_orders_gap_every_n_unique(self, order):
        """All pass-structured orders gap every n_unique items (unchanged behaviour)."""
        assert cycle_boundary_indices(2, 3, order) == {2, 4}

    def test_single_cycle_has_no_boundaries(self):
        """n_cycles=1: nothing to gap between, for every order."""
        for order in ExperimentOrder:
            assert cycle_boundary_indices(3, 1, order) == frozenset()

    def test_single_config_sequential_falls_back_to_pass_rule(self):
        """Single config [A,A,A] has no distinct blocks; each rep is a full cycle.

        Sequential must behave identically to interleave here (the sequences are
        identical), gapping between every repetition at indices 1 and 2.
        """
        seq = cycle_boundary_indices(1, 3, ExperimentOrder.SEQUENTIAL)
        inter = cycle_boundary_indices(1, 3, ExperimentOrder.INTERLEAVE)
        assert seq == inter == {1, 2}

    def test_final_boundary_never_included(self):
        """The end-of-sequence position is never a cycle boundary."""
        seq_len = 2 * 3
        assert all(i < seq_len for i in cycle_boundary_indices(2, 3, ExperimentOrder.SEQUENTIAL))
        assert all(i < seq_len for i in cycle_boundary_indices(2, 3, ExperimentOrder.INTERLEAVE))

    def test_zero_configs_empty(self):
        """Defensive: no configs -> no boundaries."""
        assert cycle_boundary_indices(0, 3, ExperimentOrder.SEQUENTIAL) == frozenset()


# =============================================================================
# count_sweep_structure() tests
# =============================================================================


class TestCountSweepStructure:
    def test_empty_sweep(self):
        assert count_sweep_structure({}) == (0, 0)

    def test_axes_only(self):
        sweep = {
            "engine": ["transformers", "vllm"],
            "transformers.engine_params.dtype": ["float16", "bfloat16"],
        }
        assert count_sweep_structure(sweep) == (2, 0)

    def test_groups_only(self):
        sweep = {
            "quant_group": [
                {"transformers.engine_params.dtype": "float16"},
                {"transformers.engine_params.dtype": "bfloat16"},
            ]
        }
        assert count_sweep_structure(sweep) == (0, 1)

    def test_mixed_axes_and_groups(self):
        sweep = {
            "engine": ["transformers", "vllm"],
            "transformers.engine_params.dtype": ["float16", "bfloat16"],
            "transformers.compilation": [
                {"transformers.llem_execution.torch_compile": True},
                {"transformers.llem_execution.torch_compile": False},
            ],
        }
        assert count_sweep_structure(sweep) == (2, 1)

    def test_scalar_counted_as_axis(self):
        sweep = {"engine": "transformers"}
        assert count_sweep_structure(sweep) == (1, 0)
