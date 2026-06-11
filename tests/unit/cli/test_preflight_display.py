"""Unit tests for the CLI pre-flight display (panel + text summary).

GPU-free: exercises format_preflight_summary and build_preflight_panel by
rendering against StudyConfig fixtures and asserting on the rendered text.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from llenergymeasure.cli._preflight_display import (
    build_preflight_panel,
    format_preflight_summary,
)
from llenergymeasure.config.grid import SkippedConfig
from llenergymeasure.config.models import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    StudyConfig,
)
from llenergymeasure.config.ssot import Engine

# =============================================================================
# format_preflight_summary() tests
# =============================================================================


def _make_study_config(
    n_configs: int = 4,
    n_cycles: int = 3,
    experiment_order: str = "interleave",
    study_hash: str = "abc123def456abcd",
    skipped_configs: list | None = None,
) -> StudyConfig:
    """Helper: build a StudyConfig with the given parameters."""
    experiments = [
        ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=i + 1)})
        for i in range(n_configs * n_cycles)
    ]
    return StudyConfig(
        experiments=experiments,
        study_execution=ExecutionConfig(n_cycles=n_cycles, experiment_order=experiment_order),
        study_design_hash=study_hash,
        skipped_configs=skipped_configs or [],
    )


class TestFormatPreflightSummary:
    def test_basic_format(self):
        """Pre-flight string shows config count, cycle count, total runs, order."""
        sc = _make_study_config(n_configs=4, n_cycles=3, experiment_order="interleave")
        summary = format_preflight_summary(sc)
        assert "4 configs x 3 cycles = 12 runs" in summary
        assert "Order: interleave" in summary

    def test_hash_displayed(self):
        """study_design_hash appears in the summary line."""
        sc = _make_study_config(study_hash="deadbeef01234567")
        summary = format_preflight_summary(sc)
        assert "deadbeef01234567" in summary

    def test_no_skipped_no_warning(self):
        """Summary with no skipped configs has no Skipping or WARNING lines."""
        sc = _make_study_config()
        summary = format_preflight_summary(sc)
        assert "Skipping" not in summary
        assert "WARNING" not in summary

    def test_with_skipped_shows_skip_line(self):
        """When skipped_configs populated, Skipping line with reasons appears."""
        skipped = [
            {
                "raw_config": {"engine": "transformers", "transformers": {"dtype": "float32"}},
                "reason": "cross-validation failed",
                "short_label": f"{Engine.TRANSFORMERS}, float32",
                "errors": [],
            }
        ]
        sc = _make_study_config(n_configs=3, n_cycles=1, skipped_configs=skipped)
        summary = format_preflight_summary(sc)
        assert "Skipping 1/" in summary
        assert f"{Engine.TRANSFORMERS}, float32" in summary
        assert "cross-validation failed" in summary

    def test_high_skip_rate_warning(self):
        """WARNING shown when >50% of generated configs were skipped."""
        # 1 valid config, 2 skipped → 67% skip rate
        skipped = [
            {
                "raw_config": {"engine": "vllm", "vllm": {"dtype": "float16"}},
                "reason": "error A",
                "short_label": "vllm, float16",
                "errors": [],
            },
            {
                "raw_config": {"engine": "vllm", "vllm": {"dtype": "bfloat16"}},
                "reason": "error B",
                "short_label": "vllm, bfloat16",
                "errors": [],
            },
        ]
        sc = _make_study_config(n_configs=1, n_cycles=1, skipped_configs=skipped)
        summary = format_preflight_summary(sc)
        assert "WARNING" in summary
        assert "67%" in summary or "sweep" in summary.lower()

    def test_low_skip_rate_no_warning(self):
        """No WARNING when <50% of generated configs were skipped."""
        # 4 valid, 1 skipped → 20% skip rate
        skipped = [
            {
                "raw_config": {"engine": "transformers", "transformers": {"dtype": "float32"}},
                "reason": "validation error",
                "short_label": f"{Engine.TRANSFORMERS}, float32",
                "errors": [],
            }
        ]
        sc = _make_study_config(n_configs=4, n_cycles=1, skipped_configs=skipped)
        summary = format_preflight_summary(sc)
        assert "Skipping 1/" in summary
        assert "WARNING" not in summary

    def test_skipped_list_argument_takes_precedence(self):
        """If skipped list of SkippedConfig passed, uses it instead of skipped_configs."""
        skipped_obj = SkippedConfig(
            raw_config={"engine": "transformers", "transformers": {"dtype": "float16"}},
            reason="via argument",
        )
        # StudyConfig has empty skipped_configs
        sc = _make_study_config(n_configs=2, n_cycles=1)
        summary = format_preflight_summary(sc, skipped=[skipped_obj])
        assert "via argument" in summary

    def test_single_cycle_format(self):
        """1 config x 1 cycle = 1 run shown correctly."""
        sc = _make_study_config(n_configs=1, n_cycles=1, experiment_order="sequential")
        summary = format_preflight_summary(sc)
        assert "1 configs x 1 cycles = 1 runs" in summary
        assert "Order: sequential" in summary


# =============================================================================
# build_preflight_panel() tests
# =============================================================================


def _render_panel(study_config: StudyConfig, width: int = 100, **kwargs: object) -> str:
    """Helper: render a build_preflight_panel() output to a plain-text string."""
    panel = build_preflight_panel(study_config, **kwargs)  # type: ignore[arg-type]
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=width)
    console.print(panel)
    return buf.getvalue()


def _make_panel_study_config(
    models: list[str] | None = None,
    engines: list[str] | None = None,
    dtypes: list[str] | None = None,
    n_cycles: int = 1,
    experiment_order: str = "sequential",
    study_name: str = "test-study",
    study_hash: str = "abc123def456abcd",
    runners: dict | None = None,
) -> StudyConfig:
    """Build a StudyConfig for panel tests with varying fields per experiment."""
    models = models or ["gpt2"]
    engines = engines or ["transformers"]
    dtypes = dtypes or ["bfloat16"]

    # Build one experiment per combination (then replicate for cycles)
    experiments = []
    for model in models:
        for engine in engines:
            for dt in dtypes:
                # transformers nests dtype under engine_params (generated shape);
                # vllm/tensorrt keep dtype flat on their hand-written configs.
                section = (
                    {"engine_params": {"dtype": dt}} if engine == "transformers" else {"dtype": dt}
                )
                experiments.append(
                    ExperimentConfig(task={"model": model}, engine=engine, **{engine: section})
                )

    # Replicate for cycles
    all_exps = experiments * n_cycles
    return StudyConfig(
        experiments=all_exps,
        study_name=study_name,
        study_execution=ExecutionConfig(n_cycles=n_cycles, experiment_order=experiment_order),
        study_design_hash=study_hash,
        runners=runners,
    )


class TestBuildPreflightPanel:
    def test_panel_contains_study_name_in_title(self):
        """Panel border title contains the study name."""
        sc = _make_panel_study_config(study_name="test-study")
        output = _render_panel(sc)
        assert "Study: test-study" in output

    def test_panel_metadata_experiments_plural(self):
        """Panel shows n configs x n cycles = n runs (plural form)."""
        sc = _make_panel_study_config(
            models=["gpt2", "gpt2-xl"], n_cycles=3, experiment_order="interleave"
        )
        output = _render_panel(sc)
        assert "2 configs x 3 cycles = 6 runs" in output

    def test_panel_metadata_experiments_singular(self):
        """Panel shows 1 config x 1 cycle = 1 run (singular form)."""
        sc = _make_panel_study_config(models=["gpt2"], n_cycles=1, experiment_order="sequential")
        output = _render_panel(sc)
        assert "1 config x 1 cycle = 1 run" in output

    def test_panel_pluralisation_singular(self):
        """1 config x 1 cycle = 1 run (all singular)."""
        sc = _make_panel_study_config(models=["gpt2"], n_cycles=1)
        output = _render_panel(sc)
        assert "1 config x 1 cycle = 1 run" in output

    def test_panel_pluralisation_plural(self):
        """2 configs x 3 cycles = 6 runs (all plural)."""
        sc = _make_panel_study_config(
            models=["gpt2", "gpt2-xl"], n_cycles=3, experiment_order="sequential"
        )
        output = _render_panel(sc)
        assert "2 configs x 3 cycles = 6 runs" in output

    def test_panel_metadata_order(self):
        """Panel shows cycle order in Order row."""
        sc = _make_panel_study_config(n_cycles=2, experiment_order="interleave")
        output = _render_panel(sc)
        assert "interleave" in output

    def test_panel_metadata_engines_with_runners(self):
        """Panel shows engine with runner mode in Backends section."""
        sc = _make_panel_study_config(
            models=["gpt2", "gpt2"],
            engines=["transformers", "vllm"],
            runners={"transformers": "local", "vllm": "docker"},
        )
        output = _render_panel(sc)
        assert "Engines" in output
        assert "transformers" in output
        assert "local" in output
        assert "vllm" in output
        assert "docker" in output

    def test_panel_metadata_engines_default_local(self):
        """Panel shows 'local' for engines when runners is None."""
        sc = _make_panel_study_config(engines=["transformers"])
        output = _render_panel(sc)
        assert "transformers" in output
        assert "local" in output

    def test_panel_metadata_dataset(self):
        """Panel shows dataset name in Dataset row."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        assert "aienergyscore" in output

    def test_panel_metadata_energy(self):
        """Panel shows energy sampler value in the Workload section."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        # Label comes from json_schema_extra display_label ("Sampler") not hardcoded string
        assert "Sampler" in output

    def test_panel_swept_model_shows_in_workload(self):
        """Swept model field appears in Workload with all values listed."""
        sc = _make_panel_study_config(models=["gpt2", "gpt2-xl"])
        output = _render_panel(sc)
        assert "gpt2" in output
        assert "gpt2-xl" in output

    def test_panel_hash_displayed(self):
        """Panel contains the study design hash."""
        sc = _make_panel_study_config(study_hash="deadbeef01234567")
        output = _render_panel(sc)
        assert "deadbeef01234567" in output

    def test_panel_unnamed_study(self):
        """Panel with no study name shows 'unnamed' in title."""
        exps = [ExperimentConfig(task={"model": "gpt2"})]
        sc = StudyConfig(
            experiments=exps,
            study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
        )
        output = _render_panel(sc)
        assert "Study: unnamed" in output

    def test_panel_multiple_engines_sorted(self):
        """Multiple engines are sorted alphabetically."""
        exps = [
            ExperimentConfig(task={"model": "gpt2"}, engine="vllm"),
            ExperimentConfig(task={"model": "gpt2"}, engine="transformers"),
        ]
        sc = StudyConfig(
            experiments=exps,
            study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
        )
        output = _render_panel(sc)
        # Both backends appear
        assert "transformers" in output
        assert "vllm" in output

    def test_panel_has_workload_section(self):
        """Panel contains 'Workload' section header."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        assert "Workload" in output

    def test_panel_has_engines_section(self):
        """Panel contains 'Backends' section header (renamed from 'Runners')."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        assert "Engines" in output
        assert "Runners" not in output

    def test_panel_no_experimental_section(self):
        """Panel no longer contains 'Experimental' section."""
        sc = _make_panel_study_config()
        output = _render_panel(sc)
        assert "Experimental" not in output

    def test_panel_experiment_order_label(self):
        """Panel shows 'Experiment order' (not 'Cycle order')."""
        sc = _make_panel_study_config(experiment_order="shuffle")
        output = _render_panel(sc)
        assert "Experiment order" in output
        assert "shuffle" in output

    def test_panel_workload_shows_model(self):
        """Workload section contains model value when model is constant."""
        sc = _make_panel_study_config(models=["gpt2"])
        output = _render_panel(sc)
        assert "gpt2" in output

    def test_panel_swept_workload_annotated_with_plus(self):
        """A swept workload field (e.g. model) gets a '+' annotation in Workload."""
        sc = _make_panel_study_config(models=["gpt2", "gpt2-xl"])
        output = _render_panel(sc)
        assert "gpt2" in output
        assert "gpt2-xl" in output
        # "+" annotation for swept workload fields
        assert "+" in output

    def test_panel_sweep_summary_multiple_configs(self):
        """Sweep section shows dimension count and config count from sweep."""
        sc = _make_panel_study_config(models=["gpt2", "gpt2-xl"], n_cycles=2)
        output = _render_panel(sc)
        assert "Sweep" in output
        assert "2 configs from sweep" in output

    def test_panel_no_sweep_section_single_config(self):
        """Sweep section is hidden when there is only one config."""
        sc = _make_panel_study_config(models=["gpt2"], n_cycles=1)
        output = _render_panel(sc)
        # Single config = no sweep section (but "Sweep" may appear in other contexts)
        # Check that the sweep summary line is not present
        assert "unique config" not in output

    def test_panel_sweep_with_axes_and_groups(self):
        """Sweep section shows axes and groups when provided."""
        sc = _make_panel_study_config(models=["gpt2", "gpt2-xl"])
        output = _render_panel(sc, sweep_axes=3, sweep_groups=2)
        assert "3 axes . 2 groups" in output
        assert "2 configs from sweep" in output

    def test_panel_sweep_axes_only(self):
        """Sweep section shows axes only when no groups."""
        sc = _make_panel_study_config(models=["gpt2", "gpt2-xl"])
        output = _render_panel(sc, sweep_axes=4, sweep_groups=0)
        assert "4 axes" in output
        assert "groups" not in output
