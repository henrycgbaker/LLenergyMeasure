"""Unit tests for the llem run CLI command.

Tests use typer.testing.CliRunner to invoke the CLI without loading models or
touching GPU hardware. All heavy operations are mocked.

Note: typer's CliRunner routes all output (stdout + stderr) to .output.
Error messages printed to sys.stderr are captured in .output for assertions.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

import llenergymeasure.cli.run as cli_run_mod
from llenergymeasure.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result() -> MagicMock:
    """Return a minimal mock ExperimentResult with required attributes."""
    from llenergymeasure.domain.experiment import ExperimentResult

    result = MagicMock(spec=ExperimentResult)
    result.experiment_id = "test-exp-001"
    result.total_energy_j = 100.0
    result.avg_tokens_per_second = 42.0
    result.duration_sec = 5.0
    result.measurement_warnings = []
    result.energy_breakdown = None
    result.energy_adjusted_j = None
    result.total_flops = 0.0
    result.latency_stats = None
    result.warmup_excluded_samples = None
    return result


def _make_mock_config() -> MagicMock:
    """Return a minimal mock ExperimentConfig."""
    from llenergymeasure.config.models import ExperimentConfig

    config = MagicMock(spec=ExperimentConfig)
    config.engine = "transformers"
    config.active_engine_params.return_value = None
    config.transformers = None  # no engine section → dtype is None (engine default)

    # task sub-model
    config.task = MagicMock()
    config.task.model = "gpt2"
    config.task.dataset = MagicMock()
    config.task.dataset.source = "aienergyscore"
    config.task.dataset.n_prompts = 100
    config.task.dataset.order = "interleaved"
    config.task.max_input_tokens = 256
    config.task.max_output_tokens = 256

    # measurement sub-model
    config.measurement = MagicMock()
    config.measurement.baseline = MagicMock()
    config.measurement.baseline.enabled = False
    return config


def _make_experiment_yaml(tmp_path: Path) -> Path:
    """Write a minimal single-experiment YAML (no sweep/experiments keys).

    The run command reads the file for study detection before delegating to
    ``load_experiment_config`` (which callers mock), so the path must exist.
    """
    path = tmp_path / "experiment.yaml"
    path.write_text("task:\n  model: gpt2\nengine: transformers\n")
    return path


def _make_capture_load() -> tuple:
    """Return (capture_fn, captured_defaults) for study routing tests.

    The capture function mimics the api load_study facade: records the
    execution_defaults it was handed and returns a MagicMock with properly
    configured study attributes.
    """
    captured: list = []

    def _capture(path, cli_overrides=None, *, execution_defaults=None):
        captured.append(execution_defaults)
        mock = MagicMock()
        mock.experiments = [MagicMock()]
        mock.study_execution.n_cycles = 1
        mock.skipped_configs = []
        return mock

    return _capture, captured


# ---------------------------------------------------------------------------
# _build_header unit tests
# ---------------------------------------------------------------------------


def test_build_header_strips_hf_org_prefix():
    """_build_header strips the HuggingFace org prefix from model name."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.task.model = "meta-llama/Llama-3.2-1B-Instruct"
    config.engine = "vllm"
    config.vllm = None  # no engine section → dtype is None (engine default)
    config.task.dataset.n_prompts = 100  # default - should not appear

    header = _build_header(config, runner_tag="container")
    assert "Llama-3.2-1B-Instruct" in header
    assert "meta-llama" not in header
    assert "[container]" in header


def test_build_header_default_dtype_omitted():
    """_build_header omits dtype when no explicit per-engine dtype is set."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.task.model = "gpt2"
    config.engine = "transformers"
    config.transformers = None  # engine default - should not appear
    config.task.dataset.n_prompts = 100  # default - should not appear

    header = _build_header(config, runner_tag="process")
    assert "bfloat16" not in header
    assert header == "gpt2 | transformers [process]"


def test_build_header_nondefault_fields_shown():
    """_build_header includes dtype (when set per-engine) and n_prompts (when non-default)."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.task.model = "gpt2"
    config.engine = "transformers"
    engine_params = MagicMock()
    engine_params.dtype = "float16"
    config.active_engine_params.return_value = engine_params
    config.task.dataset.n_prompts = 50  # non-default - should appear

    header = _build_header(config, runner_tag="process")
    assert "float16" in header
    assert "n_prompts=50" in header


# ---------------------------------------------------------------------------
# Basic flag tests
# ---------------------------------------------------------------------------


def test_run_help():
    """llem run --help exits 0 and shows the retained session flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    assert "--output" in plain
    assert "--dry-run" in plain
    assert "--resume" in plain


def test_run_version():
    """llem --version exits 0 and prints version string."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llem v" in result.output


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------


def test_run_no_args_exits_2():
    """llem run with no args (no config, no --model) exits with code 2."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )


def test_run_config_error_exits_2():
    """ConfigError raised by load_experiment_config exits with code 2."""
    from llenergymeasure.utils.exceptions import ConfigError

    with patch.object(cli_run_mod, "load_experiment_config") as mock_load:
        mock_load.side_effect = ConfigError("bad config: unknown field 'foop'")
        result = runner.invoke(app, ["run", "nonexistent.yaml"])

    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "ConfigError" in result.output


def test_run_validation_error_exits_2(tmp_path):
    """Pydantic ValidationError from a bad field value exits with code 2."""
    # "pytorh" is a misspelled engine - Pydantic will raise ValidationError
    bad_yaml = tmp_path / "experiment.yaml"
    bad_yaml.write_text("serving_mode: offline\ntask:\n  model: gpt2\nengine: pytorh\n")
    result = runner.invoke(app, ["run", str(bad_yaml)])
    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "Config validation failed" in result.output


def test_run_preflight_error_exits_1(tmp_path):
    """PreFlightError raised by run_experiment exits with code 1."""
    from llenergymeasure.utils.exceptions import PreFlightError

    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "run_experiment") as mock_run,
    ):
        mock_run.side_effect = PreFlightError("no GPU available")
        result = runner.invoke(app, ["run", str(exp_yaml)])

    assert result.exit_code == 1, (
        f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
    )
    assert "PreFlightError" in result.output


def test_run_experiment_error_exits_1(tmp_path):
    """ExperimentError raised during run exits with code 1."""
    from llenergymeasure.utils.exceptions import ExperimentError

    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "run_experiment") as mock_run,
    ):
        mock_run.side_effect = ExperimentError("inference crashed")
        result = runner.invoke(app, ["run", str(exp_yaml)])

    assert result.exit_code == 1, (
        f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
    )
    assert "ExperimentError" in result.output


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


def test_run_dry_run_exits_0(tmp_path):
    """--dry-run exits 0 and calls print_dry_run with resolved config."""
    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)
    mock_vram = {
        "weights_gb": 0.24,
        "kv_cache_gb": 0.01,
        "overhead_gb": 0.04,
        "total_gb": 0.29,
    }

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "estimate_vram", return_value=mock_vram),
        patch.object(cli_run_mod, "get_gpu_vram_gb", return_value=None),
        patch.object(cli_run_mod, "print_dry_run") as mock_print_dry,
    ):
        result = runner.invoke(app, ["run", str(exp_yaml), "--dry-run"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_print_dry.assert_called_once()


def test_run_dry_run_calls_estimate_vram(tmp_path):
    """--dry-run calls estimate_vram and get_gpu_vram_gb with the resolved config."""
    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "estimate_vram", return_value=None) as mock_vram,
        patch.object(cli_run_mod, "get_gpu_vram_gb", return_value=None) as mock_gpu_vram,
        patch.object(cli_run_mod, "print_dry_run"),
    ):
        result = runner.invoke(app, ["run", str(exp_yaml), "--dry-run"])

    assert result.exit_code == 0
    mock_vram.assert_called_once_with(mock_config)
    mock_gpu_vram.assert_called_once()


# ---------------------------------------------------------------------------
# Study-only flags ignored on single-experiment runs
# ---------------------------------------------------------------------------


def test_single_run_warns_on_study_only_flag(tmp_path):
    """--resume on a single-experiment run warns that the flag is ignored."""
    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "estimate_vram", return_value=None),
        patch.object(cli_run_mod, "get_gpu_vram_gb", return_value=None),
        patch.object(cli_run_mod, "print_dry_run"),
    ):
        result = runner.invoke(app, ["run", str(exp_yaml), "--dry-run", "--resume"])

    assert result.exit_code == 0
    out = _strip_ansi(result.output).lower()
    assert "study-only" in out
    assert "--resume" in out


def test_single_run_no_warning_without_study_flags(tmp_path):
    """A clean single-experiment run emits no study-only-flag warning."""
    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "estimate_vram", return_value=None),
        patch.object(cli_run_mod, "get_gpu_vram_gb", return_value=None),
        patch.object(cli_run_mod, "print_dry_run"),
    ):
        result = runner.invoke(app, ["run", str(exp_yaml), "--dry-run"])

    assert result.exit_code == 0
    assert "study-only" not in _strip_ansi(result.output).lower()


def test_study_run_does_not_warn_on_study_flags(tmp_path):
    """A study run with --no-lock does not emit the single-run ignored-flag warning."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  transformers.dtype: [float32, float16]\n"
    )
    mock_study_result = make_study_result()

    with (
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.api.load_study") as mock_load,
        patch("llenergymeasure.cli._preflight_display.build_preflight_panel"),
    ):
        mock_config = MagicMock()
        mock_config.experiments = [MagicMock(), MagicMock()]
        mock_config.study_execution.n_cycles = 1
        mock_config.skipped_configs = []
        mock_load.return_value = mock_config
        result = runner.invoke(app, ["run", str(study_yaml), "--no-lock"])

    assert result.exit_code == 0, f"Output: {result.output}"
    assert "study-only" not in _strip_ansi(result.output).lower()


# ---------------------------------------------------------------------------
# Quiet flag test
# ---------------------------------------------------------------------------


def test_run_quiet_flag_accepted(tmp_path):
    """--quiet suppresses step progress display (progress=None passed to run_experiment)."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "run_experiment", return_value=mock_result) as mock_run,
        patch.object(cli_run_mod, "print_result_summary"),
    ):
        result = runner.invoke(app, ["run", str(exp_yaml), "--quiet"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    # In quiet mode, progress callback must be None
    call_kwargs = mock_run.call_args
    assert call_kwargs is not None, "run_experiment was not called"
    assert call_kwargs.kwargs.get("progress") is None, "Expected progress=None in quiet mode"


# ---------------------------------------------------------------------------
# Successful run test
# ---------------------------------------------------------------------------


def test_run_success_prints_summary(tmp_path):
    """Successful run calls print_result_summary with the returned result."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "run_experiment", return_value=mock_result),
        patch.object(cli_run_mod, "print_result_summary") as mock_summary,
    ):
        result = runner.invoke(app, ["run", str(exp_yaml)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_summary.assert_called_once_with(mock_result)


# ---------------------------------------------------------------------------
# Study CLI tests (Phase 12)
# ---------------------------------------------------------------------------


def test_study_detection_with_sweep_key(tmp_path):
    """YAML with sweep: key is detected as study mode."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text("""
name: test
model: test/model
sweep:
  transformers.dtype: [float32, float16]
""")
    import yaml

    raw = yaml.safe_load(study_yaml.read_text())
    assert "sweep" in raw


def test_study_detection_with_experiments_key(tmp_path):
    """YAML with experiments: key is detected as study mode."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text("""
name: test
experiments:
  - model: test/model-a
  - model: test/model-b
""")
    import yaml

    raw = yaml.safe_load(study_yaml.read_text())
    assert "experiments" in raw


def test_removed_flags_absent_from_help():
    """Removed semantic-override flags no longer appear in llem run --help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    for flag in (
        "--model",
        "--engine",
        "--dataset",
        "--n-prompts",
        "--cycles",
        "--order",
        "--no-gaps",
        "--timeout",
        "--no-circuit-breaker",
        "--fail-fast",
        "--no-dedup",
    ):
        assert flag not in plain, f"{flag} should have been removed from run --help"


def test_removed_flag_exits_2(tmp_path):
    """A removed flag gets Typer's standard 'No such option' (exit 2), no shim."""
    exp_yaml = _make_experiment_yaml(tmp_path)
    result = runner.invoke(app, ["run", str(exp_yaml), "--cycles", "5"])
    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "No such option" in _strip_ansi(result.output)


# ---------------------------------------------------------------------------
# Study routing tests - verify CLI actually invokes run_study for study YAMLs
# ---------------------------------------------------------------------------


def test_run_study_routing_sweep_yaml(tmp_path):
    """YAML with sweep: key routes to run_study via _run_study_impl."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  transformers.dtype: [float32, float16]\n"
    )
    mock_study_result = make_study_result()

    # run_study and load_study (api facade) are lazily imported inside
    # _run_study_impl; patch at the source modules, not at llenergymeasure.cli.run
    with (
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.api.load_study") as mock_load,
        patch("llenergymeasure.cli._preflight_display.build_preflight_panel"),
    ):
        mock_config = MagicMock()
        mock_config.experiments = [MagicMock(), MagicMock()]
        mock_config.study_execution.n_cycles = 1
        mock_config.skipped_configs = []
        mock_load.return_value = mock_config
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_run.assert_called_once()


def test_run_study_routing_experiments_yaml(tmp_path):
    """YAML with experiments: key routes to run_study."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nexperiments:\n  - model: test/model-a\n    engine: transformers\n  - model: test/model-b\n    engine: transformers\n"
    )
    mock_study_result = make_study_result()

    with (
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.api.load_study") as mock_load,
        patch("llenergymeasure.cli._preflight_display.build_preflight_panel"),
    ):
        mock_config = MagicMock()
        mock_config.experiments = [MagicMock(), MagicMock()]
        mock_config.study_execution.n_cycles = 1
        mock_config.skipped_configs = []
        mock_load.return_value = mock_config
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_run.assert_called_once()


def test_run_saves_to_output_dir(tmp_path):
    """When --output CLI flag is passed, run_experiment receives output_dir."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()
    mock_result.timeseries = None
    output_dir = tmp_path / "out"
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "run_experiment", return_value=mock_result) as mock_run,
        patch.object(cli_run_mod, "print_result_summary"),
    ):
        result = runner.invoke(app, ["run", str(exp_yaml), "--output", str(output_dir)])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs.get("output_dir") == str(output_dir)


def test_run_study_cli_defaults_applied(tmp_path):
    """Study YAML without execution block receives CLI effective defaults: n_cycles=3, experiment_order=shuffle."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  transformers.dtype: [float32, float16]\n"
    )
    mock_study_result = make_study_result()
    _capture_load, captured_defaults = _make_capture_load()

    # load_study (api facade), run_study, and build_preflight_panel are all
    # lazily imported inside _run_study_impl - patch at source modules
    with (
        patch("llenergymeasure.api.load_study", side_effect=_capture_load),
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.cli._preflight_display.build_preflight_panel"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0
    assert len(captured_defaults) == 1
    defaults = captured_defaults[0]
    assert defaults is not None
    assert defaults["n_cycles"] == 3
    assert defaults["experiment_order"] == "shuffle"


def test_run_no_config_error_points_to_study_init():
    """Missing-config error exits 2 and points the user at `llem study init`."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    out = _strip_ansi(result.output)
    assert "config file is required" in out
    assert "llem study init" in out


def test_run_resume_without_config_requires_config():
    """--resume without a config still requires a config file (exit 2, study-init hint).

    Resume runs today only when the study YAML is supplied (the resume flow lives
    in the study path, which is reached only for a real config). This preserves
    that contract: `llem run --resume` alone points at `llem study init`.
    """
    result = runner.invoke(app, ["run", "--resume"])
    assert result.exit_code == 2
    out = _strip_ansi(result.output)
    assert "config file is required" in out
    assert "llem study init" in out


def test_run_engine_error_exits_1(tmp_path):
    """EngineError raised by run_experiment exits with code 1."""
    from llenergymeasure.utils.exceptions import EngineError

    mock_config = _make_mock_config()
    exp_yaml = _make_experiment_yaml(tmp_path)

    with (
        patch.object(cli_run_mod, "load_experiment_config", return_value=mock_config),
        patch.object(cli_run_mod, "run_experiment") as mock_run,
    ):
        mock_run.side_effect = EngineError("OOM during forward pass")
        result = runner.invoke(app, ["run", str(exp_yaml)])

    assert result.exit_code == 1
    assert "EngineError" in result.output


# ---------------------------------------------------------------------------
# New robustness flag tests (Phase 40.2)
# ---------------------------------------------------------------------------


def _make_study_yaml(tmp_path, content: str | None = None) -> Path:
    """Write a minimal study YAML to tmp_path and return its path."""

    study_yaml = tmp_path / "study.yaml"
    if content is None:
        content = "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  transformers.dtype: [float32, float16]\n"
    study_yaml.write_text(content)
    return study_yaml


def _make_mock_study_result():
    """Return a minimal mock StudyResult."""
    from tests.conftest import make_study_result

    return make_study_result()


def test_resume_flag_passes_resume_to_api(tmp_path):
    """--resume flag passes resume=True to run_study (API handles auto-detect)."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    mock_study_config = MagicMock()
    mock_study_config.study_design_hash = "abc123"
    mock_study_config.skipped_configs = []
    mock_study_config.experiments = [MagicMock()]
    mock_study_config.study_name = "test"
    mock_study_config.study_execution = MagicMock()
    mock_study_config.study_execution.n_cycles = 3

    with (
        patch("llenergymeasure.api.load_study", return_value=mock_study_config),
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.cli._preflight_display.build_preflight_panel"),
        patch(
            "llenergymeasure.api.find_resumable_study",
            return_value=tmp_path / "fake-study",
        ),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--resume"])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["resume"] is True


def test_resume_dir_flag_passes_path_to_api(tmp_path):
    """--resume-dir passes the explicit directory to run_study."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    mock_study_config = MagicMock()
    mock_study_config.study_design_hash = "abc123"
    mock_study_config.skipped_configs = []
    mock_study_config.experiments = [MagicMock()]
    mock_study_config.study_name = "test"
    mock_study_config.study_execution = MagicMock()
    mock_study_config.study_execution.n_cycles = 3

    explicit_dir = tmp_path / "my_study"
    explicit_dir.mkdir()
    (explicit_dir / "manifest.json").write_text("{}")

    with (
        patch("llenergymeasure.api.load_study", return_value=mock_study_config),
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.cli._preflight_display.build_preflight_panel"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--resume-dir", str(explicit_dir)])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["resume_dir"] == explicit_dir


def test_session_flags_visible_in_help():
    """The retained session flags appear in llem run --help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    for flag in ("--resume", "--resume-dir", "--no-lock", "--skip-preflight", "--output"):
        assert flag in plain, f"{flag} should be present in run --help"
