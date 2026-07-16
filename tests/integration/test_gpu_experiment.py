"""GPU integration tests -- M1 exit criteria validation.

These tests run on real GPU hardware (A100) inside Docker containers.
They validate the complete pipeline: config -> preflight -> inference -> energy -> result.

Run: pytest tests/integration/ -m gpu -v
Requires: Docker with --gpus all, gpt2 model access
"""

from pathlib import Path

import pytest

_REPLAY_DIR = Path("/app/results/replay")


def _save_replay(result, model: str, engine: str) -> None:
    """Save ExperimentResult JSON as a replay fixture for offline unit tests."""
    _REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    slug = model.replace("/", "-").lower()
    filename = f"{slug}_{engine}_v{result.schema_version}.json"
    (_REPLAY_DIR / filename).write_text(result.model_dump_json(indent=2), encoding="utf-8")


@pytest.mark.gpu
class TestM1ExitCriteria:
    """Validates M1 success criteria with a real GPU experiment."""

    def test_run_experiment_gpt2_transformers(self, tmp_path):
        """M1 primary exit criterion: llem run --model gpt2 --engine transformers
        produces a valid ExperimentResult.

        Validates STU-05: single experiment runs in-process (no subprocess).
        """
        from llenergymeasure import ExperimentConfig, ExperimentResult, run_experiment
        from llenergymeasure.config.models import DatasetConfig

        config = ExperimentConfig(
            task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=5)},  # small for speed
            engine="transformers",
        )
        result = run_experiment(config)

        # Core result assertions
        assert isinstance(result, ExperimentResult)
        from tests.conftest import EXPERIMENT_SCHEMA_VERSION

        assert result.schema_version == EXPERIMENT_SCHEMA_VERSION
        assert result.measurement_config_hash  # non-empty string
        assert len(result.measurement_config_hash) == 16

        # Energy values populated (non-zero on real GPU)
        assert result.total_energy_j > 0
        assert result.avg_energy_per_token_j > 0

        # Throughput values populated
        assert result.avg_tokens_per_second > 0
        assert result.total_tokens > 0
        assert result.total_inference_time_sec > 0

        # FLOPs populated
        assert result.total_flops > 0

        # Save replay fixture for offline unit tests
        _save_replay(result, "gpt2", "transformers")

    def test_output_files_written(self, tmp_path):
        """Timeseries parquet file written under output_dir."""
        from llenergymeasure import ExperimentConfig, run_experiment
        from llenergymeasure.config.models import DatasetConfig

        config = ExperimentConfig(
            task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=5)},
            engine="transformers",
        )
        # Direct all output under tmp_path; run_experiment creates a timestamped
        # study subdir there, with the result + timeseries sidecar nested inside
        # per-experiment directories (results/persistence.save_result layout).
        run_experiment(config, output_dir=tmp_path)

        # At least one entry created under the output dir (the study subdir).
        assert list(tmp_path.iterdir()), f"no output written under {tmp_path}"
        # Timeseries parquet sidecar written somewhere in the study tree
        # (save_timeseries defaults True; NVML telemetry is collected on a real GPU).
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) >= 1, (
            f"expected a timeseries parquet under {tmp_path}, "
            f"found: {[str(p) for p in tmp_path.rglob('*')]}"
        )

    def test_cli_run_produces_valid_output(self, tmp_path):
        """llem run <experiment.yaml> via CLI produces valid output."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        exp_yaml = tmp_path / "experiment.yaml"
        exp_yaml.write_text("task:\n  model: gpt2\nengine: transformers\n")

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                str(exp_yaml),
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0

    def test_cli_config_shows_gpu_info(self):
        """llem config shows GPU and engine information."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "GPU" in result.output or "gpu" in result.output

    def test_cli_version(self):
        """llem --version prints version string."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        from llenergymeasure import __version__

        assert __version__ in result.output or "llem" in result.output.lower()
