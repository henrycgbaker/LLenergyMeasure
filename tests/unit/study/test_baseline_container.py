"""Tests for the host-side baseline container dispatch helper.

All tests patch ``subprocess.Popen`` at the module boundary - nothing actually
talks to Docker.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.ssot import ENV_BASELINE_SPEC_PATH, ENV_ENGINE, ENV_ENTRY_MODULE
from llenergymeasure.study import baseline_container

_MODULE = "llenergymeasure.study.baseline_container"


def _write_result(exchange_dir: Path, power_w: float = 42.59) -> None:
    (exchange_dir / "baseline_result.json").write_text(
        json.dumps(
            {
                "power_w": power_w,
                "timestamp": time.time(),
                "gpu_indices": [0],
                "sample_count": 288,
                "duration_sec": 30.0,
                "mode": "measure",
            }
        ),
        encoding="utf-8",
    )


def _make_fake_popen(
    lines: list[str],
    returncode: int = 0,
    on_open: Callable[[], object] | None = None,
) -> Callable[..., MagicMock]:
    """Build a MagicMock that quacks like a Popen object for our iterator.

    - ``process.stdout`` iterates ``lines`` (must include trailing newlines).
    - ``process.wait`` returns ``returncode``.
    - ``on_open`` lets a test simulate the container writing files on start
      (e.g. baseline_result.json or baseline_error.json).
    """

    def _factory(cmd, **kwargs):
        if on_open is not None:
            on_open()
        mock = MagicMock()
        mock.stdout = iter(lines)
        mock.stderr = None
        mock.returncode = returncode

        def _wait(timeout=None):
            mock.returncode = returncode
            return returncode

        mock.wait = _wait
        mock.kill = MagicMock()
        return mock

    return _factory


class TestBuildBaselineDockerCmd:
    def test_cmd_contains_mode_env_and_gpu_filter(self, tmp_path: Path):
        cmd = baseline_container.build_baseline_docker_cmd(
            image="ghcr.io/foo/bar:v1",
            exchange_dir=str(tmp_path),
            gpu_indices=[0, 2],
            engine="vllm",
        )
        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "--gpus" in cmd
        # env vars
        assert any(f"{ENV_BASELINE_SPEC_PATH}=" in part for part in cmd)
        assert any("CUDA_VISIBLE_DEVICES=0,2" in part for part in cmd)
        # image is the LAST arg (the entrypoint script invokes the module itself)
        assert cmd[-1] == "ghcr.io/foo/bar:v1"
        # default --gpus request is "all"
        assert cmd[cmd.index("--gpus") + 1] == "all"

    def test_cmd_honours_llem_docker_gpus(self, tmp_path: Path, monkeypatch):
        """LLEM_DOCKER_GPUS overrides the --gpus value (shared-host pinning)."""
        monkeypatch.setenv("LLEM_DOCKER_GPUS", "device=2")
        cmd = baseline_container.build_baseline_docker_cmd(
            image="ghcr.io/foo/bar:v1",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="vllm",
        )
        assert cmd[cmd.index("--gpus") + 1] == "device=2"

    def test_cmd_scopes_gpus_from_config_indices(self, tmp_path: Path, monkeypatch):
        """config_gpu_indices scopes the baseline --gpus to the same physical
        devices as the experiment container (single -> device=N)."""
        monkeypatch.delenv("LLEM_DOCKER_GPUS", raising=False)
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="vllm",
            config_gpu_indices=[2],
        )
        assert cmd[cmd.index("--gpus") + 1] == "device=2"

    def test_cmd_scopes_gpus_from_config_indices_multi(self, tmp_path: Path, monkeypatch):
        """Multi config indices are quoted for docker's CSV parser."""
        monkeypatch.delenv("LLEM_DOCKER_GPUS", raising=False)
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0, 1],
            engine="vllm",
            config_gpu_indices=[2, 3],
        )
        assert cmd[cmd.index("--gpus") + 1] == '"device=2,3"'

    def test_env_overrides_config_gpu_indices(self, tmp_path: Path, monkeypatch):
        """LLEM_DOCKER_GPUS still wins over config_gpu_indices for the baseline."""
        monkeypatch.setenv("LLEM_DOCKER_GPUS", "device=5")
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="vllm",
            config_gpu_indices=[2, 3],
        )
        assert cmd[cmd.index("--gpus") + 1] == "device=5"

    def test_cmd_makes_package_importable(self, tmp_path: Path):
        """The baseline command must mount the host package + route through the
        entrypoint script so the upstream engine image can import the package.

        Without this, every Docker baseline run fails with ModuleNotFoundError:
        upstream engine images do not ship the llenergymeasure package.
        """
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="vllm",
        )
        # Package source bind-mount (makes the package importable via PYTHONPATH).
        # The package dir is mounted at the nested target so /llem-src exposes
        # only llenergymeasure, never a host site-packages sibling.
        assert any(arg.endswith(":/llem-src/llenergymeasure:ro") for arg in cmd)
        # Dispatch routes through the shared in-container bootstrap script.
        ep_idx = cmd.index("--entrypoint")
        assert cmd[ep_idx + 1] == "/llem-entry.sh"
        # The script exec's the baseline module (not the experiment one).
        assert f"{ENV_ENTRY_MODULE}=llenergymeasure.entrypoints.baseline_measure" in cmd
        # Engine is propagated so the script can route tensorrt correctly.
        assert f"{ENV_ENGINE}=vllm" in cmd
        # The old, broken trailing "python3 -m ..." form must be gone.
        assert "python3" not in cmd

    def test_cmd_tensorrt_engine_propagated(self, tmp_path: Path):
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="tensorrt",
        )
        assert f"{ENV_ENGINE}=tensorrt" in cmd

    def test_cmd_empty_gpu_indices(self, tmp_path: Path):
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[],
            engine="transformers",
        )
        assert any("CUDA_VISIBLE_DEVICES=" in part for part in cmd)

    def test_nccl_vars_forwarded(self, tmp_path: Path, monkeypatch):
        """Host NCCL_* vars are forwarded into the baseline container, matching
        the experiment path (e.g. NCCL_P2P_DISABLE=1 on PCIe hosts without P2P)."""
        monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
        monkeypatch.setenv("NCCL_IB_DISABLE", "1")
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0, 1],
            engine="vllm",
        )
        assert "NCCL_P2P_DISABLE=1" in cmd
        assert "NCCL_IB_DISABLE=1" in cmd

    def test_non_nccl_var_not_forwarded(self, tmp_path: Path, monkeypatch):
        """A non-NCCL host var must not be forwarded into the baseline container."""
        monkeypatch.setenv("SOME_UNRELATED_VAR", "leak")
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="vllm",
        )
        assert not any("SOME_UNRELATED_VAR" in part for part in cmd)

    def test_nccl_vars_sorted(self, tmp_path: Path, monkeypatch):
        """NCCL_* vars are emitted in sorted key order (deterministic argv)."""
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        monkeypatch.setenv("NCCL_DEBUG", "INFO")
        monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[0],
            engine="vllm",
        )
        # Filter to the three we set so a stray host NCCL_* var can't perturb
        # the assertion; their relative order must be alphabetical.
        expected = ["NCCL_DEBUG=INFO", "NCCL_P2P_DISABLE=1", "NCCL_SOCKET_IFNAME=eth0"]
        mine = [p for p in cmd if p in set(expected)]
        assert mine == expected


class TestParseStageLine:
    def test_non_marker_returns_none(self):
        assert baseline_container.parse_stage_line("hello world") is None
        assert baseline_container.parse_stage_line("") is None

    def test_bare_stage(self):
        parsed = baseline_container.parse_stage_line("[llem.baseline] stage=container_ready")
        assert parsed == ("container_ready", {})

    def test_stage_with_kv(self):
        parsed = baseline_container.parse_stage_line(
            "[llem.baseline] stage=sampling_done power_w=42.68 samples=289 duration=30.00"
        )
        assert parsed is not None
        name, kv = parsed
        assert name == "sampling_done"
        assert kv == {"power_w": "42.68", "samples": "289", "duration": "30.00"}

    def test_malformed_tokens_dropped(self):
        parsed = baseline_container.parse_stage_line(
            "[llem.baseline] stage=cuda_primed junk_no_equals power=5.0"
        )
        assert parsed is not None
        _, kv = parsed
        assert kv == {"power": "5.0"}


class TestRunBaselineContainerSuccess:
    def test_run_success_returns_baseline_cache(self, tmp_path: Path, monkeypatch):
        # Force mkdtemp to land inside tmp_path so the test can write the result file.
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        popen_factory = _make_fake_popen(
            lines=[],
            returncode=0,
            on_open=lambda: _write_result(exchange_dir, power_w=42.6),
        )

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=30.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert result is not None
        assert result.power_w == pytest.approx(42.6)
        assert result.gpu_indices == [0]

    def test_exchange_dir_cleaned_on_success(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        popen_factory = _make_fake_popen(
            lines=[],
            returncode=0,
            on_open=lambda: _write_result(exchange_dir),
        )

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory):
            baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert not exchange_dir.exists()

    def test_stage_markers_forwarded_to_callback(self, tmp_path: Path, monkeypatch):
        """Streamed stage markers reach the on_stage callback in order with
        monotonic elapsed times and parsed key-value tags."""
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        stage_lines = [
            "[llem.baseline] stage=container_ready\n",
            "Some unrelated stdout line\n",
            "[llem.baseline] stage=cuda_primed\n",
            "[llem.baseline] stage=sampling_started duration=30.0\n",
            "[llem.baseline] stage=sampling_done power_w=42.68 samples=289 duration=30.00\n",
            "[llem.baseline] stage=result_written\n",
        ]
        popen_factory = _make_fake_popen(
            lines=stage_lines,
            returncode=0,
            on_open=lambda: _write_result(exchange_dir),
        )

        events: list[tuple[str, dict[str, str]]] = []
        elapsed_seen: list[float] = []

        def on_stage(name: str, elapsed: float, kv: dict[str, str]) -> None:
            events.append((name, kv))
            elapsed_seen.append(elapsed)

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory):
            baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=30.0,
                gpu_indices=[0],
                engine="vllm",
                on_stage=on_stage,
            )

        names = [e[0] for e in events]
        assert names == [
            "container_ready",
            "cuda_primed",
            "sampling_started",
            "sampling_done",
            "result_written",
        ]
        # sampling_done carries the measurement summary.
        sampling_done = next(kv for name, kv in events if name == "sampling_done")
        assert sampling_done == {
            "power_w": "42.68",
            "samples": "289",
            "duration": "30.00",
        }
        # Elapsed values are monotonically non-decreasing - a mock may fire
        # them in the same tick, so we allow equality.
        assert elapsed_seen == sorted(elapsed_seen)

    def test_callback_exception_does_not_abort_run(self, tmp_path: Path, monkeypatch):
        """A broken on_stage callback must not crash the baseline container."""
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        popen_factory = _make_fake_popen(
            lines=["[llem.baseline] stage=container_ready\n"],
            returncode=0,
            on_open=lambda: _write_result(exchange_dir),
        )

        def broken(*_args, **_kwargs):
            raise RuntimeError("callback exploded")

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=30.0,
                gpu_indices=[0],
                engine="vllm",
                on_stage=broken,
            )

        assert result is not None  # measurement still succeeded


class TestRunBaselineContainerFailure:
    def test_nonzero_exit_reads_error_json(self, tmp_path: Path, monkeypatch, caplog):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _write_error():
            (exchange_dir / "baseline_error.json").write_text(
                json.dumps(
                    {
                        "type": "RuntimeError",
                        "message": "NVML unavailable",
                        "traceback": "...",
                    }
                ),
                encoding="utf-8",
            )

        popen_factory = _make_fake_popen(
            lines=["nvml error\n"],
            returncode=1,
            on_open=_write_error,
        )

        with (
            caplog.at_level("WARNING"),
            patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory),
        ):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert result is None
        assert "NVML unavailable" in caplog.text
        # Exchange dir preserved for post-mortem
        assert exchange_dir.exists()

    def test_missing_result_returns_none(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        popen_factory = _make_fake_popen(lines=[], returncode=0)

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert result is None

    def test_timeout_returns_none(self, tmp_path: Path, monkeypatch, caplog):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _factory(cmd, **kwargs):
            mock = MagicMock()

            def _never_ending():
                # Simulate a wedged container: block indefinitely in stdout.
                # We yield one line then sleep - but our loop checks the
                # timeout every iteration, so we just yield nothing and
                # let the outer timeout check trigger. Instead, raise
                # TimeoutExpired on wait().
                return
                yield  # pragma: no cover

            mock.stdout = iter([])
            mock.stderr = None

            def _wait(timeout=None):
                raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1.0)

            mock.wait = _wait
            mock.kill = MagicMock()
            mock.returncode = None
            return mock

        with (
            caplog.at_level("WARNING"),
            patch(f"{_MODULE}.subprocess.Popen", side_effect=_factory),
        ):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert result is None
        assert "timed out" in caplog.text.lower()

    def test_malformed_result_returns_none(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _write_bad():
            (exchange_dir / "baseline_result.json").write_text(
                "not valid json{{{", encoding="utf-8"
            )

        popen_factory = _make_fake_popen(
            lines=[],
            returncode=0,
            on_open=_write_bad,
        )

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=popen_factory):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert result is None

    def test_docker_binary_missing_returns_none(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        with patch(f"{_MODULE}.subprocess.Popen", side_effect=FileNotFoundError("docker")):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
                engine="vllm",
            )

        assert result is None
        # FileNotFoundError path cleans up (no diagnostic value in keeping dir)
        assert not exchange_dir.exists()
