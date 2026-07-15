"""Unit tests for DockerRunner dispatch lifecycle.

All tests mock subprocess.run - no real Docker invocations.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    ENV_CONFIG_PATH,
    ENV_HF_TOKEN,
    ENV_OUTPUT_DIR,
    ENV_SAVE_TIMESERIES,
    Engine,
)
from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerImagePullError,
    DockerOOMError,
    DockerPermissionError,
    DockerTimeoutError,
)
from llenergymeasure.infra.docker_runner import DockerRunner, _env_file, _mask_secrets
from tests.conftest import make_config, make_result
from tests.unit.docker.conftest import make_subprocess_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE = "ghcr.io/llenergymeasure/vllm:1.19.0-cuda12"


def _docker_config_hash(config) -> str:
    """Return the config hash as DockerRunner computes it: from the clean config.

    DockerRunner computes the hash from the unmodified config (output_dir is no
    longer a config field). Tests that write result files must use this helper
    to match the actual file name.
    """
    from llenergymeasure.domain.experiment import compute_declared_config_hash

    return compute_declared_config_hash(config)


def _subprocess_run_with_image_cached(*run_results: MagicMock):
    """Create a subprocess.run side_effect that handles _ensure_image's inspect call.

    The first subprocess.run call is always ``docker image inspect`` from
    ``_ensure_image`` - returns exit 0 (image cached). Subsequent calls
    return the provided results in order.
    """
    results = iter(
        [make_subprocess_result(0), *run_results]
    )  # image inspect → 0, then user results
    return lambda *a, **k: next(results)


# ---------------------------------------------------------------------------
# Test 1: Success path
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_returns_experiment_result_on_success(self, tmp_path):
        """Successful container exit returns an ExperimentResult and cleans up."""
        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-abc123"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree") as mock_rmtree,
        ):
            # Write a valid result JSON before runner reads it
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned, _ts = runner.run(config)

        assert returned.experiment_id == result.experiment_id
        # Exchange dir should be cleaned up on success
        mock_rmtree.assert_called_once()

    def test_hash_independent_of_output_settings(self, tmp_path):
        """Config hash is computed from the clean config without output path influence.

        output_dir has been removed from ExperimentConfig entirely. The hash is
        deterministic and stable regardless of any output settings (which are now
        passed via env vars, not config fields).
        """
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        config = make_config()
        result = make_result()

        # Hash is the same whether computed directly or via the helper
        direct_hash = compute_declared_config_hash(config)
        helper_hash = _docker_config_hash(config)
        assert direct_hash == helper_hash

        exchange_dir = tmp_path / "llem-hashfix"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            # Write result under the clean hash - both host and container agree
            result_path = exchange_dir / f"{direct_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned, _ts = runner.run(config)

        assert returned.experiment_id == result.experiment_id


# ---------------------------------------------------------------------------
# Test 2: Container failure - image not found
# ---------------------------------------------------------------------------


class TestContainerFailure:
    def test_image_pull_error_raised_on_no_such_image(self, tmp_path):
        """Non-zero exit with 'No such image' raises DockerImagePullError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-pull"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=make_subprocess_result(
                    1, stderr="Error: No such image: ghcr.io/example:latest"
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

    def test_exchange_dir_preserved_on_failure(self, tmp_path):
        """Exchange dir is NOT cleaned up when the container fails."""
        config = make_config()
        exchange_dir = tmp_path / "llem-preserved"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=make_subprocess_result(1, stderr="Error: No such image"),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree") as mock_rmtree,
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        # rmtree must NOT be called - dir preserved for debugging
        mock_rmtree.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: OOM error
# ---------------------------------------------------------------------------


class TestOOMError:
    def test_oom_error_raised_on_exit_137(self, tmp_path):
        """Exit 137 with OOM keyword raises DockerOOMError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-oom"
        exchange_dir.mkdir()

        # First subprocess.run call is _ensure_image (docker image inspect) → return 0 (cached).
        # Second call is the actual docker run → return 137 (OOM).
        calls = iter(
            [
                make_subprocess_result(0),  # docker image inspect - image cached
                make_subprocess_result(137, stderr="OOM killer invoked: killed by container OOM"),
            ]
        )

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=lambda *a, **k: next(calls),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerOOMError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 4: Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_docker_timeout_error(self, tmp_path):
        """subprocess.TimeoutExpired is translated to DockerTimeoutError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-timeout"
        exchange_dir.mkdir()

        # First call is image inspect (return 0), second is docker run (timeout)
        results = iter(
            [
                make_subprocess_result(0),  # docker image inspect
                None,  # sentinel: raise TimeoutExpired
            ]
        )

        def fake_run(*a, **k):
            r = next(results)
            if r is None:
                raise subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=30)
            return r

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=fake_run,
            ),
        ):
            runner = DockerRunner(image=IMAGE, timeout=30)
            with pytest.raises(DockerTimeoutError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 5: Permission error
# ---------------------------------------------------------------------------


class TestPermissionError:
    def test_permission_error_raised(self, tmp_path):
        """Exit 1 with 'permission denied' raises DockerPermissionError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-perm"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    make_subprocess_result(
                        1,
                        stderr="Got permission denied while trying to connect to the Docker daemon socket",
                    )
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerPermissionError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 6: Missing result file
# ---------------------------------------------------------------------------


class TestMissingResultFile:
    def test_missing_result_raises_container_error(self, tmp_path):
        """Container exits 0 but writes no result file → DockerContainerError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-nofile"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=make_subprocess_result(0),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError, match="no result file"):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 7: Error payload from container
# ---------------------------------------------------------------------------


class TestErrorPayloadFromContainer:
    def test_error_dict_returned_when_container_writes_error_json(self, tmp_path):
        """Container exits 0 but result file contains an error payload → return dict."""
        config = make_config()
        exchange_dir = tmp_path / "llem-errpayload"
        exchange_dir.mkdir()

        error_payload = {
            "type": "RuntimeError",
            "message": "CUDA out of memory inside container",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: CUDA OOM",
        }

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=make_subprocess_result(0),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(json.dumps(error_payload), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned, _ts = runner.run(config)

        # Error payloads are returned as dicts, not ExperimentResult
        assert isinstance(returned, dict)
        assert returned["type"] == "RuntimeError"
        assert "traceback" in returned


# ---------------------------------------------------------------------------
# Test 8: Docker command structure
# ---------------------------------------------------------------------------


class TestDockerCommandStructure:
    def test_command_includes_required_flags(self, tmp_path):
        """docker run command includes --rm, --gpus all, --shm-size 8g, mounts, env vars, entrypoint."""
        config = make_config()
        exchange_dir = tmp_path / "llem-cmd"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")  # fail fast

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]

        assert "docker" in cmd
        assert "run" in cmd
        assert "--rm" in cmd
        assert "--gpus" in cmd
        assert "all" in cmd
        assert "--shm-size" in cmd
        assert "8g" in cmd

        # Volume mount: exchange_dir:{CONTAINER_EXCHANGE_DIR}
        joined = " ".join(cmd)
        assert f":{CONTAINER_EXCHANGE_DIR}" in joined

        # Config path env var
        assert any(ENV_CONFIG_PATH in arg for arg in cmd)

        # Entrypoint script (the in-container bootstrap that primes runtime
        # deps and exec's the framework entrypoint module).
        ep_idx = cmd.index("--entrypoint")
        assert cmd[ep_idx + 1] == "/llem-entry.sh"

        # Image tag
        assert IMAGE in cmd


# ---------------------------------------------------------------------------
# Test 9: HF_TOKEN propagation
# ---------------------------------------------------------------------------


class TestHFTokenPropagation:
    def test_hf_token_uses_env_file_not_cli_arg(self, tmp_path, monkeypatch):
        """HF_TOKEN is forwarded via --env-file, not as a -e CLI argument."""
        monkeypatch.setenv(ENV_HF_TOKEN, "hf_test_secret_token")
        config = make_config()
        exchange_dir = tmp_path / "llem-hftoken"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        joined = " ".join(cmd)
        # Token value must NOT appear in the command args (security requirement)
        assert "hf_test_secret_token" not in joined
        # Must use --env-file instead
        assert "--env-file" in cmd

    def test_hf_token_absent_when_not_set(self, tmp_path, monkeypatch):
        """HF_TOKEN is not added to docker command when env var is absent."""
        monkeypatch.delenv(ENV_HF_TOKEN, raising=False)
        config = make_config()
        exchange_dir = tmp_path / "llem-nohf"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        joined = " ".join(cmd)
        assert ENV_HF_TOKEN not in joined


# ---------------------------------------------------------------------------
# Test 10: DockerRunner.run() return shape
# ---------------------------------------------------------------------------


class TestRunnerMetadata:
    def test_run_returns_tuple(self, tmp_path):
        """DockerRunner.run() returns (result, ts_tmpdir) tuple."""
        config = make_config()
        result = make_result()
        exchange_dir = tmp_path / "llem-tuple"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE, source="yaml")
            returned, ts_dir = runner.run(config)

        assert returned.experiment_id == result.experiment_id
        assert ts_dir is None  # no timeseries.parquet in exchange dir


# ---------------------------------------------------------------------------
# Test 11: Cleanup warning on rmtree failure
# ---------------------------------------------------------------------------


class TestCleanupWarning:
    def test_cleanup_failure_logs_warning_not_raises(self, tmp_path, caplog):
        """shutil.rmtree failure on cleanup emits a warning but does not raise."""
        import logging

        config = make_config()
        result = make_result()
        exchange_dir = tmp_path / "llem-cleanupwarn"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.shutil.rmtree",
                side_effect=PermissionError("permission denied"),
            ),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            # Should NOT raise even though rmtree fails
            returned, _ts = runner.run(config)

        # Still returns a valid result
        assert returned.experiment_id == result.experiment_id
        # Warning was logged
        assert any("Could not remove" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Test 12: S1 security - HF_TOKEN env-file pattern
# ---------------------------------------------------------------------------


class TestHFTokenSecure:
    def test_hf_token_not_in_cmd_args(self, tmp_path, monkeypatch):
        """HF_TOKEN value never appears in docker run command args (S1 security fix)."""
        monkeypatch.setenv(ENV_HF_TOKEN, "hf_test_secret_token_12345")
        config = make_config()
        exchange_dir = tmp_path / "llem-s1-cmd"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        # Token value must not appear in any cmd element
        assert not any("hf_test_secret_token_12345" in arg for arg in cmd)
        # Token must not be passed as -e KEY=VALUE
        assert not any(ENV_HF_TOKEN in arg and "=" in arg for arg in cmd)
        # --env-file must be present
        assert "--env-file" in cmd

    def test_hf_token_env_file_content(self):
        """_env_file creates a file with KEY=VALUE format; file is deleted after context exits."""
        secrets = {ENV_HF_TOKEN: "hf_test_secret_value"}

        captured_path: list[Path] = []

        with _env_file(secrets) as path:
            assert path is not None
            captured_path.append(path)
            content = path.read_text()
            assert content == "HF_TOKEN=hf_test_secret_value\n"

        # File must be deleted after context exits
        assert not captured_path[0].exists()

    def test_env_file_without_hf_token_still_has_output_vars(self, tmp_path, monkeypatch):
        """When HF_TOKEN is absent, env-file still exists (LLEM_OUTPUT_DIR etc.) but has no HF_TOKEN."""
        monkeypatch.delenv(ENV_HF_TOKEN, raising=False)
        config = make_config()
        exchange_dir = tmp_path / "llem-s1-notoken"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        # Env-file is still used for ENV_OUTPUT_DIR and ENV_SAVE_TIMESERIES
        assert "--env-file" in cmd
        # But HF_TOKEN must not appear anywhere in the command
        joined = " ".join(cmd)
        assert ENV_HF_TOKEN not in joined

    def test_env_file_cleanup_on_failure(self):
        """Temp env-file is deleted even when an exception is raised inside the context."""
        secrets = {ENV_HF_TOKEN: "hf_cleanup_test_value"}
        captured_path: list[Path] = []

        try:
            with _env_file(secrets) as path:
                assert path is not None
                captured_path.append(path)
                # File exists inside context
                assert path.exists()
                raise RuntimeError("Simulated failure inside context")
        except RuntimeError:
            pass

        # File must be deleted even after the exception
        assert not captured_path[0].exists()

    def test_mask_secrets(self):
        """_mask_secrets replaces long secret values with *** in strings."""
        # Long values (>4 chars) are masked
        result = _mask_secrets(
            f"docker run -e {ENV_HF_TOKEN}=abc123xyz",
            {ENV_HF_TOKEN: "abc123xyz"},
        )
        assert result == f"docker run -e {ENV_HF_TOKEN}=***"

        # Short values (<=4 chars) are NOT masked - avoids false positives
        result_short = _mask_secrets(
            "some text with abcd",
            {"KEY": "abcd"},
        )
        assert "abcd" in result_short


# ---------------------------------------------------------------------------
# Test 13: multi-GPU tensorrt dispatch is NEVER wrapped in mpirun
# ---------------------------------------------------------------------------


class TestEngineDispatchEnvVars:
    """Verify the env vars passed to the in-container entrypoint script
    (LLEM_ENGINE) and the always-on entrypoint pointer.

    No engine is launched under mpirun: TensorRT-LLM's LLM API self-manages
    tensor parallelism (setting tensor_parallel_size spawns its own workers),
    so LLEM_MPI_NP is never emitted, even at TP>1. Wrapping the container in
    mpirun -n N previously ran the whole experiment on every rank and corrupted
    the trt build cache; the signal was removed entirely.
    """

    def _capture_cmd(self, config, tmp_path) -> list[str]:
        """Run DockerRunner.run() with a fake subprocess and capture the docker cmd."""
        exchange_dir = tmp_path / "llem-mpi"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        return captured_cmds[0]

    def _env_value(self, cmd: list[str], key: str) -> str | None:
        """Return the value of the first -e KEY=VALUE pair matching *key*, or None."""
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                pair = cmd[i + 1]
                if pair.startswith(f"{key}="):
                    return pair.split("=", 1)[1]
        return None

    def test_entrypoint_is_always_script(self):
        """All engines dispatch through /llem-entry.sh regardless of TP."""
        runner = DockerRunner(image=IMAGE)
        for cfg in (
            make_config(),  # transformers
            make_config(engine="vllm"),
            make_config(engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}),
            make_config(engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 4}}),
        ):
            cmd = runner._build_docker_cmd(cfg, "abc123", "/tmp/llem-test")
            ep_idx = cmd.index("--entrypoint")
            assert cmd[ep_idx + 1] == "/llem-entry.sh"

    def test_mpi_np_absent_for_tensorrt_tp2(self, tmp_path):
        """TRT-LLM at tensor_parallel_size=2 does NOT set LLEM_MPI_NP (no mpirun wrap)."""
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 2}}
        )
        cmd = self._capture_cmd(config, tmp_path)
        assert self._env_value(cmd, "LLEM_MPI_NP") is None

    def test_mpi_np_absent_for_tensorrt_tp4(self, tmp_path):
        """TRT-LLM at tensor_parallel_size=4 does NOT set LLEM_MPI_NP (no mpirun wrap)."""
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 4}}
        )
        cmd = self._capture_cmd(config, tmp_path)
        assert self._env_value(cmd, "LLEM_MPI_NP") is None

    def test_mpi_np_absent_for_tensorrt_tp1(self, tmp_path):
        """TRT-LLM with tensor_parallel_size=1 omits LLEM_MPI_NP (single-GPU path)."""
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
        )
        cmd = self._capture_cmd(config, tmp_path)
        assert self._env_value(cmd, "LLEM_MPI_NP") is None

    def test_mpi_np_absent_for_tensorrt_tp_none(self, tmp_path):
        """TRT-LLM with tensor_parallel_size=None (default) omits LLEM_MPI_NP."""
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": None}}
        )
        cmd = self._capture_cmd(config, tmp_path)
        assert self._env_value(cmd, "LLEM_MPI_NP") is None

    def test_mpi_np_absent_for_vllm(self, tmp_path):
        """vLLM dispatch never sets LLEM_MPI_NP."""
        config = make_config(engine="vllm")
        cmd = self._capture_cmd(config, tmp_path)
        assert self._env_value(cmd, "LLEM_MPI_NP") is None

    def test_mpi_np_absent_for_transformers(self, tmp_path):
        """transformers dispatch never sets LLEM_MPI_NP."""
        config = make_config()
        cmd = self._capture_cmd(config, tmp_path)
        assert self._env_value(cmd, "LLEM_MPI_NP") is None

    def test_llem_engine_matches_config(self):
        """LLEM_ENGINE env var matches config.engine (script uses it for nvidia_entrypoint routing)."""
        runner = DockerRunner(image=IMAGE)
        cases = [
            (make_config(), "transformers"),
            (make_config(engine="vllm"), "vllm"),
            (
                make_config(
                    engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
                ),
                "tensorrt",
            ),
        ]
        for cfg, expected in cases:
            cmd = runner._build_docker_cmd(cfg, "abc123", "/tmp/llem-test")
            assert self._env_value(cmd, "LLEM_ENGINE") == expected

    def test_host_uid_gid_passed(self):
        """LLEM_HOST_UID/LLEM_HOST_GID are passed so the entrypoint can chown the deps cache."""
        import os

        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")
        assert self._env_value(cmd, "LLEM_HOST_UID") == str(os.getuid())
        assert self._env_value(cmd, "LLEM_HOST_GID") == str(os.getgid())


# ---------------------------------------------------------------------------
# Test 14: Extra volume mounts
# ---------------------------------------------------------------------------


class TestExtraMounts:
    """Verify extra_mounts produce correct -v flags and TRT-LLM auto-cache-mount works."""

    def _build_cmd(self, config, tmp_path, runner: DockerRunner) -> list[str]:
        """Call _build_docker_cmd directly on the runner (no subprocess needed)."""
        return runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

    def test_extra_mounts_appear_in_docker_cmd(self, tmp_path):
        """-v flags for extra_mounts appear in command before image name."""
        config = make_config()
        runner = DockerRunner(image=IMAGE, extra_mounts=[("/host/cache", "/root/.cache")])
        cmd = self._build_cmd(config, tmp_path, runner)

        assert "-v" in cmd
        assert "/host/cache:/root/.cache" in cmd

        # Mount must appear BEFORE the image name
        mount_idx = cmd.index("/host/cache:/root/.cache")
        image_idx = cmd.index(IMAGE)
        assert mount_idx < image_idx

    def test_multiple_extra_mounts(self, tmp_path):
        """Multiple extra mounts all appear in the command."""
        config = make_config()
        runner = DockerRunner(
            image=IMAGE,
            extra_mounts=[("/host/a", "/container/a"), ("/host/b", "/container/b")],
        )
        cmd = self._build_cmd(config, tmp_path, runner)

        assert "/host/a:/container/a" in cmd
        assert "/host/b:/container/b" in cmd

    def test_no_extra_mounts_no_extra_volumes(self, tmp_path):
        """Without extra_mounts, six -v flags appear for a non-TRT engine: exchange dir, HF cache,
        package source, pyproject.toml (single-file), entrypoint script (single-file), deps cache."""
        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = self._build_cmd(config, tmp_path, runner)

        # Six -v flags: exchange dir, HF cache auto-mount, llem-src, pyproject,
        # entrypoint script, deps cache.
        v_count = cmd.count("-v")
        assert v_count == 6
        assert f"/tmp/llem-test:{CONTAINER_EXCHANGE_DIR}" in cmd
        assert any(arg.endswith(":/llem-src:ro") for arg in cmd)
        assert any(arg.endswith(":/root/.cache/huggingface") for arg in cmd)
        assert any(arg.endswith("/pyproject.toml:/llem-pyproject.toml:ro") for arg in cmd)
        assert any(arg.endswith("/container_entrypoint.sh:/llem-entry.sh:ro") for arg in cmd)
        assert any(arg.endswith(":/llem-runtime-deps") for arg in cmd)

    def test_tensorrt_auto_cache_mount(self, tmp_path):
        """TRT-LLM engine auto-mounts ~/.cache/trt-llm:/root/.cache/trt-llm."""
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
        )
        runner = DockerRunner(image=IMAGE)

        with patch(
            "llenergymeasure.infra.docker_runner.Path.home",
            return_value=Path("/home/testuser"),
        ):
            cmd = self._build_cmd(config, tmp_path, runner)

        assert "-v" in cmd
        assert "/home/testuser/.cache/trt-llm:/root/.cache/trt-llm" in cmd

    def test_tensorrt_auto_cache_mount_not_duplicated(self, tmp_path):
        """User's custom /root/.cache/trt-llm path prevents auto-mount duplication."""
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
        )
        runner = DockerRunner(
            image=IMAGE,
            extra_mounts=[("/custom/cache", "/root/.cache/trt-llm")],
        )

        with patch(
            "llenergymeasure.infra.docker_runner.Path.home",
            return_value=Path("/home/testuser"),
        ):
            cmd = self._build_cmd(config, tmp_path, runner)

        # /root/.cache/trt-llm appears as a container path exactly once
        trt_mounts = [arg for arg in cmd if ":/root/.cache/trt-llm" in arg]
        assert len(trt_mounts) == 1
        # And it's the user's custom host path, not the auto-generated one
        assert "/custom/cache:/root/.cache/trt-llm" in cmd

    def test_non_tensorrt_no_auto_cache_mount(self, tmp_path):
        """PyTorch engine does not get the TRT-LLM auto-cache mount."""
        config = make_config()  # engine="transformers"
        runner = DockerRunner(image=IMAGE)
        cmd = self._build_cmd(config, tmp_path, runner)

        assert not any("trt-llm" in arg for arg in cmd)

    def test_tensorrt_cache_path_defaults_to_mount(self, tmp_path, monkeypatch):
        """Unset LLEM_TRT_BUILD_CACHE_PATH defaults to the mounted cache dir.

        TRT-LLM's own default root is unmounted, so without this the cache would
        die with each container. The runner pins it to the bind-mount target.
        """
        monkeypatch.delenv("LLEM_TRT_BUILD_CACHE_PATH", raising=False)
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
        )
        runner = DockerRunner(image=IMAGE)
        cmd = self._build_cmd(config, tmp_path, runner)

        vals = [a for a in cmd if a.startswith("LLEM_TRT_BUILD_CACHE_PATH=")]
        assert vals == ["LLEM_TRT_BUILD_CACHE_PATH=/root/.cache/trt-llm"]

    def test_tensorrt_cache_path_host_override_wins(self, tmp_path, monkeypatch):
        """A host-set LLEM_TRT_BUILD_CACHE_PATH is forwarded last (docker -e last-wins)."""
        monkeypatch.setenv("LLEM_TRT_BUILD_CACHE_PATH", "/mnt/scratch/trt")
        config = make_config(
            engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
        )
        runner = DockerRunner(image=IMAGE)
        cmd = self._build_cmd(config, tmp_path, runner)

        # Both the mount default and the forwarded override are present; the
        # override must come later so docker's last-wins semantics pick it.
        idx_default = cmd.index("LLEM_TRT_BUILD_CACHE_PATH=/root/.cache/trt-llm")
        idx_override = cmd.index("LLEM_TRT_BUILD_CACHE_PATH=/mnt/scratch/trt")
        assert idx_default < idx_override


# ---------------------------------------------------------------------------
# Test: --entrypoint per (engine, tp_size)
# ---------------------------------------------------------------------------


class TestEntrypointPerEngine:
    """All engines dispatch through ``/llem-entry.sh`` regardless of tp_size.

    The in-container entrypoint script reads ``LLEM_ENGINE`` and performs the
    engine-conditional routing internally - nvidia_entrypoint.sh wrap for
    tensorrt - so docker_runner's --entrypoint flag is constant. No engine is
    wrapped in mpirun (the TRT-LLM LLM API self-manages tensor parallelism).
    """

    @staticmethod
    def _entrypoint(cmd: list[str]) -> str | None:
        """Return the value of the first ``--entrypoint`` flag, or None if absent."""
        try:
            idx = cmd.index("--entrypoint")
        except ValueError:
            return None
        return cmd[idx + 1]

    @pytest.mark.parametrize(
        "engine,tp_size",
        [
            (Engine.TRANSFORMERS, 1),
            (Engine.VLLM, 1),
            (Engine.TENSORRT, 1),
            (Engine.TENSORRT, 2),
            (Engine.TENSORRT, 4),
        ],
    )
    def test_entrypoint_always_script(self, engine: Engine, tp_size: int, tmp_path):
        if engine is Engine.TENSORRT:
            config = make_config(
                engine=engine, tensorrt={"engine_params": {"tensor_parallel_size": tp_size}}
            )
        else:
            config = make_config(engine=engine)

        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        assert self._entrypoint(cmd) == "/llem-entry.sh"


# ---------------------------------------------------------------------------
# Test 15: container.log persistence on failure
# ---------------------------------------------------------------------------


class TestContainerLogPersistence:
    def test_container_log_written_on_failure(self, tmp_path):
        """Non-zero exit writes stderr_text to container.log in exchange dir."""
        config = make_config()
        exchange_dir = tmp_path / "llem-logtest"
        exchange_dir.mkdir()

        stderr_content = "ERROR: CUDA out of memory\nKilled by OOM\nstack trace here"

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    make_subprocess_result(137, stderr=stderr_content)
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerOOMError):
                runner.run(config)

        log_path = exchange_dir / "container.log"
        assert log_path.exists()
        assert log_path.read_text(encoding="utf-8") == stderr_content

    def test_exchange_dir_set_on_error(self, tmp_path):
        """Error carries exchange_dir attribute for log discovery."""
        config = make_config()
        exchange_dir = tmp_path / "llem-logmsg"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    make_subprocess_result(1, stderr="some error text")
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError) as exc_info:
                runner.run(config)

            assert exc_info.value.exchange_dir == str(exchange_dir)


# ---------------------------------------------------------------------------
# Test: Timeseries parquet rescue before exchange dir cleanup
# ---------------------------------------------------------------------------


class TestTimeseriesParquetRescue:
    """DockerRunner must rescue timeseries.parquet from the exchange dir before cleanup.

    The harness inside the container writes timeseries.parquet to /run/llem
    (= exchange_dir on host). DockerRunner must move it to a temp dir and
    inject effective_config["output_dir"] pointing there, so the study
    runner's _save_and_record can find it.
    """

    def test_parquet_rescued_to_temp_dir(self, tmp_path):
        """When timeseries.parquet exists, it is rescued to a temp dir and output_dir set."""
        config = make_config()
        result = make_result(
            timeseries="timeseries.parquet",
        )

        exchange_dir = tmp_path / "llem-ts-rescue"
        exchange_dir.mkdir()
        rescue_dir = tmp_path / "llem-ts-rescued"
        rescue_dir.mkdir()

        # tempfile.mkdtemp is called twice: once for exchange dir, once for rescue dir
        mkdtemp_returns = iter([str(exchange_dir), str(rescue_dir)])

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                side_effect=lambda **kw: next(mkdtemp_returns),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.shutil.rmtree",
            ) as mock_rmtree,
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            # Simulate harness writing timeseries.parquet inside container
            ts_path = exchange_dir / "timeseries.parquet"
            ts_path.write_bytes(b"PARQUET_CONTENT")

            runner = DockerRunner(image=IMAGE)
            _returned, ts_dir = runner.run(config)

        # ts_dir must point to the rescue dir containing the parquet sidecar
        assert ts_dir == rescue_dir
        assert (rescue_dir / "timeseries.parquet").exists()
        assert (rescue_dir / "timeseries.parquet").read_bytes() == b"PARQUET_CONTENT"

        # Exchange dir must still be cleaned up
        mock_rmtree.assert_called_once()

    def test_no_parquet_no_rescue(self, tmp_path):
        """When no timeseries.parquet exists, ts_dir is None."""
        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-no-ts"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            _returned, ts_dir = runner.run(config)

        # No parquet rescue means ts_dir is None
        assert ts_dir is None

    def test_config_serialised_without_output_dir(self, tmp_path):
        """Config JSON written to exchange dir must not contain output_dir (passed via env)."""
        config = make_config()

        result = make_result()
        exchange_dir = tmp_path / "llem-cfg-check"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        # Read back the config JSON that was written for the container
        config_path = exchange_dir / f"{config_hash}_config.json"
        written_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert "output_dir" not in written_config


# ---------------------------------------------------------------------------
# Test 16: Error JSON on non-zero exit
# ---------------------------------------------------------------------------


class TestErrorJsonOnNonZeroExit:
    def test_error_json_raises_with_payload(self, tmp_path):
        """When container exits non-zero AND error JSON exists, raises DockerContainerError
        with error_payload attribute containing the structured traceback."""
        config = make_config()
        exchange_dir = tmp_path / "llem-errorjson"
        exchange_dir.mkdir()

        config_hash = _docker_config_hash(config)
        error_payload = {
            "type": "RuntimeError",
            "message": "CUDA out of memory inside container",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: CUDA OOM",
        }

        error_json_path = exchange_dir / f"{config_hash}_error.json"
        error_json_path.write_text(json.dumps(error_payload), encoding="utf-8")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    make_subprocess_result(1, stderr="some docker stderr noise")
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError) as exc_info:
                runner.run(config)

        exc = exc_info.value
        # Structured payload attached to the exception
        assert hasattr(exc, "error_payload")
        assert exc.error_payload is not None
        assert exc.error_payload["type"] == "RuntimeError"
        assert exc.error_payload["message"] == "CUDA out of memory inside container"
        assert "traceback" in exc.error_payload
        # Error message includes the Python exception info
        assert "RuntimeError" in str(exc)
        assert "CUDA out of memory" in str(exc)
        # exchange_dir preserved for debugging
        assert exc.exchange_dir == str(exchange_dir)
        # container.log still written
        assert (exchange_dir / "container.log").exists()
        assert (exchange_dir / "container.log").read_text(encoding="utf-8") == (
            "some docker stderr noise"
        )

    def test_stderr_fallback_when_no_error_json(self, tmp_path):
        """When container exits non-zero and NO error JSON exists, DockerError is raised."""
        config = make_config()
        exchange_dir = tmp_path / "llem-noerrorjson"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    make_subprocess_result(1, stderr="Error: container failed with unknown reason")
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Version mismatch warning
# ---------------------------------------------------------------------------


class TestVersionMismatchWarning:
    def test_warns_when_container_version_none(self, tmp_path, caplog):
        """Warning emitted when container result has no llenergymeasure_version."""
        import logging

        config = make_config()
        result = make_result()  # llenergymeasure_version defaults to None

        exchange_dir = tmp_path / "llem-version-none"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        assert any("rebuild Docker images" in r.message for r in caplog.records)

    def test_warns_when_container_version_differs(self, tmp_path, caplog):
        """Warning emitted when container version differs from host version."""
        import logging

        config = make_config()
        result = make_result(llenergymeasure_version="0.1.0")

        exchange_dir = tmp_path / "llem-version-diff"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        assert any("rebuild Docker images" in r.message for r in caplog.records)

    def test_no_warning_when_versions_match(self, tmp_path, caplog):
        """No warning when container version matches host version."""
        import logging

        from llenergymeasure._version import __version__

        config = make_config()
        result = make_result(llenergymeasure_version=__version__)

        exchange_dir = tmp_path / "llem-version-match"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        assert not any("rebuild Docker images" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: mount-pivot - host package source bind-mounted into container (#511)
# ---------------------------------------------------------------------------


class TestMountPivot:
    """Verify the host bind-mounts that feed the in-container entrypoint script.

    All three engines run via ``/llem-entry.sh``, which reads
    ``/llem-pyproject.toml`` to diff runtime deps, primes any missing ones
    into ``/llem-runtime-deps``, sets ``PYTHONPATH`` itself (including the
    cache + ``/llem-src``), and exec's the framework entrypoint module.
    docker_runner doesn't need to set ``PYTHONPATH`` directly; the script
    owns that step. ``PYTHONDONTWRITEBYTECODE=1`` keeps the container
    from writing root-owned ``__pycache__`` into the bind-mounted source.
    """

    @staticmethod
    def _build(engine_name: str, tmp_path) -> list[str]:
        if engine_name == "tensorrt":
            config = make_config(
                engine="tensorrt", tensorrt={"engine_params": {"tensor_parallel_size": 1}}
            )
        else:
            config = make_config(engine=engine_name)
        return DockerRunner(image=IMAGE)._build_docker_cmd(config, "abc123", "/tmp/llem-test")

    @pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
    def test_llem_src_mount_triple_present(self, engine, tmp_path):
        """Each engine's docker cmd contains the ``-v {pkg_parent}:/llem-src:ro`` triple."""
        cmd = self._build(engine, tmp_path)
        mount_args = [arg for arg in cmd if arg.endswith(":/llem-src:ro")]
        assert len(mount_args) == 1
        # The triple is ordered: -v {host}:{container}
        mount_idx = cmd.index(mount_args[0])
        assert cmd[mount_idx - 1] == "-v"

    @pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
    def test_pyproject_mount_present(self, engine, tmp_path):
        """pyproject.toml is bind-mounted at /llem-pyproject.toml for the entrypoint's dep diff."""
        cmd = self._build(engine, tmp_path)
        mount_args = [arg for arg in cmd if arg.endswith(":/llem-pyproject.toml:ro")]
        assert len(mount_args) == 1
        host_path = mount_args[0].split(":/llem-pyproject.toml:ro")[0]
        assert host_path.endswith("/pyproject.toml")

    @pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
    def test_entry_script_mount_present(self, engine, tmp_path):
        """The container entrypoint script is bind-mounted at /llem-entry.sh."""
        cmd = self._build(engine, tmp_path)
        mount_args = [arg for arg in cmd if arg.endswith(":/llem-entry.sh:ro")]
        assert len(mount_args) == 1
        host_path = mount_args[0].split(":/llem-entry.sh:ro")[0]
        assert host_path.endswith("scripts/container_entrypoint.sh")

    @pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
    def test_deps_cache_mount_present(self, engine, tmp_path):
        """The runtime-deps cache is bind-mounted at /llem-runtime-deps."""
        cmd = self._build(engine, tmp_path)
        mount_args = [arg for arg in cmd if arg.endswith(":/llem-runtime-deps")]
        assert len(mount_args) == 1

    @pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
    def test_pythonpath_not_set_at_docker_level(self, engine, tmp_path):
        """PYTHONPATH is set by the entrypoint script (which knows the cache path),
        not by docker_runner. Docker-level PYTHONPATH would interfere with the
        script's logic that prepends /llem-runtime-deps/py{N.M}/ to the path.
        """
        cmd = self._build(engine, tmp_path)
        # No -e PYTHONPATH=... should appear; the script handles it.
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                assert not cmd[i + 1].startswith("PYTHONPATH="), (
                    f"docker_runner should not set PYTHONPATH; got {cmd[i + 1]!r}"
                )

    @pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
    def test_pythondontwritebytecode_env_set(self, engine, tmp_path):
        """``PYTHONDONTWRITEBYTECODE=1`` keeps __pycache__ out of the read-only mount."""
        cmd = self._build(engine, tmp_path)
        assert "PYTHONDONTWRITEBYTECODE=1" in cmd
        idx = cmd.index("PYTHONDONTWRITEBYTECODE=1")
        assert cmd[idx - 1] == "-e"

    def test_resolve_package_parent_dir_contains_llenergymeasure(self):
        """The helper points at the directory containing the ``llenergymeasure`` package."""
        from llenergymeasure.infra.docker_runner import _resolve_package_parent_dir

        pkg_parent = _resolve_package_parent_dir()
        assert (pkg_parent / "llenergymeasure").is_dir()
        assert (pkg_parent / "llenergymeasure" / "infra" / "docker_runner.py").is_file()

    def test_mount_host_path_matches_resolver(self, tmp_path):
        """The host side of the bind-mount equals ``_resolve_package_parent_dir()``."""
        from llenergymeasure.infra.docker_runner import _resolve_package_parent_dir

        cmd = self._build("transformers", tmp_path)
        expected_mount = f"{_resolve_package_parent_dir()}:/llem-src:ro"
        assert expected_mount in cmd


# ---------------------------------------------------------------------------
# Test: reserved exchange env keys are not clobbered by the LLEM_* forward loop
# ---------------------------------------------------------------------------


class TestReservedEnvForwarding:
    """A stray host ``LLEM_*`` reserved exchange var must not override the
    intended dispatch value. Docker applies ``-e`` last-wins over ``--env-file``,
    so forwarding a host ``LLEM_CONFIG_PATH`` (etc.) would silently clobber the
    value the builder set deliberately. The forward loop must skip reserved keys.
    """

    @staticmethod
    def _forwarded_value(cmd: list[str], key: str) -> str | None:
        """Last ``-e KEY=VALUE`` value for *key* (last-wins, as docker applies)."""
        value = None
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                pair = cmd[i + 1]
                if pair.startswith(f"{key}="):
                    value = pair.split("=", 1)[1]
        return value

    def test_stray_reserved_keys_not_forwarded(self, monkeypatch):
        """Stray host reserved keys are excluded from the blanket forward loop,
        so the builder's intended exchange values win."""
        config = make_config()
        # Stray host values that would clobber the dispatch if forwarded.
        monkeypatch.setenv(ENV_CONFIG_PATH, "/host/stray/config.json")
        monkeypatch.setenv(ENV_OUTPUT_DIR, "/host/stray/out")
        monkeypatch.setenv(ENV_SAVE_TIMESERIES, "0")

        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        # CONFIG_PATH is set once, to the intended exchange-dir path - not the stray.
        config_vals = [
            cmd[i + 1].split("=", 1)[1]
            for i, arg in enumerate(cmd)
            if arg == "-e" and cmd[i + 1].startswith(f"{ENV_CONFIG_PATH}=")
        ]
        assert config_vals == [f"{CONTAINER_EXCHANGE_DIR}/abc123_config.json"]
        assert "/host/stray/config.json" not in config_vals
        # OUTPUT_DIR / SAVE_TIMESERIES travel via --env-file, never via -e here,
        # so the stray host values must not appear as -e args.
        assert self._forwarded_value(cmd, ENV_OUTPUT_DIR) is None
        assert self._forwarded_value(cmd, ENV_SAVE_TIMESERIES) is None

    def test_non_reserved_llem_var_still_forwarded(self, monkeypatch):
        """Non-reserved framework-default LLEM_* vars are still forwarded."""
        config = make_config()
        monkeypatch.setenv("LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP", "auto")

        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        assert self._forwarded_value(cmd, "LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP") == "auto"


class TestDockerGpusOverride:
    def test_llem_docker_gpus_overrides_gpus_value(self, tmp_path, monkeypatch):
        """LLEM_DOCKER_GPUS pins llem containers to specific devices (shared host)."""
        monkeypatch.setenv("LLEM_DOCKER_GPUS", "device=2")
        config = make_config()
        exchange_dir = tmp_path / "llem-gpus"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["docker", "image", "inspect"]:
                return make_subprocess_result(0)
            captured_cmds.append(cmd)
            return make_subprocess_result(1, stderr="No such image")  # fail fast

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        assert cmd[cmd.index("--gpus") + 1] == "device=2"
        assert "all" not in cmd[: cmd.index("--gpus") + 2]


# ---------------------------------------------------------------------------
# Test: NCCL_* host env vars are forwarded into the experiment container
# ---------------------------------------------------------------------------


class TestNcclEnvForwarding:
    """Host ``NCCL_*`` env vars are forwarded into the container so multi-GPU
    tuning/workaround settings (e.g. ``NCCL_P2P_DISABLE=1`` on PCIe hosts
    without functional GPU P2P) reach the engine process inside the container.
    Non-NCCL host vars are NOT forwarded by this loop.
    """

    @staticmethod
    def _e_pairs(cmd: list[str]) -> list[str]:
        """Return every ``VALUE`` following a ``-e`` flag in *cmd*."""
        return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-e" and i + 1 < len(cmd)]

    def test_nccl_vars_forwarded(self, monkeypatch):
        """Host NCCL_* vars appear as -e KEY=VALUE args in the docker command."""
        monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
        monkeypatch.setenv("NCCL_IB_DISABLE", "1")
        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        pairs = self._e_pairs(cmd)
        assert "NCCL_P2P_DISABLE=1" in pairs
        assert "NCCL_IB_DISABLE=1" in pairs

    def test_non_nccl_var_not_forwarded(self, monkeypatch):
        """A non-NCCL, non-LLEM host var must not be forwarded into the container."""
        monkeypatch.setenv("SOME_UNRELATED_VAR", "leak")
        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        assert not any("SOME_UNRELATED_VAR" in arg for arg in cmd)

    def test_nccl_vars_sorted_deterministically(self, monkeypatch):
        """NCCL_* vars are emitted in sorted key order (argv is deterministic)."""
        # Set in non-alphabetical order; the forward loop must sort them.
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        monkeypatch.setenv("NCCL_DEBUG", "INFO")
        monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        # Filter to just the three we set so a stray host NCCL_* var can't
        # perturb the assertion; their relative order must be alphabetical.
        expected = ["NCCL_DEBUG=INFO", "NCCL_P2P_DISABLE=1", "NCCL_SOCKET_IFNAME=eth0"]
        mine = [p for p in self._e_pairs(cmd) if p in set(expected)]
        assert mine == expected

    def test_empty_nccl_var_skipped(self, monkeypatch):
        """An NCCL_* var set to an empty string is not forwarded (matches LLEM_*)."""
        monkeypatch.setenv("NCCL_P2P_DISABLE", "")
        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

        assert not any(arg.startswith("NCCL_P2P_DISABLE") for arg in cmd)


# ---------------------------------------------------------------------------
# Test: config.json sidecar rescue before exchange dir cleanup
# ---------------------------------------------------------------------------


class TestConfigSidecarRescue:
    """DockerRunner must rescue config.json from the exchange dir before cleanup.

    config.json is the sole home of provenance and engine/model/methodology
    identity. The harness inside the container writes it to /run/llem
    (= exchange_dir on host) on every successful run, independent of
    save_timeseries. Previously only timeseries.parquet was rescued, so
    _cleanup_exchange_dir destroyed config.json and a successful docker run
    landed a result.json with no provenance.
    """

    def test_config_sidecar_rescued_when_no_timeseries(self, tmp_path):
        """config.json is rescued even when no timeseries.parquet was written."""
        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-cfg-exchange"
        exchange_dir.mkdir()
        rescue_dir = tmp_path / "llem-cfg-rescued"
        rescue_dir.mkdir()

        # mkdtemp: 1st = exchange dir, 2nd = artefact rescue dir (created lazily
        # when the first artefact - config.json - is found).
        mkdtemp_returns = iter([str(exchange_dir), str(rescue_dir)])

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                side_effect=lambda **kw: next(mkdtemp_returns),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree") as mock_rmtree,
        ):
            config_hash = _docker_config_hash(config)
            (exchange_dir / f"{config_hash}_result.json").write_text(
                result.model_dump_json(), encoding="utf-8"
            )
            # Container wrote config.json but no timeseries.parquet.
            (exchange_dir / "config.json").write_text(
                json.dumps({"schema_version": "2.0", "engine": "transformers"}),
                encoding="utf-8",
            )

            runner = DockerRunner(image=IMAGE)
            _returned, artefact_dir = runner.run(config)

        assert artefact_dir == rescue_dir
        assert (rescue_dir / "config.json").exists(), (
            "config.json must be rescued from the exchange dir before cleanup"
        )
        # Exchange dir must still be cleaned up on success.
        mock_rmtree.assert_called_once()

    def test_config_and_timeseries_both_rescued(self, tmp_path):
        """Both config.json and timeseries.parquet land in the single rescue dir."""
        config = make_config()
        result = make_result(timeseries="timeseries.parquet")

        exchange_dir = tmp_path / "llem-both-exchange"
        exchange_dir.mkdir()
        rescue_dir = tmp_path / "llem-both-rescued"
        rescue_dir.mkdir()

        mkdtemp_returns = iter([str(exchange_dir), str(rescue_dir)])

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                side_effect=lambda **kw: next(mkdtemp_returns),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            (exchange_dir / f"{config_hash}_result.json").write_text(
                result.model_dump_json(), encoding="utf-8"
            )
            (exchange_dir / "config.json").write_text(
                json.dumps({"schema_version": "2.0"}), encoding="utf-8"
            )
            (exchange_dir / "timeseries.parquet").write_bytes(b"PARQUET_CONTENT")

            runner = DockerRunner(image=IMAGE)
            _returned, artefact_dir = runner.run(config)

        assert artefact_dir == rescue_dir
        assert (rescue_dir / "config.json").exists()
        assert (rescue_dir / "timeseries.parquet").read_bytes() == b"PARQUET_CONTENT"

    def test_config_sidecar_lands_in_experiment_dir_with_provenance(self, tmp_path):
        """End-to-end: a successful docker run lands config.json (with provenance)
        in the experiment dir.

        Exercises the real DockerRunner rescue feeding the real _save_and_record
        move - the two halves that together close the docker provenance hole.
        """
        from unittest.mock import MagicMock

        from llenergymeasure.study.runner import _save_and_record

        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-e2e-exchange"
        exchange_dir.mkdir()
        rescue_dir = tmp_path / "llem-e2e-rescued"
        rescue_dir.mkdir()
        study_dir = tmp_path / "study"
        study_dir.mkdir()

        mkdtemp_returns = iter([str(exchange_dir), str(rescue_dir)])

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                side_effect=lambda **kw: next(mkdtemp_returns),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(make_subprocess_result(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            (exchange_dir / f"{config_hash}_result.json").write_text(
                result.model_dump_json(), encoding="utf-8"
            )
            (exchange_dir / "config.json").write_text(
                json.dumps({"schema_version": "2.0", "engine": "transformers"}),
                encoding="utf-8",
            )

            runner = DockerRunner(image=IMAGE)
            returned, artefact_dir = runner.run(config)

        # Feed the rescued dir to _save_and_record exactly as the study runner does.
        resolution_log = {"task.model": {"effective": "gpt2", "source": "yaml"}}
        manifest = MagicMock()
        result_files: list[str] = []
        _save_and_record(
            returned,
            study_dir,
            manifest,
            config_hash,
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
            ts_source_dir=artefact_dir,
            resolution_log=resolution_log,
        )

        assert len(result_files) == 1
        dest_config = Path(result_files[0]).parent / "config.json"
        assert dest_config.exists(), "config.json must land in the experiment dir"
        payload = json.loads(dest_config.read_text())
        assert payload["provenance"] == resolution_log, (
            "provenance must be folded into the materialised config.json"
        )
