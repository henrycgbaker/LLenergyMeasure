"""GPU-free unit tests for Docker pre-flight checks.

All subprocess.run and shutil.which calls are mocked — no Docker or GPU needed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.exceptions import (
    DockerPreFlightError,
    LLEMError,
    PreFlightError,
)
from llenergymeasure.infra.docker_preflight import (
    _CUDA_COMPAT_URL,
    _DOCKER_INSTALL_URL,
    _NVIDIA_TOOLKIT_INSTALL_URL,
    _PROBE_IMAGE,
    run_docker_preflight,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Create a mock subprocess.CompletedProcess-like object."""
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


def _make_successful_probe(gpu_name: str = "Tesla A100", driver: str = "525.89.02") -> MagicMock:
    """Probe result with GPU name and driver version (tier 2 success)."""
    return _make_subprocess_result(returncode=0, stdout=f"{gpu_name}, {driver}\n")


def _make_nvidia_smi_result(driver: str = "525.89.02") -> MagicMock:
    """Host nvidia-smi result returning a driver version string."""
    return _make_subprocess_result(returncode=0, stdout=f"{driver}\n")


# ---------------------------------------------------------------------------
# TestSkipPreflight
# ---------------------------------------------------------------------------


class TestSkipPreflight:
    def test_skip_true_returns_immediately(self, caplog: pytest.LogCaptureFixture) -> None:
        """run_docker_preflight(skip=True) returns without running any checks."""
        with patch("shutil.which") as mock_which, patch("subprocess.run") as mock_run:
            with caplog.at_level(logging.WARNING):
                run_docker_preflight(skip=True)

            # No subprocess or which calls should be made
            mock_which.assert_not_called()
            mock_run.assert_not_called()

    def test_skip_true_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """run_docker_preflight(skip=True) logs a warning message."""
        with (
            patch("shutil.which"),
            patch("subprocess.run"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_preflight"),
        ):
            run_docker_preflight(skip=True)

        assert any(
            "skip" in r.message.lower() or "skipped" in r.message.lower() for r in caplog.records
        )

    def test_skip_default_is_false(self) -> None:
        """Default skip=False means checks run (patch to avoid actual Docker calls)."""
        with patch("shutil.which", return_value=None), pytest.raises(DockerPreFlightError):
            run_docker_preflight()  # skip defaults to False


# ---------------------------------------------------------------------------
# TestTier1DockerCheck
# ---------------------------------------------------------------------------


class TestTier1DockerCheck:
    def test_docker_not_on_path_raises(self) -> None:
        """Missing docker binary → DockerPreFlightError with 'Docker not found'."""
        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", return_value=None),
            pytest.raises(DockerPreFlightError) as exc_info,
        ):
            run_docker_preflight()

        msg = str(exc_info.value)
        assert "Docker not found" in msg

    def test_docker_not_on_path_includes_install_url(self) -> None:
        """Error message contains the Docker install URL."""
        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", return_value=None),
            pytest.raises(DockerPreFlightError) as exc_info,
        ):
            run_docker_preflight()

        assert _DOCKER_INSTALL_URL in str(exc_info.value)

    def test_docker_not_on_path_no_container_probe(self) -> None:
        """When Docker is missing, the container probe must never run."""
        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", return_value=None),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            with pytest.raises(DockerPreFlightError):
                run_docker_preflight()

            # No docker run should have been invoked
            for c in mock_run.call_args_list:
                args = c[0][0] if c[0] else c[1].get("args", [])
                if isinstance(args, list):
                    assert args[:2] != ["docker", "run"], (
                        "Container probe must not run when Docker is missing"
                    )


# ---------------------------------------------------------------------------
# TestTier1NvidiaToolkit
# ---------------------------------------------------------------------------


class TestTier1NvidiaToolkit:
    def _which_docker_only(self, name: str) -> str | None:
        """Return a path for docker only; all NVIDIA tools return None."""
        if name == "docker":
            return "/usr/bin/docker"
        return None

    def test_no_nvidia_toolkit_raises(self) -> None:
        """All NVIDIA toolkit binaries absent → DockerPreFlightError."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_docker_only,
            ),
            pytest.raises(DockerPreFlightError) as exc_info,
        ):
            run_docker_preflight()

        msg = str(exc_info.value)
        assert "NVIDIA Container Toolkit" in msg

    def test_no_nvidia_toolkit_includes_install_url(self) -> None:
        """Error message contains the NVIDIA toolkit install URL."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_docker_only,
            ),
            pytest.raises(DockerPreFlightError) as exc_info,
        ):
            run_docker_preflight()

        assert _NVIDIA_TOOLKIT_INSTALL_URL in str(exc_info.value)

    def test_nvidia_ctk_found_passes_tier1_toolkit_check(self) -> None:
        """nvidia-ctk on PATH counts as NVIDIA toolkit present."""

        def which_docker_and_ctk(name: str) -> str | None:
            if name in ("docker", "nvidia-ctk"):
                return f"/usr/bin/{name}"
            return None

        # Patch subprocess.run to simulate successful container probe
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=which_docker_and_ctk,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            # nvidia-smi probe (tier 1) + container probe (tier 2)
            mock_run.side_effect = [
                _make_nvidia_smi_result(),  # host nvidia-smi
                _make_successful_probe(),  # container probe
            ]
            # Should not raise (toolkit is present via nvidia-ctk)
            run_docker_preflight()


# ---------------------------------------------------------------------------
# TestTier1HostNvidiaSmi
# ---------------------------------------------------------------------------


class TestTier1HostNvidiaSmi:
    def _which_all_present(self, name: str) -> str | None:
        """All binaries present including nvidia-smi."""
        known = {"docker", "nvidia-container-runtime", "nvidia-smi"}
        return f"/usr/bin/{name}" if name in known else None

    def _which_no_nvidia_smi(self, name: str) -> str | None:
        """Docker and toolkit present, but no nvidia-smi."""
        known = {"docker", "nvidia-container-runtime"}
        return f"/usr/bin/{name}" if name in known else None

    def test_missing_nvidia_smi_warns_but_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing host nvidia-smi logs warning but does not hard-block."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_no_nvidia_smi,
            ),
            patch(
                "llenergymeasure.infra.docker_preflight.subprocess.run",
                return_value=_make_successful_probe(),
            ),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_preflight"),
        ):
            # Should NOT raise — missing nvidia-smi is warn-only
            run_docker_preflight()

        # Warning must be logged
        assert any(
            "nvidia-smi" in r.message.lower() or "remote" in r.message.lower()
            for r in caplog.records
        )

    def test_nvidia_smi_driver_version_parsed(self) -> None:
        """nvidia-smi output is parsed and driver version extracted."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result("535.104.05"),  # host nvidia-smi
                _make_successful_probe(driver="535.104.05"),  # container probe
            ]
            # Should complete without error
            run_docker_preflight()

            # First call should be nvidia-smi for driver version
            first_call_args = mock_run.call_args_list[0][0][0]
            assert "nvidia-smi" in first_call_args

    def test_nvidia_smi_timeout_warns_and_continues(self, caplog: pytest.LogCaptureFixture) -> None:
        """nvidia-smi timeout is a warn-only condition, not a hard error."""
        import subprocess as _subprocess

        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
                _make_successful_probe(),  # container probe succeeds
            ]
            with caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_preflight"):
                run_docker_preflight()  # Should not raise

        assert any("timed out" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# TestTier1MultipleFailures
# ---------------------------------------------------------------------------


class TestTier1MultipleFailures:
    def test_docker_and_toolkit_both_missing_single_error_with_two_items(self) -> None:
        """Both Docker and NVIDIA CT missing → single DockerPreFlightError listing both."""
        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", return_value=None),
            pytest.raises(DockerPreFlightError) as exc_info,
        ):
            run_docker_preflight()

        msg = str(exc_info.value)
        # Should mention both failures
        assert "Docker not found" in msg
        assert "NVIDIA Container Toolkit" in msg
        # Should use numbered list format
        assert "1." in msg
        assert "2." in msg

    def test_multiple_failures_count_in_message(self) -> None:
        """Error header says N issue(s) found with correct count."""
        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", return_value=None),
            pytest.raises(DockerPreFlightError) as exc_info,
        ):
            run_docker_preflight()

        msg = str(exc_info.value)
        assert "2 issue(s) found" in msg


# ---------------------------------------------------------------------------
# TestTier2GPUVisibility
# ---------------------------------------------------------------------------


class TestTier2GPUVisibility:
    def _which_all_present(self, name: str) -> str | None:
        known = {"docker", "nvidia-container-runtime", "nvidia-smi"}
        return f"/usr/bin/{name}" if name in known else None

    def test_container_probe_fails_gpu_not_accessible(self) -> None:
        """Container probe with returncode=125 and 'could not select device driver' → DockerPreFlightError."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result(),  # host nvidia-smi
                _make_subprocess_result(
                    returncode=125,
                    stderr='could not select device driver "" with capabilities: [[gpu]]',
                ),  # container probe fails
            ]
            with pytest.raises(DockerPreFlightError) as exc_info:
                run_docker_preflight()

        msg = str(exc_info.value)
        assert "GPU not accessible" in msg or "NVIDIA Container Toolkit" in msg

    def test_container_probe_fails_includes_toolkit_url(self) -> None:
        """GPU access error message includes NVIDIA toolkit install URL."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result(),
                _make_subprocess_result(returncode=125, stderr="could not select device driver"),
            ]
            with pytest.raises(DockerPreFlightError) as exc_info:
                run_docker_preflight()

        assert _NVIDIA_TOOLKIT_INSTALL_URL in str(exc_info.value)

    def test_container_probe_timeout_raises(self) -> None:
        """Container probe timeout → DockerPreFlightError."""
        import subprocess as _subprocess

        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result(),
                _subprocess.TimeoutExpired(cmd="docker run", timeout=30),
            ]
            with pytest.raises(DockerPreFlightError) as exc_info:
                run_docker_preflight()

        msg = str(exc_info.value)
        assert "timed out" in msg.lower() or "timeout" in msg.lower()


# ---------------------------------------------------------------------------
# TestTier2GPUVisibilitySuccess
# ---------------------------------------------------------------------------


class TestTier2GPUVisibilitySuccess:
    def _which_all_present(self, name: str) -> str | None:
        known = {"docker", "nvidia-container-runtime", "nvidia-smi"}
        return f"/usr/bin/{name}" if name in known else None

    def test_all_checks_pass_no_exception(self) -> None:
        """All tier 1 and tier 2 checks pass → no exception raised."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result("525.89.02"),
                _make_successful_probe(),
            ]
            # Must not raise
            run_docker_preflight()

    def test_all_checks_pass_no_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Silent on success — no ERROR or WARNING logs when all checks pass."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result(),
                _make_successful_probe(),
            ]
            with caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_preflight"):
                run_docker_preflight()

        # No WARNING or ERROR level records from the preflight logger
        bad_records = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == "llenergymeasure.infra.docker_preflight"
        ]
        assert bad_records == [], f"Unexpected warning/error logs: {bad_records}"


# ---------------------------------------------------------------------------
# TestTier2CUDADriverCompat
# ---------------------------------------------------------------------------


class TestTier2CUDADriverCompat:
    def _which_all_present(self, name: str) -> str | None:
        known = {"docker", "nvidia-container-runtime", "nvidia-smi"}
        return f"/usr/bin/{name}" if name in known else None

    def test_cuda_driver_mismatch_raises(self) -> None:
        """Container probe failure with CUDA/driver error → DockerPreFlightError with versions."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result("470.57.02"),  # old host driver
                _make_subprocess_result(
                    returncode=1,
                    stderr="Failed to initialize NVML: Driver/library version mismatch\ncuda error",
                ),
            ]
            with pytest.raises(DockerPreFlightError) as exc_info:
                run_docker_preflight()

        msg = str(exc_info.value)
        assert "cuda" in msg.lower() or "driver" in msg.lower()

    def test_cuda_driver_mismatch_includes_compat_url(self) -> None:
        """CUDA/driver error message includes the CUDA compatibility URL."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result("470.57.02"),
                _make_subprocess_result(
                    returncode=1,
                    stderr="CUDA version incompatible with driver",
                ),
            ]
            with pytest.raises(DockerPreFlightError) as exc_info:
                run_docker_preflight()

        assert _CUDA_COMPAT_URL in str(exc_info.value)

    def test_driver_error_keyword_triggers_cuda_path(self) -> None:
        """Stderr containing 'driver' (not 'cuda') still triggers CUDA compat error path."""
        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which",
                side_effect=self._which_all_present,
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result("470.57.02"),
                _make_subprocess_result(
                    returncode=1,
                    stderr="Failed to initialize NVML: Driver/library version mismatch",
                ),
            ]
            with pytest.raises(DockerPreFlightError) as exc_info:
                run_docker_preflight()

        # Should get the CUDA compat URL path, not the generic GPU access path
        assert _CUDA_COMPAT_URL in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestTier2SkippedWhenTier1Fails
# ---------------------------------------------------------------------------


class TestTier2SkippedWhenTier1Fails:
    def test_tier2_not_invoked_when_docker_absent(self) -> None:
        """When Docker is missing (tier 1 failure), container probe must never run."""
        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", return_value=None),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            with pytest.raises(DockerPreFlightError):
                run_docker_preflight()

            # Verify no 'docker run' subprocess calls were made
            for c in mock_run.call_args_list:
                args = c[0][0] if c[0] else []
                if isinstance(args, list) and len(args) >= 2:
                    assert not (args[0] == "docker" and args[1] == "run"), (
                        "Tier 2 container probe must not be invoked when tier 1 fails"
                    )

    def test_tier2_not_invoked_when_toolkit_absent(self) -> None:
        """When NVIDIA toolkit is missing (tier 1 failure), container probe must not run."""

        def which_docker_only(name: str) -> str | None:
            return "/usr/bin/docker" if name == "docker" else None

        with (
            patch(
                "llenergymeasure.infra.docker_preflight.shutil.which", side_effect=which_docker_only
            ),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            with pytest.raises(DockerPreFlightError):
                run_docker_preflight()

            for c in mock_run.call_args_list:
                args = c[0][0] if c[0] else []
                if isinstance(args, list) and len(args) >= 2:
                    assert not (args[0] == "docker" and args[1] == "run"), (
                        "Tier 2 must be skipped when tier 1 fails"
                    )


# ---------------------------------------------------------------------------
# TestInheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_docker_preflight_error_is_preflight_error(self) -> None:
        """DockerPreFlightError must inherit from PreFlightError."""
        assert issubclass(DockerPreFlightError, PreFlightError)

    def test_docker_preflight_error_is_llem_error(self) -> None:
        """DockerPreFlightError must inherit from LLEMError."""
        assert issubclass(DockerPreFlightError, LLEMError)

    def test_docker_preflight_error_can_be_caught_as_preflight_error(self) -> None:
        """DockerPreFlightError is catchable via PreFlightError."""
        exc = DockerPreFlightError("test")
        assert isinstance(exc, PreFlightError)
        assert isinstance(exc, LLEMError)

    def test_docker_preflight_error_can_be_raised_and_caught(self) -> None:
        """Raise DockerPreFlightError, catch as PreFlightError."""
        with pytest.raises(PreFlightError):
            raise DockerPreFlightError("pre-flight failed: Docker not found")

    def test_docker_preflight_error_message_preserved(self) -> None:
        """Exception message is accessible via str()."""
        msg = "Docker pre-flight failed: 1 issue(s) found\n  1. Docker not found on PATH"
        exc = DockerPreFlightError(msg)
        assert str(exc) == msg


# ---------------------------------------------------------------------------
# TestProbeImageConstant
# ---------------------------------------------------------------------------


class TestProbeImageConstant:
    def test_probe_image_is_string(self) -> None:
        assert isinstance(_PROBE_IMAGE, str)
        assert len(_PROBE_IMAGE) > 0

    def test_probe_uses_correct_image(self) -> None:
        """Container probe must use the declared _PROBE_IMAGE constant."""

        def which_all(name: str) -> str | None:
            known = {"docker", "nvidia-container-runtime", "nvidia-smi"}
            return f"/usr/bin/{name}" if name in known else None

        with (
            patch("llenergymeasure.infra.docker_preflight.shutil.which", side_effect=which_all),
            patch("llenergymeasure.infra.docker_preflight.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _make_nvidia_smi_result(),
                _make_successful_probe(),
            ]
            run_docker_preflight()

            # Find the docker run call
            docker_run_call = None
            for c in mock_run.call_args_list:
                args = c[0][0] if c[0] else []
                if isinstance(args, list) and len(args) >= 2 and args[0] == "docker":
                    docker_run_call = args
                    break

            assert docker_run_call is not None, "Expected a docker run call"
            assert _PROBE_IMAGE in docker_run_call, (
                f"Expected probe image {_PROBE_IMAGE!r} in docker run args: {docker_run_call}"
            )
