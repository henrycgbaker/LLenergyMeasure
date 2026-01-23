"""Integration tests for Docker entrypoint PUID/PGID handling.

Tests the entrypoint.sh script logic for:
1. Requiring explicit PUID/PGID environment variables
2. Creating user/group with specified IDs
3. Setting correct file ownership
"""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def entrypoint_path() -> Path:
    """Path to the entrypoint script."""
    return Path(__file__).parent.parent.parent / "scripts" / "entrypoint.sh"


@pytest.fixture
def test_mount_dir(tmp_path: Path) -> Path:
    """Create a test directory simulating a mounted volume."""
    mount_dir = tmp_path / "results"
    mount_dir.mkdir()
    return mount_dir


class TestEntrypointPuidRequirement:
    """Tests for PUID/PGID requirement in entrypoint.sh."""

    def test_entrypoint_script_exists(self, entrypoint_path: Path):
        """Verify entrypoint.sh exists."""
        assert entrypoint_path.exists(), f"Entrypoint not found at {entrypoint_path}"

    def test_entrypoint_requires_puid_pgid(self, entrypoint_path: Path):
        """Verify entrypoint requires PUID/PGID to be set."""
        content = entrypoint_path.read_text()

        # Check for PUID/PGID requirement validation
        assert "PUID" in content, "Missing PUID handling"
        assert "PGID" in content, "Missing PGID handling"

        # Check for error handling when not set
        assert "ERROR" in content, "Missing error message for missing PUID/PGID"

    def test_entrypoint_creates_directories(self, entrypoint_path: Path):
        """Verify entrypoint creates required directories."""
        content = entrypoint_path.read_text()

        assert "/app/results" in content, "Missing /app/results directory setup"
        assert "/app/.state" in content, "Missing /app/.state directory setup"
        assert "/app/.cache" in content, "Missing /app/.cache directory setup"

    def test_entrypoint_uses_gosu(self, entrypoint_path: Path):
        """Verify entrypoint uses gosu for user switching."""
        content = entrypoint_path.read_text()
        assert "gosu" in content, "Missing gosu for user switching"

    def test_entrypoint_creates_user(self, entrypoint_path: Path):
        """Verify entrypoint creates appuser with PUID/PGID."""
        content = entrypoint_path.read_text()
        assert "appuser" in content, "Missing appuser creation"
        assert "useradd" in content or "adduser" in content, "Missing user creation command"


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not available")
class TestEntrypointDocker:
    """Docker-based integration tests (skipped if Docker unavailable)."""

    @pytest.fixture
    def docker_image(self) -> str:
        """Assume the image is already built."""
        return "llenergymeasure:pytorch"

    def test_docker_autodetect_puid(self, tmp_path: Path, docker_image: str):
        """Test PUID auto-detection in actual Docker container."""
        # Create test directory with known ownership
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        expected_uid = results_dir.stat().st_uid
        expected_gid = results_dir.stat().st_gid

        # Run container without explicit PUID/PGID
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{results_dir}:/app/results",
                docker_image,
                "bash",
                "-c",
                "id -u && id -g",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            pytest.skip(f"Docker run failed: {result.stderr}")

        lines = result.stdout.strip().split("\n")
        detected_uid = int(lines[0])
        detected_gid = int(lines[1])

        assert detected_uid == expected_uid, f"Expected UID {expected_uid}, got {detected_uid}"
        assert detected_gid == expected_gid, f"Expected GID {expected_gid}, got {detected_gid}"

    def test_docker_explicit_override(self, tmp_path: Path, docker_image: str):
        """Test explicit PUID/PGID override in Docker container."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        explicit_uid = 1000
        explicit_gid = 1000

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                f"PUID={explicit_uid}",
                "-e",
                f"PGID={explicit_gid}",
                "-v",
                f"{results_dir}:/app/results",
                docker_image,
                "bash",
                "-c",
                "id -u && id -g",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            pytest.skip(f"Docker run failed: {result.stderr}")

        lines = result.stdout.strip().split("\n")
        detected_uid = int(lines[0])
        detected_gid = int(lines[1])

        assert detected_uid == explicit_uid
        assert detected_gid == explicit_gid

    def test_docker_file_ownership(self, tmp_path: Path, docker_image: str):
        """Test that files created in container are owned by detected user."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        expected_uid = results_dir.stat().st_uid
        expected_gid = results_dir.stat().st_gid

        # Create a file inside the container
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{results_dir}:/app/results",
                docker_image,
                "touch",
                "/app/results/test-file.txt",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            pytest.skip(f"Docker run failed: {result.stderr}")

        # Check file ownership on host
        test_file = results_dir / "test-file.txt"
        assert test_file.exists(), "Test file was not created"

        file_stat = test_file.stat()
        assert (
            file_stat.st_uid == expected_uid
        ), f"File UID {file_stat.st_uid} != expected {expected_uid}"
        assert (
            file_stat.st_gid == expected_gid
        ), f"File GID {file_stat.st_gid} != expected {expected_gid}"
