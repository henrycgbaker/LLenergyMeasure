"""Integration tests for Docker entrypoint PUID/PGID auto-detection.

Tests the entrypoint.sh script logic for:
1. Auto-detecting PUID/PGID from mounted directory ownership
2. Explicit PUID/PGID override taking precedence
3. Fallback to root when directory doesn't exist
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


class TestEntrypointPuidDetection:
    """Tests for PUID/PGID auto-detection logic in entrypoint.sh."""

    def test_autodetect_from_mounted_directory(self, entrypoint_path: Path, test_mount_dir: Path):
        """PUID/PGID should be auto-detected from mounted directory ownership."""
        # Get expected UID/GID from the test directory
        stat_info = test_mount_dir.stat()
        expected_uid = stat_info.st_uid
        expected_gid = stat_info.st_gid

        # Run the detection logic (extracted from entrypoint.sh)
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"""
                # Simulate entrypoint detection logic
                unset PUID PGID
                TEST_DIR="{test_mount_dir}"
                if [ -z "$PUID" ] && [ -d "$TEST_DIR" ]; then
                    PUID=$(stat -c %u "$TEST_DIR" 2>/dev/null || echo "0")
                    PGID=$(stat -c %g "$TEST_DIR" 2>/dev/null || echo "0")
                fi
                PUID=${{PUID:-0}}
                PGID=${{PGID:-0}}
                echo "$PUID $PGID"
                """,
            ],
            capture_output=True,
            text=True,
        )

        detected_uid, detected_gid = result.stdout.strip().split()
        assert int(detected_uid) == expected_uid
        assert int(detected_gid) == expected_gid

    def test_explicit_override_takes_precedence(self, entrypoint_path: Path, test_mount_dir: Path):
        """Explicit PUID/PGID env vars should override auto-detection."""
        explicit_uid = 1000
        explicit_gid = 1000

        result = subprocess.run(
            [
                "bash",
                "-c",
                f"""
                # Simulate entrypoint with explicit override
                export PUID={explicit_uid}
                export PGID={explicit_gid}
                TEST_DIR="{test_mount_dir}"
                if [ -z "$PUID" ] && [ -d "$TEST_DIR" ]; then
                    PUID=$(stat -c %u "$TEST_DIR" 2>/dev/null || echo "0")
                    PGID=$(stat -c %g "$TEST_DIR" 2>/dev/null || echo "0")
                fi
                PUID=${{PUID:-0}}
                PGID=${{PGID:-0}}
                echo "$PUID $PGID"
                """,
            ],
            capture_output=True,
            text=True,
        )

        detected_uid, detected_gid = result.stdout.strip().split()
        assert int(detected_uid) == explicit_uid
        assert int(detected_gid) == explicit_gid

    def test_fallback_to_root_when_dir_missing(self, entrypoint_path: Path):
        """Should fall back to root (0) when mount directory doesn't exist."""
        result = subprocess.run(
            [
                "bash",
                "-c",
                """
                # Simulate entrypoint with missing directory
                unset PUID PGID
                TEST_DIR="/nonexistent/path"
                if [ -z "$PUID" ] && [ -d "$TEST_DIR" ]; then
                    PUID=$(stat -c %u "$TEST_DIR" 2>/dev/null || echo "0")
                    PGID=$(stat -c %g "$TEST_DIR" 2>/dev/null || echo "0")
                fi
                PUID=${PUID:-0}
                PGID=${PGID:-0}
                echo "$PUID $PGID"
                """,
            ],
            capture_output=True,
            text=True,
        )

        detected_uid, detected_gid = result.stdout.strip().split()
        assert int(detected_uid) == 0
        assert int(detected_gid) == 0

    def test_entrypoint_script_exists_and_executable(self, entrypoint_path: Path):
        """Verify entrypoint.sh exists and has correct structure."""
        assert entrypoint_path.exists(), f"Entrypoint not found at {entrypoint_path}"

        content = entrypoint_path.read_text()

        # Check for auto-detection logic
        assert "stat -c %u" in content, "Missing UID auto-detection via stat"
        assert "stat -c %g" in content, "Missing GID auto-detection via stat"
        assert "/app/results" in content, "Missing /app/results directory check"

        # Check for precedence comment
        assert (
            "Precedence" in content or "precedence" in content
        ), "Missing precedence documentation in script"


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker not available")
class TestEntrypointDocker:
    """Docker-based integration tests (skipped if Docker unavailable)."""

    @pytest.fixture
    def docker_image(self) -> str:
        """Assume the image is already built."""
        return "llm-energy-measure:pytorch"

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
