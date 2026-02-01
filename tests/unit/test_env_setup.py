"""Tests for env_setup module."""

import os
from unittest.mock import patch

from llenergymeasure.config.env_setup import ensure_env_file


class TestEnvFileCreation:
    """Tests for .env file generation."""

    def test_ensure_env_file_creates_when_missing(self, tmp_path):
        """Creates .env when it doesn't exist."""
        env_file = ensure_env_file(tmp_path)

        assert env_file.exists()
        assert env_file == tmp_path / ".env"

        # Verify content
        content = env_file.read_text()
        assert "PUID=" in content
        assert "PGID=" in content

    def test_ensure_env_file_skips_when_exists(self, tmp_path):
        """Skips creation if .env already exists (never overwrites)."""
        # Create existing .env
        existing_env = tmp_path / ".env"
        existing_content = "CUSTOM_VAR=value\nPUID=9999\nPGID=9999\n"
        existing_env.write_text(existing_content)

        env_file = ensure_env_file(tmp_path)

        # Should return path but not modify (both PUID and PGID present)
        assert env_file == existing_env
        assert env_file.read_text() == existing_content

    def test_ensure_env_file_content_format(self, tmp_path):
        """Generated .env has correct format."""
        env_file = ensure_env_file(tmp_path)
        content = env_file.read_text()

        # Check structure
        assert "PUID=" in content
        assert "PGID=" in content
        assert "Auto-generated" in content

        # Verify values are numeric (UID/GID)
        lines = content.strip().split("\n")
        puid_line = next(line for line in lines if line.startswith("PUID="))
        pgid_line = next(line for line in lines if line.startswith("PGID="))

        puid_value = puid_line.split("=")[1]
        pgid_value = pgid_line.split("=")[1]

        assert puid_value.isdigit()
        assert pgid_value.isdigit()

    def test_ensure_env_file_uses_current_user(self, tmp_path):
        """Generated .env uses current user's UID/GID."""
        env_file = ensure_env_file(tmp_path)
        content = env_file.read_text()

        expected_puid = os.getuid()
        expected_pgid = os.getgid()

        assert f"PUID={expected_puid}" in content
        assert f"PGID={expected_pgid}" in content


class TestProjectRootInference:
    """Tests for project root detection."""

    def test_ensure_env_file_infers_project_root_pyproject(self, tmp_path):
        """Infers project root from pyproject.toml."""
        # Create nested directory structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").touch()

        subdir = project_root / "src" / "package"
        subdir.mkdir(parents=True)

        # Change to subdir and call without explicit root
        with patch("pathlib.Path.cwd", return_value=subdir):
            env_file = ensure_env_file()

        # Should create .env in project root, not subdir
        assert env_file == project_root / ".env"
        assert env_file.exists()

    def test_ensure_env_file_infers_project_root_git(self, tmp_path):
        """Infers project root from .git directory."""
        # Create nested directory structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        subdir = project_root / "tests" / "unit"
        subdir.mkdir(parents=True)

        # Change to subdir and call without explicit root
        with patch("pathlib.Path.cwd", return_value=subdir):
            env_file = ensure_env_file()

        # Should create .env in project root, not subdir
        assert env_file == project_root / ".env"
        assert env_file.exists()

    def test_ensure_env_file_explicit_root(self, tmp_path):
        """Uses explicit project_root when provided."""
        env_file = ensure_env_file(tmp_path)

        assert env_file == tmp_path / ".env"
        assert env_file.exists()
