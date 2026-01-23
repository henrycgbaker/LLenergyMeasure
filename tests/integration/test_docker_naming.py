"""Integration tests for Docker image and service naming.

Verifies that Docker configuration uses the correct 'llenergymeasure' naming
convention after the rename from 'llm-energy-measure'.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Path to the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def docker_compose_path(project_root: Path) -> Path:
    """Path to docker-compose.yml."""
    return project_root / "docker-compose.yml"


@pytest.fixture
def dockerfiles_dir(project_root: Path) -> Path:
    """Path to docker directory."""
    return project_root / "docker"


class TestDockerComposeNaming:
    """Tests for docker-compose.yml naming conventions."""

    def test_compose_file_exists(self, docker_compose_path: Path) -> None:
        """Verify docker-compose.yml exists."""
        assert docker_compose_path.exists(), "docker-compose.yml not found"

    def test_base_image_name(self, docker_compose_path: Path) -> None:
        """Verify base image uses llenergymeasure-base name."""
        content = yaml.safe_load(docker_compose_path.read_text())
        services = content.get("services", {})

        base_service = services.get("base", {})
        image = base_service.get("image", "")

        assert (
            image == "llenergymeasure-base:latest"
        ), f"Expected 'llenergymeasure-base:latest', got '{image}'"

    def test_backend_image_names(self, docker_compose_path: Path) -> None:
        """Verify all backend services use llenergymeasure:* naming."""
        content = yaml.safe_load(docker_compose_path.read_text())
        services = content.get("services", {})

        backends = ["pytorch", "vllm", "tensorrt"]
        for backend in backends:
            if backend in services:
                image = services[backend].get("image", "")
                expected = f"llenergymeasure:{backend}"
                assert (
                    image == expected
                ), f"Service '{backend}' has image '{image}', expected '{expected}'"

    def test_dev_image_names(self, docker_compose_path: Path) -> None:
        """Verify dev services use llenergymeasure:*-dev naming."""
        content = yaml.safe_load(docker_compose_path.read_text())
        services = content.get("services", {})

        dev_backends = ["pytorch-dev", "vllm-dev", "tensorrt-dev"]
        for backend in dev_backends:
            if backend in services:
                image = services[backend].get("image", "")
                expected = f"llenergymeasure:{backend}"
                assert (
                    image == expected
                ), f"Service '{backend}' has image '{image}', expected '{expected}'"

    def test_base_image_arg_in_backends(self, docker_compose_path: Path) -> None:
        """Verify backend services reference correct base image in build args."""
        content = yaml.safe_load(docker_compose_path.read_text())
        services = content.get("services", {})

        backends = ["pytorch", "vllm", "tensorrt", "pytorch-dev", "vllm-dev", "tensorrt-dev"]
        for backend in backends:
            if backend in services:
                build = services[backend].get("build", {})
                args = build.get("args", {})
                base_image = args.get("BASE_IMAGE", "")

                if base_image:
                    assert base_image == "llenergymeasure-base:latest", (
                        f"Service '{backend}' has BASE_IMAGE '{base_image}', "
                        "expected 'llenergymeasure-base:latest'"
                    )

    def test_no_old_naming(self, docker_compose_path: Path) -> None:
        """Verify no references to old 'llm-energy-measure' naming."""
        content = docker_compose_path.read_text()

        # Check for old image names
        old_patterns = [
            r"llm-energy-measure:",
            r"llm-energy-measure-base",
            r"llm_energy_measure",
        ]

        for pattern in old_patterns:
            matches = re.findall(pattern, content)
            assert not matches, f"Found old naming pattern '{pattern}' in docker-compose.yml"

    def test_named_volumes_use_lem_prefix(self, docker_compose_path: Path) -> None:
        """Verify named volumes use 'lem-' prefix."""
        content = yaml.safe_load(docker_compose_path.read_text())
        volumes = content.get("volumes", {})

        expected_volumes = ["hf-cache", "trt-engine-cache", "experiment-state"]
        for vol in expected_volumes:
            if vol in volumes:
                vol_config = volumes[vol]
                if isinstance(vol_config, dict):
                    name = vol_config.get("name", "")
                    assert name.startswith(
                        "lem-"
                    ), f"Volume '{vol}' name '{name}' should start with 'lem-'"


class TestDockerfileNaming:
    """Tests for Dockerfile naming conventions."""

    def test_dockerfiles_exist(self, dockerfiles_dir: Path) -> None:
        """Verify expected Dockerfiles exist."""
        expected = [
            "Dockerfile.base",
            "Dockerfile.pytorch",
            "Dockerfile.vllm",
            "Dockerfile.tensorrt",
        ]
        for dockerfile in expected:
            path = dockerfiles_dir / dockerfile
            assert path.exists(), f"Dockerfile not found: {path}"

    def test_dockerfile_base_image_default(self, dockerfiles_dir: Path) -> None:
        """Verify Dockerfiles use correct default BASE_IMAGE."""
        dockerfiles = ["Dockerfile.pytorch", "Dockerfile.vllm", "Dockerfile.tensorrt"]

        for dockerfile in dockerfiles:
            path = dockerfiles_dir / dockerfile
            if path.exists():
                content = path.read_text()
                # Check ARG BASE_IMAGE default
                if "ARG BASE_IMAGE=" in content:
                    assert (
                        "llenergymeasure-base:latest" in content
                    ), f"{dockerfile} should have BASE_IMAGE default of 'llenergymeasure-base:latest'"

    def test_no_old_naming_in_dockerfiles(self, dockerfiles_dir: Path) -> None:
        """Verify no references to old naming in Dockerfiles."""
        dockerfiles = list(dockerfiles_dir.glob("Dockerfile.*"))

        old_patterns = [
            r"llm-energy-measure",
            r"llm_energy_measure",
        ]

        for dockerfile in dockerfiles:
            content = dockerfile.read_text()
            for pattern in old_patterns:
                matches = re.findall(pattern, content)
                assert not matches, f"Found old naming '{pattern}' in {dockerfile.name}"


class TestSetupScript:
    """Tests for setup.sh naming conventions."""

    def test_setup_script_exists(self, project_root: Path) -> None:
        """Verify setup.sh exists."""
        setup_path = project_root / "setup.sh"
        assert setup_path.exists(), "setup.sh not found"

    def test_setup_uses_correct_image_names(self, project_root: Path) -> None:
        """Verify setup.sh references correct image names."""
        setup_path = project_root / "setup.sh"
        content = setup_path.read_text()

        # Should reference new naming
        assert (
            "llenergymeasure" in content.lower() or "lem" in content
        ), "setup.sh should reference 'llenergymeasure' or 'lem'"

    def test_lem_wrapper_references(self, project_root: Path) -> None:
        """Verify the lem wrapper script references correct names."""
        lem_path = project_root / "lem"
        if lem_path.exists():
            content = lem_path.read_text()
            # Should not have old references
            assert "llm-energy-measure:" not in content, "lem wrapper has old image naming"


class TestPyprojectNaming:
    """Tests for pyproject.toml naming."""

    def test_package_name(self, project_root: Path) -> None:
        """Verify package name is 'llenergymeasure'."""
        pyproject = project_root / "pyproject.toml"
        content = pyproject.read_text()

        assert 'name = "llenergymeasure"' in content, "Package name should be 'llenergymeasure'"

    def test_module_include(self, project_root: Path) -> None:
        """Verify module include is correct."""
        pyproject = project_root / "pyproject.toml"
        content = pyproject.read_text()

        assert (
            'include = "llenergymeasure"' in content
        ), "Module include should be 'llenergymeasure'"

    def test_cli_entry_points(self, project_root: Path) -> None:
        """Verify CLI entry points reference correct module."""
        pyproject = project_root / "pyproject.toml"
        content = pyproject.read_text()

        # Both entry points should reference llenergymeasure
        assert 'llenergymeasure = "llenergymeasure.cli:app"' in content
        assert 'lem = "llenergymeasure.cli:app"' in content


class TestSourceDirectory:
    """Tests for source directory naming."""

    def test_module_directory_exists(self, project_root: Path) -> None:
        """Verify src/llenergymeasure directory exists."""
        module_dir = project_root / "src" / "llenergymeasure"
        assert module_dir.exists(), "src/llenergymeasure directory should exist"
        assert module_dir.is_dir(), "src/llenergymeasure should be a directory"

    def test_old_directory_removed(self, project_root: Path) -> None:
        """Verify old src/llm_energy_measure directory is removed."""
        old_dir = project_root / "src" / "llm_energy_measure"
        assert not old_dir.exists(), "Old directory src/llm_energy_measure should not exist"

    def test_init_exists(self, project_root: Path) -> None:
        """Verify __init__.py exists in module."""
        init_path = project_root / "src" / "llenergymeasure" / "__init__.py"
        assert init_path.exists(), "src/llenergymeasure/__init__.py should exist"
