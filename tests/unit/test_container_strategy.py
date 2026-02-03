"""Tests for container strategy and ContainerManager."""

from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.orchestration.container import ContainerManager, ContainerState


class TestContainerState:
    """Tests for ContainerState dataclass."""

    def test_initial_state(self) -> None:
        """ContainerState initializes correctly."""
        state = ContainerState(service="pytorch", status="stopped")
        assert state.service == "pytorch"
        assert state.status == "stopped"
        assert state.restart_count == 0


class TestContainerManager:
    """Tests for ContainerManager class."""

    def test_initialization(self) -> None:
        """ContainerManager initializes with services."""
        manager = ContainerManager(services=["pytorch", "vllm"])

        assert manager.services == ["pytorch", "vllm"]
        assert "pytorch" in manager._states
        assert "vllm" in manager._states
        assert manager._states["pytorch"].status == "stopped"
        assert manager._started is False

    def test_unknown_service_raises(self) -> None:
        """exec() with unknown service raises ValueError."""
        manager = ContainerManager(services=["pytorch"])

        with pytest.raises(ValueError, match="Unknown service"):
            manager.exec("vllm", ["echo", "test"])

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_start_all_success(self, mock_run: MagicMock) -> None:
        """start_all() calls docker compose up."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ContainerManager(services=["pytorch", "vllm"])
        result = manager.start_all()

        assert result is True
        assert manager._started is True
        assert manager._states["pytorch"].status == "running"
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "up" in call_args
        assert "-d" in call_args

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_start_all_failure(self, mock_run: MagicMock) -> None:
        """start_all() returns False on failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        manager = ContainerManager(services=["pytorch"])
        result = manager.start_all()

        assert result is False
        assert manager._started is False
        assert manager._states["pytorch"].status == "error"

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_stop_all(self, mock_run: MagicMock) -> None:
        """stop_all() calls docker compose down."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ContainerManager(services=["pytorch"])
        manager._started = True
        manager._states["pytorch"].status = "running"

        result = manager.stop_all()

        assert result is True
        assert manager._started is False
        assert manager._states["pytorch"].status == "stopped"

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_exec_builds_correct_command(self, mock_run: MagicMock) -> None:
        """exec() builds correct docker compose exec command."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ContainerManager(services=["pytorch"])
        manager._started = True
        manager._states["pytorch"].status = "running"

        manager.exec("pytorch", ["lem", "experiment", "config.yaml"])

        call_args = mock_run.call_args[0][0]
        assert call_args[:3] == ["docker", "compose", "exec"]
        assert "pytorch" in call_args
        assert "lem" in call_args

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_exec_with_env_vars(self, mock_run: MagicMock) -> None:
        """exec() passes environment variables with -e flags."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ContainerManager(services=["pytorch"])
        manager._started = True
        manager._states["pytorch"].status = "running"

        manager.exec(
            "pytorch",
            ["lem", "experiment"],
            env={"LEM_CAMPAIGN_ID": "abc123"},
        )

        call_args = mock_run.call_args[0][0]
        assert "-e" in call_args
        assert "LEM_CAMPAIGN_ID=abc123" in call_args

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_context_manager(self, mock_run: MagicMock) -> None:
        """ContainerManager works as context manager."""
        mock_run.return_value = MagicMock(returncode=0)

        with ContainerManager(services=["pytorch"]) as manager:
            assert manager._started is True

        # stop_all called on exit
        assert mock_run.call_count == 2  # start + stop

    @patch("llenergymeasure.orchestration.container.subprocess.run")
    def test_restart_service(self, mock_run: MagicMock) -> None:
        """restart_service() increments restart count."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ContainerManager(services=["pytorch"])
        manager._states["pytorch"].status = "running"

        result = manager.restart_service("pytorch")

        assert result is True
        assert manager._states["pytorch"].restart_count == 1
