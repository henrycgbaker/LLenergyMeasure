"""Tests for docker_detection module."""

from unittest.mock import MagicMock, patch

from llenergymeasure.config.docker_detection import (
    is_inside_docker,
    should_use_docker_for_campaign,
)


class TestDockerDetection:
    """Tests for Docker container detection."""

    @patch("llenergymeasure.config.docker_detection.Path")
    def test_is_inside_docker_false_on_host(self, mock_path):
        """Host machine (no Docker markers) returns False."""
        # Mock /.dockerenv does not exist
        mock_path.return_value.exists.return_value = False

        # Mock /proc/1/cgroup read raises FileNotFoundError
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = is_inside_docker()

        assert result is False

    @patch("llenergymeasure.config.docker_detection.Path")
    def test_is_inside_docker_true_dockerenv(self, mock_path):
        """Presence of /.dockerenv indicates Docker."""
        # Mock /.dockerenv exists
        mock_path.return_value.exists.return_value = True

        result = is_inside_docker()

        assert result is True

    @patch("llenergymeasure.config.docker_detection.Path")
    def test_is_inside_docker_true_cgroup(self, mock_path):
        """Docker string in /proc/1/cgroup indicates Docker."""
        # Mock /.dockerenv does not exist
        mock_path.return_value.exists.return_value = False

        # Mock /proc/1/cgroup contains "docker"
        cgroup_content = "12:memory:/docker/abc123\n11:cpuset:/docker/abc123\n"
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = cgroup_content

        with patch("builtins.open", return_value=mock_file):
            result = is_inside_docker()

        assert result is True

    @patch("llenergymeasure.config.docker_detection.Path")
    def test_is_inside_docker_true_containerd(self, mock_path):
        """Containerd string in /proc/1/cgroup indicates container."""
        # Mock /.dockerenv does not exist
        mock_path.return_value.exists.return_value = False

        # Mock /proc/1/cgroup contains "containerd"
        cgroup_content = "12:memory:/containerd/xyz789\n"
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = cgroup_content

        with patch("builtins.open", return_value=mock_file):
            result = is_inside_docker()

        assert result is True


class TestDockerCampaignDispatch:
    """Tests for Docker campaign dispatch logic."""

    @patch("llenergymeasure.config.docker_detection.is_inside_docker")
    def test_should_use_docker_inside_container(self, mock_is_inside):
        """Already in Docker → no nested containers."""
        mock_is_inside.return_value = True

        result = should_use_docker_for_campaign(["pytorch"])

        assert result is False

    @patch("llenergymeasure.config.docker_detection.is_inside_docker")
    @patch("llenergymeasure.config.backend_detection.is_backend_available")
    def test_should_use_docker_single_backend_available(
        self, mock_backend_available, mock_is_inside
    ):
        """Single backend installed locally → run locally."""
        mock_is_inside.return_value = False
        mock_backend_available.return_value = True

        result = should_use_docker_for_campaign(["pytorch"])

        assert result is False
        mock_backend_available.assert_called_once_with("pytorch")

    @patch("llenergymeasure.config.docker_detection.is_inside_docker")
    @patch("llenergymeasure.config.backend_detection.is_backend_available")
    def test_should_use_docker_multi_backend(self, mock_backend_available, mock_is_inside):
        """Multi-backend campaign → use Docker."""
        mock_is_inside.return_value = False
        # Even if both installed locally, multi-backend uses Docker
        mock_backend_available.return_value = True

        result = should_use_docker_for_campaign(["pytorch", "vllm"])

        assert result is True

    @patch("llenergymeasure.config.docker_detection.is_inside_docker")
    @patch("llenergymeasure.config.backend_detection.is_backend_available")
    def test_should_use_docker_single_backend_unavailable(
        self, mock_backend_available, mock_is_inside
    ):
        """Single backend not installed → use Docker."""
        mock_is_inside.return_value = False
        mock_backend_available.return_value = False

        result = should_use_docker_for_campaign(["vllm"])

        assert result is True
        mock_backend_available.assert_called_once_with("vllm")
