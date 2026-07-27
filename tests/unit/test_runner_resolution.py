"""Unit tests for runner resolution precedence chain.

Tests cover:
  - parse_runner_value: "process", "container", "container:image" forms, plus the
    clean-break rejection of the legacy "local"/"docker"/"docker:image" vocabulary
  - is_docker_available: PATH inspection for docker + NVIDIA CT tools
  - is_running_in_container / is_container_socket_available: file-existence detection
  - resolve_runner: full precedence chain (env > yaml > user_config > auto > default),
    including the container-self-aware auto-detection branch
  - resolve_study_runners: multi-engine resolution
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.config.ssot import ENV_RUNNER_PREFIX
from llenergymeasure.config.user_config import UserRunnersConfig
from llenergymeasure.infra.runner_resolution import (
    is_container_socket_available,
    is_docker_available,
    is_running_in_container,
    parse_runner_value,
    resolve_runner,
    resolve_study_runners,
)


@pytest.fixture(autouse=True)
def _clear_detection_caches(monkeypatch):
    """Clear detection caches and default to the HOST topology (not in a container).

    Auto-detection is container-self-aware: it consults ``is_running_in_container``
    and ``is_container_socket_available``. The existing precedence tests assert the
    host topology, so both default to "on the host" here, making them deterministic
    regardless of where the suite runs (host, CI VM, or a dev container). Tests that
    exercise the in-container branches override these explicitly.
    """
    is_docker_available.cache_clear()
    is_running_in_container.cache_clear()
    is_container_socket_available.cache_clear()
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_running_in_container", lambda: False
    )
    yield
    is_docker_available.cache_clear()
    is_running_in_container.cache_clear()
    is_container_socket_available.cache_clear()


# ---------------------------------------------------------------------------
# parse_runner_value - canonical vocabulary
# ---------------------------------------------------------------------------


class TestParseRunnerValue:
    def test_process_returns_process_mode_no_image(self):
        mode, image = parse_runner_value("process")
        assert mode == "process"
        assert image is None

    def test_bare_container_returns_container_mode_no_image(self):
        mode, image = parse_runner_value("container")
        assert mode == "container"
        assert image is None

    def test_container_with_image_returns_container_mode_and_image(self):
        mode, image = parse_runner_value("container:ghcr.io/custom/img:v1")
        assert mode == "container"
        assert image == "ghcr.io/custom/img:v1"

    def test_container_with_complex_image_tag(self):
        mode, image = parse_runner_value("container:nvcr.io/nvidia/pytorch:23.10-py3")
        assert mode == "container"
        assert image == "nvcr.io/nvidia/pytorch:23.10-py3"

    def test_container_colon_empty_raises(self):
        """'container:' with empty image raises ValueError."""
        with pytest.raises(ValueError, match="empty image name"):
            parse_runner_value("container:")

    def test_unknown_value_raises(self):
        """Unrecognised runner value raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised runner value"):
            parse_runner_value("singularity:myimage")

    def test_unknown_value_plain_raises(self):
        """Unrecognised plain string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised runner value"):
            parse_runner_value("kubernetes")

    def test_container_with_image_containing_colon(self):
        """Image tag containing colon (e.g. ghcr.io/org/img:1.0) is preserved."""
        mode, image = parse_runner_value("container:ghcr.io/llem/vllm:1.19.0-cuda12")
        assert mode == "container"
        assert image == "ghcr.io/llem/vllm:1.19.0-cuda12"


# ---------------------------------------------------------------------------
# parse_runner_value - legacy vocabulary is a clean break (rejected with a hint)
# ---------------------------------------------------------------------------


class TestParseRunnerValueLegacyRejected:
    def test_legacy_local_rejected_with_migration_hint(self):
        with pytest.raises(ValueError, match=r"'local' was renamed in v0.7 - use 'process'"):
            parse_runner_value("local")

    def test_legacy_docker_rejected_with_migration_hint(self):
        with pytest.raises(ValueError, match=r"'docker' was renamed in v0.7 - use 'container'"):
            parse_runner_value("docker")

    def test_legacy_docker_image_rejected_with_migration_hint(self):
        # The message must name the user's ACTUAL input, not a placeholder.
        with pytest.raises(
            ValueError,
            match=r"'docker:ghcr.io/custom/img:v1' was renamed.*use 'container:ghcr.io/custom/img:v1'",
        ):
            parse_runner_value("docker:ghcr.io/custom/img:v1")

    def test_legacy_docker_colon_empty_rejected_with_migration_hint(self):
        with pytest.raises(ValueError, match=r"'docker:' was renamed.*use 'container:<image>'"):
            parse_runner_value("docker:")


# ---------------------------------------------------------------------------
# is_docker_available
# ---------------------------------------------------------------------------


class TestIsDockerAvailable:
    def test_returns_true_when_docker_and_nvidia_ctk_on_path(self):
        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("docker", "nvidia-ctk") else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is True

    def test_returns_true_when_docker_and_nvidia_container_runtime_on_path(self):
        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("docker", "nvidia-container-runtime") else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is True

    def test_returns_true_when_docker_and_nvidia_container_cli_on_path(self):
        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("docker", "nvidia-container-cli") else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is True

    def test_returns_false_when_docker_not_on_path(self):
        with patch("llenergymeasure.infra.runner_resolution.shutil.which", return_value=None):
            assert is_docker_available() is False

    def test_returns_false_when_docker_present_but_no_nvidia_tool(self):
        def mock_which(name: str) -> str | None:
            return "/usr/bin/docker" if name == "docker" else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is False


# ---------------------------------------------------------------------------
# resolve_runner - precedence chain
# ---------------------------------------------------------------------------


class TestResolveRunner:
    """Test resolve_runner with each precedence layer."""

    # --- Env var (highest) ---

    def test_env_var_wins_over_everything(self, monkeypatch):
        """LLEM_RUNNER_VLLM=container:custom/img wins over yaml and user_config."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}VLLM", "container:custom/img")
        yaml_runners = {"vllm": "process"}
        user_config = UserRunnersConfig(vllm="process")

        spec = resolve_runner("vllm", yaml_runners=yaml_runners, user_config=user_config)

        assert spec.source == "env"
        assert spec.mode == "container"
        assert spec.image == "custom/img"

    def test_env_var_bare_container(self, monkeypatch):
        """LLEM_RUNNER_TRANSFORMERS=container (bare) sets mode=container, image=None."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "container")

        spec = resolve_runner("transformers")

        assert spec.source == "env"
        assert spec.mode == "container"
        assert spec.image is None

    def test_env_var_process_overrides_yaml_container(self, monkeypatch):
        """Env var 'process' takes precedence even when yaml says 'container'."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "process")
        spec = resolve_runner("transformers", yaml_runners={"transformers": "container"})
        assert spec.source == "env"
        assert spec.mode == "process"

    # --- YAML runners ---

    def test_yaml_runners_wins_over_user_config(self):
        """yaml_runners={'transformers': 'container'} wins over user_config with 'process'."""
        user_config = UserRunnersConfig(transformers="process")

        spec = resolve_runner(
            "transformers", yaml_runners={"transformers": "container"}, user_config=user_config
        )

        assert spec.source == "yaml"
        assert spec.mode == "container"
        assert spec.image is None

    def test_yaml_runners_container_with_image(self):
        """yaml_runners with container:image resolves image correctly."""
        spec = resolve_runner(
            "vllm",
            yaml_runners={"vllm": "container:ghcr.io/myorg/vllm:latest"},
        )
        assert spec.source == "yaml"
        assert spec.mode == "container"
        assert spec.image == "ghcr.io/myorg/vllm:latest"

    def test_yaml_runners_legacy_docker_rejected_with_migration_hint(self):
        """A legacy YAML 'docker' value is rejected with a migration error (clean break)."""
        with pytest.raises(ValueError, match=r"'docker' was renamed in v0.7 - use 'container'"):
            resolve_runner("vllm", yaml_runners={"vllm": "docker"})

    def test_yaml_runners_missing_engine_falls_through(self):
        """If engine not in yaml_runners, falls through to lower layers."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner(
                "tensorrt",
                yaml_runners={"transformers": "container"},  # tensorrt not listed
            )
        assert spec.source == "default"

    def test_yaml_runners_none_falls_through(self):
        """yaml_runners=None skips YAML layer entirely."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("transformers", yaml_runners=None)
        assert spec.source == "default"

    # --- User config ---

    def test_user_config_container_with_image_wins_over_auto_detection(self):
        """user_config.transformers='container:myimg' wins over auto-detection."""
        user_config = UserRunnersConfig(transformers="container:myimg")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "user_config"
        assert spec.mode == "container"
        assert spec.image == "myimg"

    def test_explicit_process_in_user_config_respected_not_overridden_by_auto_detect(self):
        """Explicit 'process' in user_config wins; auto-detection is not applied."""
        user_config = UserRunnersConfig(transformers="process")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "user_config"
        assert spec.mode == "process"

    def test_user_config_bare_container_sets_mode_container_image_none(self):
        """user_config.vllm='container' resolves to mode=container, image=None."""
        user_config = UserRunnersConfig(vllm="container")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("vllm", user_config=user_config)

        assert spec.source == "user_config"
        assert spec.mode == "container"
        assert spec.image is None

    # --- Auto-detection ---

    def test_auto_detected_when_docker_available_and_no_config(self):
        """When Docker available and no explicit config, source='auto_detected'."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers")  # no yaml_runners, no user_config

        assert spec.source == "auto_detected"
        assert spec.mode == "container"
        assert spec.image is None

    def test_auto_user_config_default_falls_through_to_auto_detection(self):
        """user_config=UserRunnersConfig() (all defaults to 'auto') falls through to auto-detection."""
        user_config = UserRunnersConfig()  # all fields default to "auto"

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        # "auto" falls through - Docker auto-detection applies
        assert spec.source == "auto_detected"
        assert spec.mode == "container"

    def test_explicit_auto_in_user_config_falls_through_to_auto_detection(self):
        """Explicit 'auto' in user_config falls through to auto-detection."""
        user_config = UserRunnersConfig(transformers="auto")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "auto_detected"
        assert spec.mode == "container"

    def test_auto_user_config_no_docker_falls_to_default(self):
        """user_config defaults to 'auto', Docker unavailable → falls to default."""
        user_config = UserRunnersConfig()  # all fields default to "auto"

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "default"
        assert spec.mode == "process"

    # --- Default (process fallback) ---

    def test_default_process_when_docker_unavailable_and_no_config(self):
        """When Docker not available and no config, source='default', mode='process'."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("transformers")

        assert spec.source == "default"
        assert spec.mode == "process"
        assert spec.image is None

    # --- Parse integration ---

    def test_parse_runner_value_integration_container_custom_image(self, monkeypatch):
        """parse_runner_value integration: 'container:ghcr.io/custom:v1' resolves image."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "container:ghcr.io/custom:v1")
        spec = resolve_runner("transformers")
        assert spec.mode == "container"
        assert spec.image == "ghcr.io/custom:v1"
        assert spec.source == "env"


# ---------------------------------------------------------------------------
# resolve_study_runners
# ---------------------------------------------------------------------------


class TestResolveStudyRunners:
    def test_resolves_each_engine(self):
        """resolve_study_runners returns spec for each engine."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            result = resolve_study_runners(["transformers", "vllm"])

        assert set(result.keys()) == {"transformers", "vllm"}
        assert all(isinstance(v, RunnerSpec) for v in result.values())

    def test_yaml_runners_applied_per_engine(self):
        """yaml_runners are applied to each engine correctly."""
        yaml_runners = {"transformers": "process", "vllm": "container:myimg"}

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            result = resolve_study_runners(["transformers", "vllm"], yaml_runners=yaml_runners)

        assert result["transformers"].mode == "process"
        assert result["transformers"].source == "yaml"
        assert result["vllm"].mode == "container"
        assert result["vllm"].image == "myimg"

    def test_empty_engines_list_returns_empty_dict(self):
        result = resolve_study_runners([])
        assert result == {}

    def test_single_engine(self):
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            result = resolve_study_runners(["tensorrt"])

        assert "tensorrt" in result
        assert result["tensorrt"].source == "auto_detected"

    def test_mixed_auto_and_explicit_per_engine(self):
        """Study with one engine explicitly set and another using auto-detection.

        Simulates a researcher who forces transformers=process but lets vllm auto-detect
        to a container. Each engine resolves independently through the precedence chain.
        """
        user_config = UserRunnersConfig(transformers="process", vllm="auto", tensorrt="auto")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            result = resolve_study_runners(
                ["transformers", "vllm", "tensorrt"], user_config=user_config
            )

        # transformers: explicit "process" -> user_config source
        assert result["transformers"].mode == "process"
        assert result["transformers"].source == "user_config"
        # vllm: "auto" falls through -> container auto-detected
        assert result["vllm"].mode == "container"
        assert result["vllm"].source == "auto_detected"
        # tensorrt: "auto" falls through -> container auto-detected
        assert result["tensorrt"].mode == "container"
        assert result["tensorrt"].source == "auto_detected"


# ---------------------------------------------------------------------------
# Container-self detection helpers (file-existence / env, no shelling out)
# ---------------------------------------------------------------------------


def _fake_path(existing: set[str]):
    """Return a drop-in for ``pathlib.Path`` where ``.exists()`` is True only for
    the paths in *existing*. Used to make the file-existence probes hermetic."""

    def factory(p):
        obj = MagicMock()
        obj.exists.return_value = str(p) in existing
        return obj

    return factory


class TestIsRunningInContainer:
    """File-existence self-detection: /.dockerenv (docker) or /run/.containerenv (podman)."""

    def test_false_when_no_marker_files(self):
        with patch("llenergymeasure.infra.runner_resolution.Path", new=_fake_path(set())):
            assert is_running_in_container() is False

    def test_true_when_dockerenv_present(self):
        with patch(
            "llenergymeasure.infra.runner_resolution.Path",
            new=_fake_path({"/.dockerenv"}),
        ):
            assert is_running_in_container() is True

    def test_true_when_podman_containerenv_present(self):
        with patch(
            "llenergymeasure.infra.runner_resolution.Path",
            new=_fake_path({"/run/.containerenv"}),
        ):
            assert is_running_in_container() is True


class TestIsContainerSocketAvailable:
    """DooD signal: DOCKER_HOST env set, or /var/run/docker.sock present."""

    def test_true_when_docker_host_env_set(self, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")
        # Even with no socket file, DOCKER_HOST alone signals a control endpoint.
        with patch("llenergymeasure.infra.runner_resolution.Path", new=_fake_path(set())):
            assert is_container_socket_available() is True

    def test_true_when_socket_path_present(self, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        with patch(
            "llenergymeasure.infra.runner_resolution.Path",
            new=_fake_path({"/var/run/docker.sock"}),
        ):
            assert is_container_socket_available() is True

    def test_false_when_no_env_and_no_socket(self, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        with patch("llenergymeasure.infra.runner_resolution.Path", new=_fake_path(set())):
            assert is_container_socket_available() is False


# ---------------------------------------------------------------------------
# resolve_runner - container-self-aware auto-detection (the hardening slice)
# ---------------------------------------------------------------------------


class TestContainerSelfAwareAutoDetection:
    """Auto-detection resolves by container-context, not blind PATH inspection.

    The two known-good topologies are locked here: on the host Docker on PATH still
    means container; inside a socketless container we now resolve process instead of
    attempting docker-in-docker.
    """

    def test_host_with_docker_resolves_container_unchanged(self):
        """Known-good topology 1: on the host, Docker on PATH -> container (as before)."""
        # is_running_in_container defaults to False via the autouse fixture (host).
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("vllm")
        assert spec.mode == "container"
        assert spec.source == "auto_detected"

    def test_in_container_no_socket_resolves_process_despite_docker_on_path(self, caplog):
        """Known-good topology 2 fixed: in-container + no socket -> process, even with
        the docker CLI on PATH (today's blindness would have attempted DinD)."""
        with (
            patch(
                "llenergymeasure.infra.runner_resolution.is_running_in_container",
                return_value=True,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_container_socket_available",
                return_value=False,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_docker_available",
                return_value=True,  # docker CLI present, but no usable socket
            ),
            caplog.at_level("INFO", logger="llenergymeasure.infra.runner_resolution"),
        ):
            spec = resolve_runner("vllm")
        assert spec.mode == "process"
        assert spec.source == "default"
        assert any(
            "inside a container without a Docker socket" in r.message for r in caplog.records
        )

    def test_in_container_with_socket_resolves_container(self):
        """In-container + Docker socket -> container (DooD siblings via host daemon).

        Socket presence drives the decision; the NVIDIA-toolkit PATH check
        (is_docker_available) does not apply inside llem's container."""
        with (
            patch(
                "llenergymeasure.infra.runner_resolution.is_running_in_container",
                return_value=True,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_container_socket_available",
                return_value=True,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_docker_available",
                return_value=False,  # NVIDIA CT not on PATH inside llem's container
            ),
        ):
            spec = resolve_runner("vllm")
        assert spec.mode == "container"
        assert spec.source == "auto_detected"

    # --- Explicit pins are unaffected by container-context in every case ---

    def test_env_pin_process_unaffected_in_container_with_socket(self, monkeypatch):
        """An env pin short-circuits before auto-detection: process stays process even
        in a container-with-socket that auto-detection would resolve to container."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}VLLM", "process")
        with (
            patch(
                "llenergymeasure.infra.runner_resolution.is_running_in_container",
                return_value=True,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_container_socket_available",
                return_value=True,
            ),
        ):
            spec = resolve_runner("vllm")
        assert spec.mode == "process"
        assert spec.source == "env"

    def test_user_config_container_pin_unaffected_in_socketless_container(self):
        """A user-config container pin short-circuits before auto-detection: container
        stays container even in a socketless container that would auto-resolve process."""
        user_config = UserRunnersConfig(vllm="container:ghcr.io/org/vllm:v1")
        with (
            patch(
                "llenergymeasure.infra.runner_resolution.is_running_in_container",
                return_value=True,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_container_socket_available",
                return_value=False,
            ),
        ):
            spec = resolve_runner("vllm", user_config=user_config)
        assert spec.mode == "container"
        assert spec.image == "ghcr.io/org/vllm:v1"
        assert spec.source == "user_config"

    def test_yaml_pin_container_unaffected_in_socketless_container(self):
        """A YAML container pin short-circuits before the container-aware auto branch."""
        with (
            patch(
                "llenergymeasure.infra.runner_resolution.is_running_in_container",
                return_value=True,
            ),
            patch(
                "llenergymeasure.infra.runner_resolution.is_container_socket_available",
                return_value=False,
            ),
        ):
            spec = resolve_runner("vllm", yaml_runners={"vllm": "container"})
        assert spec.mode == "container"
        assert spec.source == "yaml"
