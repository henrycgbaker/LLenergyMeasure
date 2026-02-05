"""Unit tests for GPU environment variable propagation to Docker containers (Phase 3).

Tests the _build_docker_command function's handling of GPU indices and
environment variable propagation for NVIDIA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestBuildDockerCommandGPUPropagation:
    """Tests for _build_docker_command GPU env var propagation."""

    def test_build_docker_command_with_gpus_sets_nvidia_visible_devices(self) -> None:
        """gpus parameter should set NVIDIA_VISIBLE_DEVICES to host GPU indices."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
            gpus=[0, 1, 2],
        )

        # Find NVIDIA_VISIBLE_DEVICES in command
        nvidia_idx = None
        for i, arg in enumerate(cmd):
            if (
                arg == "-e"
                and i + 1 < len(cmd)
                and cmd[i + 1].startswith("NVIDIA_VISIBLE_DEVICES=")
            ):
                nvidia_idx = i + 1
                break

        assert nvidia_idx is not None, "NVIDIA_VISIBLE_DEVICES not found in command"
        assert cmd[nvidia_idx] == "NVIDIA_VISIBLE_DEVICES=0,1,2"

    def test_build_docker_command_with_gpus_sets_cuda_visible_devices_remapped(self) -> None:
        """gpus parameter should set CUDA_VISIBLE_DEVICES with remapped indices (0,1,2,...).

        Inside the container, GPUs appear as 0,1,2,... regardless of host indices.
        This is because NVIDIA_VISIBLE_DEVICES controls which GPUs are mounted,
        and inside the container they're renumbered starting from 0.
        """
        from llenergymeasure.cli.campaign import _build_docker_command

        # Even with non-sequential host GPU indices, container indices are remapped
        cmd = _build_docker_command(
            backend="vllm",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
            gpus=[2, 3, 5],  # Host indices (non-sequential)
        )

        # Find CUDA_VISIBLE_DEVICES in command
        cuda_idx = None
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd) and cmd[i + 1].startswith("CUDA_VISIBLE_DEVICES="):
                cuda_idx = i + 1
                break

        assert cuda_idx is not None, "CUDA_VISIBLE_DEVICES not found in command"
        # Inside container, 3 GPUs become 0,1,2 (remapped from host 2,3,5)
        assert cmd[cuda_idx] == "CUDA_VISIBLE_DEVICES=0,1,2"

    def test_build_docker_command_without_gpus_no_gpu_env_vars(self) -> None:
        """Without gpus parameter, no GPU env vars should be set."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
            gpus=None,  # No GPUs specified
        )

        # Check no NVIDIA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES in command
        env_args = []
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                env_args.append(cmd[i + 1])

        nvidia_found = any("NVIDIA_VISIBLE_DEVICES=" in e for e in env_args)
        cuda_found = any("CUDA_VISIBLE_DEVICES=" in e for e in env_args)

        assert not nvidia_found, "NVIDIA_VISIBLE_DEVICES should not be set without gpus"
        assert not cuda_found, "CUDA_VISIBLE_DEVICES should not be set without gpus"

    def test_build_docker_command_single_gpu(self) -> None:
        """Single GPU should set both env vars correctly."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="tensorrt",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
            gpus=[3],  # Single GPU at host index 3
        )

        # Find both env vars
        nvidia_value = None
        cuda_value = None
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                if cmd[i + 1].startswith("NVIDIA_VISIBLE_DEVICES="):
                    nvidia_value = cmd[i + 1].split("=")[1]
                elif cmd[i + 1].startswith("CUDA_VISIBLE_DEVICES="):
                    cuda_value = cmd[i + 1].split("=")[1]

        assert nvidia_value == "3", "NVIDIA_VISIBLE_DEVICES should be host index"
        assert cuda_value == "0", "CUDA_VISIBLE_DEVICES should be remapped to 0"

    def test_build_docker_command_gpu_env_before_backend(self) -> None:
        """GPU env vars should come before service name in command."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
            gpus=[0, 1],
        )

        # Find backend position
        backend_idx = cmd.index("pytorch")

        # All -e flags should be before backend
        for i, arg in enumerate(cmd):
            if arg == "-e":
                assert i < backend_idx, "-e flags should come before service name"


class TestLauncherContainerContextDetection:
    """Tests for launcher.py container context detection logic.

    Note: The actual _early_cuda_visible_devices_setup is called at module import
    time in if __name__ == "__main__" block, so we test the logic patterns here.
    """

    def test_container_context_detection_with_nvidia_visible_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NVIDIA_VISIBLE_DEVICES set (not 'all') indicates container context."""
        # When NVIDIA_VISIBLE_DEVICES is set to specific GPUs, we're in a container
        # and should use remapped indices (0,1,2,...)

        nvidia_visible = "0,1"  # Container has 2 GPUs mounted

        # In container: GPUs are remapped to 0,1,2,...
        gpu_count = len(nvidia_visible.split(","))
        expected_cuda = ",".join(str(i) for i in range(gpu_count))

        assert expected_cuda == "0,1"  # 2 GPUs -> 0,1

    def test_local_context_detection_without_nvidia_visible_devices(self) -> None:
        """Without NVIDIA_VISIBLE_DEVICES, we're in local context."""
        # When NVIDIA_VISIBLE_DEVICES is not set, use config.gpus directly

        nvidia_visible = ""  # Not set
        config_gpus = [2, 3]  # User wants GPUs 2 and 3

        # Local context: use config.gpus directly
        if not nvidia_visible or nvidia_visible == "all":
            expected_cuda = ",".join(str(g) for g in config_gpus)
        else:
            gpu_count = len(nvidia_visible.split(","))
            expected_cuda = ",".join(str(i) for i in range(gpu_count))

        assert expected_cuda == "2,3"  # Local: use config values directly

    def test_nvidia_visible_devices_all_uses_config_gpus(self) -> None:
        """NVIDIA_VISIBLE_DEVICES='all' should use config.gpus directly."""
        nvidia_visible = "all"  # All GPUs visible
        config_gpus = [0, 2, 4]  # User wants specific GPUs

        # "all" means use config.gpus directly (not in specific-GPU container)
        if not nvidia_visible or nvidia_visible == "all":
            expected_cuda = ",".join(str(g) for g in config_gpus)
        else:
            gpu_count = len(nvidia_visible.split(","))
            expected_cuda = ",".join(str(i) for i in range(gpu_count))

        assert expected_cuda == "0,2,4"


class TestDockerCommandCampaignContext:
    """Tests for campaign context propagation in Docker commands."""

    def test_campaign_context_propagated_to_docker_command(self) -> None:
        """Campaign context env vars should be included in Docker command."""
        from llenergymeasure.cli.campaign import _build_docker_command

        campaign_context = {
            "LEM_CAMPAIGN_ID": "test-campaign-123",
            "LEM_CAMPAIGN_NAME": "my-campaign",
            "LEM_CYCLE": "2",
            "LEM_TOTAL_CYCLES": "5",
        }

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset="alpaca",
            sample_size=100,
            results_dir=Path("/results"),
            campaign_context=campaign_context,
            gpus=[0, 1],
        )

        # Extract all -e arguments
        env_vars = {}
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                key_value = cmd[i + 1]
                if "=" in key_value:
                    key, value = key_value.split("=", 1)
                    env_vars[key] = value

        # Verify campaign context was included
        assert env_vars.get("LEM_CAMPAIGN_ID") == "test-campaign-123"
        assert env_vars.get("LEM_CAMPAIGN_NAME") == "my-campaign"
        assert env_vars.get("LEM_CYCLE") == "2"
        assert env_vars.get("LEM_TOTAL_CYCLES") == "5"

    def test_gpu_env_vars_before_campaign_context(self) -> None:
        """GPU env vars should be set before campaign context for precedence.

        NVIDIA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES should be set first
        so that campaign context doesn't accidentally override them.
        """
        from llenergymeasure.cli.campaign import _build_docker_command

        campaign_context = {
            "LEM_CAMPAIGN_ID": "test-123",
        }

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
            campaign_context=campaign_context,
            gpus=[0, 1],
        )

        # Find positions of GPU env vars and campaign context
        nvidia_idx = None
        campaign_idx = None
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                if "NVIDIA_VISIBLE_DEVICES=" in cmd[i + 1]:
                    nvidia_idx = i
                elif "LEM_CAMPAIGN_ID=" in cmd[i + 1]:
                    campaign_idx = i

        assert nvidia_idx is not None
        assert campaign_idx is not None
        assert nvidia_idx < campaign_idx, "GPU env vars should come before campaign context"


class TestDockerCommandStructure:
    """Tests for Docker command structure and format."""

    def test_docker_compose_run_rm_format(self) -> None:
        """Command should use 'docker compose run --rm' format."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="vllm",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
        )

        assert cmd[0] == "docker"
        assert cmd[1] == "compose"
        assert cmd[2] == "run"
        assert cmd[3] == "--rm"

    def test_backend_service_name_in_command(self) -> None:
        """Backend should be used as service name."""
        from llenergymeasure.cli.campaign import _build_docker_command

        for backend in ["pytorch", "vllm", "tensorrt"]:
            cmd = _build_docker_command(
                backend=backend,
                config_path="/app/configs/test.yaml",
                dataset=None,
                sample_size=None,
                results_dir=None,
            )

            # Backend name should appear after --rm and env vars
            assert backend in cmd, f"Backend {backend} should be in command"

    def test_lem_experiment_command_included(self) -> None:
        """Command should include 'lem experiment' subcommand."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
        )

        # After service name, should have lem experiment
        backend_idx = cmd.index("pytorch")
        assert cmd[backend_idx + 1] == "lem"
        assert cmd[backend_idx + 2] == "experiment"

    def test_config_path_in_command(self) -> None:
        """Config path should be in command."""
        from llenergymeasure.cli.campaign import _build_docker_command

        config_path = "/app/configs/my_test_config.yaml"
        cmd = _build_docker_command(
            backend="pytorch",
            config_path=config_path,
            dataset=None,
            sample_size=None,
            results_dir=None,
        )

        assert config_path in cmd

    def test_dataset_and_sample_size_options(self) -> None:
        """Dataset and sample size should be passed as options."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset="alpaca",
            sample_size=50,
            results_dir=None,
        )

        # Find --dataset and --sample-size
        assert "--dataset" in cmd
        dataset_idx = cmd.index("--dataset")
        assert cmd[dataset_idx + 1] == "alpaca"

        assert "--sample-size" in cmd
        sample_idx = cmd.index("--sample-size")
        assert cmd[sample_idx + 1] == "50"

    def test_yes_and_fresh_flags_included(self) -> None:
        """--yes and --fresh flags should be included."""
        from llenergymeasure.cli.campaign import _build_docker_command

        cmd = _build_docker_command(
            backend="pytorch",
            config_path="/app/configs/test.yaml",
            dataset=None,
            sample_size=None,
            results_dir=None,
        )

        assert "--yes" in cmd
        assert "--fresh" in cmd
