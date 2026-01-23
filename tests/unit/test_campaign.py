"""Unit tests for campaign functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_energy_measure.config.campaign_config import (
    CampaignConfig,
    CampaignExecutionConfig,
    generate_campaign_id,
)
from llm_energy_measure.orchestration.campaign import CampaignRunner


class TestCampaignExecutionConfig:
    """Tests for CampaignExecutionConfig model."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = CampaignExecutionConfig()
        assert config.cycles == 3
        assert config.structure == "interleaved"
        assert config.warmup_prompts == 5
        assert config.warmup_timeout_seconds == 30.0
        assert config.config_gap_seconds == 60.0
        assert config.cycle_gap_seconds == 300.0

    def test_structure_options(self) -> None:
        """Test all structure options are valid."""
        for structure in ["interleaved", "shuffled", "grouped"]:
            config = CampaignExecutionConfig(structure=structure)
            assert config.structure == structure

    def test_cycles_validation(self) -> None:
        """Test cycles validation bounds."""
        # Valid range
        for cycles in [1, 5, 10, 20]:
            config = CampaignExecutionConfig(cycles=cycles)
            assert config.cycles == cycles

        # Invalid: too low
        with pytest.raises(ValueError):
            CampaignExecutionConfig(cycles=0)

        # Invalid: too high
        with pytest.raises(ValueError):
            CampaignExecutionConfig(cycles=21)


class TestCampaignConfig:
    """Tests for CampaignConfig model."""

    @pytest.fixture
    def temp_configs(self, tmp_path: Path) -> list[Path]:
        """Create temporary config files for testing."""
        configs = []
        for name in ["pytorch_base", "vllm_base"]:
            config_path = tmp_path / f"{name}.yaml"
            config_path.write_text(
                yaml.dump(
                    {
                        "config_name": name,
                        "model_name": "test/model",
                        "backend": name.split("_")[0],
                    }
                )
            )
            configs.append(config_path)
        return configs

    def test_minimal_config(self, temp_configs: list[Path]) -> None:
        """Test minimal valid config."""
        config = CampaignConfig(
            campaign_name="test-campaign",
            configs=[str(p) for p in temp_configs],
        )
        assert config.campaign_name == "test-campaign"
        assert len(config.configs) == 2
        assert config.execution.cycles == 3  # Default

    def test_campaign_id_generation(self) -> None:
        """Test campaign_id is deterministic."""
        id1 = generate_campaign_id("test-campaign")
        id2 = generate_campaign_id("test-campaign")
        assert id1 == id2
        assert len(id1) == 8

        # Different names produce different IDs
        id3 = generate_campaign_id("other-campaign")
        assert id3 != id1

    def test_campaign_id_property(self, temp_configs: list[Path]) -> None:
        """Test campaign_id property matches generate function."""
        config = CampaignConfig(
            campaign_name="my-campaign",
            configs=[str(p) for p in temp_configs],
        )
        assert config.campaign_id == generate_campaign_id("my-campaign")

    def test_config_paths_validation(self, tmp_path: Path) -> None:
        """Test config paths must be YAML files."""
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("invalid")

        with pytest.raises(ValueError, match="YAML file"):
            CampaignConfig(
                campaign_name="test",
                configs=[str(txt_file)],
            )

    def test_config_existence_validation(self) -> None:
        """Test config files must exist."""
        with pytest.raises(ValueError, match="not found"):
            CampaignConfig(
                campaign_name="test",
                configs=["nonexistent.yaml"],
            )

    def test_get_config_names(self, temp_configs: list[Path]) -> None:
        """Test get_config_names extracts stems."""
        config = CampaignConfig(
            campaign_name="test",
            configs=[str(p) for p in temp_configs],
        )
        names = config.get_config_names()
        assert names == ["pytorch_base", "vllm_base"]


class TestCampaignRunner:
    """Tests for CampaignRunner execution order generation."""

    @pytest.fixture
    def campaign(self, tmp_path: Path) -> CampaignConfig:
        """Create a campaign config for testing."""
        configs = []
        for name in ["config_a", "config_b", "config_c"]:
            config_path = tmp_path / f"{name}.yaml"
            config_path.write_text(
                yaml.dump(
                    {
                        "config_name": name,
                        "model_name": "test/model",
                        "backend": "pytorch",
                    }
                )
            )
            configs.append(str(config_path))

        return CampaignConfig(
            campaign_name="test-runner",
            configs=configs,
            execution=CampaignExecutionConfig(cycles=3),
        )

    def test_interleaved_order(self, campaign: CampaignConfig) -> None:
        """Test interleaved structure produces correct order."""
        campaign = campaign.model_copy(
            update={"execution": campaign.execution.model_copy(update={"structure": "interleaved"})}
        )
        runner = CampaignRunner(campaign)
        order = runner.generate_execution_order()

        # Should be: A,B,C, A,B,C, A,B,C
        assert len(order) == 9
        for cycle in range(3):
            assert order[cycle * 3 + 0].config_name == "config_a"
            assert order[cycle * 3 + 1].config_name == "config_b"
            assert order[cycle * 3 + 2].config_name == "config_c"

    def test_grouped_order(self, campaign: CampaignConfig) -> None:
        """Test grouped structure produces correct order."""
        campaign = campaign.model_copy(
            update={"execution": campaign.execution.model_copy(update={"structure": "grouped"})}
        )
        runner = CampaignRunner(campaign)
        order = runner.generate_execution_order()

        # Should be: A,A,A, B,B,B, C,C,C
        assert len(order) == 9
        for i in range(3):
            assert order[i].config_name == "config_a"
            assert order[i].cycle_index == i
        for i in range(3, 6):
            assert order[i].config_name == "config_b"
        for i in range(6, 9):
            assert order[i].config_name == "config_c"

    def test_shuffled_order_with_seed(self, campaign: CampaignConfig) -> None:
        """Test shuffled structure is deterministic with seed."""
        campaign = campaign.model_copy(
            update={"execution": campaign.execution.model_copy(update={"structure": "shuffled"})}
        )

        runner1 = CampaignRunner(campaign, seed=42)
        order1 = runner1.generate_execution_order()

        runner2 = CampaignRunner(campaign, seed=42)
        order2 = runner2.generate_execution_order()

        # Same seed = same order
        for i in range(len(order1)):
            assert order1[i].config_name == order2[i].config_name

    def test_shuffled_order_different_seeds(self, campaign: CampaignConfig) -> None:
        """Test shuffled structure varies with different seeds."""
        campaign = campaign.model_copy(
            update={"execution": campaign.execution.model_copy(update={"structure": "shuffled"})}
        )

        runner1 = CampaignRunner(campaign, seed=1)
        order1 = runner1.generate_execution_order()

        runner2 = CampaignRunner(campaign, seed=2)
        order2 = runner2.generate_execution_order()

        # Different seeds should produce different orders (almost certainly)
        names1 = [e.config_name for e in order1]
        names2 = [e.config_name for e in order2]
        assert names1 != names2

    def test_total_experiments(self, campaign: CampaignConfig) -> None:
        """Test total_experiments calculation."""
        runner = CampaignRunner(campaign)
        assert runner.total_experiments == 9  # 3 configs x 3 cycles

    def test_is_cycle_complete(self, campaign: CampaignConfig) -> None:
        """Test cycle completion detection."""
        runner = CampaignRunner(campaign)
        runner.generate_execution_order()

        # With 3 configs per cycle:
        # Index 2 (3rd experiment) completes cycle 1
        # Index 5 (6th experiment) completes cycle 2
        # Index 8 (9th experiment) completes cycle 3
        assert not runner.is_cycle_complete(0)
        assert not runner.is_cycle_complete(1)
        assert runner.is_cycle_complete(2)
        assert not runner.is_cycle_complete(3)
        assert not runner.is_cycle_complete(4)
        assert runner.is_cycle_complete(5)


class TestCampaignYAML:
    """Tests for campaign YAML parsing."""

    def test_full_campaign_yaml(self, tmp_path: Path) -> None:
        """Test parsing a full campaign YAML."""
        # Create experiment configs
        for name in ["pytorch", "vllm"]:
            (tmp_path / f"{name}.yaml").write_text(
                yaml.dump({"config_name": name, "model_name": "test/model", "backend": name})
            )

        # Create campaign YAML
        campaign_yaml = {
            "campaign_name": "full-test",
            "dataset": "alpaca",
            "num_samples": 100,
            "configs": [
                str(tmp_path / "pytorch.yaml"),
                str(tmp_path / "vllm.yaml"),
            ],
            "execution": {
                "cycles": 5,
                "structure": "shuffled",
                "warmup_prompts": 10,
                "warmup_timeout_seconds": 60,
                "config_gap_seconds": 120,
                "cycle_gap_seconds": 600,
            },
        }

        config = CampaignConfig(**campaign_yaml)

        assert config.campaign_name == "full-test"
        assert config.dataset == "alpaca"
        assert config.num_samples == 100
        assert config.execution.cycles == 5
        assert config.execution.structure == "shuffled"
        assert config.execution.warmup_prompts == 10
        assert config.execution.warmup_timeout_seconds == 60
        assert config.execution.config_gap_seconds == 120
        assert config.execution.cycle_gap_seconds == 600
