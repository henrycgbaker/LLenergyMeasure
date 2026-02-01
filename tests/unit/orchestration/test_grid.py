"""Unit tests for campaign grid expansion and validation."""

from __future__ import annotations

from typing import Any

from llenergymeasure.config.campaign_config import CampaignGridConfig
from llenergymeasure.orchestration.grid import (
    GridExpansionResult,
    expand_campaign_grid,
    validate_campaign_grid,
)


def _make_grid(**kwargs: Any) -> CampaignGridConfig:
    """Create a CampaignGridConfig with sensible defaults."""
    defaults: dict[str, Any] = {
        "backends": ["pytorch"],
        "shared": {},
        "backend_overrides": {},
    }
    defaults.update(kwargs)
    return CampaignGridConfig(**defaults)


class TestExpandCampaignGrid:
    """Tests for expand_campaign_grid — pure cartesian product logic."""

    def test_expand_single_backend(self) -> None:
        """1 backend x 2 shared params = 2 configs."""
        grid = _make_grid(
            backends=["pytorch"],
            shared={"fp_precision": ["float16", "bfloat16"]},
        )
        configs = expand_campaign_grid(grid)

        assert len(configs) == 2
        precisions = {c["fp_precision"] for c in configs}
        assert precisions == {"float16", "bfloat16"}
        assert all(c["backend"] == "pytorch" for c in configs)

    def test_expand_multi_backend(self) -> None:
        """2 backends x 2 shared params = 4 configs."""
        grid = _make_grid(
            backends=["pytorch", "vllm"],
            shared={"fp_precision": ["float16", "bfloat16"]},
        )
        configs = expand_campaign_grid(grid)

        assert len(configs) == 4
        backend_precision = {(c["backend"], c["fp_precision"]) for c in configs}
        assert ("pytorch", "float16") in backend_precision
        assert ("pytorch", "bfloat16") in backend_precision
        assert ("vllm", "float16") in backend_precision
        assert ("vllm", "bfloat16") in backend_precision

    def test_expand_with_backend_overrides(self) -> None:
        """Backend-specific params expand correctly per backend."""
        grid = _make_grid(
            backends=["pytorch", "vllm"],
            backend_overrides={
                "pytorch": {"batch_size": [1, 4]},
                "vllm": {"max_num_seqs": [8, 16]},
            },
        )
        configs = expand_campaign_grid(grid)

        # pytorch: 2 overrides, vllm: 2 overrides = 4 configs
        assert len(configs) == 4

        pytorch_configs = [c for c in configs if c["backend"] == "pytorch"]
        assert len(pytorch_configs) == 2
        batch_sizes = {c["pytorch"]["batch_size"] for c in pytorch_configs}
        assert batch_sizes == {1, 4}

        vllm_configs = [c for c in configs if c["backend"] == "vllm"]
        assert len(vllm_configs) == 2
        seqs = {c["vllm"]["max_num_seqs"] for c in vllm_configs}
        assert seqs == {8, 16}

    def test_expand_with_models(self) -> None:
        """Models axis included in cartesian product."""
        grid = _make_grid(
            backends=["pytorch"],
            models=["model-a", "model-b"],
            shared={"fp_precision": ["float16"]},
        )
        configs = expand_campaign_grid(grid)

        # 1 backend x 2 models x 1 precision = 2
        assert len(configs) == 2
        model_names = {c["model_name"] for c in configs}
        assert model_names == {"model-a", "model-b"}

    def test_expand_empty_grid(self) -> None:
        """Empty shared params produces 1 config per backend."""
        grid = _make_grid(
            backends=["pytorch", "vllm"],
            shared={},
        )
        configs = expand_campaign_grid(grid)

        assert len(configs) == 2
        backends = {c["backend"] for c in configs}
        assert backends == {"pytorch", "vllm"}

    def test_expand_with_base_config(self) -> None:
        """Base config values are preserved in expanded configs."""
        grid = _make_grid(
            backends=["pytorch"],
            shared={"fp_precision": ["float16"]},
        )
        base = {"model_name": "base-model", "max_output_tokens": 128}
        configs = expand_campaign_grid(grid, base_config=base)

        assert len(configs) == 1
        assert configs[0]["model_name"] == "base-model"
        assert configs[0]["max_output_tokens"] == 128
        assert configs[0]["fp_precision"] == "float16"

    def test_expand_nested_keys(self) -> None:
        """Dot-notation shared keys set nested values."""
        grid = _make_grid(
            backends=["pytorch"],
            shared={"decoder.preset": ["greedy", "sampling"]},
        )
        configs = expand_campaign_grid(grid)

        assert len(configs) == 2
        presets = {c["decoder"]["preset"] for c in configs}
        assert presets == {"greedy", "sampling"}


class TestValidateCampaignGrid:
    """Tests for validate_campaign_grid — Pydantic dry-run validation."""

    def test_validate_keeps_valid(self) -> None:
        """Valid config dicts pass through validation."""
        # ExperimentConfig requires config_name, model_name, backend
        config_dicts = [
            {"config_name": "test1", "model_name": "test/model", "backend": "pytorch"},
            {"config_name": "test2", "model_name": "test/model", "backend": "vllm"},
        ]
        result = validate_campaign_grid(config_dicts)

        assert len(result.valid_configs) == 2
        assert len(result.filtered_configs) == 0
        assert result.total_generated == 2

    def test_validate_filters_invalid(self) -> None:
        """Invalid configs (e.g., bad backend) filtered with reason."""
        config_dicts = [
            {"config_name": "test1", "model_name": "test/model", "backend": "pytorch"},
            {"config_name": "test2", "model_name": "test/model", "backend": "not_a_real_backend"},
        ]
        result = validate_campaign_grid(config_dicts)

        assert len(result.valid_configs) == 1
        assert len(result.filtered_configs) == 1
        assert result.filtered_configs[0].severity == "error"
        assert len(result.filtered_configs[0].reason) > 0


class TestGridExpansionResult:
    """Tests for GridExpansionResult model."""

    def test_grid_expansion_result_summary(self) -> None:
        """Summary string shows correct counts."""
        result = GridExpansionResult(
            valid_configs=[{"backend": "pytorch"}],
            filtered_configs=[],
            warnings=[],
            total_generated=1,
        )
        summary = result.summary
        assert "Generated 1 experiments" in summary
        assert "1 valid" in summary

    def test_grid_expansion_result_summary_with_filtered(self) -> None:
        """Summary includes filtered count and reasons."""
        from llenergymeasure.orchestration.grid import GridValidationIssue

        result = GridExpansionResult(
            valid_configs=[{"backend": "pytorch"}],
            filtered_configs=[
                GridValidationIssue(
                    config_desc="bad/config",
                    reason="Invalid backend",
                    severity="error",
                )
            ],
            warnings=[],
            total_generated=2,
        )
        summary = result.summary
        assert "Generated 2 experiments" in summary
        assert "1 filtered" in summary
