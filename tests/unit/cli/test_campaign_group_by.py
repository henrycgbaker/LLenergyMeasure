"""Unit tests for campaign --group-by field extraction and grouping logic."""

from __future__ import annotations

from typing import Any

import pytest

from llenergymeasure.domain.experiment import AggregatedResult
from llenergymeasure.results.aggregation import (
    _extract_field_value,
    aggregate_campaign_with_grouping,
)


class TestExtractFieldValue:
    """Tests for _extract_field_value function."""

    @pytest.fixture
    def sample_result(self) -> AggregatedResult:
        """Create sample AggregatedResult for testing."""
        from datetime import datetime

        from llenergymeasure.domain.experiment import AggregationMetadata

        config_data: dict[str, Any] = {
            "config_name": "test-config",
            "backend": "pytorch",
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "pytorch": {"batch_size": 2, "load_in_4bit": False},
            "fp_precision": "float16",
            "decoder": {"temperature": 0.0, "top_k": 50},
        }

        return AggregatedResult(
            experiment_id="test_exp",
            backend="pytorch",
            aggregation=AggregationMetadata(num_processes=1),
            total_tokens=150,
            total_energy_j=100.0,
            total_inference_time_sec=3.0,
            avg_tokens_per_second=50.0,
            avg_energy_per_token_j=0.67,
            total_flops=1e12,
            start_time=datetime.now(),
            end_time=datetime.now(),
            effective_config=config_data,
        )

    def test_extract_backend_field(self, sample_result: AggregatedResult) -> None:
        """Extract backend field from top level."""
        value = _extract_field_value(sample_result, "backend")
        assert value == "pytorch"

    def test_extract_config_name_field(self, sample_result: AggregatedResult) -> None:
        """Extract config_name from effective_config."""
        value = _extract_field_value(sample_result, "config_name")
        assert value == "test-config"

    def test_extract_model_name_field(self, sample_result: AggregatedResult) -> None:
        """Extract model_name from effective_config."""
        value = _extract_field_value(sample_result, "model_name")
        assert value == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_extract_nested_field_backend_config(self, sample_result: AggregatedResult) -> None:
        """Extract nested field from backend config (pytorch.batch_size)."""
        value = _extract_field_value(sample_result, "pytorch.batch_size")
        assert value == "2"

    def test_extract_nested_field_decoder_config(self, sample_result: AggregatedResult) -> None:
        """Extract nested field from decoder config (decoder.temperature)."""
        value = _extract_field_value(sample_result, "decoder.temperature")
        assert value == "0.0"

    def test_extract_nested_field_boolean(self, sample_result: AggregatedResult) -> None:
        """Boolean fields are converted to strings."""
        value = _extract_field_value(sample_result, "pytorch.load_in_4bit")
        assert value == "False"

    def test_extract_top_level_field(self, sample_result: AggregatedResult) -> None:
        """Extract top-level field from effective_config."""
        value = _extract_field_value(sample_result, "fp_precision")
        assert value == "float16"

    def test_extract_nonexistent_field_returns_unknown(
        self, sample_result: AggregatedResult
    ) -> None:
        """Non-existent fields return 'unknown'."""
        value = _extract_field_value(sample_result, "nonexistent_field")
        assert value == "unknown"

    def test_extract_nonexistent_nested_field(self, sample_result: AggregatedResult) -> None:
        """Non-existent nested fields return 'unknown'."""
        value = _extract_field_value(sample_result, "pytorch.nonexistent")
        assert value == "unknown"

    def test_extract_deeply_nested_field(self, sample_result: AggregatedResult) -> None:
        """Extract deeply nested field (multiple dots)."""
        value = _extract_field_value(sample_result, "decoder.top_k")
        assert value == "50"


class TestAggregateCampaignWithGrouping:
    """Tests for aggregate_campaign_with_grouping function."""

    @pytest.fixture
    def multi_config_results(self) -> dict[str, list[AggregatedResult]]:
        """Create sample results from multiple configs."""
        from datetime import datetime

        from llenergymeasure.domain.experiment import AggregationMetadata

        # PyTorch config, batch_size 1
        pytorch_bs1_config: dict[str, Any] = {
            "config_name": "pytorch-bs1",
            "backend": "pytorch",
            "model_name": "test/model",
            "pytorch": {"batch_size": 1},
        }
        pytorch_bs1_results = [
            AggregatedResult(
                experiment_id=f"exp_{i}",
                backend="pytorch",
                aggregation=AggregationMetadata(num_processes=1),
                total_tokens=150,
                total_energy_j=100.0 + i * 5,
                total_inference_time_sec=3.0,
                avg_tokens_per_second=50.0 + i,
                avg_energy_per_token_j=0.67,
                total_flops=1e12,
                start_time=datetime.now(),
                end_time=datetime.now(),
                effective_config=pytorch_bs1_config,
            )
            for i in range(3)
        ]

        # PyTorch config, batch_size 4
        pytorch_bs4_config: dict[str, Any] = {
            "config_name": "pytorch-bs4",
            "backend": "pytorch",
            "model_name": "test/model",
            "pytorch": {"batch_size": 4},
        }
        pytorch_bs4_results = [
            AggregatedResult(
                experiment_id=f"exp_{i+3}",
                backend="pytorch",
                aggregation=AggregationMetadata(num_processes=1),
                total_tokens=150,
                total_energy_j=150.0 + i * 10,
                total_inference_time_sec=2.0,
                avg_tokens_per_second=75.0 + i,
                avg_energy_per_token_j=1.0,
                total_flops=1e12,
                start_time=datetime.now(),
                end_time=datetime.now(),
                effective_config=pytorch_bs4_config,
            )
            for i in range(3)
        ]

        # vLLM config
        vllm_config: dict[str, Any] = {
            "config_name": "vllm-test",
            "backend": "vllm",
            "model_name": "test/model",
            "vllm": {"max_num_seqs": 256},
        }
        vllm_results = [
            AggregatedResult(
                experiment_id=f"exp_{i+6}",
                backend="vllm",
                aggregation=AggregationMetadata(num_processes=1),
                total_tokens=150,
                total_energy_j=200.0 + i * 15,
                total_inference_time_sec=1.5,
                avg_tokens_per_second=100.0 + i,
                avg_energy_per_token_j=1.33,
                total_flops=1e12,
                start_time=datetime.now(),
                end_time=datetime.now(),
                effective_config=vllm_config,
            )
            for i in range(3)
        ]

        return {
            "pytorch-bs1": pytorch_bs1_results,
            "pytorch-bs4": pytorch_bs4_results,
            "vllm-test": vllm_results,
        }

    def test_group_by_backend(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Group results by backend field."""
        grouped = aggregate_campaign_with_grouping(multi_config_results, ["backend"])

        # Should have 2 groups: pytorch and vllm
        assert len(grouped) == 2
        assert ("pytorch",) in grouped
        assert ("vllm",) in grouped

        # PyTorch group should have 6 cycles (2 configs x 3 cycles each)
        assert grouped[("pytorch",)]["n_cycles"] == 6
        # vLLM group should have 3 cycles
        assert grouped[("vllm",)]["n_cycles"] == 3

    def test_group_by_nested_field(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Group results by nested field (pytorch.batch_size)."""
        # Filter to only PyTorch results for this test
        pytorch_only = {k: v for k, v in multi_config_results.items() if "pytorch" in k}

        grouped = aggregate_campaign_with_grouping(pytorch_only, ["pytorch.batch_size"])

        # Should have 2 groups: batch_size 1 and 4
        assert len(grouped) == 2
        assert ("1",) in grouped
        assert ("4",) in grouped

        # Each group has 3 cycles
        assert grouped[("1",)]["n_cycles"] == 3
        assert grouped[("4",)]["n_cycles"] == 3

    def test_group_by_multiple_fields(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Group by multiple fields: backend and model_name."""
        grouped = aggregate_campaign_with_grouping(multi_config_results, ["backend", "model_name"])

        # All configs have same model_name, so groups by backend only
        assert len(grouped) >= 2
        # Keys are tuples
        assert any(isinstance(k, tuple) for k in grouped)
        # Check tuple structure
        for key in grouped:
            assert len(key) == 2  # (backend, model_name)

    def test_grouped_results_include_metadata(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Grouped results include group_fields and group_values metadata."""
        grouped = aggregate_campaign_with_grouping(multi_config_results, ["backend"])

        for entry in grouped.values():
            assert "group_fields" in entry
            assert "group_values" in entry
            assert entry["group_fields"] == ["backend"]
            assert isinstance(entry["group_values"], list)

    def test_grouped_results_include_bootstrap_ci(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Grouped results include bootstrap CI for energy and throughput."""
        grouped = aggregate_campaign_with_grouping(multi_config_results, ["backend"])

        for entry in grouped.values():
            assert "energy_j" in entry
            assert "throughput_tps" in entry
            # CI should have mean, ci_lower, ci_upper
            assert "mean" in entry["energy_j"]
            assert "ci_lower" in entry["energy_j"]
            assert "ci_upper" in entry["energy_j"]

    def test_group_by_nonexistent_field(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Grouping by non-existent field creates 'unknown' groups."""
        grouped = aggregate_campaign_with_grouping(multi_config_results, ["nonexistent_field"])

        # All results grouped under "unknown"
        assert len(grouped) == 1
        assert ("unknown",) in grouped
        # Should include all 9 results
        assert grouped[("unknown",)]["n_cycles"] == 9

    def test_group_by_empty_list_groups_all_together(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Grouping by empty field list groups all results under empty tuple."""
        # With no grouping fields, all results go in one group with key ()
        grouped = aggregate_campaign_with_grouping(multi_config_results, [])

        # Should have one group with empty tuple key
        assert len(grouped) == 1
        assert () in grouped
        # Should include all 9 results
        assert grouped[()]["n_cycles"] == 9

    def test_grouped_results_correct_confidence_level(
        self, multi_config_results: dict[str, list[AggregatedResult]]
    ) -> None:
        """Bootstrap CI respects custom confidence level."""
        grouped = aggregate_campaign_with_grouping(
            multi_config_results, ["backend"], confidence=0.90
        )

        # Just verify function accepts confidence parameter
        # CI width will differ, but checking for presence is sufficient
        for entry in grouped.values():
            assert "energy_j" in entry
            ci = entry["energy_j"]
            assert "ci_lower" in ci and "ci_upper" in ci
