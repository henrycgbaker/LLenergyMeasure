"""Tests for inference engine utilities."""

import pytest

from llm_energy_measure.core.inference import calculate_inference_metrics
from llm_energy_measure.domain.metrics import InferenceMetrics


class TestCalculateInferenceMetrics:
    """Tests for calculate_inference_metrics."""

    def test_basic_metrics(self):
        result = calculate_inference_metrics(
            num_prompts=10,
            latencies_ms=[100.0, 100.0],  # 200ms total = 0.2s
            total_input_tokens=100,
            total_generated_tokens=200,
        )

        assert isinstance(result, InferenceMetrics)
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.total_tokens == 300
        assert result.inference_time_sec == pytest.approx(0.2)
        assert result.tokens_per_second == pytest.approx(1000.0)  # 200 tokens / 0.2s

    def test_empty_latencies(self):
        result = calculate_inference_metrics(
            num_prompts=0,
            latencies_ms=[],
            total_input_tokens=0,
            total_generated_tokens=0,
        )

        assert result.inference_time_sec == 0.0
        assert result.tokens_per_second == 0.0
        assert result.latency_per_token_ms == 0.0

    def test_zero_tokens_generated(self):
        result = calculate_inference_metrics(
            num_prompts=5,
            latencies_ms=[100.0],
            total_input_tokens=50,
            total_generated_tokens=0,
        )

        assert result.tokens_per_second == 0.0
        assert result.latency_per_token_ms == 0.0

    def test_latency_per_token_calculation(self):
        result = calculate_inference_metrics(
            num_prompts=1,
            latencies_ms=[1000.0],  # 1 second
            total_input_tokens=10,
            total_generated_tokens=100,
        )

        # 100 tokens / 1 second = 100 tokens/sec
        assert result.tokens_per_second == pytest.approx(100.0)
        # 1000 ms / 100 tokens = 10 ms/token
        assert result.latency_per_token_ms == pytest.approx(10.0)

    def test_multiple_batches(self):
        result = calculate_inference_metrics(
            num_prompts=100,
            latencies_ms=[50.0, 60.0, 70.0, 80.0],  # 260ms total
            total_input_tokens=1000,
            total_generated_tokens=5000,
        )

        assert result.inference_time_sec == pytest.approx(0.26)
        expected_tps = 5000 / 0.26
        assert result.tokens_per_second == pytest.approx(expected_tps)
