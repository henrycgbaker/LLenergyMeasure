"""Inference metrics utilities for LLM benchmarking."""

from __future__ import annotations

from llenergymeasure.domain.metrics import InferenceMetrics


def calculate_inference_metrics(
    num_prompts: int,
    latencies_ms: list[float],
    total_input_tokens: int,
    total_generated_tokens: int,
) -> InferenceMetrics:
    """Calculate inference performance metrics.

    Args:
        num_prompts: Number of prompts processed.
        latencies_ms: List of per-batch latencies in milliseconds.
        total_input_tokens: Total input tokens processed.
        total_generated_tokens: Total tokens generated.

    Returns:
        InferenceMetrics with calculated values.
    """
    total_time_sec = sum(latencies_ms) / 1000.0 if latencies_ms else 0.0
    tokens_per_sec = total_generated_tokens / total_time_sec if total_time_sec > 0 else 0.0
    latency_per_token_ms = 1000.0 / tokens_per_sec if tokens_per_sec > 0 else 0.0

    return InferenceMetrics(
        total_tokens=total_input_tokens + total_generated_tokens,
        input_tokens=total_input_tokens,
        output_tokens=total_generated_tokens,
        inference_time_sec=total_time_sec,
        tokens_per_second=tokens_per_sec,
        latency_per_token_ms=latency_per_token_ms,
    )
