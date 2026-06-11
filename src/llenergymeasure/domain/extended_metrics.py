"""Extended efficiency metrics computation (pure math, Layer 0).

Computes extended metrics from raw inference data. All metrics use graceful
degradation - null values when data is unavailable, never errors.

This module is pure: it depends only on ``domain.metrics`` and numpy, so it sits
at the bottom of the architectural layering and can be called from the harness.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from llenergymeasure.domain.metrics import (
    BatchEfficiencyMetrics,
    ExtendedEfficiencyMetrics,
    GPUUtilisationMetrics,
    KVCacheEfficiencyMetrics,
    MemoryEfficiencyMetrics,
    RequestLatencyMetrics,
)

logger = logging.getLogger(__name__)


def compute_extended_metrics(
    output_tokens: int,
    total_energy_j: float,
    tokens_per_second: float,
    precision_factor: float = 1.0,
    itl_mean_ms: float | None = None,
    per_request_latencies_ms: list[float] | None = None,
    gpu_utilisation_samples: list[float] | None = None,
    memory_bandwidth_samples: list[float] | None = None,
    memory_stats: dict[str, float] | None = None,
    batch_stats: dict[str, Any] | None = None,
    kv_cache_stats: dict[str, Any] | None = None,
) -> ExtendedEfficiencyMetrics:
    """Compute extended efficiency metrics from raw data.

    All inputs are optional. Metrics are null when underlying data unavailable.

    Args:
        output_tokens: Number of output tokens generated.
        total_energy_j: Total energy consumed in Joules.
        tokens_per_second: Throughput in tokens/second.
        precision_factor: Precision factor (1.0 for FP16/FP32, 0.5 for INT8, etc.).
        itl_mean_ms: Mean inter-token latency in ms (for TPOT).
        per_request_latencies_ms: Per-request E2E latencies in ms.
        gpu_utilisation_samples: GPU SM utilisation samples (0-100).
        memory_bandwidth_samples: GPU memory-bandwidth utilisation samples (0-100).
        memory_stats: Dict with peak_mb, total_vram_mb, model_mb, kv_cache_mb.
        batch_stats: Dict with effective_batch_size, padding_overhead, num_batches.
        kv_cache_stats: Dict with hit_rate, blocks_used, blocks_total (vLLM).

    Returns:
        ExtendedEfficiencyMetrics with computed values (nulls where unavailable).
    """
    metrics = ExtendedEfficiencyMetrics()

    if output_tokens == 0:
        logger.warning("Output tokens is 0 - derived efficiency metrics will be unavailable")
        return metrics

    # TPOT (Time Per Output Token) - industry standard alias for ITL mean
    if itl_mean_ms is not None:
        metrics.tpot_ms = itl_mean_ms

    # Token Efficiency Index - composite score
    if tokens_per_second > 0 and total_energy_j > 0 and output_tokens > 0:
        tokens_per_joule = output_tokens / total_energy_j
        metrics.token_efficiency_index = tokens_per_second * tokens_per_joule * precision_factor

    # Memory efficiency
    metrics.memory = _compute_memory_metrics(output_tokens, memory_stats)

    # GPU utilisation
    metrics.gpu_utilisation = _compute_gpu_utilisation_metrics(
        gpu_utilisation_samples, memory_bandwidth_samples
    )

    # Batch efficiency
    metrics.batch = _compute_batch_metrics(batch_stats)

    # KV cache efficiency (vLLM only)
    metrics.kv_cache = _compute_kv_cache_metrics(kv_cache_stats)

    # Request latency statistics
    metrics.request_latency = _compute_request_latency_metrics(per_request_latencies_ms)

    return metrics


def _compute_memory_metrics(
    output_tokens: int,
    memory_stats: dict[str, float] | None,
) -> MemoryEfficiencyMetrics:
    """Compute memory efficiency metrics."""
    mem = MemoryEfficiencyMetrics()

    if not memory_stats:
        return mem

    # Raw values
    mem.total_vram_mb = memory_stats.get("total_vram_mb", 0.0)
    mem.model_memory_mb = memory_stats.get("model_mb", 0.0)
    mem.peak_memory_mb = memory_stats.get("peak_mb", 0.0)
    mem.kv_cache_mb = memory_stats.get("kv_cache_mb")

    # Derived metrics
    if mem.peak_memory_mb > 0 and output_tokens > 0:
        # Tokens per GB of VRAM used
        peak_gb = mem.peak_memory_mb / 1024
        if peak_gb > 0:
            mem.tokens_per_gb_vram = output_tokens / peak_gb

    if mem.total_vram_mb > 0 and mem.model_memory_mb > 0:
        mem.model_memory_utilisation = mem.model_memory_mb / mem.total_vram_mb

    if mem.kv_cache_mb is not None and mem.peak_memory_mb > 0:
        mem.kv_cache_memory_ratio = mem.kv_cache_mb / mem.peak_memory_mb

    return mem


def _compute_gpu_utilisation_metrics(
    samples: list[float] | None,
    memory_bandwidth_samples: list[float] | None = None,
) -> GPUUtilisationMetrics:
    """Compute GPU utilisation metrics from samples.

    Args:
        samples: SM utilisation samples (0-100), one per NVML poll tick.
        memory_bandwidth_samples: Memory-bandwidth utilisation samples (0-100).
            NVML proxy: percent of time the memory controller was active.
    """
    gpu = GPUUtilisationMetrics()

    if samples:
        gpu.sm_utilisation_mean = float(np.mean(samples))
        gpu.sm_utilisation_samples = len(samples)

    if memory_bandwidth_samples:
        gpu.memory_bandwidth_utilisation = float(np.mean(memory_bandwidth_samples))

    return gpu


def _compute_batch_metrics(
    batch_stats: dict[str, Any] | None,
) -> BatchEfficiencyMetrics:
    """Compute batch efficiency metrics."""
    batch = BatchEfficiencyMetrics()

    if not batch_stats:
        return batch

    batch.effective_batch_size = batch_stats.get("effective_batch_size")
    batch.padding_overhead = batch_stats.get("padding_overhead")

    num_batches = batch_stats.get("num_batches")
    if num_batches is not None:
        batch.num_batches = int(num_batches)

    # Batch utilisation requires configured batch size
    configured = batch_stats.get("configured_batch_size")
    if configured and batch.effective_batch_size:
        batch.batch_utilisation = batch.effective_batch_size / configured

    return batch


def _compute_kv_cache_metrics(
    kv_cache_stats: dict[str, Any] | None,
) -> KVCacheEfficiencyMetrics:
    """Compute KV cache metrics (vLLM only)."""
    kv = KVCacheEfficiencyMetrics()

    if not kv_cache_stats:
        return kv

    kv.kv_cache_hit_rate = kv_cache_stats.get("hit_rate")
    kv.kv_cache_blocks_used = kv_cache_stats.get("blocks_used")
    kv.kv_cache_blocks_total = kv_cache_stats.get("blocks_total")

    return kv


def _compute_request_latency_metrics(
    latencies_ms: list[float] | None,
) -> RequestLatencyMetrics:
    """Compute request latency statistics."""
    req = RequestLatencyMetrics()

    if not latencies_ms:
        return req

    arr = np.array(latencies_ms)
    req.e2e_latency_mean_ms = float(np.mean(arr))
    req.e2e_latency_median_ms = float(np.median(arr))
    req.e2e_latency_p95_ms = float(np.percentile(arr, 95))
    req.e2e_latency_p99_ms = float(np.percentile(arr, 99))
    req.e2e_latency_samples = len(latencies_ms)

    return req
