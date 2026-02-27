"""Export functionality for benchmark results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from llenergymeasure.domain.experiment import ExperimentResult, RawProcessResult

logger = logging.getLogger(__name__)


def flatten_model(model: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a Pydantic model to a flat dictionary.

    Args:
        model: Pydantic model or dict to flatten.
        prefix: Key prefix for nested fields.

    Returns:
        Flat dictionary with underscore-separated keys.
    """
    result: dict[str, Any] = {}

    if hasattr(model, "model_dump"):
        data = model.model_dump()
    elif isinstance(model, dict):
        data = model
    else:
        return {prefix: model} if prefix else {"value": model}

    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_model(value, new_key))
        elif isinstance(value, list):
            # For lists, create indexed keys
            for i, item in enumerate(value):
                if isinstance(item, dict) or hasattr(item, "model_dump"):
                    result.update(flatten_model(item, f"{new_key}_{i}"))
                else:
                    result[f"{new_key}_{i}"] = item
        else:
            result[new_key] = value

    return result


def export_aggregated_to_csv(
    results: list[ExperimentResult],
    output_path: Path,
    include_process_breakdown: bool = False,
) -> Path:
    """Export aggregated results to CSV.

    Args:
        results: List of ExperimentResult (v2.0) to export.
        output_path: Path to output CSV file.
        include_process_breakdown: Include per-process columns.

    Returns:
        Path to the created CSV file.
    """
    if not results:
        logger.warning("No results to export")
        return output_path

    rows: list[dict[str, Any]] = []

    for result in results:
        row = _aggregated_to_row(result, include_process_breakdown)
        rows.append(row)

    # Get all keys from all rows
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    # Order keys logically
    ordered_keys = _order_columns(list(all_keys))

    # Fill missing values
    for row in rows:
        for key in ordered_keys:
            if key not in row:
                row[key] = "NA"

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d results to %s", len(results), output_path)
    return output_path


def _aggregated_to_row(
    result: ExperimentResult,
    include_process_breakdown: bool,
) -> dict[str, Any]:
    """Convert an ExperimentResult (v2.0) to a flat row dict.

    Columns are grouped by prefix for CSV readability:
    - Core metrics (experiment_id, tokens, energy, throughput)
    - methodology_* : Measurement methodology fields
    - energy_* : Energy breakdown
    - thermal_* : Thermal throttling info
    - env_* : Environment snapshot
    - gpu_*/latency_*/batch_*/kv_cache_* : Extended efficiency metrics
    - timeseries : Reference to time-series data file
    """
    # Aggregation fields (may be None for single-GPU)
    agg = result.aggregation
    row: dict[str, Any] = {
        "experiment_id": result.experiment_id,
        "schema_version": result.schema_version,
        "measurement_config_hash": result.measurement_config_hash,
        "start_time": result.start_time.isoformat(),
        "end_time": result.end_time.isoformat(),
        "duration_sec": result.duration_sec,
        "num_processes": agg.num_processes if agg else 1,
        "aggregation_method": agg.method if agg else "single_process",
        # Core metrics
        "total_tokens": result.total_tokens,
        "energy_raw_j": result.total_energy_j,
        "total_inference_time_sec": result.total_inference_time_sec,
        "avg_tokens_per_second": result.avg_tokens_per_second,
        "avg_energy_per_token_j": result.avg_energy_per_token_j,
        "total_flops": result.total_flops,
        "tokens_per_joule": result.tokens_per_joule,
        # Methodology
        "measurement_methodology": result.measurement_methodology,
        "steady_state_window": (
            f"{result.steady_state_window[0]}-{result.steady_state_window[1]}"
            if result.steady_state_window
            else None
        ),
        # Verification (multi-GPU only)
        "temporal_overlap_verified": agg.temporal_overlap_verified if agg else None,
        "gpu_attribution_verified": agg.gpu_attribution_verified if agg else None,
    }

    # Add aggregation warnings if any
    if agg and agg.warnings:
        row["warnings"] = "; ".join(agg.warnings)
    elif result.measurement_warnings:
        row["warnings"] = "; ".join(result.measurement_warnings)

    # --- Energy breakdown ---
    eb = result.energy_breakdown
    row["energy_adjusted_j"] = result.energy_adjusted_j or (eb.adjusted_j if eb else None)
    row["energy_baseline_w"] = result.baseline_power_w or (eb.baseline_power_w if eb else None)
    row["energy_baseline_method"] = eb.baseline_method if eb else None

    # --- Thermal throttling ---
    tt = result.thermal_throttle
    row["thermal_throttle_detected"] = tt.detected if tt else False
    row["thermal_throttle_duration_sec"] = tt.throttle_duration_sec if tt else 0.0
    row["thermal_max_temp_c"] = tt.max_temperature_c if tt else None

    # --- Environment snapshot (v2.0 field name) ---
    env = result.environment_snapshot
    row["env_gpu_name"] = env.gpu.name if env else None
    row["env_gpu_vram_mb"] = env.gpu.vram_total_mb if env else None
    row["env_cuda_version"] = env.cuda.version if env else None
    row["env_driver_version"] = env.cuda.driver_version if env else None
    row["env_gpu_temp_c"] = env.thermal.temperature_c if env else None
    row["env_power_limit_w"] = env.thermal.power_limit_w if env else None
    row["env_cpu_governor"] = env.cpu.governor if env else None
    row["env_in_container"] = env.container.detected if env else None
    row["env_summary"] = env.summary_line if env else None

    # --- Extended efficiency metrics ---
    em = result.extended_metrics
    row["gpu_util_mean_pct"] = em.gpu_utilisation.sm_utilisation_mean if em else None
    row["gpu_mem_peak_mb"] = em.memory.peak_memory_mb if em else None
    row["latency_e2e_mean_ms"] = em.request_latency.e2e_latency_mean_ms if em else None
    row["latency_e2e_p95_ms"] = em.request_latency.e2e_latency_p95_ms if em else None
    row["batch_effective_size"] = em.batch.effective_batch_size if em else None
    row["kv_cache_hit_rate"] = em.kv_cache.kv_cache_hit_rate if em else None

    # --- Timeseries reference (v2.0 field name: timeseries not timeseries_path) ---
    row["timeseries"] = result.timeseries

    # Optionally add per-process breakdown
    if include_process_breakdown:
        for proc in result.process_results:
            prefix = f"process_{proc.process_index}"
            row[f"{prefix}_gpu_id"] = proc.gpu_id
            row[f"{prefix}_tokens"] = proc.inference_metrics.total_tokens
            row[f"{prefix}_energy_j"] = proc.energy_metrics.total_energy_j
            row[f"{prefix}_tokens_per_second"] = proc.inference_metrics.tokens_per_second

    return row


def _order_columns(keys: list[str]) -> list[str]:
    """Order columns in a logical sequence.

    Groups related columns by prefix for CSV readability:
    core > methodology_ > energy_ > thermal_ > env_ > gpu_/latency_/batch_/kv_cache_ > timeseries > process_
    """
    # Priority ordering - these appear first
    priority = [
        # Core identification and metrics
        "experiment_id",
        "schema_version",
        "measurement_config_hash",
        "start_time",
        "end_time",
        "duration_sec",
        "num_processes",
        "total_tokens",
        "energy_raw_j",
        "avg_tokens_per_second",
        "tokens_per_joule",
        "avg_energy_per_token_j",
        "total_flops",
        "total_inference_time_sec",
        "aggregation_method",
        "temporal_overlap_verified",
        "gpu_attribution_verified",
        "warnings",
        # Methodology
        "measurement_methodology",
        "steady_state_window",
        # Energy breakdown
        "energy_adjusted_j",
        "energy_baseline_w",
        "energy_baseline_method",
        # Thermal throttling
        "thermal_throttle_detected",
        "thermal_throttle_duration_sec",
        "thermal_max_temp_c",
        # Environment snapshot
        "env_gpu_name",
        "env_gpu_vram_mb",
        "env_cuda_version",
        "env_driver_version",
        "env_gpu_temp_c",
        "env_power_limit_w",
        "env_cpu_governor",
        "env_in_container",
        "env_summary",
        # Extended efficiency metrics
        "gpu_util_mean_pct",
        "gpu_mem_peak_mb",
        "latency_e2e_mean_ms",
        "latency_e2e_p95_ms",
        "batch_effective_size",
        "kv_cache_hit_rate",
        # Timeseries reference
        "timeseries",
    ]

    ordered = [k for k in priority if k in keys]
    remaining = sorted([k for k in keys if k not in priority])

    return ordered + remaining


def export_raw_to_csv(
    results: list[RawProcessResult],
    output_path: Path,
) -> Path:
    """Export raw process results to CSV.

    Args:
        results: List of raw results to export.
        output_path: Path to output CSV file.

    Returns:
        Path to the created CSV file.
    """
    if not results:
        logger.warning("No results to export")
        return output_path

    rows = [flatten_model(r) for r in results]

    # Get union of all keys
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    ordered_keys = sorted(all_keys)

    # Fill missing values
    for row in rows:
        for key in ordered_keys:
            if key not in row:
                row[key] = "NA"

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d raw results to %s", len(results), output_path)
    return output_path


class ResultsExporter:
    """Unified interface for exporting results in various formats."""

    def __init__(self, output_dir: Path):
        """Initialize exporter with output directory.

        Args:
            output_dir: Directory for exported files.
        """
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_aggregated(
        self,
        results: list[ExperimentResult],
        filename: str = "results.csv",
        include_process_breakdown: bool = False,
    ) -> Path:
        """Export aggregated results to CSV."""
        path = self._output_dir / filename
        return export_aggregated_to_csv(results, path, include_process_breakdown)

    def export_raw(
        self,
        results: list[RawProcessResult],
        filename: str = "raw_results.csv",
    ) -> Path:
        """Export raw results to CSV."""
        path = self._output_dir / filename
        return export_raw_to_csv(results, path)

    def export_json(
        self,
        results: list[ExperimentResult],
        filename: str = "results.json",
    ) -> Path:
        """Export aggregated results to JSON."""
        import json

        path = self._output_dir / filename
        data = [r.model_dump(mode="json") for r in results]

        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Exported %d results to %s", len(results), path)
        return path
