"""Export functionality for benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from loguru import logger

from llm_energy_measure.domain.experiment import AggregatedResult, RawProcessResult


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
    results: list[AggregatedResult],
    output_path: Path,
    include_process_breakdown: bool = False,
) -> Path:
    """Export aggregated results to CSV.

    Args:
        results: List of aggregated results to export.
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

    logger.info(f"Exported {len(results)} results to {output_path}")
    return output_path


def _aggregated_to_row(
    result: AggregatedResult,
    include_process_breakdown: bool,
) -> dict[str, Any]:
    """Convert an aggregated result to a flat row dict."""
    row: dict[str, Any] = {
        "experiment_id": result.experiment_id,
        "start_time": result.start_time.isoformat(),
        "end_time": result.end_time.isoformat(),
        "duration_sec": result.duration_sec,
        "num_processes": result.aggregation.num_processes,
        "aggregation_method": result.aggregation.method,
        # Core metrics
        "total_tokens": result.total_tokens,
        "total_energy_j": result.total_energy_j,
        "total_inference_time_sec": result.total_inference_time_sec,
        "avg_tokens_per_second": result.avg_tokens_per_second,
        "avg_energy_per_token_j": result.avg_energy_per_token_j,
        "total_flops": result.total_flops,
        "tokens_per_joule": result.tokens_per_joule,
        # Verification
        "temporal_overlap_verified": result.aggregation.temporal_overlap_verified,
        "gpu_attribution_verified": result.aggregation.gpu_attribution_verified,
    }

    # Add warnings if any
    if result.aggregation.warnings:
        row["warnings"] = "; ".join(result.aggregation.warnings)

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
    """Order columns in a logical sequence."""
    # Priority ordering - these appear first
    priority = [
        "experiment_id",
        "start_time",
        "end_time",
        "duration_sec",
        "num_processes",
        "total_tokens",
        "total_energy_j",
        "avg_tokens_per_second",
        "tokens_per_joule",
        "avg_energy_per_token_j",
        "total_flops",
        "total_inference_time_sec",
        "aggregation_method",
        "temporal_overlap_verified",
        "gpu_attribution_verified",
        "warnings",
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

    logger.info(f"Exported {len(results)} raw results to {output_path}")
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
        results: list[AggregatedResult],
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
        results: list[AggregatedResult],
        filename: str = "results.json",
    ) -> Path:
        """Export aggregated results to JSON."""
        import json

        path = self._output_dir / filename
        data = [r.model_dump(mode="json") for r in results]

        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(results)} results to {path}")
        return path
