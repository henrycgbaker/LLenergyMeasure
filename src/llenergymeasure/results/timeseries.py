"""Time-series export for power/thermal/memory data.

Exports per-process time-series data to separate JSON files, keeping
the main results lightweight. Uses compact keys to minimise file size
while including summary statistics in the header.

File layout:
    results/raw/exp_ID/
    ├── process_0.json              # Main results
    ├── process_0_timeseries.json   # Time-series (optional)
    └── .completed_0
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from llenergymeasure.constants import SCHEMA_VERSION

if TYPE_CHECKING:
    from llenergymeasure.core.power_thermal import PowerThermalSample


def export_timeseries(
    samples: list[PowerThermalSample],
    experiment_id: str,
    process_index: int,
    output_dir: Path,
    sample_interval_ms: int = 100,
) -> Path:
    """Export power/thermal time-series to a JSON file.

    Creates a compact JSON file with per-sample data and summary statistics.
    Uses short keys (t, power_w, mem_mb, temp_c, sm_pct, throttle) to keep
    file size manageable for long experiments.

    Args:
        samples: List of PowerThermalSample from the sampler.
        experiment_id: Experiment identifier.
        process_index: Process rank in distributed setup.
        output_dir: Directory for the output file.
        sample_interval_ms: Configured sample interval in milliseconds.

    Returns:
        Path to the created JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"process_{process_index}_timeseries.json"

    if not samples:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "process_index": process_index,
            "sample_count": 0,
            "sample_interval_ms": sample_interval_ms,
            "duration_sec": 0.0,
            "summary": {},
            "samples": [],
        }
        _atomic_write_json(data, output_path)
        logger.debug(f"Timeseries: empty file for process {process_index}")
        return output_path

    base_time = samples[0].timestamp

    # Compute summary statistics
    powers = [s.power_w for s in samples if s.power_w is not None]
    memories = [s.memory_used_mb for s in samples if s.memory_used_mb is not None]
    temps = [s.temperature_c for s in samples if s.temperature_c is not None]
    throttled = [s for s in samples if s.thermal_throttle]

    summary: dict[str, Any] = {}
    if powers:
        summary["power_mean_w"] = round(sum(powers) / len(powers), 2)
        summary["power_min_w"] = round(min(powers), 2)
        summary["power_max_w"] = round(max(powers), 2)
    if memories:
        summary["memory_mean_mb"] = round(sum(memories) / len(memories), 1)
        summary["memory_max_mb"] = round(max(memories), 1)
    if temps:
        summary["temperature_mean_c"] = round(sum(temps) / len(temps), 1)
        summary["temperature_max_c"] = round(max(temps), 1)
    summary["thermal_throttle_detected"] = len(throttled) > 0
    summary["thermal_throttle_sample_count"] = len(throttled)

    # Build compact sample list
    sample_list = [
        {
            "t": round(s.timestamp - base_time, 4),
            "power_w": round(s.power_w, 2) if s.power_w is not None else None,
            "mem_mb": round(s.memory_used_mb, 1) if s.memory_used_mb is not None else None,
            "temp_c": s.temperature_c,
            "sm_pct": s.sm_utilisation,
            "throttle": s.thermal_throttle,
        }
        for s in samples
    ]

    duration = samples[-1].timestamp - base_time if len(samples) > 1 else 0.0

    data = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "process_index": process_index,
        "sample_count": len(samples),
        "sample_interval_ms": sample_interval_ms,
        "duration_sec": round(duration, 4),
        "summary": summary,
        "samples": sample_list,
    }

    _atomic_write_json(data, output_path)
    logger.debug(
        f"Timeseries: {len(samples)} samples for process {process_index} -> {output_path.name}"
    )
    return output_path


def load_timeseries(path: Path) -> dict[str, Any]:
    """Load time-series data from a JSON file.

    Args:
        path: Path to the timeseries JSON file.

    Returns:
        Parsed timeseries data dict.

    Raises:
        FileNotFoundError: If path doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Timeseries file not found: {path}")

    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    return data


def aggregate_timeseries(
    timeseries_paths: list[Path],
    output_path: Path,
    experiment_id: str,
) -> Path:
    """Bundle per-process timeseries into one aggregated file.

    Combines all per-process timeseries into a single file for convenient
    access. Does not attempt to align timestamps across processes — each
    process's data is included as-is.

    Args:
        timeseries_paths: Paths to per-process timeseries JSON files.
        output_path: Path for the aggregated output file.
        experiment_id: Experiment identifier.

    Returns:
        Path to the created aggregated file.
    """
    processes = []
    for p in timeseries_paths:
        try:
            processes.append(load_timeseries(p))
        except FileNotFoundError:
            logger.warning(f"Timeseries file missing, skipping: {p}")

    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "process_count": len(processes),
        "processes": processes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(data, output_path)
    logger.debug(f"Aggregated timeseries: {len(processes)} processes -> {output_path.name}")
    return output_path


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON atomically via temp file + rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
        prefix=path.stem,
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=None, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
