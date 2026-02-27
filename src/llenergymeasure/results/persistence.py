"""v2.0 results persistence — save, load, atomic writes.

Handles directory lifecycle ({name}_{timestamp}/), collision avoidance,
JSON serialisation (primary), and Parquet sidecar management.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.domain.experiment import ExperimentResult

logger = logging.getLogger(__name__)


def _experiment_dir_name(result: ExperimentResult) -> str:
    """Generate {model_slug}_{backend}_{timestamp} directory name.

    model_slug: from effective_config["model"], / replaced by -, lowercased.
    backend: from result.backend.
    timestamp: ISO 8601 minute-granular (2026-02-18T14-30), colons -> hyphens.
    """
    raw_model = result.effective_config.get("model", "unknown")
    model_slug = str(raw_model).replace("/", "-").lower()
    backend = result.backend
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    return f"{model_slug}_{backend}_{timestamp}"


def _find_collision_free_dir(base: Path) -> Path:
    """Return base or base_1, base_2, etc. — never overwrites.

    Creates the directory atomically to avoid race conditions.
    """
    target = base
    counter = 0
    while target.exists():
        counter += 1
        target = Path(f"{base}_{counter}")
    target.mkdir(parents=True)
    return target


def _atomic_write(content: str, path: Path) -> None:
    """Write content to path atomically via temp file + os.replace().

    Uses POSIX rename semantics — atomic on same filesystem.
    Cleans up temp file on failure.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.stem)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def save_result(
    result: ExperimentResult,
    output_dir: Path,
    timeseries_source: Path | None = None,
) -> Path:
    """Save ExperimentResult to a collision-safe subdirectory of output_dir.

    Creates: {output_dir}/{model_slug}_{backend}_{timestamp}/result.json
    If timeseries_source provided: copies to {dir}/timeseries.parquet.

    Args:
        result: The experiment result to persist.
        output_dir: Parent directory. Created if missing.
        timeseries_source: Optional path to existing .parquet file to copy in.

    Returns:
        Path to the result.json file (usable with load_result() directly).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_name = _experiment_dir_name(result)
    base_dir = output_dir / dir_name
    target_dir = _find_collision_free_dir(base_dir)

    result_path = target_dir / "result.json"
    _atomic_write(result.model_dump_json(indent=2), result_path)
    logger.debug("Saved result to %s", result_path)

    if timeseries_source is not None:
        timeseries_source = Path(timeseries_source)
        if timeseries_source.exists():
            dest = target_dir / "timeseries.parquet"
            shutil.copy2(timeseries_source, dest)
            logger.debug("Copied timeseries sidecar to %s", dest)
        else:
            logger.warning("timeseries_source %s does not exist — skipping copy", timeseries_source)

    return result_path


def load_result(path: Path) -> ExperimentResult:
    """Load ExperimentResult from a result.json path.

    Auto-discovers timeseries.parquet sidecar in the same directory.
    If the result references a sidecar but the file is missing, loads
    successfully and emits a UserWarning (graceful degradation).

    Args:
        path: Path to result.json (as returned by save_result()).

    Returns:
        ExperimentResult loaded from disk.
    """
    from llenergymeasure.domain.experiment import ExperimentResult

    path = Path(path)
    content = path.read_text(encoding="utf-8")
    result = ExperimentResult.model_validate_json(content)

    sidecar = path.parent / "timeseries.parquet"
    if result.timeseries is not None and not sidecar.exists():
        warnings.warn(
            f"Timeseries sidecar missing at {sidecar}. "
            "result.timeseries field preserved but file is not present.",
            UserWarning,
            stacklevel=2,
        )

    return result
