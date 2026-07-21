"""Exchange-directory lifecycle: create, read the result, rescue, clean up.

The exchange dir is a host tempdir bind-mounted into the container at
``/run/llem``. The harness writes its result JSON and artefact sidecars there;
this module owns creating it, reading the result back (via the shared domain
payload parser), rescuing the artefacts before teardown, and removing it.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import TEMP_PREFIX_EXCHANGE, TEMP_PREFIX_TIMESERIES
from llenergymeasure.domain.bundle_artefacts import (
    CONFIG_SIDECAR_FILENAME,
    ENVIRONMENT_FILENAME,
    TIMESERIES_FILENAME,
)
from llenergymeasure.infra.docker_errors import DockerContainerError
from llenergymeasure.utils.io import load_json

logger = logging.getLogger(__name__)


def create_exchange_dir() -> Path:
    """Create and return a fresh exchange directory (host tempdir)."""
    return Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_EXCHANGE))


def write_config(exchange_dir: Path, config: Any, config_hash: str) -> Path:
    """Serialise the experiment config into the exchange dir; return its path.

    The container reads this at ``/run/llem/{config_hash}_config.json``. Output
    dir and save_timeseries travel via env vars (not the config), so the written
    JSON is the clean declared config - the same bytes ``config_hash`` was
    computed from.
    """
    config_path = exchange_dir / f"{config_hash}_config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    return config_path


def read_result(exchange_dir: Path, config_hash: str) -> Any:
    """Read and parse the result JSON written by the container.

    Args:
        exchange_dir: Host path of the temporary exchange directory.
        config_hash:  Hash prefix for the result file name.

    Returns:
        ExperimentResult if the file contains a valid result, or a dict
        error payload if the container wrote an error JSON.

    Raises:
        DockerContainerError: If the result file does not exist.
    """
    # Lazy import to avoid pulling the heavy domain result models at module
    # load time (the parser imports ExperimentResult transitively).
    from llenergymeasure.domain.result_payload import parse_experiment_result_payload

    result_path = exchange_dir / f"{config_hash}_result.json"
    if not result_path.exists():
        raise DockerContainerError(
            message=f"Container exited 0 but no result file found at {result_path}",
            fix_suggestion="Check container logs for errors during experiment execution.",
        )

    raw = load_json(result_path)

    # Container may write an error payload even on exit 0 (defensive check).
    # Error payloads have "type" and "traceback" keys (mirror StudyRunner worker
    # format). This detection stays here: it is exchange-specific IPC, not a
    # property of a persisted result.
    if isinstance(raw, dict) and "type" in raw and "traceback" in raw:
        return raw

    # Cross-version IPC: strip fields unknown to the host schema (container may
    # run an older/newer version) and warn on a host/container version skew.
    # Both behaviours live in the shared parser now, keyed by tolerant=True and
    # the host version as expected_version.
    return parse_experiment_result_payload(raw, tolerant=True, expected_version=__version__)


def rescue_artefacts(exchange_dir: Path) -> Path | None:
    """Move the harness artefacts out of the exchange dir before cleanup.

    The harness inside the container wrote its artefacts to /run/llem
    (= exchange_dir on host): config.json always (the sole home of provenance
    and the authoritative home of identity), environment.json (the accurate
    in-container environment snapshot - the host's own snapshot describes the
    dispatching host, not this container), and timeseries.parquet when enabled.
    Move any that are present to a fresh temp dir so the caller can copy them
    into the study directory before the exchange dir is destroyed. config.json
    must survive too - otherwise a successful docker run lands a result.json
    with no provenance.

    Returns the rescue temp dir, or ``None`` if no artefacts were present.
    """
    artefact_tmpdir: Path | None = None
    for name in (CONFIG_SIDECAR_FILENAME, ENVIRONMENT_FILENAME, TIMESERIES_FILENAME):
        src = exchange_dir / name
        if src.exists():
            if artefact_tmpdir is None:
                artefact_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES))
            shutil.move(str(src), str(artefact_tmpdir / name))
    return artefact_tmpdir


def cleanup_exchange_dir(exchange_dir: Path) -> None:
    """Remove the temporary exchange directory.

    Logs a warning on failure but never raises - cleanup must not mask
    real errors from the caller.

    Args:
        exchange_dir: Path to remove.
    """
    try:
        shutil.rmtree(exchange_dir)
    except Exception as exc:
        logger.warning("Could not remove exchange dir %s: %s", exchange_dir, exc)
