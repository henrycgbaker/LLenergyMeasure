"""Container failure classification: log tail, error payload, error mapping.

When a container exits non-zero, this module persists the stderr tail to a
``container.log`` in the exchange dir (so it survives for post-mortem), then
builds the right :class:`DockerError`: it prefers the structured error JSON the
container entrypoint writes over docker's own stderr (which can carry misleading
daemon messages), falling back to keyword-based translation of docker stderr.
"""

from __future__ import annotations

import logging
from pathlib import Path

from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    capture_stderr_snippet,
    translate_docker_error,
)
from llenergymeasure.utils.exceptions import DockerError
from llenergymeasure.utils.io import load_json

logger = logging.getLogger(__name__)


def classify_container_failure(
    *,
    returncode: int,
    stderr_text: str,
    image: str,
    exchange_dir: Path,
    config_hash: str,
) -> DockerError:
    """Persist the log tail and return the classified failure error.

    Writes ``container.log`` in the exchange dir, then returns a
    :class:`DockerError` with ``exchange_dir`` attached for debug discovery. The
    exchange dir is NOT cleaned up by the caller after a failure - it is
    preserved for post-mortem, so ``container.log`` and the error JSON survive.
    """
    logger.debug("Container failed (exit %d). Debug artifacts at %s", returncode, exchange_dir)

    # Persist container stderr to a log file in the exchange dir so it survives
    # for post-mortem debugging.
    container_log_path = exchange_dir / "container.log"
    try:
        container_log_path.write_text(stderr_text or "(no stderr captured)", encoding="utf-8")
        logger.debug("Container log written to %s", container_log_path)
    except Exception as write_exc:
        logger.warning("Failed to write container.log: %s", write_exc)

    # Prefer structured error JSON written by the container entrypoint over
    # Docker's stderr, which can contain misleading daemon messages.
    error_json_path = exchange_dir / f"{config_hash}_error.json"
    error: DockerError
    if error_json_path.exists():
        payload = load_json(error_json_path)
        error = DockerContainerError(
            message=f"{payload.get('type', 'UnknownError')}: {payload.get('message', '')}",
            fix_suggestion="Check the error traceback in the error JSON for details.",
            stderr_snippet=capture_stderr_snippet(stderr_text) if stderr_text else None,
        )
        error.error_payload = payload
    else:
        error = translate_docker_error(returncode, stderr_text, image)

    error.exchange_dir = str(exchange_dir)
    return error
