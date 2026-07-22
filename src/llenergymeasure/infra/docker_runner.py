"""DockerRunner - dispatches a single experiment to an ephemeral Docker container.

The DockerRunner manages the full container lifecycle:

1. Create a temporary exchange directory (``tempfile.mkdtemp(prefix='llem-')``)
2. Serialise ExperimentConfig to JSON in the exchange dir
3. Start ``docker run --rm --gpus all`` with the exchange dir mounted as /run/llem
4. Block until the container exits
5. Read the result JSON written by the container entrypoint
6. Clean up the exchange dir on success; preserve it on failure for debugging

This module is the facade: the concerns are decomposed into the
``llenergymeasure.infra.docker`` package (``command`` builds the argv,
``lifecycle`` runs the container process and owns the watchdog, ``exchange``
owns the exchange-dir lifecycle + result read + rescue, ``diagnostics``
classifies failures). ``DockerRunner`` composes them; its constructor and public
method signatures are unchanged.

It is consumed by StudyRunner as the dispatch mechanism when ``runner=docker``
is resolved by runner_resolution.resolve_runner().
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.domain.progress import ProgressCallback

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    ENV_HF_TOKEN,
    ENV_OUTPUT_DIR,
    ENV_SAVE_TIMESERIES,
    TEMP_PREFIX_ENV_FILE,
)
from llenergymeasure.infra.docker import command, diagnostics, exchange, lifecycle
from llenergymeasure.utils.exceptions import DockerError

__all__ = ["DockerRunner"]

logger = logging.getLogger(__name__)


@contextmanager
def _env_file(secrets: dict[str, str]) -> Iterator[Path | None]:
    """Write secrets to a temp env-file, yield path, delete on exit.

    Creates a temp file with mode 0600 (owner read-write only) via mkstemp.
    Yields None if secrets dict is empty.
    Cleanup is guaranteed via finally block (crash/SIGINT/normal exit).

    Args:
        secrets: Dict of env var name -> value pairs.

    Yields:
        Path to temp file, or None if no secrets.
    """
    if not secrets:
        yield None
        return

    fd, path_str = tempfile.mkstemp(prefix=TEMP_PREFIX_ENV_FILE, suffix=".env")
    path = Path(path_str)
    try:
        with os.fdopen(fd, "w") as f:
            for key, value in secrets.items():
                f.write(f"{key}={value}\n")
        yield path
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def _mask_secrets(text: str, secrets: dict[str, str]) -> str:
    """Replace secret values with *** in a string."""
    for v in secrets.values():
        if v and len(v) > 4:
            text = text.replace(v, "***")
    return text


class DockerRunner:
    """Dispatches a single experiment to an ephemeral Docker container.

    Lifecycle:
        1. Create temp exchange dir (``tempfile.mkdtemp(prefix='llem-')``)
        2. Write ExperimentConfig as JSON to ``{config_hash}_config.json``
        3. ``docker run --rm --gpus all -v {exchange_dir}:/run/llem``
               ``-v {pkg_dir}:/llem-src/llenergymeasure:ro``
               ``-v {requirements_file}:/llem-requirements.txt:ro``
               ``-v {entry_script}:/llem-entry.sh:ro``
               ``-v {deps_cache}:/llem-runtime-deps``
               ``-e LLEM_ENGINE={engine}``
               ``-e LLEM_CONFIG_PATH=/run/llem/{config_hash}_config.json``
               ``--entrypoint /llem-entry.sh --shm-size 8g {image}``
               (``--gpus`` and ``--shm-size`` values are configurable - see
               ``gpu_indices`` / ``LLEM_DOCKER_GPUS`` and ``LLEM_DOCKER_SHM_SIZE``).
           The entrypoint script primes any missing runtime deps to the
           bind-mounted cache, then exec's
           ``python3 -m llenergymeasure.entrypoints.container`` (routing
           through ``/opt/nvidia/nvidia_entrypoint.sh`` for TRT-LLM).
        4. Read ``{config_hash}_result.json`` from exchange dir
        5. Clean up temp dir on success; preserve on failure with debug path logged

    Args:
        image:   Docker image to run (e.g. ``"ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19.0-cuda12"``).
        timeout: Optional wall-clock timeout in seconds. None = no timeout.
        silence_timeout: Optional stdout-silence ceiling in seconds. The
                 streaming-mode watchdog kills the container if no
                 stdout/stderr line arrives within this window. None or 0
                 disables the silence watchdog (wall-clock only). Values
                 >= ``timeout`` are accepted but redundant with the
                 wall-clock - the watchdog raises whichever fires first.
        source:  Runner resolution source string (e.g. ``"yaml"``, ``"auto_detected"``).
                 Recorded in the result's runner_provenance for traceability.
        gpu_indices: Optional host GPU indices to scope the container to via
                 ``--gpus device=<indices>`` (see ``utils.env_config.ENV_DOCKER_GPUS``
                 for the index space). Sourced from ``study_execution.gpu_indices``;
                 the ``LLEM_DOCKER_GPUS`` env var overrides it (env>config).
                 ``None`` / empty preserves the default ``--gpus all``.
    """

    def __init__(
        self,
        image: str,
        timeout: float | None = None,
        silence_timeout: float | None = None,
        source: str = "unknown",
        extra_mounts: list[tuple[str, str]] | None = None,
        container_name: str | None = None,
        labels: dict[str, str] | None = None,
        gpu_indices: list[int] | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout
        # 0 / None / negative all map to "disabled" so call-sites and
        # config-side handling stay simple.
        self.silence_timeout: float | None = (
            silence_timeout if silence_timeout and silence_timeout > 0 else None
        )
        self.source = source
        self.extra_mounts = extra_mounts or []
        self._container_name = container_name
        self._labels = labels or {}
        self.gpu_indices = gpu_indices

    @property
    def short_image(self) -> str:
        """Short image tag for display (e.g. 'transformers:v0.12.0')."""
        from llenergymeasure.utils.formatting import short_name

        return short_name(self.image)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        config: Any,
        progress: ProgressCallback | None = None,
        save_timeseries: bool = True,
        skip_image_check: bool = False,
    ) -> tuple[Any, Path | None]:
        """Run an experiment inside an ephemeral Docker container.

        When a progress callback is provided, streams container stdout line by
        line and forwards JSON progress events to the callback. Lines starting
        with ``{"event":`` are parsed as progress events; other lines are
        forwarded as container log output (visible at -v).

        Args:
            config: ExperimentConfig to dispatch.
            progress: Optional ProgressCallback for step-by-step progress reporting.

        Returns:
            Tuple of (result, artefact_tmpdir):
            - result: ExperimentResult on success, or a dict error payload if the
              container wrote an error JSON.
            - artefact_tmpdir: Path to temp dir containing the rescued artefacts
              (config.json and environment.json, plus timeseries.parquet when it
              was written), or None when none were present. Caller is responsible
              for cleanup.

        Raises:
            DockerTimeoutError:    Container exceeded ``self.timeout`` seconds.
            DockerImagePullError:  Image not found or could not be pulled.
            DockerGPUAccessError:  NVIDIA Container Toolkit misconfigured.
            DockerOOMError:        Container ran out of memory.
            DockerPermissionError: Permission denied on Docker socket.
            DockerContainerError:  Generic container failure (non-zero exit).
        """
        # Lazy import to avoid heavy domain imports at module load time
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        exchange_dir = exchange.create_exchange_dir()

        # Collect secrets for env-file (never pass as CLI args)
        secrets: dict[str, str] = {}
        hf_token = os.environ.get(ENV_HF_TOKEN)
        if hf_token:
            secrets[ENV_HF_TOKEN] = hf_token

        _p = progress  # short alias

        try:
            # --- Ensure image is available (pull with visible output if needed) ---
            if not skip_image_check:
                lifecycle.ensure_image(self.image, progress=_p)

            # --- Write config JSON ---
            # Compute config_hash from the clean config (no output path mutation).
            # Output dir and save_timeseries are passed via env vars, not config.
            # The exchange dir owns the write (it owns that dir's lifecycle).
            config_hash = compute_declared_config_hash(config)
            exchange.write_config(exchange_dir, config, config_hash)

            # Pass output params via env vars so the container entrypoint can
            # forward them to the harness as runtime params.
            secrets[ENV_OUTPUT_DIR] = CONTAINER_EXCHANGE_DIR
            secrets[ENV_SAVE_TIMESERIES] = "1" if save_timeseries else "0"

            # --- Build and execute docker command ---
            t0_container: float | None = None
            if _p:
                # Show short image tag (e.g. "transformers:v0.12.0") not the full registry path
                short_image = self.short_image
                _p.on_step_start("container_start", "Starting", short_image)
                t0_container = time.perf_counter()

            # Secrets are passed via a temp env-file (mode 0600) that is deleted after
            # the container exits - they never appear in the command argument list.
            with _env_file(secrets) as env_path:
                cmd = self._build_docker_cmd(
                    config, config_hash, str(exchange_dir), env_path=env_path
                )
                logger.debug("Running docker command: %s", _mask_secrets(str(cmd), secrets))

                # Use streaming launch + wait when a progress callback is provided.
                # Container inner events (baseline, model, warmup, measure, save)
                # are forwarded as top-level steps for granular progress display.
                if _p:
                    returncode, stderr_text = self._run_container_streaming(
                        cmd,
                        _p,
                        _mask_secrets_fn=lambda t: _mask_secrets(t, secrets),
                        container_start_time=t0_container,
                    )
                else:
                    # Classic mode: blocking run until exit (backward compatible)
                    returncode, stderr_text = lifecycle.run_blocking(cmd, self.timeout)

            # --- Handle non-zero exit ---
            if returncode != 0:
                error: DockerError = diagnostics.classify_container_failure(
                    returncode=returncode,
                    stderr_text=stderr_text,
                    image=self.image,
                    exchange_dir=exchange_dir,
                    config_hash=config_hash,
                )
                # Do NOT clean up - preserve for debugging
                exchange_dir = None  # type: ignore[assignment]
                raise error

            # --- Read result ---
            result = exchange.read_result(exchange_dir, config_hash)

            # --- Rescue artefacts before cleanup ---
            # config.json / environment.json / timeseries.parquet must survive the
            # exchange-dir teardown so the caller can land them in the study dir.
            artefact_tmpdir: Path | None = None
            if not isinstance(result, dict):
                artefact_tmpdir = exchange.rescue_artefacts(exchange_dir)

            # --- Success: clean up ---
            exchange.cleanup_exchange_dir(exchange_dir)
            exchange_dir = None  # type: ignore[assignment]

            # Error payload dicts ({type, message, traceback}) are returned as-is.
            if isinstance(result, dict):
                return result, None

            return result, artefact_tmpdir

        finally:
            # Exchange dir is set to None when we've handed off or already cleaned up.
            # If it's still set here, an unexpected exception occurred - preserve for debugging.
            if exchange_dir is not None:
                logger.debug("Preserving exchange dir for debugging: %s", exchange_dir)

    # ------------------------------------------------------------------
    # Private helpers (thin delegation to the docker/ concern modules)
    # ------------------------------------------------------------------

    def _build_docker_cmd(
        self,
        config: Any,
        config_hash: str,
        exchange_dir: str,
        env_path: Path | None = None,
    ) -> list[str]:
        """Build the ``docker run`` command list (delegates to ``docker.command``)."""
        return command.build_docker_cmd(
            image=self.image,
            config=config,
            config_hash=config_hash,
            exchange_dir=exchange_dir,
            env_path=env_path,
            extra_mounts=self.extra_mounts,
            container_name=self._container_name,
            labels=self._labels,
            gpu_indices=self.gpu_indices,
        )

    def _run_container_streaming(
        self,
        cmd: list[str],
        progress: ProgressCallback | None = None,
        _mask_secrets_fn: Callable[[str], str] | None = None,
        container_start_time: float | None = None,
    ) -> tuple[int, str]:
        """Launch the container then wait for exit, streaming progress.

        The launch (:func:`docker.lifecycle.launch`) is separable from the wait
        (:func:`docker.lifecycle.wait_to_completion`, which owns the watchdog);
        this method composes them into the block-until-exit mode.
        """
        proc = lifecycle.launch(cmd)
        return lifecycle.wait_to_completion(
            proc,
            timeout=self.timeout,
            silence_timeout=self.silence_timeout,
            progress=progress,
            mask_secrets_fn=_mask_secrets_fn,
            container_start_time=container_start_time,
        )
