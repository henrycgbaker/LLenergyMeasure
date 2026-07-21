"""Experiment session seam - context-managed engine lifetimes for the sweep loop.

An :class:`ExperimentSession` is one measurement session: a context manager
whose ``__enter__`` acquires the session's resources (spawn a worker subprocess,
prepare a container), whose ``run()`` produces the session's result payload, and
whose ``__exit__`` releases everything - ALWAYS, including on the SIGINT,
circuit-break, and exception paths.

The two offline (batch) implementations here wrap today's dispatch paths:

- :class:`SubprocessSession` - one freshly spawned worker subprocess.
- :class:`DockerSession` - one blocking ``DockerRunner.run`` container.

Offline sessions produce EXACTLY ONE result per session (one engine lifetime =
one measured window). The protocol shape - a lifetime that a work method drives -
is deliberately open to a future server session (v0.8.0) that produces N results
(one per request window) over the same lifetime, without any offline-batch
assumption baked into the seam. That is the one-dispatch:N-results capability the
sweep loop is re-keyed to admit (constraint C3); this slice adds no window-spec
vocabulary and no server session.

The sweep loop in :class:`~llenergymeasure.study.runner.StudyRunner` consumes
sessions uniformly through ``_run_one`` / ``_run_one_docker``: the manifest,
circuit breaker, GPU locks, and SIGINT handling all key off the ``(session,
result)`` outcome rather than off "the dispatch function returned once".
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    TEMP_PREFIX_TIMESERIES,
    TIMEOUT_SIGTERM_GRACE,
    TIMEOUT_THREAD_JOIN,
)
from llenergymeasure.domain.progress import STEPS_LOCAL, docker_steps
from llenergymeasure.study._progress import _consume_progress_events

# Re-imported into this module's namespace so the worker-surface helpers resolve
# at this session's USE site: tests patch e.g.
# ``llenergymeasure.study.session._collect_result`` (patch-at-use-site).
from llenergymeasure.study.worker import (
    _UNSET,
    COLLECT_RESULT_PROCESS_CRASH,
    COLLECT_RESULT_TIMEOUT,
    _collect_result,
    _derive_exit_reason,
    _kill_process_group,
    _run_experiment_worker,
)

if TYPE_CHECKING:
    import threading

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.infra.runner_resolution import RunnerSpec
    from llenergymeasure.study.runner import StudyRunner

logger = logging.getLogger(__name__)


@runtime_checkable
class ExperimentSession(Protocol):
    """A context-managed engine lifetime that a work method drives to results.

    Lifecycle:
    - ``__enter__`` acquires the session's resources (env prep + subprocess
      spawn, or container preparation) and returns the session.
    - ``run()`` performs the session's measured work and returns its result
      payload. Offline sessions return exactly one result (an
      :class:`~llenergymeasure.domain.experiment.ExperimentResult` on success,
      or a failure dict); a future server session returns one per request
      window over the same lifetime.
    - ``__exit__`` releases every resource the session acquired, on the normal,
      SIGINT, circuit-break, and exception paths alike, exactly once.
    """

    def __enter__(self) -> ExperimentSession: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None: ...

    def run(self) -> Any: ...


class _OfflineSession:
    """Shared scaffolding for the two offline (one-result) session implementations.

    Holds the driving ``StudyRunner`` (the source of the shared study-level
    caches and services: env snapshot, baseline, manifest, progress, result
    handling) plus the per-dispatch identity. ``__exit__`` funnels through
    ``_cleanup()`` under a one-shot guard so teardown runs exactly once whether
    the body returned normally or raised.
    """

    def __init__(
        self,
        runner: StudyRunner,
        config: ExperimentConfig,
        *,
        config_hash: str,
        cycle: int,
        index: int,
    ) -> None:
        self._runner = runner
        self.config = config
        self.config_hash = config_hash
        self.cycle = cycle
        self.index = index
        self._torn_down = False

    def __enter__(self) -> _OfflineSession:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def _cleanup(self) -> None:  # pragma: no cover - overridden
        """Release the session's resources. Called exactly once by __exit__."""

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
        if self._torn_down:
            return None
        self._torn_down = True
        self._cleanup()
        return None


class SubprocessSession(_OfflineSession):
    """Offline session: one experiment in a freshly spawned worker subprocess.

    ``__enter__`` stages the timeseries dir, resolves the cached env snapshot +
    baseline, wires the parent<-child Pipe and the progress-event Queue, spawns
    the worker (``daemon=False`` for clean CUDA teardown), starts the progress
    consumer, marks the experiment running, and starts the process after the
    pre-dispatch GPU-memory residual check. ``run()`` drains the pipe before
    joining (H5 deadlock fix), honours the SIGINT grace -> SIGKILL escalation,
    collects the result, writes the parent-side sentinel for crash/timeout, and
    hands the result to the runner. ``__exit__`` stops the consumer, closes the
    read end, clears the active-process handle, and removes the staging dir.
    """

    def __init__(
        self,
        runner: StudyRunner,
        config: ExperimentConfig,
        mp_ctx: Any,
        *,
        config_hash: str,
        cycle: int,
        index: int,
    ) -> None:
        super().__init__(runner, config, config_hash=config_hash, cycle=cycle, index=index)
        self._mp_ctx = mp_ctx
        self.timeout = runner.study.study_execution.experiment_timeout_seconds
        self._local_spec = runner._runner_specs.get(config.engine) if runner._runner_specs else None
        # Populated in __enter__.
        self._ts_tmpdir: Path | None = None
        self._parent_conn: Any = None
        self._progress_queue: Any = None
        self._consumer: threading.Thread | None = None
        self._process: Any = None
        self._exp_start: float = 0.0
        self._consumer_stopped = False

    def __enter__(self) -> SubprocessSession:
        import threading

        runner = self._runner
        config = self.config

        # Signal study display: new experiment starting (subprocess = local steps)
        if runner._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            runner._progress.begin_experiment(
                self.index,
                format_experiment_header(config),
                list(STEPS_LOCAL),
                runner_info=self._local_spec.to_runner_info() if self._local_spec else None,
            )

        self._exp_start = time.monotonic()

        # Create a temp dir for harness artefacts. The harness receives it as
        # output_dir (a runtime param, not from config) and writes config.json
        # there always, plus timeseries.parquet when save_timeseries is on. The
        # staging dir is created regardless of save_timeseries so the config.json
        # sidecar - sole home of provenance, authoritative home of identity -
        # always materialises; __exit__ removes it after _handle_result copies
        # the artefacts into the study directory.
        save_ts = runner.study.output.save_timeseries
        self._ts_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES))

        # Resolve cached snapshot in parent - serialised to subprocess via Pipe
        snapshot = runner._get_env_snapshot()

        # Resolve cached baseline in parent - avoids 30s re-measurement per subprocess
        baseline = runner._get_baseline(config) if config.measurement.baseline.enabled else None

        self._parent_conn, child_conn = self._mp_ctx.Pipe(duplex=False)
        self._progress_queue = self._mp_ctx.Queue()

        self._process = self._mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, self._progress_queue, snapshot),
            kwargs={
                "output_dir": str(self._ts_tmpdir),
                "save_timeseries": save_ts,
                "baseline": baseline,
                "study_dir": str(runner.study_dir),
                "study_run_id": runner.study_run_id,
                "cycle": self.cycle,
                "config_hash": self.config_hash,
            },
            daemon=False,  # daemon=False: clean CUDA teardown if parent exits unexpectedly
        )

        self._consumer = threading.Thread(
            target=_consume_progress_events,
            args=(self._progress_queue, runner._progress),
            daemon=True,
        )
        self._consumer.start()

        runner.manifest.mark_running(self.config_hash, self.cycle)
        runner._active_process = self._process

        # Pre-dispatch GPU memory residual check (MEAS-01, MEAS-02)
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

        check_gpu_memory_residual()

        self._process.start()
        child_conn.close()
        return self

    def run(self) -> Any:
        runner = self._runner
        config = self.config
        p = self._process
        parent_conn = self._parent_conn
        timeout = self.timeout

        # Drain pipe BEFORE join to prevent buffer deadlock (H5).
        # If pickled ExperimentResult > 64 KB, child blocks on conn.send()
        # while parent blocks in p.join() - classic deadlock.
        pipe_payload = _UNSET
        if parent_conn.poll(timeout=timeout):
            try:
                pipe_payload = parent_conn.recv()
            except Exception:
                pipe_payload = _UNSET

        # Non-blocking join after pipe is drained (grace for teardown)
        p.join(timeout=TIMEOUT_THREAD_JOIN)

        # SIGINT was received during join: SIGTERM was already sent by handler.
        # Grace period for clean CUDA teardown, then SIGKILL.
        if runner._interrupt_event.is_set() and p.is_alive():
            p.join(timeout=TIMEOUT_SIGTERM_GRACE)
            if p.is_alive():
                _kill_process_group(p.pid, signal.SIGKILL)
                p.join()

        runner._active_process = None

        # Sentinel stops consumer thread - covers SIGKILL path too
        self._stop_consumer()

        result = _collect_result(p, parent_conn, config, timeout, pipe_payload=pipe_payload)

        # Parent writes the sentinel record for SIGKILL / timeout - the
        # worker's context manager can't flush when its ``__exit__`` never
        # ran. ``write_sentinel`` is itself best-effort and swallows OSError.
        if isinstance(result, dict) and result.get("type") in {
            COLLECT_RESULT_PROCESS_CRASH,
            COLLECT_RESULT_TIMEOUT,
        }:
            from llenergymeasure.study.runtime_observations import write_sentinel

            exit_reason = (
                "timeout"
                if result.get("type") == COLLECT_RESULT_TIMEOUT
                else _derive_exit_reason(p.exitcode)
            )
            write_sentinel(
                config,
                study_dir=runner.study_dir,
                study_run_id=runner.study_run_id,
                cycle=self.cycle,
                config_hash=self.config_hash,
                exit_reason=exit_reason,
                exit_code=p.exitcode,
            )

        exp_elapsed = time.monotonic() - self._exp_start
        runner._handle_result(
            result,
            config,
            self.config_hash,
            self.cycle,
            self.index,
            exp_elapsed,
            ts_source_dir=self._ts_tmpdir,
            environment_snapshot=(
                runner._get_env_snapshot() if not isinstance(result, dict) else None
            ),
            runner_provenance=_provenance_from_spec(self._local_spec),
            runner_environment=(
                _runner_environment(self._local_spec) if not isinstance(result, dict) else None
            ),
        )
        return result

    def _stop_consumer(self) -> None:
        """Enqueue the consumer sentinel and join it, exactly once."""
        if self._consumer_stopped:
            return
        self._consumer_stopped = True
        if self._progress_queue is not None:
            self._progress_queue.put(None)
        if self._consumer is not None:
            self._consumer.join()

    def _cleanup(self) -> None:
        runner = self._runner
        # Clear the active-process handle so a late SIGINT can't act on a
        # process this session already finished with.
        runner._active_process = None
        # Exception path: the process may still be alive (run() never reached
        # its join/kill sequence). Reap it so a spawn leak can't outlive the run.
        if self._process is not None:
            with contextlib.suppress(Exception):
                if self._process.is_alive():
                    _kill_process_group(self._process.pid, signal.SIGKILL)
                    self._process.join()
        # Stop the progress consumer (no-op if run() already did).
        self._stop_consumer()
        # Close the read end of the Pipe (C4 FD-leak fix): exactly once, here.
        if self._parent_conn is not None:
            with contextlib.suppress(Exception):
                self._parent_conn.close()
        # Remove the timeseries staging dir after _handle_result copied the parquet.
        if self._ts_tmpdir is not None:
            shutil.rmtree(self._ts_tmpdir, ignore_errors=True)


class DockerSession(_OfflineSession):
    """Offline session: one experiment in a blocking ``DockerRunner.run`` container.

    ``__enter__`` resolves the image, container name/labels, emits the study
    begin-experiment events (before baseline resolution so baseline step events
    fire against a registered index), assembles the bind mounts (including the
    per-cache-key baseline cache), builds the ``DockerRunner`` facade, runs the
    pre-dispatch GPU-memory residual check, and marks the experiment running.
    ``run()`` performs the blocking container dispatch - translating a
    ``DockerError`` into a non-fatal failure dict and persisting the failure
    artefacts - and hands the result to the runner. ``__exit__`` removes the
    container's timeseries staging dir.

    The container lifecycle is DockerRunner's; this session does not change any
    ``DockerRunner`` call signature (S13 reworks its internals behind a frozen
    facade).
    """

    def __init__(
        self,
        runner: StudyRunner,
        config: ExperimentConfig,
        spec: RunnerSpec,
        *,
        config_hash: str,
        cycle: int,
        index: int,
    ) -> None:
        super().__init__(runner, config, config_hash=config_hash, cycle=cycle, index=index)
        self.spec = spec
        # Populated in __enter__.
        self._image: str | None = None
        self._docker_runner: Any = None
        self._exp_start: float = 0.0
        self._docker_ts_dir: Path | None = None

    def __enter__(self) -> DockerSession:
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.container_lifecycle import (
            generate_container_labels,
            generate_container_name,
        )
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

        runner = self._runner
        config = self.config
        spec = self.spec

        # Image is pre-resolved during preflight (resolve_image precedence chain).
        # Fall back to get_default_image() only for direct DockerRunner usage
        # outside the study path.
        self._image = spec.image if spec.image is not None else get_default_image(config.engine)

        study_id = runner.study.study_design_hash or "unknown"
        container_name = generate_container_name(study_id, self.index)
        labels = generate_container_labels(study_id)

        # begin_experiment MUST run before _get_baseline so baseline step events
        # fire against a registered experiment index.
        if runner._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            host_baseline = (
                config.measurement.baseline.enabled
                and config.measurement.baseline.strategy != "fresh"
            )
            steps = docker_steps(
                images_prepared=runner._images_prepared,
                host_baseline=host_baseline,
            )
            runner._progress.begin_experiment(
                self.index,
                format_experiment_header(config),
                steps,
                runner_info=spec.to_runner_info(),
            )
            # Host-side preflight doesn't run in Docker path - checked inside container
            runner._progress.on_step_skip("preflight", "checked inside container")

        extra_mounts = list(spec.extra_mounts) if spec.extra_mounts else []
        cache_key = runner._baseline_cache_key(config)
        baseline = runner._get_baseline(config) if config.measurement.baseline.enabled else None
        if baseline is not None:
            # Experiment container reads /run/llem/baseline_cache.json; the host
            # picks the right per-cache-key file at dispatch time. Docker parses
            # relative bind-mount sources as named volumes, so resolve first.
            baseline_cache_path = runner._get_baseline_cache_path(cache_key)
            extra_mounts.append(
                (
                    str(baseline_cache_path.resolve()),
                    f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json",
                )
            )

        self._docker_runner = DockerRunner(
            image=self._image,
            timeout=runner.study.study_execution.experiment_timeout_seconds,
            silence_timeout=runner.study.study_execution.stdout_silence_timeout_seconds,
            source=spec.source,
            extra_mounts=extra_mounts,
            container_name=container_name,
            labels=labels,
            gpu_indices=runner.study.study_execution.gpu_indices,
        )

        # Pre-dispatch GPU memory residual check (same as local path)
        check_gpu_memory_residual()

        runner.manifest.mark_running(self.config_hash, self.cycle)
        self._exp_start = time.monotonic()
        return self

    def run(self) -> Any:
        from llenergymeasure.infra.docker_errors import docker_exc_to_failure
        from llenergymeasure.study.container_lifecycle import persist_failure_artefacts
        from llenergymeasure.utils.exceptions import DockerError

        runner = self._runner
        config = self.config

        result: Any
        try:
            # Pass study progress as step callback - DockerRunner calls on_step_*
            # skip_image_check=True when images were verified at study level.
            result, self._docker_ts_dir = self._docker_runner.run(
                config,
                progress=runner._progress,
                save_timeseries=runner.study.output.save_timeseries,
                skip_image_check=runner._images_prepared,
            )
        except DockerError as exc:
            # Translate to a non-fatal failure dict (silence / timeout / structured
            # payload classification handled in the shared helper) and persist the
            # container.log + error JSON so the failure is debuggable.
            result = docker_exc_to_failure(exc, self.config_hash)
            persist_failure_artefacts(exc, runner.study_dir, self.config_hash, self.cycle, result)

        exp_elapsed = time.monotonic() - self._exp_start
        runner._handle_result(
            result,
            config,
            self.config_hash,
            self.cycle,
            self.index,
            exp_elapsed,
            ts_source_dir=self._docker_ts_dir,
            environment_snapshot=(
                runner._get_env_snapshot() if not isinstance(result, dict) else None
            ),
            runner_provenance=_provenance_from_spec(self.spec),
            runner_environment=(
                _runner_environment(self.spec, resolved_image=self._image)
                if not isinstance(result, dict)
                else None
            ),
        )
        return result

    def _cleanup(self) -> None:
        # Clean up the temp dir after _handle_result has copied the parquet.
        if self._docker_ts_dir is not None:
            shutil.rmtree(self._docker_ts_dir, ignore_errors=True)


def _provenance_from_spec(spec: RunnerSpec | None) -> Any:
    """Delegate to the runner-module builder (single home for provenance mapping)."""
    from llenergymeasure.study.runner import _provenance_from_spec as _impl

    return _impl(spec)


def _runner_environment(spec: RunnerSpec | None, *, resolved_image: str | None = None) -> Any:
    """Delegate to the runner-module builder (single home for the runner block)."""
    from llenergymeasure.study.runner import _runner_environment as _impl

    return _impl(spec, resolved_image=resolved_image)
