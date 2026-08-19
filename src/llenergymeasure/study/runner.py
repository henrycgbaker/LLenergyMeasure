"""StudyRunner - orchestrates per-experiment subprocess/Docker dispatch.

``StudyRunner`` owns the study-level run loop: SIGINT handling, GPU locks,
container lifecycle, circuit breaker, wall-clock timeout, and result handling.
The separable concerns live in sibling modules and are mixed in or imported:

- ``study.session``: context-managed per-experiment dispatch lifetimes
  (``ExperimentSession`` + the offline subprocess/docker implementations the
  run loop drives, one per experiment).
- ``study.worker``: subprocess-worker surface (child entry point, result
  collection, process-group signalling).
- ``study._progress``: cross-process progress plumbing.
- ``study.baseline_measure`` (``_BaselineMixin``): baseline measurement,
  caching, and drift validation.
- ``study.image_prep`` (``_ImageMixin``): Docker image preparation and
  schema-fingerprint verification.
- ``study.container_lifecycle``: container naming/labels/cleanup plus failure
  artefact persistence.

Each experiment runs in a freshly spawned subprocess with a clean CUDA context
(or in a Docker container). Results travel parent<-child via multiprocessing.Pipe
or DockerRunner. The parent survives experiment failures, timeouts, and SIGINT
without data corruption.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os  # noqa: F401 - patch target: tests patch study.runner.os.{killpg,setpgrp}
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from llenergymeasure.config.grid import ExperimentOrder, cycle_boundary_indices
from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    RUNNER_CONTAINER,
    RUNNER_PROCESS,
    SOURCE_IMPLICIT,
    TIMEOUT_ENV_SNAPSHOT,
    TIMEOUT_INTERRUPT_POLL,
    engine_str,
)
from llenergymeasure.domain.bundle_artefacts import (
    CONFIG_SIDECAR_FILENAME,
    EQUIVALENCE_GROUPS_FILENAME,
)

# Re-imported into this module's namespace so existing
# ``from llenergymeasure.study.runner import X`` sites (tests, study.single)
# keep resolving here even though the dispatch mechanics now live in
# study.session / study._progress.
from llenergymeasure.study._progress import _consume_progress_events
from llenergymeasure.study.baseline_measure import _BaselineMixin
from llenergymeasure.study.gaps import run_gap
from llenergymeasure.study.image_prep import _ImageMixin
from llenergymeasure.study.worker import _kill_process_group, _run_experiment_worker
from llenergymeasure.utils.io import load_json

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig
    from llenergymeasure.config.runner_spec import RunnerSpec
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.progress import StudyProgressCallback
    from llenergymeasure.domain.provenance import RunnerProvenance
    from llenergymeasure.study.manifest import ManifestWriter

__all__ = [
    "StudyRunner",
    "_consume_progress_events",
    "_kill_process_group",
    "_run_experiment_worker",
    "_save_and_record",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level helpers
# =============================================================================


def _provenance_from_spec(
    spec: RunnerSpec | None, *, resolved_image: str | None = None
) -> RunnerProvenance:
    """Build the unified RunnerProvenance from a resolved RunnerSpec.

    The config-layer ``RunnerSpec`` cannot live on a domain result (config and
    domain are independent sibling layers), so its execution-mode fields are
    mirrored onto the domain-layer ``RunnerProvenance``. The one model is
    serialised into both result.json (self-contained provenance) and the
    system.json ``runner`` block, so this single builder populates the full
    superset - including the registry digest the system block anchors on.

    For container specs the digest is resolved host-side via ``docker image inspect``
    on the image that actually ran (``resolved_image`` when the spec left the
    image implicit). Digest resolution is best-effort: an unresolved digest
    records None and is debug-logged, never raising - so provenance never fails a
    run. Process specs (and no spec at all) record ``mode="process"`` with no image,
    digest, or image_source and the spec's source (``"implicit"`` when no spec was
    resolved - the resolution chain never ran, distinct from the ``"default"``
    precedence layer, which is a real fall-through).
    """
    from llenergymeasure.domain.provenance import RunnerProvenance

    if spec is None or spec.mode != RUNNER_CONTAINER:
        return RunnerProvenance(
            mode=RUNNER_PROCESS,
            image=None,
            source=spec.source if spec is not None else SOURCE_IMPLICIT,
            image_source=None,
            image_digest=None,
        )

    from llenergymeasure.infra.image_registry import resolve_image_digest

    image = resolved_image if resolved_image is not None else spec.image
    digest = resolve_image_digest(image) if image is not None else None
    if image is not None and digest is None:
        logger.debug(
            "Could not resolve registry digest for runner image %s; "
            "runner provenance image_digest will be null.",
            image,
        )
    return RunnerProvenance(
        mode=spec.mode,
        image=image,
        source=spec.source,
        image_source=spec.image_source,
        image_digest=digest,
    )


def _save_and_record(
    result: Any,
    study_dir: Path,
    manifest: ManifestWriter,
    config_hash: str,
    cycle: int,
    result_files: list[str],
    *,
    model_name: str,
    engine: str,
    experiment_index: int | None = None,
    ts_source_dir: Path | None = None,
    environment_snapshot: Any | None = None,
    resolution_log: dict[str, Any] | None = None,
    resolved_config_hash: str | None = None,
    runner_provenance: RunnerProvenance | None = None,
) -> None:
    """Assemble the results bundle and update the manifest.

    Thin orchestrator over :class:`llenergymeasure.results.bundle.BundleWriter`,
    which owns the assembly policy (dir creation, result + provenance write,
    environment rescue-preference + runner-block patch, config-sidecar move +
    patch, and the registry-driven loudness backstops). This function only wires
    the inputs together, appends the result path to ``result_files``, and records
    the summary metrics on the manifest.

    Args:
        model_name / engine: Identity for the experiment directory slug
            (authoritative home: config.json; the result keeps a convenience copy).
        ts_source_dir: Staging dir where the harness/container wrote config.json,
            timeseries.parquet, and (under docker) the rescued system.json.
        environment_snapshot: Host EnvironmentSnapshot for system.json (a
            rescued in-container snapshot is preferred over it when present).
        resolution_log / resolved_config_hash: Patched into the config.json sidecar.
        runner_provenance: The unified runner provenance (mode, image, source,
            image_source, image_digest); folded into result.json and written as
            the system.json ``runner`` block.

    On any save failure, marks the experiment failed on the manifest.
    """
    try:
        from llenergymeasure.domain.session import SessionBlock
        from llenergymeasure.results.bundle import BundleWriter

        # Offline degenerates to the one-window session: a fresh
        # session id, window_count=1, all phase raws null (offline pre-window
        # phases are not instrumented). Present in both modes so readers see one
        # shape.
        session = SessionBlock(
            session_id=uuid.uuid4().hex,
            window_count=1,
            level_count=1,
        )

        writer = BundleWriter(
            study_dir,
            model_name=model_name,
            engine=engine,
            config_hash=config_hash,
            cycle=cycle,
            experiment_index=experiment_index,
            ts_source_dir=ts_source_dir,
        )
        result_path = writer.write_result(
            result, runner_provenance=runner_provenance, session=session
        )
        writer.write_system(
            host_snapshot=environment_snapshot,
            runner=runner_provenance,
            session=session,
        )
        writer.move_config_sidecar(
            resolved_config_hash=resolved_config_hash,
            resolution_log=resolution_log,
        )
        writer.finalize()

        result_files.append(str(result_path))
        rel_path = str(result_path.relative_to(study_dir))

        # Extract summary metrics for manifest (used by --resume display).
        elapsed_sec = getattr(result, "total_inference_time_sec", None)
        manifest.mark_completed(
            config_hash,
            cycle,
            rel_path,
            elapsed_seconds=elapsed_sec,
            inference_seconds=elapsed_sec,
            energy_joules=getattr(result, "total_energy_j", None),
            adj_energy_joules=getattr(result, "energy_adjusted_j", None),
            throughput_tok_s=getattr(result, "avg_tokens_per_second", None),
            energy_per_token_mj=getattr(result, "energy_per_token_mj_adjusted", None)
            or getattr(result, "energy_per_token_mj_total", None),
        )
    except Exception as exc:
        manifest.mark_failed(config_hash, cycle, type(exc).__name__, str(exc))


# =============================================================================
# StudyRunner
# =============================================================================


class StudyRunner(_BaselineMixin, _ImageMixin):
    """Dispatcher: runs each experiment in a freshly spawned subprocess.

    Uses multiprocessing.get_context('spawn') - never fork.
    Results travel via Pipe. Failures are structured and non-fatal.
    Handles SIGINT (Ctrl+C) with two-stage escalation: SIGTERM → 2s grace → SIGKILL.
    """

    def __init__(
        self,
        study: StudyConfig,
        manifest_writer: ManifestWriter,
        study_dir: Path,
        runner_specs: dict[str, RunnerSpec] | None = None,
        progress: StudyProgressCallback | None = None,
        no_lock: bool = False,
        skip_set: set[tuple[str, int]] | None = None,
        resolution_logs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.study = study
        self.manifest = manifest_writer
        self.study_dir = study_dir
        self.result_files: list[str] = []
        # Pre-resolved runner specs per engine (None = all experiments use subprocess path)
        self._runner_specs = runner_specs
        # Live study progress display (None = no live output)
        self._progress = progress
        # When True, skip GPU advisory lock acquisition
        self._no_lock = no_lock
        # Set of (config_hash, cycle) pairs to skip (resume mode)
        self._skip_set: set[tuple[str, int]] = skip_set or set()
        # Pre-built resolution logs keyed by config_hash (folded into the
        # config.json sidecar's provenance section by _save_and_record).
        self._resolution_logs: dict[str, dict[str, Any]] = resolution_logs or {}
        # Declared-config-hash → resolved-config-hash mapping.
        # Built from study.experiments (post-dedup unique configs) so _save_and_record
        # can patch the resolved_config_hash into each config.json sidecar.
        self._resolved_hashes: dict[str, str] = self._build_resolved_hashes(study)
        # SIGINT state - initialised here, set live in run()
        self._interrupt_event: threading.Event = threading.Event()
        self._active_process: Any = None  # multiprocessing.Process | None
        self._interrupt_count: int = 0
        # Per-config_hash cycle counters - reset at the start of each run()
        self._cycle_counters: dict[str, int] = {}
        # Study-level environment snapshot cache - collected once, reused across experiments
        self._env_snapshot_future: Future[EnvironmentSnapshot] | None = None
        # Study-level baseline cache, keyed per runner target ("local_<engine>" or
        # "image_<sanitized>") so multi-engine studies don't cross-contaminate.
        self._baselines: dict[str, Any] = {}  # dict[str, BaselineCache]
        self._experiments_since_validation: dict[str, int] = {}
        # Study-level image preparation: True after _prepare_images() succeeds
        self._images_prepared: bool = False
        # Per-run UUID tagging every runtime-observation record from this
        # StudyRunner.run() invocation. Generated once so re-runs against
        # the same study_dir can be disambiguated by downstream consumers.
        self.study_run_id: str = str(uuid.uuid4())

    def run(self) -> list[Any]:
        """Run all experiments in order; return list of results or failure dicts.

        Installs a SIGINT handler for the duration of the run. First Ctrl+C sends
        SIGTERM to the active subprocess and sets interrupt_event. Second Ctrl+C (or
        grace period expiry) sends SIGKILL. After the loop exits, if interrupted,
        calls manifest.mark_interrupted() and sys.exit(130).

        Acquires per-GPU advisory file locks before image preparation (unless
        no_lock=True). Releases locks in the finally block regardless of outcome.

        Integrates circuit breaker (closed -> open -> half-open -> closed/abort) and
        wall-clock timeout: both mark remaining experiments as skipped and update
        the manifest status before returning.

        Note: study.experiments is already the fully-ordered execution sequence produced
        by apply_cycles() in load_study_config(). The runner must not call apply_cycles()
        again - doing so would multiply the count by n_cycles a second time.
        """
        from llenergymeasure.domain.experiment import compute_declared_config_hash
        from llenergymeasure.study.circuit_breaker import CircuitBreaker

        # study.experiments is already cycled by load_study_config(); use as-is.
        ordered = self.study.experiments

        # n_unique: count of distinct configs (for cycle-gap detection).
        # Do not use len(ordered) - that includes repetitions.
        seen_hashes = {compute_declared_config_hash(c) for c in self.study.experiments}
        n_unique = len(seen_hashes)

        # Cycle-gap boundaries: the positions in `ordered` at which the larger
        # cycle gap fires. Computed once here (not from positional modulo in the
        # loop) so the boundaries track experiment_order - they differ between
        # sequential (between config blocks) and pass-structured orders.
        cycle_gap_indices = cycle_boundary_indices(
            n_unique,
            self.study.study_execution.n_cycles,
            ExperimentOrder(self.study.study_execution.experiment_order),
        )

        # Session grouping: fold consecutive rate-only-varying server cells
        # into one session (one launch, a rate level per cell). group_starts maps the
        # FIRST index of a multi-cell server group to its member indices; single
        # cells (offline, or a lone server cell) are dispatched one at a time as
        # before, so the offline path is untouched.
        from llenergymeasure.study.server_session import partition_server_groups

        units = partition_server_groups(ordered)
        group_starts = {unit[0]: unit for unit in units if len(unit) > 1}
        consumed_by_group: set[int] = set()

        # spawn: CUDA-safe; fork causes silent CUDA corruption
        mp_ctx = multiprocessing.get_context("spawn")

        # Reset interrupt state for this run
        self._interrupt_event.clear()
        self._interrupt_count = 0
        self._active_process = None
        self._cycle_counters = {}

        original_sigint, original_sigterm, gpu_locks = self._install_run_handlers(ordered)

        # Intentionally covers server-mode container images too, not just offline
        # dispatch. It is the only early-fail + visible-pull point for a pure-server
        # study: launch_container_server's `docker run -d` pulls a missing image
        # silently with no subprocess timeout and before the readiness clock starts,
        # so gating this off would hide pull progress and defer failure.
        self._prepare_images()

        # Circuit breaker: tracks consecutive failures, decides abort/probe.
        breaker = CircuitBreaker(
            max_failures=self.study.study_execution.max_consecutive_failures,
            cooldown_seconds=self.study.study_execution.circuit_breaker_cooldown_seconds,
        )

        # Wall-clock deadline: computed once before the loop.
        deadline: float | None = None
        if self.study.study_execution.wall_clock_timeout_hours:
            deadline = time.monotonic() + (
                self.study.study_execution.wall_clock_timeout_hours * 3600
            )

        try:
            results: list[Any] = []
            # Track whether the loop was aborted by timeout or circuit breaker.
            # Used to skip mark_study_completed() on non-clean exits.
            _aborted = False

            for i, config in enumerate(ordered):
                if self._interrupt_event.is_set():
                    break

                # Consumed by a server group that a prior iteration dispatched.
                if i in consumed_by_group:
                    continue

                is_group = i in group_starts

                # Resume skip-set: single cells use the per-cell skip; a server group
                # handles resume per-member inside its own dispatch (some members may
                # have completed in a prior run).
                if not is_group and self._resume_should_skip(config, i, len(ordered)):
                    continue

                # Wall-clock timeout check: mark remaining experiments skipped.
                if deadline is not None and time.monotonic() > deadline:
                    # consumed_by_group is defensive symmetry here: the timeout check
                    # runs BEFORE dispatching group i, so no already-consumed index can
                    # fall in the [i, N) sweep (a spanning group would have made i itself
                    # consumed and continued above). The circuit-breaker site (index+1,
                    # mid-group) is where the consumed filter is load-bearing.
                    self._mark_remaining_skipped(
                        ordered, i, compute_declared_config_hash, consumed_by_group
                    )
                    self.manifest.mark_study_timed_out()
                    logger.warning(
                        "Study timed out after %.1f hours",
                        self.study.study_execution.wall_clock_timeout_hours,
                    )
                    _aborted = True
                    break

                # Inter-experiment + per-cycle gaps (break if interrupted during a gap)
                if self._run_inter_experiment_gaps(i, cycle_gap_indices):
                    break

                if is_group:
                    group = group_starts[i]
                    consumed_by_group.update(group)
                    result = self._run_one_server_group([ordered[j] for j in group], index=i + 1)
                    if result is None:
                        continue  # whole group already completed on resume
                else:
                    result = self._run_one(config, mp_ctx, index=i + 1)
                results.append(result)

                # Circuit breaker integration: update state based on result.
                if self._apply_circuit_breaker(breaker, result, ordered, i, consumed_by_group):
                    _aborted = True
                    break

            # Mark study completed on clean exit (no interrupt, timeout, or circuit break).
            if not self._interrupt_event.is_set() and not _aborted:
                self.manifest.mark_study_completed()
                # Write equivalence_groups.json - post-run observed-config-hash groups.
                self._write_equivalence_groups_sidecar()

        finally:
            self._restore_run_handlers(original_sigint, original_sigterm, gpu_locks)

        if self._interrupt_event.is_set():
            completed = sum(1 for r in results if not isinstance(r, dict))
            total = len(ordered)
            print(
                f"\n{completed}/{total} experiments completed. "
                "Results in study directory. Manifest: interrupted."
            )
            self.manifest.mark_interrupted()
            sys.exit(130)

        return results

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        """SIGINT handler installed for the duration of run().

        First Ctrl+C sets the interrupt event and sends SIGTERM to the active
        subprocess group; a second sends SIGKILL.
        """
        self._interrupt_count += 1
        self._interrupt_event.set()
        if self._interrupt_count == 1:
            print(
                "\nInterrupt received. Waiting for experiment to finish cleanly "
                "(Ctrl+C again to force)..."
            )
            if self._active_process is not None and self._active_process.is_alive():
                _kill_process_group(
                    self._active_process.pid, signal.SIGTERM
                )  # SIGTERM - gentle first attempt
        else:
            print("\nForce-killing experiment subprocess...")
            if self._active_process is not None and self._active_process.is_alive():
                _kill_process_group(self._active_process.pid, signal.SIGKILL)  # SIGKILL

    def _install_run_handlers(
        self, ordered: list[Any]
    ) -> tuple[Any, signal.Handlers | None, list[Any]]:
        """Install the SIGINT handler, acquire per-GPU advisory locks, and wire the
        Docker container lifecycle + SIGTERM bridge.

        A container-dispatching study is refused up front when it has no design
        hash: that hash is the ownership key for every container the study
        launches, so the failure fires before any handler is installed, any lock
        is taken, and any container exists.

        Returns ``(original_sigint, original_sigterm, gpu_locks)`` for restoration in
        run()'s finally block via _restore_run_handlers.
        """
        config_gpu_indices = self.study.study_execution.gpu_indices
        uses_docker = bool(
            self._runner_specs
            and any(s.mode == RUNNER_CONTAINER for s in self._runner_specs.values())
        )

        study_id: str | None = None
        if uses_docker:
            from llenergymeasure.study.container_lifecycle import require_study_id

            study_id = require_study_id(self.study.study_design_hash)

        original_sigint = signal.signal(signal.SIGINT, self._handle_sigint)

        # Acquire per-GPU advisory locks before image preparation.
        # Lock names use the PHYSICAL device the study occupies, resolved from
        # the effective GPU selector (env LLEM_DOCKER_GPUS, else config
        # gpu_indices): two studies on different physical GPUs must not share a
        # lock. When there is no docker-level pinning (all / unset), logical ==
        # physical, so fall back to the in-container logical indices.
        # Measurement-side index resolution is unchanged - _resolve_gpu_indices
        # still yields the logical indices that address the energy samplers.
        # Sorted acquisition prevents deadlocks when multiple studies share GPUs.
        gpu_locks: list[Any] = []
        if not self._no_lock and ordered:
            from llenergymeasure.study.gpu_locks import acquire_gpu_locks
            from llenergymeasure.utils.env_config import pinned_gpu_lock_ids

            lock_ids = pinned_gpu_lock_ids(config_gpu_indices)
            if lock_ids is None:
                from llenergymeasure.device.gpu_info import _resolve_gpu_indices

                lock_ids = [str(i) for i in _resolve_gpu_indices(ordered[0])]
            gpu_locks = acquire_gpu_locks(lock_ids)

        # Container lifecycle: reap orphaned containers, register cleanup, install SIGTERM bridge.
        # Only activated for studies that use Docker runners.
        original_sigterm: signal.Handlers | None = None
        if study_id is not None:
            from llenergymeasure.study.container_lifecycle import (
                install_sigterm_bridge,
                reap_orphaned_containers,
                register_container_cleanup,
            )

            reap_orphaned_containers()
            register_container_cleanup(study_id)
            original_sigterm = install_sigterm_bridge()

        return original_sigint, original_sigterm, gpu_locks

    def _restore_run_handlers(
        self,
        original_sigint: Any,
        original_sigterm: signal.Handlers | None,
        gpu_locks: list[Any],
    ) -> None:
        """Restore signal handlers and release GPU locks acquired by _install_run_handlers."""
        signal.signal(signal.SIGINT, original_sigint)
        if original_sigterm is not None:
            signal.signal(signal.SIGTERM, original_sigterm)
        if gpu_locks:
            from llenergymeasure.study.gpu_locks import release_gpu_locks

            release_gpu_locks(gpu_locks)

    def _resume_should_skip(self, config: Any, index: int, total: int) -> bool:
        """Return True when ``config`` already completed in a prior run (resume skip-set).

        Advances the per-config-hash cycle counter as a side effect when skipping, so the
        skip-set stays aligned with the cycle the runner would otherwise execute next.
        """
        if not self._skip_set:
            return False
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        config_hash_pre = compute_declared_config_hash(config)
        next_cycle = self._cycle_counters.get(config_hash_pre, 0) + 1
        if (config_hash_pre, next_cycle) in self._skip_set:
            self._cycle_counters[config_hash_pre] = next_cycle
            logger.info(
                "Skipping completed experiment %d/%d (resumed)",
                index + 1,
                total,
            )
            return True
        return False

    def _run_inter_experiment_gaps(self, index: int, cycle_gap_indices: frozenset[int]) -> bool:
        """Run the inter-experiment gap and, on cycle boundaries, the per-cycle gap.

        ``cycle_gap_indices`` is the set of positions in the ordered execution
        sequence at which the larger cycle gap must fire, precomputed once by
        :func:`llenergymeasure.config.grid.cycle_boundary_indices` so the
        boundaries track the active experiment_order (they differ between
        sequential and pass-structured orders, and never include index 0).

        Returns True if an interrupt arrived during a gap (caller should break).
        """
        # Config gap: between every consecutive experiment pair
        if index > 0:
            gap_secs = float(self.study.study_execution.experiment_gap_seconds or 0)
            if gap_secs > 0:
                self._run_gap(gap_secs, "Experiment gap")
                if self._interrupt_event.is_set():
                    return True

        # Cycle gap: at the cycle boundaries for the active experiment_order
        if index in cycle_gap_indices:
            cycle_gap_secs = float(self.study.study_execution.cycle_gap_seconds or 0)
            if cycle_gap_secs > 0:
                self._run_gap(cycle_gap_secs, "Cycle gap")
                if self._interrupt_event.is_set():
                    return True
        return False

    def _apply_circuit_breaker(
        self,
        breaker: Any,
        result: Any,
        ordered: list[Any],
        index: int,
        consumed: frozenset[int] | set[int] = frozenset(),
    ) -> bool:
        """Update the circuit breaker from an experiment result.

        Records failure/success, applies cooldown + probe on a trip, and on a failed
        probe marks the remaining experiments skipped. Returns True when the study must
        abort (caller sets _aborted and breaks). ``consumed`` carries the server-group
        indices the skip sweep must not re-process.
        """
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        if isinstance(result, dict) and "type" in result:
            error_type = result.get("type", "UnknownError")
            error_msg = result.get("message", "")
            action = breaker.record_failure(error_type, error_msg)

            if action == "tripped":
                for line in breaker.get_failure_summary():
                    logger.warning("Circuit breaker: %s", line)
                if breaker.cooldown_seconds > 0:
                    logger.info("Circuit breaker cooldown: %.0fs", breaker.cooldown_seconds)
                    time.sleep(breaker.cooldown_seconds)
                breaker.start_probe()
                # Next loop iteration is the probe experiment.

            elif action == "abort":
                # Probe failed - abort the study immediately.
                self._mark_remaining_skipped(
                    ordered, index + 1, compute_declared_config_hash, consumed
                )
                self.manifest.mark_study_circuit_breaker()
                logger.error("Circuit breaker: probe experiment failed, aborting study")
                return True

        else:
            # Success path: reset circuit breaker (if not disabled).
            if not breaker.is_disabled:
                breaker.record_success()
        return False

    def _write_equivalence_groups_sidecar(self) -> None:
        """Write ``equivalence_groups.json`` to the study directory after run completion.

        Bundles:
        - Pre-run groups from ``StudyConfig.pre_run_equivalence_groups`` (resolved-config-hash
          dedup computed at sweep-expansion time).
        - Post-run observed-config-hash collision groups built by scanning all ``config.json``
          sidecars in the study directory.

        Best-effort - failures are logged at DEBUG to avoid masking study results.
        """
        try:
            from llenergymeasure.study.equivalence_groups import (
                EquivalenceGroups,
                ObservedCollisionGroup,
                PreRunGroup,
                find_observed_collisions,
                write_equivalence_groups,
            )

            study = self.study
            study_id = study.study_design_hash or "unknown"
            study_name = study.study_name or "unnamed-study"
            raw_mode = getattr(study, "dedup_mode", "off")
            dedup_mode: Literal["resolved", "off"] = "resolved" if raw_mode == "resolved" else "off"

            # Deserialise pre-run groups from StudyConfig (stored as raw dicts)
            pre_run_groups: list[PreRunGroup] = []
            for g in getattr(study, "pre_run_equivalence_groups", []):
                with contextlib.suppress(Exception):
                    pre_run_groups.append(
                        PreRunGroup(
                            resolved_config_hash=str(g.get("resolved_config_hash", "")),
                            canonical_config_excerpt=dict(g.get("canonical_config_excerpt", {})),
                            member_experiment_ids=tuple(g.get("member_experiment_ids", [])),
                            member_count=int(g.get("member_count", 0)),
                            representative_experiment_id=str(
                                g.get("representative_experiment_id", "")
                            ),
                            would_dedup=bool(g.get("would_dedup", False)),
                            deduplicated=bool(g.get("deduplicated", False)),
                        )
                    )

            # Scan config.json sidecars for post-run observed-config-hash groups
            sidecars: list[dict[str, Any]] = []
            for config_json in self.study_dir.rglob(CONFIG_SIDECAR_FILENAME):
                with contextlib.suppress(Exception):
                    sidecars.append(load_json(config_json))
            post_run_groups: list[ObservedCollisionGroup] = find_observed_collisions(sidecars)

            groups = EquivalenceGroups(
                study_id=study_id,
                study_name=study_name,
                dedup_mode=dedup_mode,
                groups=pre_run_groups,
                observed_collision_groups=post_run_groups,
            )
            write_equivalence_groups(groups, self.study_dir / EQUIVALENCE_GROUPS_FILENAME)
            logger.debug("Wrote equivalence_groups.json to %s", self.study_dir)
        except Exception as exc:
            logger.debug("equivalence_groups.json write failed (non-fatal): %s", exc)

    @staticmethod
    def _build_resolved_hashes(study: Any) -> dict[str, str]:
        """Build a declared_config_hash → resolved_config_hash mapping.

        Iterates the unique post-dedup experiments in ``study.experiments``
        (cycles produce duplicates; seen_hashes deduplicates them) and
        computes each experiment's resolved hash via the same resolved-config
        pipeline used at sweep-expansion time.

        Returns an empty dict on any failure - _save_and_record treats a
        missing resolved_config_hash as best-effort and writes ``None``.
        """
        try:
            from llenergymeasure.domain.experiment import compute_declared_config_hash
            from llenergymeasure.study.hashing import build_resolved_view, hash_config

            result: dict[str, str] = {}
            seen: set[str] = set()
            for exp in study.experiments:
                declared_h = compute_declared_config_hash(exp)
                if declared_h in seen:
                    continue
                seen.add(declared_h)
                resolved_h = hash_config(build_resolved_view(exp))
                result[declared_h] = resolved_h
            return result
        except Exception as exc:
            logger.debug("_build_resolved_hashes failed (non-fatal): %s", exc)
            return {}

    def _mark_remaining_skipped(
        self,
        ordered: list[Any],
        start_index: int,
        hash_fn: Any,
        consumed: frozenset[int] | set[int] = frozenset(),
    ) -> None:
        """Mark all experiments from start_index onwards as skipped in the manifest.

        Increments cycle counters to assign the correct cycle number for each
        remaining experiment before marking it skipped.

        Indices already consumed by a dispatched server group are SKIPPED here
        without touching their cycle counters: their session already resolved them
        (completed / failed) and advanced their counters, so re-processing would
        double-advance and mark a non-existent (future-cycle) entry - a KeyError
        that would abort the study uncleanly.

        Args:
            ordered: Full ordered experiment list (study.experiments).
            start_index: Index of the first experiment to mark as skipped.
            hash_fn: compute_declared_config_hash callable.
            consumed: Indices a server group already dispatched (do not re-process).
        """
        for j in range(start_index, len(ordered)):
            if j in consumed:
                continue
            cfg = ordered[j]
            h = hash_fn(cfg)
            c = self._cycle_counters.get(h, 0) + 1
            self._cycle_counters[h] = c
            self.manifest.mark_skipped(h, c)

    def _get_env_snapshot(self) -> EnvironmentSnapshot:
        """Return cached environment snapshot, collecting on first call.

        Uses background-threaded collection on first call. Subsequent calls
        return the resolved snapshot immediately (study-level cache).
        """
        if self._env_snapshot_future is None:
            from llenergymeasure.harness.environment import collect_environment_snapshot_async

            self._env_snapshot_future = collect_environment_snapshot_async()
        return self._env_snapshot_future.result(timeout=TIMEOUT_ENV_SNAPSHOT)

    def _run_gap(self, seconds: float, label: str) -> None:
        """Run a thermal gap, rendering countdown in the live display or terminal."""
        if self._progress:
            from llenergymeasure.study.gaps import format_gap_duration

            for remaining in range(int(seconds), 0, -1):
                if self._interrupt_event.is_set():
                    break
                self._progress.show_gap(f"{label}: {format_gap_duration(remaining)}")
                self._interrupt_event.wait(timeout=TIMEOUT_INTERRUPT_POLL)
            self._progress.clear_gap()
        else:
            # Fall back to terminal countdown
            run_gap(seconds, label, self._interrupt_event)

    def _run_one(self, config: ExperimentConfig, mp_ctx: Any, index: int) -> Any:
        """Dispatch one experiment via Docker or subprocess, collect result or failure dict.

        Assigns this experiment's cycle number, then drives the appropriate
        ``ExperimentSession`` (``DockerSession`` when the engine's runner spec is
        Docker, else ``SubprocessSession``). The session's context-manager
        lifecycle owns setup and teardown; ``run()`` produces the single result
        (offline = one result per session). The grace SIGTERM -> SIGKILL
        escalation on interrupt lives in ``SubprocessSession.run()``.
        """
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        config_hash = compute_declared_config_hash(config)
        # Increment per-config_hash counter: 1st run → cycle=1, 2nd → cycle=2, etc.
        cycle = self._cycle_counters.get(config_hash, 0) + 1
        self._cycle_counters[config_hash] = cycle

        # Server dispatch path: a server config drives a ServerSession - launch
        # the engine server, warm up, run the window manager to N results, shut down
        # - a sibling of the two offline dispatch paths. Routed FIRST so the offline
        # subprocess/docker paths below stay byte-identical for offline configs.
        if config.serving_mode == "server":
            return self._run_one_server(config, config_hash=config_hash, cycle=cycle, index=index)

        # Docker dispatch path - check runner spec for this engine
        spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        if spec is not None and spec.mode == RUNNER_CONTAINER:
            return self._run_one_docker(
                config, spec, config_hash=config_hash, cycle=cycle, index=index
            )

        from llenergymeasure.study.session import SubprocessSession

        # Subprocess dispatch: a SubprocessSession owns the freshly spawned
        # worker's lifetime - env prep + spawn on __enter__, teardown on __exit__
        # (always, including the SIGINT/exception paths). The sweep loop keys off
        # the single result the session produces (offline = one result/session).
        with SubprocessSession(
            self, config, mp_ctx, config_hash=config_hash, cycle=cycle, index=index
        ) as session:
            return session.run()

    def _run_one_server(
        self,
        config: ExperimentConfig,
        *,
        config_hash: str,
        cycle: int,
        index: int,
    ) -> Any:
        """Dispatch one server-mode experiment via a ServerSession (sibling of the offline paths).

        The session launches the engine server, warms up, drives the window
        manager to N window results, and shuts the server down on every exit path.
        Any host-side construction / launch / readiness failure (a bad dataset, a
        transport error, the server never coming up) is translated to a non-fatal
        failure dict - recorded per-cell and the study continues, mirroring the
        Docker path's DockerError translation - so one bad server cell never kills a
        whole multi-experiment study. Interrupts (KeyboardInterrupt / SystemExit)
        are BaseException-not-Exception and propagate. The N window results ride
        back as a ServerSessionResult (or a failure dict); the manifest transitions
        are the session's own, and reap is guaranteed by the session's
        __enter__-failure cleanup above.
        """
        from llenergymeasure.study.server_session import ServerSession

        spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        try:
            with ServerSession(
                self, config, spec, config_hash=config_hash, cycle=cycle, index=index
            ) as session:
                return session.run()
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            self.manifest.mark_failed(config_hash, cycle, failure["type"], failure["message"])
            if self._progress:
                self._progress.end_experiment_fail(index, 0.0, error=failure["message"])
            return failure

    def _run_one_server_group(self, member_configs: list[ExperimentConfig], *, index: int) -> Any:
        """Dispatch a rate-only-varying server group as one session.

        Assigns each member its (config_hash, cycle), skipping members already
        completed in a prior run (resume) while still advancing their cycle counters
        so subsequent cycles stay aligned. The surviving cells drive one server
        lifetime with a rate level each (one launch, re-warm per level, the warmup
        protocol unchanged); the per-cell manifest lifecycle is the session's own. Returns the
        session result (a ServerSessionResult or a failure dict), or None when the
        whole group was already completed on resume. Host-side construction / launch
        failures translate to a non-fatal failure dict per member, mirroring the
        single-cell path.
        """
        from llenergymeasure.domain.experiment import compute_declared_config_hash
        from llenergymeasure.study.server_session import ServerCell, ServerSession

        cells: list[ServerCell] = []
        for config in member_configs:
            config_hash = compute_declared_config_hash(config)
            cycle = self._cycle_counters.get(config_hash, 0) + 1
            self._cycle_counters[config_hash] = cycle
            if (config_hash, cycle) in self._skip_set:
                logger.info("Skipping completed server cell (resumed)")
                continue
            cells.append(ServerCell(config, config_hash, cycle))
        if not cells:
            return None

        spec = self._runner_specs.get(member_configs[0].engine) if self._runner_specs else None
        try:
            with ServerSession.for_group(self, cells, spec, index=index) as session:
                return session.run()
        except Exception as exc:
            error_type = type(exc).__name__
            message = str(exc)
            # "cells" carries the group size so the study summary counts a whole-group
            # launch failure as N failed cells, not one. The per-cell manifest
            # marks below remain the authoritative cell-level record.
            failure: dict[str, Any] = {"type": error_type, "message": message, "cells": len(cells)}
            # A late fault (e.g. a teardown fault escaping the session guard) must not
            # rewrite history: a cell already recorded completed has valid bundles on
            # disk, so downgrade only the cells not already completed.
            for cell in cells:
                if self.manifest.entry_status(cell.config_hash, cell.cycle) == "completed":
                    continue
                self.manifest.mark_failed(cell.config_hash, cell.cycle, error_type, message)
            if self._progress:
                self._progress.end_experiment_fail(index, 0.0, error=message)
            return failure

    def _handle_result(
        self,
        result: Any,
        config: ExperimentConfig,
        config_hash: str,
        cycle: int,
        index: int,
        elapsed: float,
        ts_source_dir: Path | None = None,
        environment_snapshot: Any | None = None,
        runner_provenance: RunnerProvenance | None = None,
    ) -> None:
        """Update manifest and signal study display based on experiment outcome."""
        model_name = config.task.model
        engine = engine_str(config.engine)
        if isinstance(result, dict) and "type" in result:
            error_type = result.get("type", "UnknownError")
            error_message = result.get("message", "")
            log_file = result.get("log_file")
            # Local/subprocess dispatch: the worker captured a full traceback but
            # only its type/message were being kept. Persist it into failed-runs/
            # so a local failure is as debuggable as the Docker path (which sets
            # log_file via persist_failure_artefacts). Docker failure dicts carry
            # no "traceback" key, so this only fires for local dispatch.
            if log_file is None and result.get("traceback"):
                from llenergymeasure.study.container_lifecycle import persist_failure_traceback

                persist_failure_traceback(
                    self.study_dir, config_hash, cycle, result["traceback"], result
                )
                log_file = result.get("log_file")
            self.manifest.mark_failed(
                config_hash, cycle, error_type, error_message, log_file=log_file
            )
            if self._progress:
                self._progress.end_experiment_fail(index, elapsed, error=error_message)
        else:
            _save_and_record(
                result,
                self.study_dir,
                self.manifest,
                config_hash,
                cycle,
                self.result_files,
                model_name=model_name,
                engine=engine,
                experiment_index=index,
                ts_source_dir=ts_source_dir,
                environment_snapshot=environment_snapshot,
                resolution_log=self._resolution_logs.get(config_hash),
                resolved_config_hash=self._resolved_hashes.get(config_hash),
                runner_provenance=runner_provenance,
            )
            if self._progress:
                host_path = self.result_files[-1] if self.result_files else None
                # Emit save paths as substeps BEFORE end_experiment_ok (which clears
                # inner step display). This makes paths visible inline in TTY mode.
                if host_path is not None:
                    # Docker experiments: show container path first, then host path.
                    # The original container path is /run/llem; by this point output_dir
                    # has been rewritten to the host temp dir, so use the known constant.
                    spec = self._runner_specs.get(engine) if self._runner_specs else None
                    is_docker = spec is not None and spec.mode == RUNNER_CONTAINER
                    if is_docker:
                        self._progress.on_substep("save", f"container: {CONTAINER_EXCHANGE_DIR}")
                    self._progress.on_substep("save", f"host: {host_path}")

                energy_j = getattr(result, "total_energy_j", None)
                throughput = getattr(result, "avg_tokens_per_second", None)
                infer_sec = getattr(result, "total_inference_time_sec", None)
                adj_energy_j = getattr(result, "energy_adjusted_j", None)
                energy_per_token_mj_adjusted = getattr(result, "energy_per_token_mj_adjusted", None)
                energy_per_token_mj_total = getattr(result, "energy_per_token_mj_total", None)
                self._progress.end_experiment_ok(
                    index,
                    elapsed,
                    energy_j=energy_j if energy_j and energy_j > 0 else None,
                    throughput_tok_s=throughput if throughput and throughput > 0 else None,
                    inference_time_sec=infer_sec if infer_sec and infer_sec > 0 else None,
                    adj_energy_j=adj_energy_j if adj_energy_j and adj_energy_j > 0 else None,
                    energy_per_token_mj_adjusted=energy_per_token_mj_adjusted,
                    energy_per_token_mj_total=energy_per_token_mj_total,
                )
                # Also store for finish() footer
                if host_path is not None:
                    self._progress.on_experiment_saved(index, host_path)

    def _run_one_docker(
        self,
        config: ExperimentConfig,
        spec: RunnerSpec,
        *,
        config_hash: str,
        cycle: int,
        index: int,
    ) -> Any:
        """Dispatch one experiment to a Docker container via DockerRunner.

        Blocking dispatch - no subprocess or thread overhead.
        DockerErrors are caught and converted to non-fatal failure dicts so the
        study continues even when a container fails.

        Args:
            config:      ExperimentConfig to run.
            spec:        Resolved RunnerSpec (mode="container") for this engine.
            config_hash: Pre-computed config hash (avoids recomputing).
            cycle:       Current cycle number for manifest tracking.
            index:       1-based position in study for progress display.

        Returns:
            ExperimentResult on success, or a failure dict on error.
        """
        from llenergymeasure.study.session import DockerSession

        # Docker dispatch: a DockerSession owns the container lifetime - image +
        # mount + facade prep on __enter__, teardown on __exit__. The DockerError
        # -> non-fatal failure-dict translation lives in the session's run(), so
        # the sweep loop keys off the single result uniformly with the local path.
        with DockerSession(
            self, config, spec, config_hash=config_hash, cycle=cycle, index=index
        ) as session:
            return session.run()
