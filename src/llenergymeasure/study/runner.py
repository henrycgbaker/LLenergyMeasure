"""StudyRunner - orchestrates per-experiment subprocess/Docker dispatch.

``StudyRunner`` owns the study-level run loop: SIGINT handling, GPU locks,
container lifecycle, circuit breaker, wall-clock timeout, and result handling.
The separable concerns live in sibling modules and are mixed in or imported:

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
import json
import logging
import multiprocessing
import os  # noqa: F401 - patch target: tests patch study.runner.os.{killpg,setpgrp}
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    RUNNER_DOCKER,
    TEMP_PREFIX_TIMESERIES,
    TIMEOUT_ENV_SNAPSHOT,
    TIMEOUT_INTERRUPT_POLL,
    TIMEOUT_SIGTERM_GRACE,
    TIMEOUT_THREAD_JOIN,
    engine_str,
)
from llenergymeasure.domain.bundle_artefacts import (
    CONFIG_SIDECAR_FILENAME,
    EQUIVALENCE_GROUPS_FILENAME,
)
from llenergymeasure.domain.progress import STEPS_LOCAL, docker_steps
from llenergymeasure.study._progress import _consume_progress_events
from llenergymeasure.study.baseline_measure import _BaselineMixin
from llenergymeasure.study.gaps import run_gap
from llenergymeasure.study.image_prep import _ImageMixin

# Re-imported into this module's namespace so that (a) existing
# ``from llenergymeasure.study.runner import X`` sites keep working and
# (b) ``patch("llenergymeasure.study.runner.<name>")`` intercepts the
# bare-name call sites inside ``_run_one`` / ``run``.
from llenergymeasure.study.worker import (
    _UNSET,
    COLLECT_RESULT_PROCESS_CRASH,
    COLLECT_RESULT_TIMEOUT,
    _collect_result,
    _derive_exit_reason,
    _kill_process_group,
    _run_experiment_worker,
)
from llenergymeasure.utils.io import load_json

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.experiment import RunnerProvenance
    from llenergymeasure.domain.progress import StudyProgressCallback
    from llenergymeasure.infra.runner_resolution import RunnerSpec
    from llenergymeasure.study.manifest import ManifestWriter

__all__ = [
    "StudyRunner",
    "_kill_process_group",
    "_run_experiment_worker",
    "_save_and_record",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level helpers
# =============================================================================


def _provenance_from_spec(spec: RunnerSpec | None) -> RunnerProvenance:
    """Build a RunnerProvenance from a resolved RunnerSpec.

    The infra-layer ``RunnerSpec`` cannot live on the domain result (layer
    violation), so its execution-mode fields are mirrored onto the domain-layer
    ``RunnerProvenance``. When no spec is available (pure in-process local run),
    records ``mode="local"`` with ``source="local"`` and no image.
    """
    from llenergymeasure.domain.experiment import RunnerProvenance

    if spec is None:
        return RunnerProvenance(mode="local", image=None, source="local", image_source=None)
    return RunnerProvenance(
        mode=spec.mode,
        image=spec.image,
        source=spec.source,
        image_source=spec.image_source,
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
    """Save result to disk and update manifest. Appends result path to result_files.

    Resolves the timeseries parquet sidecar from the result object and passes it
    to save_result() so it is copied into the experiment subdirectory. The stale
    flat file written by MeasurementHarness is removed after the copy.

    Args:
        model_name: Model name/path for the experiment directory slug
            (authoritative home: config.json; the result keeps a convenience copy).
        engine: Inference engine name for the experiment directory slug
            (authoritative home: config.json; the result keeps a convenience copy).
        ts_source_dir: Directory where the harness wrote timeseries.parquet.
        environment_snapshot: EnvironmentSnapshot for per-experiment environment.json sidecar.
        resolution_log: Pre-built per-field resolution log for this experiment,
            folded into the config.json sidecar's ``provenance`` section.
        runner_provenance: How the experiment was executed (local vs docker). Attached to the
            frozen result via model_copy before saving so it persists into result.json.

    On save failure, marks the experiment as completed with empty path.
    """
    try:
        from llenergymeasure.results.persistence import save_environment, save_result

        # Attach runner provenance to the frozen result before saving (it
        # serialises into result.json, unlike the environment sidecar).
        if runner_provenance is not None and hasattr(result, "model_copy"):
            result = result.model_copy(update={"runner_provenance": runner_provenance})

        # Resolve timeseries sidecar from result fields.
        # MeasurementHarness writes timeseries.parquet to the output_dir and
        # sets result.timeseries = "timeseries.parquet". Both must be present for
        # the copy to proceed.
        ts_source: Path | None = None
        ts_filename = getattr(result, "timeseries", None)
        if ts_filename and ts_source_dir is not None:
            candidate = ts_source_dir / ts_filename
            if candidate.exists():
                ts_source = candidate

        result_path = save_result(
            result,
            study_dir,
            model_name=model_name,
            engine=engine,
            timeseries_source=ts_source,
            experiment_index=experiment_index,
            cycle=cycle,
        )

        # Write per-experiment environment.json sidecar
        if environment_snapshot is not None:
            save_environment(
                environment_snapshot,
                result.experiment_id,
                config_hash,
                result_path.parent,
            )

        # Move config.json sidecar (written by harness to temp dir) to experiment dir.
        # Patch in two fields the harness subprocess cannot compute: the
        # resolved_config_hash from StudyConfig, and the per-field provenance
        # (source + effective + default) from the pre-built resolution log. The
        # provenance section replaces the retired _resolution.json sidecar.
        if ts_source_dir is not None:
            config_sidecar_src = ts_source_dir / CONFIG_SIDECAR_FILENAME
            if config_sidecar_src.exists():
                try:
                    _payload = load_json(config_sidecar_src)
                    if resolved_config_hash is not None:
                        _payload["resolved_config_hash"] = resolved_config_hash
                    if resolution_log:
                        _payload["provenance"] = resolution_log
                    from llenergymeasure.results.persistence import _atomic_write

                    _atomic_write(
                        json.dumps(_payload, indent=2, default=str),
                        result_path.parent / CONFIG_SIDECAR_FILENAME,
                    )
                except Exception as exc:  # pragma: no cover - best-effort
                    logger.debug("config.json sidecar move failed: %s", exc)
                finally:
                    config_sidecar_src.unlink(missing_ok=True)

        # Loudness backstop: config.json is the sole home of provenance and
        # the authoritative home of identity. A completed experiment that lands
        # without one is silent data loss - warn (never fail) so the gap is
        # visible rather than discovered later at analysis time.
        if not (result_path.parent / CONFIG_SIDECAR_FILENAME).exists():
            logger.warning(
                "No config.json materialised for %s (cycle %d) at %s - provenance "
                "and authoritative engine/model identity are missing from this result.",
                config_hash,
                cycle,
                result_path.parent,
            )

        # Clean up the stale flat parquet file after it has been copied into the
        # experiment subdirectory (mirrors cli/run.py line 288).
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)

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
            mj_per_tok=getattr(result, "mj_per_tok_adjusted", None)
            or getattr(result, "mj_per_tok_total", None),
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
        # Study-level baseline cache, keyed per runner target ("local" or
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

        # spawn: CUDA-safe; fork causes silent CUDA corruption (CP-1)
        mp_ctx = multiprocessing.get_context("spawn")

        # Reset interrupt state for this run
        self._interrupt_event.clear()
        self._interrupt_count = 0
        self._active_process = None
        self._cycle_counters = {}

        original_sigint, original_sigterm, gpu_locks = self._install_run_handlers(ordered)

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

                # Resume skip-set: skip experiments that completed in a previous run.
                if self._resume_should_skip(config, i, len(ordered)):
                    continue

                # Wall-clock timeout check: mark remaining experiments skipped.
                if deadline is not None and time.monotonic() > deadline:
                    self._mark_remaining_skipped(ordered, i, compute_declared_config_hash)
                    self.manifest.mark_study_timed_out()
                    logger.warning(
                        "Study timed out after %.1f hours",
                        self.study.study_execution.wall_clock_timeout_hours,
                    )
                    _aborted = True
                    break

                # Inter-experiment + per-cycle gaps (break if interrupted during a gap)
                if self._run_inter_experiment_gaps(i, n_unique):
                    break

                result = self._run_one(config, mp_ctx, index=i + 1)
                results.append(result)

                # Circuit breaker integration: update state based on result.
                if self._apply_circuit_breaker(breaker, result, ordered, i):
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

        Returns ``(original_sigint, original_sigterm, gpu_locks)`` for restoration in
        run()'s finally block via _restore_run_handlers.
        """
        original_sigint = signal.signal(signal.SIGINT, self._handle_sigint)

        # Acquire per-GPU advisory locks before image preparation.
        # Lock names use the PHYSICAL device the study occupies, parsed from the
        # docker --gpus pinning (LLEM_DOCKER_GPUS): two studies on different
        # physical GPUs must not share a lock. When there is no docker-level
        # pinning (all / unset), logical == physical, so fall back to the
        # in-container logical indices. Measurement-side index resolution is
        # unchanged - _resolve_gpu_indices still yields the logical indices that
        # address the energy samplers.
        # Sorted acquisition prevents deadlocks when multiple studies share GPUs.
        gpu_locks: list[Any] = []
        if not self._no_lock and ordered:
            from llenergymeasure.study.gpu_locks import acquire_gpu_locks
            from llenergymeasure.utils.env_config import pinned_gpu_lock_ids

            lock_ids = pinned_gpu_lock_ids()
            if lock_ids is None:
                from llenergymeasure.device.gpu_info import _resolve_gpu_indices

                lock_ids = [str(i) for i in _resolve_gpu_indices(ordered[0])]
            gpu_locks = acquire_gpu_locks(lock_ids)

        # Container lifecycle: reap orphaned containers, register cleanup, install SIGTERM bridge.
        # Only activated for studies that use Docker runners.
        original_sigterm: signal.Handlers | None = None
        if self._runner_specs and any(s.mode == RUNNER_DOCKER for s in self._runner_specs.values()):
            from llenergymeasure.study.container_lifecycle import (
                install_sigterm_bridge,
                reap_orphaned_containers,
                register_container_cleanup,
            )

            study_id = self.study.study_design_hash or "unknown"
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

    def _run_inter_experiment_gaps(self, index: int, n_unique: int) -> bool:
        """Run the inter-experiment gap and, on cycle boundaries, the per-cycle gap.

        Returns True if an interrupt arrived during a gap (caller should break).
        """
        # Config gap: between every consecutive experiment pair
        if index > 0:
            gap_secs = float(self.study.study_execution.experiment_gap_seconds or 0)
            if gap_secs > 0:
                self._run_gap(gap_secs, "Experiment gap")
                if self._interrupt_event.is_set():
                    return True

        # Cycle gap: after every complete round of N unique configs
        if n_unique > 0 and index > 0 and index % n_unique == 0:
            cycle_gap_secs = float(self.study.study_execution.cycle_gap_seconds or 0)
            if cycle_gap_secs > 0:
                self._run_gap(cycle_gap_secs, "Cycle gap")
                if self._interrupt_event.is_set():
                    return True
        return False

    def _apply_circuit_breaker(
        self, breaker: Any, result: Any, ordered: list[Any], index: int
    ) -> bool:
        """Update the circuit breaker from an experiment result.

        Records failure/success, applies cooldown + probe on a trip, and on a failed
        probe marks the remaining experiments skipped. Returns True when the study must
        abort (caller sets _aborted and breaks).
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
                self._mark_remaining_skipped(ordered, index + 1, compute_declared_config_hash)
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
    ) -> None:
        """Mark all experiments from start_index onwards as skipped in the manifest.

        Increments cycle counters to assign the correct cycle number for each
        remaining experiment before marking it skipped.

        Args:
            ordered: Full ordered experiment list (study.experiments).
            start_index: Index of the first experiment to mark as skipped.
            hash_fn: compute_declared_config_hash callable.
        """
        for j in range(start_index, len(ordered)):
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

        Checks runner_specs for this experiment's engine first. If a Docker spec is
        found, delegates to _run_one_docker(). Otherwise falls through to the existing
        subprocess dispatch path.

        If interrupt_event is set after join, attempts graceful SIGTERM → 2s grace →
        SIGKILL before collecting whatever result is available.
        """
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        config_hash = compute_declared_config_hash(config)
        # Increment per-config_hash counter: 1st run → cycle=1, 2nd → cycle=2, etc.
        cycle = self._cycle_counters.get(config_hash, 0) + 1
        self._cycle_counters[config_hash] = cycle

        # Docker dispatch path - check runner spec for this engine
        spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        if spec is not None and spec.mode == RUNNER_DOCKER:
            return self._run_one_docker(
                config, spec, config_hash=config_hash, cycle=cycle, index=index
            )

        timeout = self.study.study_execution.experiment_timeout_seconds

        # Signal study display: new experiment starting (subprocess = local steps)
        if self._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            local_spec = self._runner_specs.get(config.engine) if self._runner_specs else None
            self._progress.begin_experiment(
                index,
                format_experiment_header(config),
                list(STEPS_LOCAL),
                runner_info=local_spec.to_runner_info() if local_spec else None,
            )

        exp_start = time.monotonic()

        # Create a temp dir for harness artefacts. The harness receives it as
        # output_dir (a runtime param, not from config) and writes config.json
        # there always, plus timeseries.parquet when save_timeseries is on. The
        # staging dir is created regardless of save_timeseries so the config.json
        # sidecar - sole home of provenance, authoritative home of identity -
        # always materialises;
        # it is cleaned up after _handle_result copies the artefacts into the
        # study directory.
        save_ts = self.study.output.save_timeseries
        ts_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES))

        # Resolve cached snapshot in parent - serialised to subprocess via Pipe
        snapshot = self._get_env_snapshot()

        # Resolve cached baseline in parent - avoids 30s re-measurement per subprocess
        baseline = self._get_baseline(config) if config.measurement.baseline.enabled else None

        parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
        progress_queue = mp_ctx.Queue()

        p = mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, progress_queue, snapshot),
            kwargs={
                "output_dir": str(ts_tmpdir),
                "save_timeseries": save_ts,
                "baseline": baseline,
                "study_dir": str(self.study_dir),
                "study_run_id": self.study_run_id,
                "cycle": cycle,
                "config_hash": config_hash,
            },
            daemon=False,  # daemon=False: clean CUDA teardown if parent exits unexpectedly
        )

        consumer = threading.Thread(
            target=_consume_progress_events,
            args=(progress_queue, self._progress),
            daemon=True,
        )
        consumer.start()

        self.manifest.mark_running(config_hash, cycle)
        self._active_process = p

        # Pre-dispatch GPU memory residual check (MEAS-01, MEAS-02)
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

        check_gpu_memory_residual()

        p.start()
        child_conn.close()

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
        if self._interrupt_event.is_set() and p.is_alive():
            p.join(timeout=TIMEOUT_SIGTERM_GRACE)
            if p.is_alive():
                _kill_process_group(p.pid, signal.SIGKILL)
                p.join()

        self._active_process = None

        # Sentinel stops consumer thread - covers SIGKILL path too
        progress_queue.put(None)
        consumer.join()

        result = _collect_result(p, parent_conn, config, timeout, pipe_payload=pipe_payload)
        parent_conn.close()

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
                study_dir=self.study_dir,
                study_run_id=self.study_run_id,
                cycle=cycle,
                config_hash=config_hash,
                exit_reason=exit_reason,
                exit_code=p.exitcode,
            )

        exp_elapsed = time.monotonic() - exp_start
        local_spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        self._handle_result(
            result,
            config,
            config_hash,
            cycle,
            index,
            exp_elapsed,
            ts_source_dir=ts_tmpdir,
            environment_snapshot=self._get_env_snapshot() if not isinstance(result, dict) else None,
            runner_provenance=_provenance_from_spec(local_spec),
        )

        # Clean up the temp dir created for timeseries parquet output.
        # _save_and_record already copied the parquet into the study dir.
        if ts_tmpdir is not None:
            shutil.rmtree(ts_tmpdir, ignore_errors=True)

        return result

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
                    is_docker = spec is not None and spec.mode == RUNNER_DOCKER
                    if is_docker:
                        self._progress.on_substep("save", f"container: {CONTAINER_EXCHANGE_DIR}")
                    self._progress.on_substep("save", f"host: {host_path}")

                energy_j = getattr(result, "total_energy_j", None)
                throughput = getattr(result, "avg_tokens_per_second", None)
                infer_sec = getattr(result, "total_inference_time_sec", None)
                adj_energy_j = getattr(result, "energy_adjusted_j", None)
                mj_per_tok_adjusted = getattr(result, "mj_per_tok_adjusted", None)
                mj_per_tok_total = getattr(result, "mj_per_tok_total", None)
                self._progress.end_experiment_ok(
                    index,
                    elapsed,
                    energy_j=energy_j if energy_j and energy_j > 0 else None,
                    throughput_tok_s=throughput if throughput and throughput > 0 else None,
                    inference_time_sec=infer_sec if infer_sec and infer_sec > 0 else None,
                    adj_energy_j=adj_energy_j if adj_energy_j and adj_energy_j > 0 else None,
                    mj_per_tok_adjusted=mj_per_tok_adjusted,
                    mj_per_tok_total=mj_per_tok_total,
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
            spec:        Resolved RunnerSpec (mode="docker") for this engine.
            config_hash: Pre-computed config hash (avoids recomputing).
            cycle:       Current cycle number for manifest tracking.
            index:       1-based position in study for progress display.

        Returns:
            ExperimentResult on success, or a failure dict on error.
        """
        from llenergymeasure.infra.docker_errors import docker_exc_to_failure
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.container_lifecycle import (
            generate_container_labels,
            generate_container_name,
            persist_failure_artefacts,
        )
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual
        from llenergymeasure.utils.exceptions import DockerError

        # Image is pre-resolved during preflight (resolve_image precedence chain).
        # Fall back to get_default_image() only for direct DockerRunner usage
        # outside the study path.
        image = spec.image if spec.image is not None else get_default_image(config.engine)

        study_id = self.study.study_design_hash or "unknown"
        container_name = generate_container_name(study_id, index)
        labels = generate_container_labels(study_id)

        # begin_experiment MUST run before _get_baseline so baseline step events
        # fire against a registered experiment index.
        if self._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            host_baseline = (
                config.measurement.baseline.enabled
                and config.measurement.baseline.strategy != "fresh"
            )
            steps = docker_steps(
                images_prepared=self._images_prepared,
                host_baseline=host_baseline,
            )
            self._progress.begin_experiment(
                index,
                format_experiment_header(config),
                steps,
                runner_info=spec.to_runner_info(),
            )
            # Host-side preflight doesn't run in Docker path - checked inside container
            self._progress.on_step_skip("preflight", "checked inside container")

        extra_mounts = list(spec.extra_mounts) if spec.extra_mounts else []
        cache_key = self._baseline_cache_key(config)
        baseline = self._get_baseline(config) if config.measurement.baseline.enabled else None
        if baseline is not None:
            # Experiment container reads /run/llem/baseline_cache.json; the host
            # picks the right per-cache-key file at dispatch time. Docker parses
            # relative bind-mount sources as named volumes, so resolve first.
            baseline_cache_path = self._get_baseline_cache_path(cache_key)
            extra_mounts.append(
                (
                    str(baseline_cache_path.resolve()),
                    f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json",
                )
            )

        docker_runner = DockerRunner(
            image=image,
            timeout=self.study.study_execution.experiment_timeout_seconds,
            silence_timeout=self.study.study_execution.stdout_silence_timeout_seconds,
            source=spec.source,
            extra_mounts=extra_mounts,
            container_name=container_name,
            labels=labels,
        )

        # Pre-dispatch GPU memory residual check (same as local path)
        check_gpu_memory_residual()

        self.manifest.mark_running(config_hash, cycle)

        exp_start = time.monotonic()

        result: Any
        docker_ts_dir: Path | None = None
        try:
            # Pass study progress as step callback - DockerRunner calls on_step_*
            # skip_image_check=True when images were verified at study level.
            result, docker_ts_dir = docker_runner.run(
                config,
                progress=self._progress,
                save_timeseries=self.study.output.save_timeseries,
                skip_image_check=self._images_prepared,
            )
        except DockerError as exc:
            # Translate to a non-fatal failure dict (silence / timeout / structured
            # payload classification handled in the shared helper) and persist the
            # container.log + error JSON so the failure is debuggable.
            result = docker_exc_to_failure(exc, config_hash)
            persist_failure_artefacts(exc, self.study_dir, config_hash, cycle, result)

        exp_elapsed = time.monotonic() - exp_start
        self._handle_result(
            result,
            config,
            config_hash,
            cycle,
            index,
            exp_elapsed,
            ts_source_dir=docker_ts_dir,
            environment_snapshot=self._get_env_snapshot() if not isinstance(result, dict) else None,
            runner_provenance=_provenance_from_spec(spec),
        )

        # Clean up the temp dir after _save_and_record has copied the parquet.
        if docker_ts_dir is not None:
            shutil.rmtree(docker_ts_dir, ignore_errors=True)

        return result
