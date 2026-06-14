"""Study-level baseline measurement, caching, and drift validation.

``_BaselineMixin`` holds the stateful baseline methods mixed into
``StudyRunner``. Baselines are keyed per runner target ("local" or
``image_<slug>``) so multi-engine studies don't cross-contaminate, and are
persisted to disk so mid-study restarts reuse a still-valid measurement.

For Docker targets the measurement runs inside a short-lived container of the
same engine image, so the CUDA init footprint matches the experiment container
(see ``.product/research/baseline-measurement-location.md``).

These methods read and write ``self._baselines``, ``self._progress``,
``self._runner_specs``, and ``self._experiments_since_validation``, which are
initialised in ``StudyRunner.__init__``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import RUNNER_DOCKER
from llenergymeasure.domain.progress import STEP_BASELINE
from llenergymeasure.study.image_prep import _sanitize_image_for_filename

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.progress import StudyProgressCallback
    from llenergymeasure.infra.runner_resolution import RunnerSpec

logger = logging.getLogger(__name__)


class _BaselineMixin:
    """Stateful baseline measurement/caching/validation methods for StudyRunner.

    Relies on attributes set up by ``StudyRunner.__init__``:
    ``study_dir``, ``_runner_specs``, ``_progress``, ``_baselines``, and
    ``_experiments_since_validation``.
    """

    # Attributes provided by StudyRunner.__init__ (declared for the type checker).
    study_dir: Path
    _runner_specs: dict[str, RunnerSpec] | None
    _progress: StudyProgressCallback | None
    _baselines: dict[str, Any]
    _experiments_since_validation: dict[str, int]

    def _baseline_cache_key(self, config: ExperimentConfig) -> str:
        """Return the cache key for this experiment's runner target.

        Baselines are keyed per runner target because the container's CUDA init
        footprint (~8.7 W/GPU on A100) is process-local and may differ between
        engine images. See ``.product/research/baseline-measurement-location.md``.

        The ``image_`` prefix uses an underscore (not ``:``) so the key is
        safe to embed directly in both filesystem paths and Docker bind-mount
        sources. A ``:`` in the mount source string would be parsed by Docker
        as the mount-mode separator and fail with ``invalid mode``.
        """
        spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        if spec is None or spec.mode != RUNNER_DOCKER or not spec.image:
            return "local"
        return f"image_{_sanitize_image_for_filename(spec.image)}"

    def _get_baseline(self, config: ExperimentConfig) -> Any:
        """Return baseline power according to the configured strategy.

        Strategies: ``cached`` (measure once per runner target, persist to disk,
        reuse within TTL), ``validated`` (same, with periodic drift spot-check),
        and ``fresh`` (returns None; harness measures in-container per experiment).

        For Docker targets the measurement runs inside a short-lived container
        of the same engine image, so the CUDA init state matches the experiment
        container (see ``.product/research/baseline-measurement-location.md``).
        """
        strategy = config.measurement.baseline.strategy

        if strategy == "fresh":
            return None

        cache_key = self._baseline_cache_key(config)
        location = "host" if cache_key == "local" else "baseline container"
        cached = self._resolve_cached_baseline(config, cache_key, strategy)

        # In-memory hit → emit "Reusing" and return
        if cached is not None:
            if self._progress is not None:
                self._progress.on_step_start(
                    STEP_BASELINE,
                    "Reusing",
                    f"cached {cached.power_w:.1f}W · {location}",
                )
                self._emit_baseline_result_substeps(cached, elapsed=0.0, mode="cached")
                self._progress.on_step_done(STEP_BASELINE, 0.0)
            return cached

        # Try loading from disk first (handles mid-study restarts)
        disk_path = self._get_baseline_cache_path(cache_key)
        loaded = self._load_disk_baseline(config, cache_key, location, strategy, disk_path)
        if loaded is not None:
            return loaded

        # Measure fresh baseline (in-container for Docker targets)
        return self._measure_fresh_baseline(config, cache_key, strategy, disk_path)

    def _resolve_cached_baseline(
        self, config: ExperimentConfig, cache_key: str, strategy: str
    ) -> Any:
        """Return the in-memory cached baseline for ``cache_key`` if still usable.

        Drops the cache on TTL expiry and, for the ``validated`` strategy, runs a
        periodic drift spot-check (which may re-measure). Returns None when no valid
        in-memory baseline is available.
        """
        cached = self._baselines.get(cache_key)

        # TTL expiry check
        if cached is not None:
            age = time.time() - cached.timestamp
            if age >= config.measurement.baseline.cache_ttl_seconds:
                logger.info(
                    "Baseline expired (age=%.0fs > ttl=%.0fs). Re-measuring.",
                    age,
                    config.measurement.baseline.cache_ttl_seconds,
                )
                cached = None
                self._baselines.pop(cache_key, None)

        # validated: periodic spot-check for drift (only if baseline still valid)
        if strategy == "validated" and cached is not None:
            self._experiments_since_validation[cache_key] = (
                self._experiments_since_validation.get(cache_key, 0) + 1
            )
            if (
                self._experiments_since_validation[cache_key]
                >= config.measurement.baseline.validation_interval
            ):
                self._validate_baseline(config, cache_key)
                cached = self._baselines.get(cache_key)  # may have been re-measured
        return cached

    def _load_disk_baseline(
        self,
        config: ExperimentConfig,
        cache_key: str,
        location: str,
        strategy: str,
        disk_path: Path,
    ) -> Any:
        """Load a persisted baseline from ``disk_path`` (mid-study restart path).

        Emits Loading progress, registers a valid load in memory, and returns it, or
        None when the file is absent or its on-disk cache is no longer valid.
        """
        if not disk_path.exists():
            return None
        from llenergymeasure.harness.baseline import load_baseline_cache

        if self._progress is not None:
            self._progress.on_step_start(STEP_BASELINE, "Loading", f"baseline cache · {location}")
            t0_load = time.perf_counter()

        loaded = load_baseline_cache(disk_path, ttl=config.measurement.baseline.cache_ttl_seconds)

        if self._progress is not None:
            load_elapsed = time.perf_counter() - t0_load
            if loaded is not None:
                self._emit_baseline_result_substeps(loaded, elapsed=load_elapsed, mode="disk")
            self._progress.on_step_done(STEP_BASELINE, load_elapsed)

        if loaded is not None:
            if loaded.method is None:
                loaded.method = strategy
            self._baselines[cache_key] = loaded
            self._experiments_since_validation.setdefault(cache_key, 0)
            logger.debug("Loaded baseline from disk cache: %.1fW", loaded.power_w)
            return loaded
        return None

    def _measure_fresh_baseline(
        self, config: ExperimentConfig, cache_key: str, strategy: str, disk_path: Path
    ) -> Any:
        """Measure a fresh baseline (in-container for Docker targets), persist it to
        ``disk_path``, and emit Calibrating progress.

        Returns the measured baseline, or None on measurement failure.
        """
        dur = config.measurement.baseline.duration_seconds
        # Prefer the image tag over engine name so users see which container
        # is running in multi-engine studies.
        spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        target_label = (spec.image if spec and spec.image else config.engine) or "baseline"
        if self._progress is not None:
            self._progress.on_step_start(STEP_BASELINE, "Calibrating", "sampling idle GPU draw")
            t0_meas = time.perf_counter()
            # Seed first substep before Popen so the docker cold-start is visible.
            if cache_key != "local":
                self._progress.on_substep_start(
                    STEP_BASELINE,
                    f"launching separate {target_label} baseline container",
                )

        on_stage = (
            self._make_baseline_stage_callback(duration_sec=dur, target_label=target_label)
            if self._progress is not None
            else None
        )
        measured = self._measure_baseline(config, cache_key, on_stage=on_stage)

        if measured is not None:
            measured.method = strategy
            self._baselines[cache_key] = measured
            self._experiments_since_validation[cache_key] = 0

            from llenergymeasure.harness.baseline import save_baseline_cache

            save_baseline_cache(disk_path, measured)

        if self._progress is not None:
            elapsed = time.perf_counter() - t0_meas
            if measured is None:
                # Freeze any live substep with the failure text + accurate
                # elapsed so users don't see a silent tick hiding a crash.
                self._progress.on_substep_done(
                    STEP_BASELINE,
                    text=(
                        f"measurement failed after {elapsed:.1f}s - see log warnings "
                        f"(experiment container will re-measure fresh)"
                    ),
                    elapsed_sec=elapsed,
                )
            elif cache_key == "local":
                # No container subprocess → emit a retroactive sampling substep
                # so the local path still gets a breakdown. The Docker path
                # already emitted substeps live via the on_stage callback.
                self._emit_baseline_result_substeps(
                    measured,
                    elapsed=elapsed,
                    mode="fresh",
                    is_containerised=False,
                )
            self._progress.on_step_done(STEP_BASELINE, elapsed)

        return measured

    def _make_baseline_stage_callback(
        self, *, duration_sec: float, target_label: str = "baseline"
    ) -> Any:
        """Build a stage-marker callback that drives live baseline sub-bullets.

        Each transition ``on_substep_done``s the prior substep (freezing with
        a tick) and ``on_substep_start``s the next one. The very first substep
        ("launching <target_label> container") is seeded in ``_get_baseline``
        before ``subprocess.Popen`` so the docker cold-start is visible.
        """
        dur_label = f"{duration_sec:.0f}s"

        def on_stage(name: str, elapsed: float, kv: dict[str, str]) -> None:
            if self._progress is None:
                return
            if name == "container_ready":
                self._progress.on_substep_done(
                    STEP_BASELINE, f"{target_label} baseline container ready"
                )
                self._progress.on_substep_start(
                    STEP_BASELINE, "initialising CUDA runtime inside baseline container"
                )
            elif name == "cuda_primed":
                self._progress.on_substep_done(STEP_BASELINE, "CUDA runtime primed")
                self._progress.on_substep_start(
                    STEP_BASELINE, f"sampling idle GPU draw ({dur_label})"
                )
            elif name == "sampling_done":
                power_w = kv.get("power_w", "?")
                samples = kv.get("samples", "?")
                sampled = kv.get("duration", "?")
                self._progress.on_substep_done(
                    STEP_BASELINE,
                    f"sampled idle GPU draw · {sampled}s ({power_w}W · {samples} samples)",
                )
                self._progress.on_substep_start(
                    STEP_BASELINE,
                    f"writing baseline result · tearing down {target_label} baseline container",
                )
            elif name == "result_written":
                self._progress.on_substep_done(
                    STEP_BASELINE,
                    f"baseline result cached · {target_label} baseline container torn down",
                )

        return on_stage

    def _emit_baseline_result_substeps(
        self,
        baseline: Any,  # BaselineCache
        *,
        elapsed: float,
        mode: str,  # "fresh" | "cached" | "disk"
        is_containerised: bool = False,
    ) -> None:
        """Emit dim-bullet substeps explaining where the baseline time went.

        For a fresh Docker measurement we split the wall-clock into container
        launch/teardown (the residual) vs the NVML sampling window (recorded
        inside ``measure_baseline_power``). This answers "why did a 30s
        measurement take 37.7s?" without the user having to dig through logs.

        For in-memory and disk-loaded reuse we only emit the result summary -
        there is no sampling window to describe.
        """
        if self._progress is None:
            return
        if mode == "fresh" and is_containerised:
            # Residual captures docker run startup + CUDA prime + result write
            # + container teardown.
            overhead = max(0.0, elapsed - baseline.duration_sec)
            self._progress.on_substep(
                STEP_BASELINE,
                f"container launch + teardown: {overhead:.1f}s",
                overhead,
            )
            self._progress.on_substep(
                STEP_BASELINE,
                f"sampling: {baseline.duration_sec:.1f}s "
                f"({baseline.power_w:.1f}W · {baseline.sample_count} samples)",
                baseline.duration_sec,
            )
        elif mode == "fresh":
            self._progress.on_substep(
                STEP_BASELINE,
                f"sampling: {baseline.duration_sec:.1f}s "
                f"({baseline.power_w:.1f}W · {baseline.sample_count} samples)",
                baseline.duration_sec,
            )
        else:
            source = "in-memory" if mode == "cached" else "disk"
            self._progress.on_substep(
                STEP_BASELINE,
                f"reused from {source} cache: "
                f"{baseline.power_w:.1f}W ({baseline.sample_count} samples)",
            )

    def _measure_baseline(
        self,
        config: ExperimentConfig,
        cache_key: str,
        on_stage: Any = None,  # StageCallback | None
    ) -> Any:
        """Measure a fresh baseline on host or inside a baseline container.

        Local runner targets measure on host (no process boundary, no bias).
        Docker targets dispatch a short-lived baseline container of the engine
        image so the CUDA init state matches the experiment container. See
        ``.product/research/baseline-measurement-location.md`` for the
        empirical rationale (~8.7 W/GPU bias on A100).

        Args:
            config: Experiment config with baseline settings.
            cache_key: "local" or "image_<slug>" - chooses dispatch path.
            on_stage: Optional callback forwarded to ``run_baseline_container``
                for streaming stage markers. Ignored on the local path (there
                is no subprocess to stream from).
        """
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        gpu_indices = _resolve_gpu_indices(config)

        if cache_key == "local":
            from llenergymeasure.harness.baseline import measure_baseline_power

            return measure_baseline_power(
                duration_sec=config.measurement.baseline.duration_seconds,
                gpu_indices=gpu_indices,
            )

        # Docker: _baseline_cache_key already guaranteed a resolved image when
        # it returned a non-"local" key.
        assert self._runner_specs is not None
        spec = self._runner_specs[config.engine]
        assert spec.image is not None
        from llenergymeasure.study.baseline_container import run_baseline_container

        return run_baseline_container(
            image=spec.image,
            mode="measure",
            duration_sec=config.measurement.baseline.duration_seconds,
            gpu_indices=gpu_indices,
            on_stage=on_stage,
        )

    def _spot_check_baseline(
        self,
        config: ExperimentConfig,
        cache_key: str,
        gpu_indices: list[int],
    ) -> float | None:
        """Quick drift-check measurement for the validated strategy.

        Dispatches to host or baseline container matching the cache key. Returns
        the measured power in watts, or None on failure.
        """
        if cache_key == "local":
            from llenergymeasure.harness.baseline import measure_spot_check

            return measure_spot_check(gpu_indices=gpu_indices, duration_sec=5.0)

        spec = self._runner_specs.get(config.engine) if self._runner_specs else None
        if spec is None or spec.image is None:
            return None

        from llenergymeasure.study.baseline_container import run_baseline_container

        result = run_baseline_container(
            image=spec.image,
            mode="spot_check",
            duration_sec=5.0,
            gpu_indices=gpu_indices,
        )
        return result.power_w if result is not None else None

    def _validate_baseline(self, config: ExperimentConfig, cache_key: str) -> None:
        """Spot-check baseline for drift (strategy='validated' only).

        Performs a short measurement and compares with the cached baseline. On
        drift beyond the configured threshold, re-measures the full baseline and
        updates the disk cache. Emits a single ``Validating`` step (with an
        ``on_step_update`` mid-step when a re-measure is triggered) so the
        display step counter stays clean.
        """
        cached = self._baselines.get(cache_key)
        if cached is None:
            return

        from llenergymeasure.device.gpu_info import _resolve_gpu_indices
        from llenergymeasure.harness.baseline import save_baseline_cache

        gpu_indices = _resolve_gpu_indices(config)
        location = "host" if cache_key == "local" else "baseline container"

        if self._progress is not None:
            self._progress.on_step_start(
                STEP_BASELINE, "Validating", f"{location} · drift check (5s)"
            )
            t0 = time.perf_counter()

        spot = self._spot_check_baseline(config, cache_key, gpu_indices)
        self._experiments_since_validation[cache_key] = 0

        if spot is None:
            logger.warning("Baseline validation: spot-check measurement failed")
            if self._progress is not None:
                self._progress.on_step_done(STEP_BASELINE, time.perf_counter() - t0)
            return

        drift = abs(spot - cached.power_w) / cached.power_w
        if drift > config.measurement.baseline.drift_threshold:
            dur = config.measurement.baseline.duration_seconds
            logger.info(
                "Baseline drift detected: %.1fW -> %.1fW (%.1f%% > %.1f%% threshold). "
                "Re-measuring full baseline.",
                cached.power_w,
                spot,
                drift * 100,
                config.measurement.baseline.drift_threshold * 100,
            )
            if self._progress is not None:
                self._progress.on_step_update(
                    STEP_BASELINE,
                    f"{location} · drift {drift * 100:.1f}% > "
                    f"{config.measurement.baseline.drift_threshold * 100:.0f}%, "
                    f"re-measuring ({dur:.0f}s)",
                )

            remeasured = self._measure_baseline(config, cache_key)
            if remeasured is not None:
                remeasured.method = "validated"
                self._baselines[cache_key] = remeasured
                save_baseline_cache(self._get_baseline_cache_path(cache_key), remeasured)
        else:
            cached.method = "validated"
            logger.debug(
                "Baseline validation passed: drift=%.1f%% (threshold=%.1f%%)",
                drift * 100,
                config.measurement.baseline.drift_threshold * 100,
            )

        if self._progress is not None:
            self._progress.on_step_done(STEP_BASELINE, time.perf_counter() - t0)

    def _get_baseline_cache_path(self, cache_key: str) -> Path:
        """Return the disk path for the baseline cache file keyed by runner target.

        File lives at ``{study_dir}/_study-artefacts/baseline_cache_{cache_key}.json``.
        Creates the artefacts directory if needed.
        """
        artefacts_dir = self.study_dir / "_study-artefacts"
        artefacts_dir.mkdir(parents=True, exist_ok=True)
        return artefacts_dir / f"baseline_cache_{cache_key}.json"
