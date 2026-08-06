"""Progress callback protocol for step-by-step measurement reporting.

Lives in domain/ (Layer 0) so every layer can import it without
violating the architectural layering rules.

Steps are grouped into phases for hierarchical display (Docker BuildKit
style). The display renders phase headers with indented sub-steps.
Steps that don't apply are shown as SKIP with a fixed total count.

The protocol also supports on_substep() for fine-grained sub-operation
reporting within an active step (e.g. CUDA check, model access check).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Callback protocol for reporting measurement step progress.

    Step names come from the step registry (``StepSpec`` / ``register_step``
    below), grouped into phases. The base vocabulary::

        Setup:       preflight, image_check, pull, container_start, container_preflight
        Measurement: baseline, model, prompts, warmup, thermal_floor,
                     energy_select, measure, flops, save

    The registry is extensible - future modes add steps and phases as registry
    entries, and every consumer (labels, phase mapping, ordered step lists)
    derives from it.

    Steps that don't apply in a given run are reported via on_step_skip()
    and rendered as SKIP in the display. This keeps a fixed [x/y] counter.

    Sub-step granularity: on_substep() reports completed sub-operations
    within an active step (e.g. "CUDA available", "model accessible").

    Implementors:
        - StepDisplay (cli/_step_display.py) -- Rich-based TTY rendering
        - StreamProgressCallback (entrypoints/container.py) -- JSON lines to stdout
        - _QueueProgressCallback (study/_progress.py) -- multiprocessing.Queue bridge
    """

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        """Signal that a named step has begun.

        Args:
            step: Step identifier from the fixed vocabulary.
            description: Human-readable verb label (e.g. "Loading model").
            detail: Additional context (e.g. model name, image tag).
        """
        ...

    def on_step_update(self, step: str, detail: str) -> None:
        """Update the detail text of the currently active step.

        Used for live progress within a step (e.g. warmup iteration count,
        CV convergence progress).

        Args:
            step: Step identifier (must match most recent on_step_start).
            detail: Updated detail text.
        """
        ...

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        """Signal that a step has completed.

        Args:
            step: Step identifier (must match most recent on_step_start).
            elapsed_sec: Wall-clock time for this step in seconds.
        """
        ...

    def on_step_skip(self, step: str, reason: str = "") -> None:
        """Signal that a step was skipped (not applicable to this run).

        Args:
            step: Step identifier from the fixed vocabulary.
            reason: Optional human-readable reason (e.g. "disabled", "cached").
        """
        ...

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Signal a completed sub-operation within the active step.

        Args:
            step: Parent step identifier (must match most recent on_step_start).
            text: Human-readable substep description (e.g. "CUDA available").
            elapsed_sec: Wall-clock time for this substep (0.0 = instantaneous).
        """
        ...

    def on_substep_start(self, step: str, text: str) -> None:
        """Start a live sub-operation within the active step.

        The substep renders as a dim indented bullet with a spinner and
        rising elapsed counter until ``on_substep_done`` arrives. Only one
        active substep per parent step is supported - calling
        ``on_substep_start`` again without a matching ``on_substep_done``
        freezes the prior substep with the previous start's text and its
        accumulated elapsed.

        Args:
            step: Parent step identifier (must match the active step).
            text: Present-tense description (e.g. "launching baseline container").
        """
        ...

    def on_substep_done(
        self,
        step: str,
        text: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Freeze the currently-active sub-operation of ``step``.

        Args:
            step: Parent step identifier.
            text: Optional final text (e.g. "42.6W · 288 samples"). If
                ``None``, the original ``on_substep_start`` text is kept.
            elapsed_sec: Optional override for the recorded duration. If
                ``None``, the monotonic delta since ``on_substep_start`` is
                used.
        """
        ...


def emit_substep(
    progress: ProgressCallback | None, step: str, text: str, elapsed_sec: float = 0.0
) -> None:
    """Emit a substep event to the progress callback when one is registered.

    Shared by the harness and the measurement bracket so the "no callback -> no-op"
    guard lives in exactly one place.
    """
    if progress is not None:
        progress.on_substep(step, text, elapsed_sec)


@runtime_checkable
class StudyProgressCallback(ProgressCallback, Protocol):
    """Extended callback for study-level experiment tracking + per-step progress.

    Adds begin/end experiment methods on top of ProgressCallback's step events.
    Used by StudyStepDisplay (cli/_step_display.py) and consumed by
    StudyRunner (study/runner.py).

    Implementors:
        - StudyStepDisplay (cli/_step_display.py)
    """

    def begin_experiment(
        self,
        index: int,
        header: str,
        steps: list[str],
        runner_info: dict[str, str | None] | None = None,
    ) -> None:
        """Signal that a new experiment is starting within the study.

        Args:
            index: 1-based experiment position in the study.
            header: Pre-built display string (e.g. "Qwen2.5-0.5B / transformers / bf16 batch=4").
                    Built by ``format_experiment_header()`` in ``utils/formatting.py``.
            steps: Ordered step names for this experiment's [x/y] counter.
            runner_info: Optional dict with runner/image provenance for display:
                ``mode`` ("process" or "container"), ``source`` (where mode was resolved),
                ``image`` (container image tag or None), ``image_source`` (where image
                was resolved: "local_build", "registry", "env", "yaml", etc.).
        """
        ...

    def end_experiment_ok(
        self,
        index: int,
        elapsed: float,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
        inference_time_sec: float | None = None,
        adj_energy_j: float | None = None,
        energy_per_token_mj_adjusted: float | None = None,
        energy_per_token_mj_total: float | None = None,
    ) -> None:
        """Signal that an experiment completed successfully.

        Args:
            index: 1-based experiment position.
            elapsed: Total wall-clock time in seconds.
            energy_j: Total energy in joules (None if unavailable).
            throughput_tok_s: Throughput in tokens/second (None if unavailable).
            inference_time_sec: Measurement window duration (None if unavailable).
            adj_energy_j: Baseline-subtracted energy in joules (None if no baseline).
            energy_per_token_mj_adjusted: mJ/tok from adjusted energy (None if unavailable).
            energy_per_token_mj_total: mJ/tok from total energy (None if unavailable).
        """
        ...

    def end_experiment_fail(self, index: int, elapsed: float, error: str = "") -> None:
        """Signal that an experiment failed.

        Args:
            index: 1-based experiment position.
            elapsed: Wall-clock time until failure in seconds.
            error: Human-readable error message.
        """
        ...

    def on_experiment_saved(
        self, index: int, host_path: str, container_path: str | None = None
    ) -> None:
        """Signal that experiment results were saved to disk.

        Args:
            index: 1-based experiment position.
            host_path: Absolute path on the host filesystem.
            container_path: Path inside the Docker container (None for local runs).
        """
        ...

    def begin_image_prep(self, engines: list[str]) -> None:
        """Signal the start of study-level Docker image preparation.

        Args:
            engines: Engine names that require Docker images.
        """
        ...

    def image_ready(
        self,
        engine: str,
        image: str,
        cached: bool,
        elapsed: float,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Signal that a Docker image is ready (found locally or pulled).

        Args:
            engine: Engine name (e.g. "transformers").
            image: Docker image reference.
            cached: True if image was found locally, False if pulled.
            elapsed: Wall-clock time for the check/pull.
            metadata: Optional image metadata (id, size, built, layers).
        """
        ...

    def image_failed(self, engine: str, image: str, error: str) -> None:
        """Signal that a Docker image could not be prepared.

        Args:
            engine: Engine name.
            image: Docker image reference that was attempted.
            error: Human-readable error message.
        """
        ...

    def end_image_prep(self) -> None:
        """Signal the end of study-level Docker image preparation."""
        ...

    def show_gap(self, text: str) -> None:
        """Show a gap countdown line in the display (e.g. 'Experiment gap: 7s')."""
        ...

    def clear_gap(self) -> None:
        """Clear the gap countdown line."""
        ...


# Phase names -- top-level groups in the hierarchical display.
PHASE_SETUP = "Setup"
PHASE_MEASUREMENT = "Measurement"

# Step vocabulary -- canonical names used across all layers.
# Setup phase
STEP_PREFLIGHT = "preflight"
STEP_IMAGE_CHECK = "image_check"
STEP_PULL = "pull"
STEP_CONTAINER_START = "container_start"
STEP_CONTAINER_PREFLIGHT = "container_preflight"

# Measurement phase
STEP_BASELINE = "baseline"
STEP_MODEL = "model"
STEP_PROMPTS = "prompts"
STEP_WARMUP = "warmup"
STEP_THERMAL_FLOOR = "thermal_floor"
STEP_ENERGY_SELECT = "energy_select"
STEP_MEASURE = "measure"
STEP_FLOPS = "flops"
STEP_SAVE = "save"

# Server-mode-only steps (online-serving measurement): launch the engine server
# and drive it to readiness before the measured windows open.
STEP_SERVER_LAUNCH = "server_launch"
STEP_SERVER_READY = "server_ready"

# -------------------------------------------------------------------------
# Step registry -- the single source of truth for the progress vocabulary.
#
# Each StepSpec fully describes one step: its identity, display label, phase,
# ordering, and the run-mode surfaces it appears in. The label map, phase map,
# and the ordered step lists every consumer renders from are all *derived* from
# this registry. Adding a step (server mode's server-start / health / ramp
# phases, for example) is one ``register_step()`` call, or one entry below - no
# renderer, dispatcher, or list builder needs editing.
# -------------------------------------------------------------------------

# Step-list surfaces -- the run-mode variants that select and order the steps a
# run emits. A step declares which surfaces it belongs to and where it sits in
# each. Future modes (server mode) add their own surface here without touching
# any consumer.
#: Surface identifiers, aligned with the runner-mode vocabulary
#: (process / container, formerly local / docker). Not serialized - internal
#: step-list keys only - so the rename rides freely.
SURFACE_PROCESS = "process"  # direct host harness run, no container
SURFACE_CONTAINER = "container"  # container run, baseline measured in-container (fresh)
SURFACE_CONTAINER_HOST_BASELINE = "container_host_baseline"  # baseline measured on host first
SURFACE_SERVER = "server"  # online-serving measurement (host-driven traffic + sampling)

#: The offline surfaces (the batch-measurement modes). Setup + offline-only
#: measurement steps belong here; ``_ALL_SURFACES`` keeps its name and membership
#: so every existing offline StepSpec is untouched by the rename.
_ALL_SURFACES = frozenset({SURFACE_PROCESS, SURFACE_CONTAINER, SURFACE_CONTAINER_HOST_BASELINE})
_DOCKER_SURFACES = frozenset({SURFACE_CONTAINER, SURFACE_CONTAINER_HOST_BASELINE})
#: The measurement-tail steps every mode shares (offline surfaces + server).
_MEASUREMENT_SHARED = _ALL_SURFACES | frozenset({SURFACE_SERVER})


@dataclass(frozen=True)
class StepSpec:
    """Full description of one progress step so every consumer renders from it.

    Attributes:
        id: Stable step identifier (the frozen ``STEP_*`` value emitted on the
            wire and matched by renderers). Never renamed - display behaviour
            depends on the exact string.
        label: Human-readable verb shown in the display (e.g. "Loading").
        phase: Phase header the step groups under (``PHASE_SETUP`` /
            ``PHASE_MEASUREMENT``).
        order: Position key within an ordered step list (lower sorts earlier).
        surfaces: Run-mode surfaces the step appears in (see ``SURFACE_*``).
        order_overrides: Per-surface ``order`` overrides for steps whose
            position depends on the run mode. Defaults to ``order`` everywhere;
            only ``STEP_BASELINE`` needs one - it is measured earlier when the
            host runs a standalone baseline before the experiment container.
    """

    id: str
    label: str
    phase: str
    order: int
    surfaces: frozenset[str]
    order_overrides: Mapping[str, int] = field(default_factory=dict)

    def order_in(self, surface: str) -> int:
        """Ordering key for this step within ``surface``."""
        return self.order_overrides.get(surface, self.order)


# The registry. Insertion order is not significant; ``order`` drives sequencing.
_STEP_SPECS: list[StepSpec] = [
    # Setup phase
    StepSpec(STEP_PREFLIGHT, "Checking", PHASE_SETUP, 10, _ALL_SURFACES),
    StepSpec(STEP_IMAGE_CHECK, "Inspecting", PHASE_SETUP, 20, _DOCKER_SURFACES),
    StepSpec(STEP_PULL, "Pulling", PHASE_SETUP, 30, _DOCKER_SURFACES),
    StepSpec(STEP_CONTAINER_START, "Starting", PHASE_SETUP, 50, _DOCKER_SURFACES),
    StepSpec(STEP_CONTAINER_PREFLIGHT, "Checking", PHASE_SETUP, 60, _ALL_SURFACES),
    # Measurement phase
    StepSpec(
        STEP_BASELINE,
        "Measuring",
        PHASE_MEASUREMENT,
        65,
        _ALL_SURFACES,
        order_overrides={SURFACE_CONTAINER_HOST_BASELINE: 40},
    ),
    StepSpec(STEP_MODEL, "Loading", PHASE_MEASUREMENT, 70, _ALL_SURFACES),
    StepSpec(STEP_PROMPTS, "Loading", PHASE_MEASUREMENT, 80, _ALL_SURFACES),
    # Server-mode setup: launch the engine server and drive it to readiness.
    StepSpec(STEP_SERVER_LAUNCH, "Launching", PHASE_SETUP, 45, frozenset({SURFACE_SERVER})),
    StepSpec(STEP_SERVER_READY, "Awaiting", PHASE_SETUP, 55, frozenset({SURFACE_SERVER})),
    # Warmup / measure / save are shared by offline and server surfaces.
    StepSpec(STEP_WARMUP, "Warming up", PHASE_MEASUREMENT, 90, _MEASUREMENT_SHARED),
    StepSpec(STEP_THERMAL_FLOOR, "Waiting", PHASE_MEASUREMENT, 100, _ALL_SURFACES),
    StepSpec(STEP_ENERGY_SELECT, "Selecting", PHASE_MEASUREMENT, 110, _ALL_SURFACES),
    StepSpec(STEP_MEASURE, "Measuring", PHASE_MEASUREMENT, 120, _MEASUREMENT_SHARED),
    StepSpec(STEP_FLOPS, "Estimating", PHASE_MEASUREMENT, 130, _ALL_SURFACES),
    StepSpec(STEP_SAVE, "Saving", PHASE_MEASUREMENT, 140, _MEASUREMENT_SHARED),
]

# Derived vocabulary maps. Mutated in place by ``register_step`` so importers
# that bound these dicts by name keep seeing later additions. Unknown steps
# default to PHASE_MEASUREMENT at the consumer lookup, not here.
STEP_LABELS: dict[str, str] = {spec.id: spec.label for spec in _STEP_SPECS}
STEP_PHASES: dict[str, str] = {spec.id: spec.phase for spec in _STEP_SPECS}


def register_step(spec: StepSpec) -> None:
    """Register an additional progress step in the vocabulary.

    Future measurement modes (server mode's server-start / health / ramp
    phases, for example) add their steps by registering specs here. Every
    consumer - labels, phase mapping, and the ordered step lists produced by
    ``steps_for_surface`` / ``docker_steps`` - derives from the registry, so no
    renderer or dispatcher changes are needed.
    """
    _STEP_SPECS.append(spec)
    STEP_LABELS[spec.id] = spec.label
    STEP_PHASES[spec.id] = spec.phase


def steps_for_surface(surface: str, *, exclude: frozenset[str] = frozenset()) -> list[str]:
    """Ordered step ids for ``surface``, sorted least-to-greatest by ``order``.

    Args:
        surface: One of the ``SURFACE_*`` values.
        exclude: Step ids to drop (e.g. image-prep steps already handled at
            study level).
    """
    specs = [s for s in _STEP_SPECS if surface in s.surfaces and s.id not in exclude]
    specs.sort(key=lambda s: s.order_in(surface))
    return [s.id for s in specs]


# Image-prep steps are omitted per-experiment when the study preflight already
# verified and pulled the engine images.
_IMAGE_PREP_STEPS = frozenset({STEP_IMAGE_CHECK, STEP_PULL})


def docker_steps(*, images_prepared: bool, host_baseline: bool) -> list[str]:
    """Assemble the Docker-path step list for the given run mode.

    Args:
        images_prepared: True when the study preflight has already verified
            and pulled engine images. Omits ``image_check`` / ``pull`` from
            the list because they were handled at study level.
        host_baseline: True when the host runner measures baseline *before*
            dispatching the experiment container - i.e. ``cached`` or
            ``validated`` strategies that fire a short-lived baseline
            container (or host-side measurement for local runners). False
            for ``fresh``, where the harness measures baseline inside the
            experiment container after ``container_preflight``.

    The two modes differ in exactly one position: where ``STEP_BASELINE`` sits
    relative to ``STEP_CONTAINER_START`` / ``STEP_CONTAINER_PREFLIGHT`` - encoded
    by the ``docker_host_baseline`` surface's baseline ``order`` override. The
    measurement-phase tail is identical in both.
    """
    surface = SURFACE_CONTAINER_HOST_BASELINE if host_baseline else SURFACE_CONTAINER
    exclude = _IMAGE_PREP_STEPS if images_prepared else frozenset()
    return steps_for_surface(surface, exclude=exclude)


def server_steps() -> list[str]:
    """Ordered step list for a server-mode (online-serving) experiment.

    Launch -> await readiness -> warm up -> measure windows -> save. Derived from
    the ``SURFACE_SERVER`` membership like every other surface, so adding a server
    step is one registry entry (no consumer edits)."""
    return steps_for_surface(SURFACE_SERVER)


# Process path: no container steps, direct host harness measurement.
STEPS_LOCAL: list[str] = steps_for_surface(SURFACE_PROCESS)


# -------------------------------------------------------------------------
# Container-boundary step aliases.
#
# The in-container harness emits ``STEP_PREFLIGHT`` for its own preflight; the
# host renders that as ``STEP_CONTAINER_PREFLIGHT`` so it does not collide with
# the host-side preflight step. docker_runner resolves inbound container step
# ids through this map, keeping the relationship in the registry rather than
# hard-coded at the dispatch boundary.
# -------------------------------------------------------------------------
CONTAINER_STEP_ALIASES: dict[str, str] = {STEP_PREFLIGHT: STEP_CONTAINER_PREFLIGHT}


def resolve_container_step(step: str) -> str:
    """Map an in-container step id to its host-side registry id."""
    return CONTAINER_STEP_ALIASES.get(step, step)
