"""Unit tests for domain/progress.py - ProgressCallback protocol and step vocabulary."""

from __future__ import annotations

from llenergymeasure.domain.progress import (
    STEP_BASELINE,
    STEP_CONTAINER_PREFLIGHT,
    STEP_CONTAINER_START,
    STEP_IMAGE_CHECK,
    STEP_LABELS,
    STEP_MEASURE,
    STEP_MODEL,
    STEP_PHASES,
    STEP_PREFLIGHT,
    STEP_PULL,
    STEP_SAVE,
    STEP_WARMUP,
    ProgressCallback,
    docker_steps,
)


def test_protocol_is_runtime_checkable():
    """ProgressCallback can be used with isinstance() at runtime."""

    class Impl:
        def on_step_start(self, step: str, description: str, detail: str = "") -> None:
            pass

        def on_step_update(self, step: str, detail: str) -> None:
            pass

        def on_step_done(self, step: str, elapsed_sec: float) -> None:
            pass

        def on_step_skip(self, step: str, reason: str = "") -> None:
            pass

        def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
            pass

        def on_substep_start(self, step: str, text: str) -> None:
            pass

        def on_substep_done(
            self, step: str, text: str | None = None, elapsed_sec: float | None = None
        ) -> None:
            pass

    assert isinstance(Impl(), ProgressCallback)


def test_non_conforming_class_fails_protocol():
    """A class missing methods does not satisfy the protocol."""

    class Incomplete:
        def on_step_start(self, step: str, description: str, detail: str = "") -> None:
            pass

        # Missing on_step_update and on_step_done

    assert not isinstance(Incomplete(), ProgressCallback)


def test_step_labels_cover_all_constants():
    """Every step constant has a corresponding label."""
    for step in [
        STEP_PREFLIGHT,
        STEP_IMAGE_CHECK,
        STEP_PULL,
        STEP_CONTAINER_START,
        STEP_CONTAINER_PREFLIGHT,
        STEP_BASELINE,
        STEP_MODEL,
        STEP_WARMUP,
        STEP_MEASURE,
        STEP_SAVE,
    ]:
        assert step in STEP_LABELS, f"Missing label for step: {step}"


def test_step_labels_are_verbs():
    """Labels should be action-oriented (basic smoke test)."""
    assert STEP_LABELS[STEP_PREFLIGHT] == "Checking"
    assert STEP_LABELS[STEP_MODEL] == "Loading"
    assert STEP_LABELS[STEP_MEASURE] == "Measuring"


def test_step_phases_cover_all_constants():
    """Every step constant has a phase assignment."""
    for step in [
        STEP_PREFLIGHT,
        STEP_IMAGE_CHECK,
        STEP_PULL,
        STEP_CONTAINER_START,
        STEP_CONTAINER_PREFLIGHT,
        STEP_BASELINE,
        STEP_MODEL,
        STEP_WARMUP,
        STEP_MEASURE,
        STEP_SAVE,
    ]:
        assert step in STEP_PHASES, f"Missing phase for step: {step}"


# -------------------------------------------------------------------------
# docker_steps() - assembles the Docker step list with strategy-dependent
# placement of STEP_BASELINE. The renderer iterates in registered order, so
# a step list out of sync with actual event order is exactly what causes the
# "janky" progress display when using cached/validated baselines.
# -------------------------------------------------------------------------


def test_docker_steps_full_with_host_baseline():
    """Baseline container runs BEFORE the experiment container.

    cached/validated: host dispatches a short-lived baseline container first
    so STEP_BASELINE must come before STEP_CONTAINER_START in the step list.
    """
    steps = docker_steps(images_prepared=False, host_baseline=True)
    assert steps == [
        STEP_PREFLIGHT,
        STEP_IMAGE_CHECK,
        STEP_PULL,
        STEP_BASELINE,
        STEP_CONTAINER_START,
        STEP_CONTAINER_PREFLIGHT,
        STEP_MODEL,
        "prompts",
        STEP_WARMUP,
        "thermal_floor",
        "energy_select",
        STEP_MEASURE,
        "flops",
        STEP_SAVE,
    ]


def test_docker_steps_full_with_fresh_baseline():
    """fresh strategy: harness measures baseline INSIDE the experiment container.

    STEP_BASELINE must sit after container_preflight (which is the last
    in-container setup event before the harness starts measuring).
    """
    steps = docker_steps(images_prepared=False, host_baseline=False)
    assert steps.index(STEP_BASELINE) > steps.index(STEP_CONTAINER_PREFLIGHT)
    assert steps[:3] == [STEP_PREFLIGHT, STEP_IMAGE_CHECK, STEP_PULL]
    # Baseline sits immediately after container_preflight, before model.
    baseline_idx = steps.index(STEP_BASELINE)
    assert steps[baseline_idx - 1] == STEP_CONTAINER_PREFLIGHT
    assert steps[baseline_idx + 1] == STEP_MODEL


def test_docker_steps_images_prepared_drops_image_check_and_pull():
    """Study-level image prep means per-experiment image_check/pull are redundant."""
    steps = docker_steps(images_prepared=True, host_baseline=True)
    assert STEP_IMAGE_CHECK not in steps
    assert STEP_PULL not in steps
    # host_baseline still positions baseline before container_start
    assert steps.index(STEP_BASELINE) < steps.index(STEP_CONTAINER_START)


def test_docker_steps_images_prepared_fresh():
    """Per-study image prep combined with fresh strategy."""
    steps = docker_steps(images_prepared=True, host_baseline=False)
    assert STEP_IMAGE_CHECK not in steps
    assert STEP_PULL not in steps
    assert steps.index(STEP_BASELINE) > steps.index(STEP_CONTAINER_PREFLIGHT)


def test_docker_steps_measurement_tail_is_identical_across_variants():
    """The measurement-phase tail (model…save) is the same in every mode.

    This locks in the single-source-of-truth property of the constructor -
    changing the tail in one variant would drift the others automatically.
    """
    tails = {}
    for images_prepared in (True, False):
        for host_baseline in (True, False):
            steps = docker_steps(images_prepared=images_prepared, host_baseline=host_baseline)
            # Everything from STEP_MODEL onwards is the measurement tail.
            tail = steps[steps.index(STEP_MODEL) :]
            tails[(images_prepared, host_baseline)] = tuple(tail)
    assert len(set(tails.values())) == 1, f"Measurement tail drifted across variants: {tails}"


# -------------------------------------------------------------------------
# Step registry extensibility (S14). The vocabulary is a registry, not a
# closed set: a new step is one register_step() call and it flows through
# labels, phase mapping, and the ordered step lists with no consumer edits.
# This is the seam server mode (v0.8.0) uses to add server-start/health/ramp.
# -------------------------------------------------------------------------


def test_register_step_flows_through_labels_phases_and_ordering():
    """A newly registered step flows through the whole registry with no
    renderer or list-builder changes."""
    import llenergymeasure.domain.progress as progress

    saved_specs = list(progress._STEP_SPECS)
    saved_labels = dict(progress.STEP_LABELS)
    saved_phases = dict(progress.STEP_PHASES)
    try:
        # A hypothetical server-mode step: it joins the existing local surface
        # (positioned between preflight=10 and container_preflight=60) and its
        # own brand-new "server" surface - the pattern server mode will use.
        progress.register_step(
            progress.StepSpec(
                id="server_health",
                label="Polling",
                phase="Server",
                order=45,
                surfaces=frozenset({progress.SURFACE_LOCAL, "server"}),
            )
        )

        # Label mapping flows.
        assert progress.STEP_LABELS["server_health"] == "Polling"
        # Phase mapping flows - a brand-new phase name is carried verbatim.
        assert progress.STEP_PHASES["server_health"] == "Server"

        # Ordering flows: the step lands at its declared position in the local
        # surface, between preflight and container_preflight.
        local = progress.steps_for_surface(progress.SURFACE_LOCAL)
        assert "server_health" in local
        assert local.index(STEP_PREFLIGHT) < local.index("server_health")
        assert local.index("server_health") < local.index(STEP_CONTAINER_PREFLIGHT)

        # Ordering flows in a brand-new surface too: no builder changes needed.
        assert progress.steps_for_surface("server") == ["server_health"]

        # A step that opts out of the docker surfaces never appears there.
        assert "server_health" not in docker_steps(images_prepared=False, host_baseline=True)
    finally:
        progress._STEP_SPECS[:] = saved_specs
        progress.STEP_LABELS.clear()
        progress.STEP_LABELS.update(saved_labels)
        progress.STEP_PHASES.clear()
        progress.STEP_PHASES.update(saved_phases)
