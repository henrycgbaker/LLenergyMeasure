"""Pre-flight validation for study configurations.

Runs before any Docker dispatch or experiment execution. Handles multi-engine
study validation and Docker pre-flight checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llenergymeasure.config.models import StudyConfig
from llenergymeasure.config.ssot import (
    ENGINE_PACKAGES,
    RUNNER_CONTAINER,
    RUNNER_PROCESS,
    SOURCE_MULTI_ENGINE_ELEVATION,
    Engine,
)
from llenergymeasure.utils.exceptions import PreFlightError

if TYPE_CHECKING:
    from llenergymeasure.config.runner_spec import RunnerSpec

logger = logging.getLogger(__name__)


def run_study_preflight(
    study: StudyConfig,
    skip_preflight: bool = False,
) -> tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]]:
    """Pre-flight checks for a study configuration.

    Single-engine studies pass through - per-experiment pre-flight runs later
    in the subprocess.

    Which config layer pinned each engine's runner and image was decided when the
    study was resolved; this reads those pins off the study rather than consulting
    the user config or the environment again.

    Multi-engine studies resolve Docker elevation by precedence: an engine the
    user explicitly pinned (env var / study YAML / user config) keeps its
    runner, while engines whose runner resolved from auto-detection or the
    built-in default are elevated to Docker for cross-engine isolation. Every
    experiment runs in its own subprocess regardless of runner, so isolation
    always holds; elevation guards *environment feasibility* (an engine's
    dependency closure being installable on the host), not isolation. Engines
    pinned to local are therefore checked for host importability, and Docker is
    only required when an auto-resolved engine actually needs elevating.

    When any experiment in the study will use a Docker runner, runs Docker
    pre-flight checks (GPU visibility, CUDA/driver compat) unless skipped.

    After runner mode resolution, resolves Docker images for all Docker engines
    on the orthogonal image axis: the study's resolved image pin, the
    ``container:<image>`` runner shorthand, then the smart default.

    Args:
        study: Resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks. The effective skip value
            is ``skip_preflight OR study.study_execution.skip_preflight`` - CLI flag
            takes priority, then YAML config.

    Returns:
        Tuple of (runner_specs, system_overrides):
        - runner_specs: Resolved runner specs dict (engine -> RunnerSpec).
          Docker runner specs have ``image`` and ``image_source`` populated.
        - system_overrides: Dict of overrides applied during preflight, keyed
          by override target (e.g. ``"runner.transformers"``). Each value has
          ``declared``, ``effective``, and ``reason`` keys.

    Raises:
        PreFlightError: Multi-engine study pins an engine to process that is not
            importable on the host; an auto-resolved engine needs container
            elevation but Docker is unavailable on the host; or llem is running
            inside a container without a Docker socket, so it cannot elevate to
            sibling containers (docker-in-docker is not supported).
        DockerPreFlightError: Docker pre-flight check failed (inherits PreFlightError).
    """
    from llenergymeasure.config.runner_spec import RunnerSpec, pins_from_resolved
    from llenergymeasure.infra.runner_resolution import resolve_study_runners

    engines = {exp.engine for exp in study.experiments}
    is_multi_engine = len(engines) > 1
    system_overrides: dict[str, dict[str, str]] = {}

    runner_pins = pins_from_resolved(study.runners, study.settings_provenance, section="runners")
    image_pins = pins_from_resolved(study.images, study.settings_provenance, section="images")

    # Turn each engine's pin (or the absence of one) into a runner. Precedence-based
    # elevation (below) reads each spec.source to tell explicit user pins
    # (env / yaml / user_config) from auto-resolved runners
    # (auto_detected / default).
    runner_specs = resolve_study_runners(list(engines), runner_pins)

    # Multi-engine precedence: explicit pins win; auto-resolved runners elevate
    # to Docker for cross-engine isolation.
    if is_multi_engine:
        system_overrides = _apply_multi_engine_precedence(runner_specs)

    # Resolve Docker images for all Docker engines (orthogonal to runner mode).
    from llenergymeasure.infra.image_registry import resolve_image

    for engine_name, spec in runner_specs.items():
        if spec.mode == RUNNER_CONTAINER:
            image, image_source = resolve_image(
                engine_name,
                spec_image=spec.image,
                pin=image_pins.get(engine_name),
            )
            runner_specs[engine_name] = RunnerSpec(
                mode=spec.mode,
                image=image,
                source=spec.source,
                image_source=image_source,
            )

    # Docker pre-flight: run once if any engine resolves to a Docker runner.
    # Effective skip = CLI flag (skip_preflight param) OR YAML config value.
    effective_skip = skip_preflight or study.study_execution.skip_preflight
    if any(spec.mode == RUNNER_CONTAINER for spec in runner_specs.values()):
        from llenergymeasure.infra.docker_preflight import run_docker_preflight

        run_docker_preflight(skip=effective_skip)

    return runner_specs, system_overrides


def _apply_multi_engine_precedence(
    runner_specs: dict[str, RunnerSpec],
) -> dict[str, dict[str, str]]:
    """Apply precedence-based Docker elevation for a multi-engine study.

    Explicit runner pins (env var / study YAML / user config) win: an engine the
    user pinned keeps its runner. Runner choice is machine-binding and recorded
    per result, so honouring an explicit local pin is a reproducibility
    contract - the user is asserting this host can run that engine. Only engines
    whose runner resolved from auto-detection or the built-in default are
    elevated to Docker (tagged ``SOURCE_MULTI_ENGINE_ELEVATION``).

    Mutates *runner_specs* in place - elevated engines get a Docker spec - and
    returns the system-override records for the elevated engines.

    Raises:
        PreFlightError: an engine explicitly pinned to process is not importable
            in the host environment; container elevation is required for an
            auto-resolved engine but Docker is unavailable on the host; or llem is
            running inside a container without a Docker socket, so it cannot elevate
            to sibling containers (docker-in-docker is not supported).
    """
    from llenergymeasure.config.runner_spec import RunnerSpec
    from llenergymeasure.infra.runner_resolution import (
        is_container_socket_available,
        is_docker_available,
        is_running_in_container,
    )

    explicit_local: list[str] = []
    kept_explicit: list[str] = []
    elevated: list[str] = []
    for engine_name, spec in runner_specs.items():
        if spec.is_explicit:
            # User pinned this engine - honour it (whether process or container).
            kept_explicit.append(engine_name)
            # A process pin still needs a host importability check before dispatch.
            if spec.mode == RUNNER_PROCESS:
                explicit_local.append(engine_name)
        else:
            # auto_detected / default -> elevate to a container for isolation.
            elevated.append(engine_name)

    # Import pre-flight: every engine pinned to process must be importable here.
    # Reuse the single-engine local-preflight availability check (find_spec on
    # the engine package) rather than duplicating it - it is a public helper on
    # harness/preflight, which owns the fact.
    import_failures: list[str] = []
    if explicit_local:
        from llenergymeasure.harness.preflight import check_engine_installed
    for engine_name in sorted(explicit_local):
        if not check_engine_installed(engine_name):
            package = ENGINE_PACKAGES.get(Engine(engine_name), engine_name)
            import_failures.append(
                f"{engine_name}: package '{package}' is not importable in the host "
                f"environment. Either install the engine extra "
                f"(pip install 'llenergymeasure[{engine_name}]'), or drop the explicit "
                f"'{engine_name}: {RUNNER_PROCESS}' runner pin so multi-engine elevation "
                f"runs it in a container."
            )
    if import_failures:
        lines = "\n".join(f"  \u2717 {f}" for f in import_failures)
        raise PreFlightError(
            f"Multi-engine study pins engine(s) to a {RUNNER_PROCESS} runner, but the host "
            f"environment cannot import them:\n{lines}"
        )

    # Container elevation is only required when an auto-resolved engine needs it.
    # All-explicit studies (process and/or container pins) do not gate here; explicit
    # container pins are validated by the Docker pre-flight downstream.
    #
    # The elevation path must be viable and must be container-self-aware: when llem
    # itself runs inside a container, the viable path is a mounted Docker socket
    # (docker-outside-of-docker siblings via the host daemon), never docker-in-docker.
    # A socketless container therefore fails with an actionable error rather than
    # silently attempting DinD. On the host, the gate is the usual Docker + NVIDIA
    # Container Toolkit availability check (unchanged).
    if elevated:
        if is_running_in_container():
            if not is_container_socket_available():
                raise PreFlightError(
                    "Multi-engine study needs container isolation to run "
                    f"{', '.join(sorted(elevated))} (auto-resolved, not explicitly "
                    "pinned), but llenergymeasure is running inside a container without "
                    "a Docker socket, so it cannot start sibling containers "
                    "(docker-in-docker is not supported). Mount the host Docker socket "
                    "(-v /var/run/docker.sock:/var/run/docker.sock), pin those engines "
                    f"to an explicit '{RUNNER_PROCESS}' runner, or use a single engine."
                )
        elif not is_docker_available():
            raise PreFlightError(
                "Multi-engine study needs Docker isolation to run "
                f"{', '.join(sorted(elevated))} (auto-resolved, not explicitly pinned), "
                "but Docker is not available. Install Docker + NVIDIA Container Toolkit, "
                "pin those engines to an explicit runner, or use a single engine."
            )

    system_overrides: dict[str, dict[str, str]] = {}
    for engine_name in elevated:
        spec = runner_specs[engine_name]
        system_overrides[f"runner.{engine_name}"] = {
            "declared": spec.mode,
            "effective": RUNNER_CONTAINER,
            "reason": "auto-elevated (multi-engine study)",
        }
        runner_specs[engine_name] = RunnerSpec(
            mode=RUNNER_CONTAINER, image=spec.image, source=SOURCE_MULTI_ENGINE_ELEVATION
        )

    logger.info(
        "Multi-engine study: elevated %s to Docker for isolation; kept explicit "
        "runner pins for %s.",
        ", ".join(sorted(elevated)) or "(none)",
        ", ".join(sorted(kept_explicit)) or "(none)",
    )

    # All engines explicitly pinned to process: allowed, but the isolation the
    # Docker path would give is now the user's responsibility. Warn once (the
    # "mixed runners" warning downstream only fires when modes actually differ,
    # so it never covers this all-process multi-engine state).
    if all(spec.mode == RUNNER_PROCESS for spec in runner_specs.values()):
        logger.warning(
            "Multi-engine study running every engine as a host process: ensure the host "
            "environment genuinely satisfies every engine; a container per engine "
            "remains the recommended isolation."
        )

    return system_overrides
