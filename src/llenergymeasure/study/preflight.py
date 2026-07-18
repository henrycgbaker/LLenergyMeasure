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
    EXPLICIT_RUNNER_SOURCES,
    RUNNER_DOCKER,
    RUNNER_LOCAL,
    SOURCE_MULTI_ENGINE_ELEVATION,
    Engine,
)
from llenergymeasure.utils.exceptions import PreFlightError

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserRunnersConfig
    from llenergymeasure.infra.runner_resolution import RunnerSpec

logger = logging.getLogger(__name__)


def run_study_preflight(
    study: StudyConfig,
    skip_preflight: bool = False,
    yaml_runners: dict[str, str] | None = None,
    user_config: UserRunnersConfig | None = None,
    yaml_images: dict[str, str] | None = None,
    user_config_images: dict[str, str] | None = None,
) -> tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]]:
    """Pre-flight checks for a study configuration.

    Single-engine studies pass through - per-experiment pre-flight runs later
    in the subprocess.

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

    After runner mode resolution, resolves Docker images for all Docker
    engines using the orthogonal image precedence chain (env > YAML > user
    config > local build > registry).

    Args:
        study: Resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks. The effective skip value
            is ``skip_preflight OR study.study_execution.skip_preflight`` - CLI flag
            takes priority, then YAML config.
        yaml_runners: Runner config from the study YAML ``runners:`` section.
            Forwarded to ``resolve_study_runners()`` so pre-flight uses the same
            runner resolution as the actual dispatch path.
        user_config: Loaded UserRunnersConfig. Forwarded to
            ``resolve_study_runners()`` to match actual dispatch precedence.
        yaml_images: Image overrides from the study YAML ``images:`` section.
        user_config_images: Image overrides from the user config ``images:``
            section.

    Returns:
        Tuple of (runner_specs, system_overrides):
        - runner_specs: Resolved runner specs dict (engine -> RunnerSpec).
          Docker runner specs have ``image`` and ``image_source`` populated.
        - system_overrides: Dict of overrides applied during preflight, keyed
          by override target (e.g. ``"runner.transformers"``). Each value has
          ``declared``, ``effective``, and ``reason`` keys.

    Raises:
        PreFlightError: Multi-engine study pins an engine to local that is not
            importable on the host, or an auto-resolved engine needs Docker
            elevation but Docker is unavailable.
        DockerPreFlightError: Docker pre-flight check failed (inherits PreFlightError).
    """
    from llenergymeasure.infra.runner_resolution import RunnerSpec, resolve_study_runners

    engines = {exp.engine for exp in study.experiments}
    is_multi_engine = len(engines) > 1
    system_overrides: dict[str, dict[str, str]] = {}

    # Resolve runners via the normal precedence chain first. Precedence-based
    # elevation (below) reads each spec.source to tell explicit user pins
    # (env / yaml / user_config) from auto-resolved runners
    # (auto_detected / default).
    runner_specs = resolve_study_runners(
        list(engines), yaml_runners=yaml_runners, user_config=user_config
    )

    # Multi-engine precedence: explicit pins win; auto-resolved runners elevate
    # to Docker for cross-engine isolation.
    if is_multi_engine:
        system_overrides = _apply_multi_engine_precedence(runner_specs)

    # Resolve Docker images for all Docker engines (orthogonal to runner mode).
    from llenergymeasure.infra.image_registry import resolve_image

    for engine_name, spec in runner_specs.items():
        if spec.mode == RUNNER_DOCKER:
            image, image_source = resolve_image(
                engine_name,
                spec_image=spec.image,
                yaml_images=yaml_images,
                user_config_images=user_config_images,
            )
            runner_specs[engine_name] = RunnerSpec(
                mode=spec.mode,
                image=image,
                source=spec.source,
                image_source=image_source,
                extra_mounts=spec.extra_mounts,
            )

    # Docker pre-flight: run once if any engine resolves to a Docker runner.
    # Effective skip = CLI flag (skip_preflight param) OR YAML config value.
    effective_skip = skip_preflight or study.study_execution.skip_preflight
    if any(spec.mode == RUNNER_DOCKER for spec in runner_specs.values()):
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
        PreFlightError: an engine explicitly pinned to local is not importable
            in the host environment, or Docker is required to elevate an
            auto-resolved engine but is unavailable.
    """
    from llenergymeasure.infra.runner_resolution import RunnerSpec, is_docker_available

    explicit_local: list[str] = []
    kept_explicit: list[str] = []
    elevated: list[str] = []
    for engine_name, spec in runner_specs.items():
        if spec.source in EXPLICIT_RUNNER_SOURCES:
            # User pinned this engine - honour it (whether local or docker).
            kept_explicit.append(engine_name)
            # A local pin still needs a host importability check before dispatch.
            if spec.mode == RUNNER_LOCAL:
                explicit_local.append(engine_name)
        else:
            # auto_detected / default -> elevate to Docker for isolation.
            elevated.append(engine_name)

    # Import pre-flight: every engine pinned to local must be importable here.
    # Reuse the single-engine local-preflight availability check (find_spec on
    # the engine package) rather than duplicating it. Cross-module private
    # import, following the documented pattern used elsewhere across the package.
    import_failures: list[str] = []
    if explicit_local:
        from llenergymeasure.harness.preflight import _check_engine_installed
    for engine_name in sorted(explicit_local):
        if not _check_engine_installed(engine_name):
            package = ENGINE_PACKAGES.get(Engine(engine_name), engine_name)
            import_failures.append(
                f"{engine_name}: package '{package}' is not importable in the host "
                f"environment. Either install the engine extra "
                f"(pip install 'llenergymeasure[{engine_name}]'), or drop the explicit "
                f"'{engine_name}: local' runner pin so multi-engine elevation runs it "
                f"in Docker."
            )
    if import_failures:
        lines = "\n".join(f"  \u2717 {f}" for f in import_failures)
        raise PreFlightError(
            "Multi-engine study pins engine(s) to a local runner, but the host "
            f"environment cannot import them:\n{lines}"
        )

    # Docker is only required when an auto-resolved engine needs elevating.
    # All-explicit studies (local and/or docker pins) do not gate on it here;
    # explicit docker pins are validated by the Docker pre-flight downstream.
    if elevated and not is_docker_available():
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
            "effective": RUNNER_DOCKER,
            "reason": "auto-elevated (multi-engine study)",
        }
        runner_specs[engine_name] = RunnerSpec(
            mode=RUNNER_DOCKER, image=spec.image, source=SOURCE_MULTI_ENGINE_ELEVATION
        )

    logger.info(
        "Multi-engine study: elevated %s to Docker for isolation; kept explicit "
        "runner pins for %s.",
        ", ".join(sorted(elevated)) or "(none)",
        ", ".join(sorted(kept_explicit)) or "(none)",
    )

    # All engines explicitly pinned to local: allowed, but the isolation the
    # Docker path would give is now the user's responsibility. Warn once (the
    # "mixed runners" warning downstream only fires when modes actually differ,
    # so it never covers this all-local multi-engine state).
    if all(spec.mode == RUNNER_LOCAL for spec in runner_specs.values()):
        logger.warning(
            "Multi-engine study running all engines locally: ensure the host "
            "environment genuinely satisfies every engine; Docker per engine "
            "remains the recommended isolation."
        )

    return system_overrides
