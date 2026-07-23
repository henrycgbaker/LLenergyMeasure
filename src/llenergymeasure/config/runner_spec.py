"""RunnerSpec - the resolved runner value object.

A ``RunnerSpec`` is a plain resolved fact about *how* an engine should be
dispatched (process vs container, which image, and the precedence source that
selected it). It lives in the config/domain foundation layer, not in ``infra``,
because it is config-derived data consumed by the study layer - the resolution
*machinery* (Docker detection, env-var reading) stays in
``infra.runner_resolution``, which imports this value object.

The precedence taxonomy the spec references (the ``source`` tags, ``RunnerMode``,
and ``EXPLICIT_RUNNER_SOURCES``) is the single source of truth in
``config.ssot``; ``is_explicit`` reads it directly, so the classification lives
next to the taxonomy it describes.
"""

from __future__ import annotations

from dataclasses import dataclass

from llenergymeasure.config.ssot import EXPLICIT_RUNNER_SOURCES, RunnerMode


@dataclass
class RunnerSpec:
    """Resolved runner specification: where and how an engine should be dispatched.

    A spec is a plain resolved value, independent of any single dispatch: it
    carries no assumption that it is consumed by exactly one ``run()`` call and
    no dispatch machinery of its own. The same spec may drive one experiment or
    be reused across many.

    Attributes:
        mode:         Execution mode - "process" or "container".
        image:        Container image to use. None for process mode or when the
                      default should be resolved at dispatch time.
        source:       Which layer of the precedence chain produced this spec:
                      "env", "yaml", "user_config", "auto_detected", "default".
        image_source: Where the container image was resolved from:
                      "env", "yaml", "runner_override", "user_config",
                      "local_build", "registry", or None (process mode / unresolved).
    """

    mode: RunnerMode
    image: str | None
    source: str
    image_source: str | None = None

    @property
    def is_explicit(self) -> bool:
        """True if this runner was an explicit user pin (env var / YAML / user config).

        Explicit pins win over multi-engine Docker elevation; auto-resolved
        runners (``auto_detected`` / ``default``) do not. Classification lives
        here, next to the ``source`` taxonomy it describes.
        """
        return self.source in EXPLICIT_RUNNER_SOURCES

    def to_runner_info(self) -> dict[str, str | None]:
        """Build runner info dict for progress display callbacks."""
        return {
            "mode": self.mode,
            "source": self.source,
            "image": self.image,
            "image_source": self.image_source,
        }
