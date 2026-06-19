"""Resolved-config view construction from resolved ExperimentConfig (study layer).

The pure hashing primitives live in :mod:`llenergymeasure.domain.hashing`
(Layer 0).  This module contains only :func:`build_resolved_view`, which
needs ``ExperimentConfig`` and therefore belongs at Layer 4 (study).

``hash_config`` is re-exported because the resolved-view path always builds a
view and then hashes it; production callers take both from here.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import engine_str
from llenergymeasure.domain.hashing import ConfigHashView, hash_config

__all__ = [
    "build_resolved_view",
    "hash_config",
]


# ---------------------------------------------------------------------------
# Resolved-config view construction - from library-resolution mechanism output
# ---------------------------------------------------------------------------


def build_resolved_view(config: ExperimentConfig) -> ConfigHashView:
    """Project a (post-library-resolution) ``ExperimentConfig`` into a resolved-config view.

    Reads the active engine section's full post-normalisation state; the
    library-resolution mechanism has already applied dormant invariants to fixpoint before this
    runs.  Callers pass the resolved config, not the declared one - resolved_config_hash is
    meaningless on a pre-resolved config.

    Engine-specific sub-models carry a ``sampling`` attribute; it is lifted
    into its own dict so the resolved-config / observed-config ordering separates
    "how the engine constructs" from "what it generates with".
    """
    engine_name = engine_str(config.engine)
    section: Any = getattr(config, engine_name, None)
    dump: dict[str, Any] = section.model_dump(mode="python") if section is not None else {}
    sampling = dump.pop("sampling", None) or {}

    return ConfigHashView(
        engine=engine_name,
        task=config.task.model_dump(mode="python"),
        observed_engine_params=dump,
        observed_sampling_params=sampling,
        passthrough_kwargs=dict(config.passthrough_kwargs or {}),
    )
