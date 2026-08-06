"""Resolved-config view construction from resolved ExperimentConfig (study layer).

The pure hashing primitives live in :mod:`llenergymeasure.domain.hashing`
(Layer 0).  This module contains only :func:`build_resolved_view`, which
needs ``ExperimentConfig`` and therefore belongs at Layer 4 (study).

``hash_config`` is re-exported because the resolved-view path always builds a
view and then hashes it; production callers take both from here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import engine_str
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.domain.hashing import ConfigHashView, hash_config

__all__ = [
    "build_resolved_view",
    "hash_config",
    "resolved_config_hash",
    "resolved_hashes_by_declared",
]


# ---------------------------------------------------------------------------
# Resolved-config view construction - from library-resolution mechanism output
# ---------------------------------------------------------------------------


def build_resolved_view(config: ExperimentConfig) -> ConfigHashView:
    """Project a (post-library-resolution) ``ExperimentConfig`` into a resolved-config view.

    Reads the active engine section's full post-normalisation state; the
    library-resolution mechanism has already applied dormant rules to fixpoint before this
    runs.  Callers pass the resolved config, not the declared one - resolved_config_hash is
    meaningless on a pre-resolved config.

    The generated engine sections split into ``engine_params`` /
    ``sampling_params`` sub-models; each is lifted into its own dict so the
    resolved-config / observed-config ordering separates "how the engine
    constructs" from "what it generates with". Any section-level extras merge
    into the engine-params view.

    The active engine's ``llem_execution`` block (llem-owned execution knobs the
    engine has no native API for - batch_size, torch_compile, allow_tf32,
    autocast) and the ``measurement`` block (methodology dials) also join the
    view: both drive execution or define distinct runs, so a sweep over either
    must produce distinct resolved hashes rather than collapsing under dedup.

    The active mode namespace's identity projection (``config.mode_section_identity()``
    - server: traffic-minus-slo + warmup + cooldown; offline: the warmup block when
    an offline: section is present, else ``{}``) joins the view too, so a
    traffic-rate or warmup-protocol sweep produces distinct hashes while two runs
    differing only in slo bounds collapse. These are llem-owned (no library-
    resolution pass, no engine observation), so the resolved and observed views
    project the SAME declared values at v0.7.
    """
    engine_name = engine_str(config.engine)
    section: Any = getattr(config, engine_name, None)
    dump: dict[str, Any] = section.model_dump(mode="python") if section is not None else {}
    engine_params = dump.pop("engine_params", None) or {}
    sampling = dump.pop("sampling_params", None) or {}
    # llem_execution goes into its own hash-identity slot, not the engine-params
    # view. The section dump already carries it (the transformers subclass has it
    # as a real field), so capture the pop directly rather than re-fetching and
    # re-dumping the same submodel via active_llem_execution().
    execution_dump = dump.pop("llem_execution", None) or {}

    return ConfigHashView(
        engine=engine_name,
        task=config.task.model_dump(mode="python"),
        serving_mode=config.serving_mode,
        mode_section=config.mode_section_identity(),
        observed_engine_params={**engine_params, **dump},
        observed_sampling_params=sampling,
        passthrough_kwargs=dict(config.passthrough_kwargs or {}),
        llem_execution=execution_dump,
        measurement=config.measurement.model_dump(mode="python"),
    )


def resolved_config_hash(config: ExperimentConfig) -> str:
    """Resolved-config hash of a (resolved) config: hash of its resolved-config view.

    The single home for ``hash_config(build_resolved_view(config))`` - the resolved
    hash carries the realised warmup protocol (the user-config overlay output),
    so it moves when the resolved protocol changes even though the declared hash
    does not.
    """
    return hash_config(build_resolved_view(config))


def resolved_hashes_by_declared(experiments: Sequence[ExperimentConfig]) -> dict[str, str]:
    """Map each declared config_hash to its resolved-config hash (first-wins on dupes).

    Cycles produce duplicate declared hashes; the first occurrence wins (they all
    resolve identically). Callers own the strictness: the manifest wraps this in a
    best-effort try/except, resume calls it directly and lets a failure raise.
    """
    result: dict[str, str] = {}
    for exp in experiments:
        declared = compute_declared_config_hash(exp)
        if declared not in result:
            result[declared] = resolved_config_hash(exp)
    return result
