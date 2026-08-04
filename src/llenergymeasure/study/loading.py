"""Study-layer finalisation of a parsed study config.

The config layer (:mod:`llenergymeasure.config.loader`) does pure
parse + sweep-expansion and returns a
:class:`~llenergymeasure.config.loader.LoadedStudyRaw`. The dedup /
study_design_hash / cycle-ordering / equivalence-group steps need the
study-layer library-resolution mechanism, so they live here - keeping the
config layer free of any upward import into ``study``.

:func:`finalise_study` is the single composition point. The public entry that
parses *and* finalises in one call is ``llenergymeasure.api.load_study``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal

from llenergymeasure.config.grid import (
    ExperimentOrder,
    apply_cycles,
    compute_study_design_hash,
)
from llenergymeasure.config.loader import LoadedStudyRaw
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.study.library_resolution import resolve_library_effective

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserConfig

__all__ = ["finalise_study"]

logger = logging.getLogger(__name__)


def finalise_study(raw: LoadedStudyRaw, *, user_config: UserConfig | None = None) -> StudyConfig:
    """Apply library-resolution dedup, design hash, and cycle ordering.

    Consumes the :class:`~llenergymeasure.config.loader.LoadedStudyRaw` produced
    by :func:`llenergymeasure.config.loader.load_study_config` and produces the
    resolved :class:`~llenergymeasure.config.models.StudyConfig` that the runner
    iterates over.

    Steps:
      0. Overlay the tool-wide user-config server warmup onto each declared server
         config (R7W), BEFORE dedup, so the resolved-config hash binds on the
         realised warmup protocol.
      1. Library-resolution mechanism + resolved-config-hash dedup of the
         declared configs (see sweep-dedup.md §2).
      2. compute_study_design_hash() over the post-dedup configs - the hash
         identifies the *unique* measurement set, not duplicate declarations.
      3. apply_cycles() to produce the execution sequence.
      4. Serialise pre-run equivalence groups for the sidecar writer.

    Args:
        raw: Parsed + sweep-expanded study material.
        user_config: Tool-wide user config whose ``server.warmup`` defaults are
            overlaid onto each declared server config. ``None`` (the default)
            applies NO overlay - callers hand in a config only at the production
            edge (``api.load_study``), keeping the finalise step hermetic for the
            unit tests that construct one directly.

    Returns:
        Resolved StudyConfig with ordered experiments, study_design_hash, dedup
        mode, and pre-run equivalence groups.
    """
    # R7W: overlay the tool-wide user-config server warmup onto each declared
    # server config BEFORE dedup, so the resolved-config hash - and hence dedup -
    # binds on the realised warmup protocol. Declared hashes are untouched (the
    # overlay is side-channel state, never a field), and the overlay rides the
    # sweep-dedup deep copies through to the runner. No-op for offline configs and
    # when the user config carries no warmup layer. The DECLARED-family drift checks
    # (validate_config_drift on study_design_hash, the skip-set on config_hash) are
    # blind to a user-config warmup change between runs; the RESOLVED-family guard
    # (resume.validate_resolved_config_drift, keyed on the manifest resolved_config_hash)
    # closes that gap, so a resumed study now rejects a changed resolved protocol
    # rather than silently skipping a differently-resolved cell.
    if user_config is not None:
        from llenergymeasure.config.precedence import apply_server_warmup_overlay

        for exp in raw.valid_experiments:
            apply_server_warmup_overlay(exp, user_config)

    execution = raw.execution

    # Apply library-resolution mechanism + resolved-config-hash dedup to the declared configs
    # before running cycles. See sweep-dedup.md §2 - this collapses measurement-equivalent
    # configs so a 6-config sweep with dormant sampling fields becomes 4 unique runs.
    dedup = resolve_library_effective(
        raw.valid_experiments,
        deduplicate=execution.deduplicate_equivalent,
    )

    run_experiments = dedup.canonical_configs
    dedup_mode: Literal["resolved", "off"] = (
        "resolved" if execution.deduplicate_equivalent else "off"
    )

    # Compute study_design_hash over the post-dedup configs - the hash
    # identifies the *unique* measurement set, not duplicate declarations.
    study_hash = compute_study_design_hash(run_experiments)

    _maybe_hint_sequential_server_singletons(
        run_experiments,
        experiment_order=execution.experiment_order,
        n_cycles=execution.n_cycles,
    )

    # Apply cycle ordering to produce execution sequence
    ordered = apply_cycles(
        run_experiments,
        n_cycles=execution.n_cycles,
        experiment_order=ExperimentOrder(execution.experiment_order),
        study_design_hash=study_hash,
        shuffle_seed=execution.shuffle_seed,
    )

    # Serialise pre-run equivalence groups for the sidecar writer. The runner's deserialiser
    # (StudyRunner._write_equivalence_groups_sidecar) reads member_experiment_ids /
    # representative_experiment_id, so map the dedup group's member indices back to their
    # declared-config-hash experiment ids and emit those keys. (This previously hand-rolled
    # member_indices / representative_index - the wrong keys and raw ints - so every loaded
    # group came back with empty members.)
    from llenergymeasure.domain.experiment import compute_declared_config_hash

    experiment_ids = [compute_declared_config_hash(exp) for exp in raw.valid_experiments]
    pre_run_groups: list[dict[str, Any]] = [
        {
            "resolved_config_hash": g.resolved_config_hash,
            "canonical_config_excerpt": g.canonical_excerpt,
            "member_experiment_ids": [experiment_ids[i] for i in g.member_indices],
            "member_count": g.member_count,
            "representative_experiment_id": experiment_ids[g.representative_index],
            "would_dedup": g.member_count > 1,
            "deduplicated": dedup.deduplicated and g.member_count > 1,
        }
        for g in dedup.groups
    ]

    return StudyConfig(
        experiments=ordered,
        study_name=raw.study_name,
        output=raw.output,
        study_execution=execution,
        runners=raw.runners,
        images=raw.images,
        study_design_hash=study_hash,
        skipped_configs=[s.to_dict() for s in raw.skipped],
        dedup_mode=dedup_mode,
        pre_run_equivalence_groups=pre_run_groups,
        declared_resolved_config_hashes=list(dedup.declared_resolved_hashes),
        dormant_observations=[asdict(obs) for obs in dedup.dormant_observations],
    )


def _maybe_hint_sequential_server_singletons(
    run_experiments: list[ExperimentConfig],
    *,
    experiment_order: str,
    n_cycles: int,
) -> None:
    """Emit a did-you-know when a foldable server sweep runs under sequential order.

    Under ``sequential`` order with ``n_cycles > 1`` each grid point's cycles are
    adjacent, so a rate/slo sweep's cells are never both consecutive AND same-cycle
    and each dispatches as its own server session (one server launch per cell per
    cycle). ``interleave`` replays the base ordering per cycle pass, so each pass
    folds the sweep into a single launch. Both are correct; this is INFO, not a
    warning.

    The foldability test reuses the session-grouping machinery on the pre-cycle
    base list: a unit of two or more cells is (by construction) a multi-cell server
    group that would fold under interleave. No key-stripping logic is duplicated.
    """
    if experiment_order != ExperimentOrder.SEQUENTIAL or n_cycles <= 1:
        return

    from llenergymeasure.study.server_session import partition_server_groups

    units = partition_server_groups(run_experiments)
    if any(len(unit) >= 2 for unit in units):
        logger.info(
            "sequential order launches one server per cell per cycle; set "
            "experiment_order: interleave to reuse one server launch per sweep pass."
        )
