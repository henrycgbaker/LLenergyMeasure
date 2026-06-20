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

from typing import Any, Literal

from llenergymeasure.config.grid import (
    ExperimentOrder,
    apply_cycles,
    compute_study_design_hash,
)
from llenergymeasure.config.loader import LoadedStudyRaw
from llenergymeasure.config.models import StudyConfig
from llenergymeasure.study.library_resolution import resolve_library_effective

__all__ = ["finalise_study"]


def finalise_study(raw: LoadedStudyRaw) -> StudyConfig:
    """Apply library-resolution dedup, design hash, and cycle ordering.

    Consumes the :class:`~llenergymeasure.config.loader.LoadedStudyRaw` produced
    by :func:`llenergymeasure.config.loader.load_study_config` and produces the
    resolved :class:`~llenergymeasure.config.models.StudyConfig` that the runner
    iterates over.

    Steps:
      1. Library-resolution mechanism + resolved-config-hash dedup of the
         declared configs (see sweep-dedup.md §2).
      2. compute_study_design_hash() over the post-dedup configs - the hash
         identifies the *unique* measurement set, not duplicate declarations.
      3. apply_cycles() to produce the execution sequence.
      4. Serialise pre-run equivalence groups for the sidecar writer.

    Args:
        raw: Parsed + sweep-expanded study material.

    Returns:
        Resolved StudyConfig with ordered experiments, study_design_hash, dedup
        mode, and pre-run equivalence groups.
    """
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
    )
