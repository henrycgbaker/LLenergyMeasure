"""The single study-resolution entry point.

The config layer (:mod:`llenergymeasure.config.loader`) does pure
parse + sweep-expansion and returns a
:class:`~llenergymeasure.config.loader.LoadedStudyRaw`. The dedup /
study_design_hash / cycle-ordering / equivalence-group steps need the
study-layer library-resolution mechanism, so they live here - keeping the
config layer free of any upward import into ``study``.

:func:`resolve_study` is the ONE entry every study passes through, whether it
arrived as a YAML file or as objects a caller built in memory (#886). A study
that reaches the orchestrator without it is unresolved - no dedup, no identity
hash, no cycle expansion - so the orchestrator rejects it rather than running it.
``llenergymeasure.api.load_study`` is the YAML front door (parse, then resolve);
:func:`resolve_study_objects` is the object front door that ``run_study`` and
``run_experiment`` use for a study a caller built in memory, touching no file.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal

from llenergymeasure.config.grid import (
    ExperimentOrder,
    apply_cycles,
    compute_study_design_hash,
)
from llenergymeasure.config.loader import LoadedStudyRaw
from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.study.library_resolution import resolve_library_effective

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserConfig

__all__ = ["resolve_study", "resolve_study_objects"]

logger = logging.getLogger(__name__)


def resolve_study(
    raw: LoadedStudyRaw,
    *,
    user_config: UserConfig | None = None,
    execution_defaults: Mapping[str, Any] | None = None,
) -> StudyConfig:
    """Resolve parsed study material into the StudyConfig the runner executes.

    Consumes a :class:`~llenergymeasure.config.loader.LoadedStudyRaw` - from
    :func:`llenergymeasure.config.loader.load_study_config` on the YAML path, or
    built directly from objects on the programmatic path - and produces the
    resolved :class:`~llenergymeasure.config.models.StudyConfig`. Both routes
    resolve identically, down to the ``study_design_hash``.

    Steps, in this order:
      0. Reject a study that mixes serving_mode values (a staging restriction of
         this release), then resolve the execution block: caller-supplied
         effective defaults beneath the study file, then the machine-local
         thermal gaps for gaps the study left unset.
      1. Overlay the tool-wide user-config server warmup onto each declared server
         config, BEFORE dedup, so the resolved-config hash binds on the
         realised warmup protocol.
      2. Library-resolution mechanism + resolved-config-hash dedup of the
         declared configs.
      3. compute_study_design_hash() over the post-dedup configs - the hash
         identifies the *unique* measurement set, not duplicate declarations.
      4. apply_cycles() to produce the execution sequence.
      5. Serialise pre-run equivalence groups for the sidecar writer.

    Args:
        raw: Parsed + sweep-expanded study material.
        user_config: Tool-wide user config supplying the ``server.warmup``
            overlay and the machine-local thermal gap defaults. ``None`` (the
            default) applies NEITHER - callers hand in a config at the production
            edge, keeping the resolution step hermetic for the unit tests that
            construct one directly.
        execution_defaults: Execution-block defaults that sit BENEATH the study
            file: a field the file wrote always wins, a field it left unset takes
            this value instead of the conservative built-in default. This is how
            the CLI applies its research-appropriate defaults (3 cycles, shuffle)
            without re-reading the study file to find out what it declared.

    Returns:
        Resolved StudyConfig with ordered experiments, study_design_hash, dedup
        mode, and pre-run equivalence groups.

    Raises:
        ConfigError: The study mixes serving_mode values, or
            ``execution_defaults`` is not a valid execution-block layer.
    """
    _validate_homogeneous_serving_mode(raw.valid_experiments)

    execution = _resolve_execution(
        raw.execution,
        user_config=user_config,
        execution_defaults=execution_defaults,
    )

    # Overlay the tool-wide user-config server warmup onto each declared
    # server config BEFORE dedup, so the resolved-config hash - and hence dedup -
    # binds on the realised warmup protocol. Declared hashes are untouched (the
    # overlay is side-channel state, never a field), and the overlay rides the
    # dedup deep copies through to the runner. No-op for offline configs and
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

    # Apply library-resolution mechanism + resolved-config-hash dedup to the declared configs
    # before running cycles. This collapses measurement-equivalent
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
    # declared-config-hash experiment ids and emit those keys.
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


def resolve_study_objects(
    study: StudyConfig,
    *,
    user_config: UserConfig | None = None,
) -> StudyConfig:
    """Resolve a StudyConfig a caller built in memory, touching no file.

    The object front door onto :func:`resolve_study`, the counterpart of the YAML
    front door ``api.load_study``: it unwraps the parts the caller assembled back
    into a :class:`~llenergymeasure.config.loader.LoadedStudyRaw` and resolves
    them, so a programmatically built study is deduplicated, hashed,
    cycle-expanded and given its equivalence groups exactly as a study file is
    (#886).

    The unwrap lives here, beside ``LoadedStudyRaw`` and the resolution it feeds,
    so a new ``StudyConfig`` field has one obvious place to be carried across
    rather than being dropped silently by a caller's own copy of this mapping.

    Args:
        study: The study as the caller assembled it. Its declared fields are never
            rewritten; see ``resolve_study`` for the side-channel state the warmup
            overlay attaches to the caller's own ExperimentConfig objects.
        user_config: Tool-wide user config, passed straight through to
            :func:`resolve_study`.

    Returns:
        The resolved StudyConfig, as a new object.
    """
    raw = LoadedStudyRaw(
        valid_experiments=list(study.experiments),
        # Skipped grid points only arise from YAML sweep expansion, so a study
        # built from objects normally has none. They are carried across after
        # resolution instead of through the raw material: they are already dicts,
        # the shape resolve_study emits, so nothing is silently dropped.
        skipped=[],
        study_name=study.study_name,
        output=study.output,
        execution=study.study_execution,
        runners=study.runners,
        images=study.images,
    )
    resolved = resolve_study(raw, user_config=user_config)
    if study.skipped_configs:
        resolved = resolved.model_copy(update={"skipped_configs": study.skipped_configs})
    return resolved


def _validate_homogeneous_serving_mode(experiments: list[ExperimentConfig]) -> None:
    """Reject a study that mixes serving_mode values (a staging restriction).

    DELIBERATELY DELETABLE: this function plus its single call site in
    :func:`resolve_study` are the ONLY thing restricting mixed serving_mode
    studies. A later release that admits the engine x serving_mode grid crossing
    deletes both and nothing else changes.

    It lives at the resolution entry point, not as a model_validator: the data
    model stays mixed-legal, so building a mixed ``StudyConfig`` in memory or
    inspecting one is unaffected, while every route that actually RUNS a study
    (the study file and the programmatic entry points alike) is gated identically.

    Args:
        experiments: The fully-expanded, already-validated experiment list.

    Raises:
        ConfigError: When the experiments span more than one serving_mode.
    """
    modes = sorted({exp.serving_mode for exp in experiments})
    if len(modes) > 1:
        from llenergymeasure.utils.exceptions import ConfigError

        raise ConfigError(
            f"This study mixes serving_mode values ({', '.join(modes)}), but a "
            "single study must use exactly one serving_mode at this release. "
            "Mixed-mode studies (a study spanning both offline and server) arrive "
            "in a later release. Split the study so every experiment shares one "
            "serving_mode: run one study per serving_mode."
        )


def _resolve_execution(
    declared: ExecutionConfig,
    *,
    user_config: UserConfig | None,
    execution_defaults: Mapping[str, Any] | None,
) -> ExecutionConfig:
    """Resolve the study execution block through the precedence layers.

    Two resolutions happen here, both of which the runner would otherwise have to
    guess at:

    - ``execution_defaults`` sit BENEATH the study file. Only the fields the file
      actually wrote (``model_fields_set``) enter the winning layer, so a caller
      default fills a field the file omitted and never overrides one it declared.
    - ``experiment_gap_seconds`` / ``cycle_gap_seconds`` are documented as
      "None = use machine default from user config", and this is where that
      default is applied. Leaving them None made the runner read them as zero, so
      experiments ran back-to-back and thermal state bled between measurements
      for anyone relying on their machine defaults (#886).

    Raises:
        ConfigError: ``execution_defaults`` names a field the execution block does
            not have, or gives one a value it will not take.
    """
    execution = declared

    if execution_defaults:
        from pydantic import ValidationError

        from llenergymeasure.config.precedence import fields_set_layer, resolve_layers
        from llenergymeasure.utils.exceptions import ConfigError

        try:
            execution = ExecutionConfig.model_validate(
                resolve_layers(
                    ExecutionConfig().model_dump(mode="python"),
                    dict(execution_defaults),
                    fields_set_layer(declared),
                )
            )
        except ValidationError as exc:
            # execution_defaults is caller-supplied, so a typo here is a caller
            # mistake and deserves a message that names the parameter rather than
            # a bare pydantic traceback about a study block the caller never wrote.
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigError(
                f"Invalid execution_defaults: {details}. Keys must be "
                f"study_execution fields ({', '.join(sorted(ExecutionConfig.model_fields))})."
            ) from exc

    if user_config is not None:
        gaps: dict[str, Any] = {}
        if execution.experiment_gap_seconds is None:
            gaps["experiment_gap_seconds"] = user_config.execution.experiment_gap_seconds
        if execution.cycle_gap_seconds is None:
            gaps["cycle_gap_seconds"] = user_config.execution.cycle_gap_seconds
        if gaps:
            execution = execution.model_copy(update=gaps)

    return execution


def _maybe_hint_sequential_server_singletons(
    run_experiments: list[ExperimentConfig],
    *,
    experiment_order: str,
    n_cycles: int,
) -> None:
    """Emit a did-you-know when a foldable server sweep runs under sequential order.

    Under ``sequential`` order with ``n_cycles > 1`` each grid point's cycles are
    adjacent, so a rate sweep's cells are never both consecutive AND same-cycle
    and each dispatches as its own server session (one server launch per cell per
    cycle). ``interleave`` replays the base ordering per cycle pass, so each pass
    folds the sweep into a single launch. Both are correct; this is INFO, not a
    warning.

    Only a rate sweep is foldable: an slo sweep never folds (slo is excluded from
    the declared config hash as a post-hoc overlay, so slo-differing cells share a
    declared hash and either dedup-collapse to one config or cycle-track as
    repeats), so this hint stays silent for an slo-only sweep under either order.

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
