"""Grid expansion for study configurations.

Orchestrates turning a raw study dict into a validated experiment list:
``base:`` inheritance, the fixed (shared) field set, sweep dispatch, and the
Pydantic-validation pass that partitions configs into valid and skipped. The
sweep-block mechanics live in :mod:`llenergymeasure.config.sweep_expansion`
and the cycle-ordering mechanics in
:mod:`llenergymeasure.config.cycle_ordering`; their public API is re-exported
here so ``llenergymeasure.config.grid`` stays the single import surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llenergymeasure.config._dict_utils import deep_merge
from llenergymeasure.config.cycle_ordering import (
    ExperimentOrder,
    apply_cycles,
    cycle_boundary_indices,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import ALL_ENGINES
from llenergymeasure.config.sweep_expansion import _expand_sweep, count_sweep_structure
from llenergymeasure.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentOrder",
    "SkippedConfig",
    "apply_cycles",
    "compute_study_design_hash",
    "count_sweep_structure",
    "cycle_boundary_indices",
    "expand_grid",
]

# Keys that belong to the study YAML structure, not to individual experiments.
# These are stripped from base: files and excluded from the fixed dict.
# "runners" and "images" are study-level metadata (per-engine runner config and
# per-engine Docker image overrides) - not experiment fields.
_STUDY_ONLY_KEYS = frozenset(
    {
        "sweep",
        "experiments",
        "study_execution",
        "base",
        "study_name",
        "version",
        "runners",
        "images",
        "output",
    }
)


@dataclass
class SkippedConfig:
    """An ExperimentConfig that failed Pydantic validation during grid expansion."""

    raw_config: dict[str, Any]
    reason: str
    errors: list[dict[str, Any]] = field(default_factory=list)
    rule_id: str | None = None
    """Engine rule that rejected this config, or None for a non-rule failure.

    Populated from the ValidationError entries via :func:`_extract_rule_id`;
    None when the config failed on a type error, unknown field, or any other
    validation that is not an engine-rule rejection.
    """

    @property
    def short_label(self) -> str:
        """Short label for display: 'engine, dtype'. dtype lives under engine_params."""
        engine = self.raw_config.get("engine", "unknown")
        section = self.raw_config.get(engine) if isinstance(engine, str) else None
        engine_params = section.get("engine_params") if isinstance(section, dict) else None
        dtype = engine_params.get("dtype", "?") if isinstance(engine_params, dict) else "?"
        return f"{engine}, {dtype}"

    @property
    def display_reason(self) -> str:
        """Compact reason for digests: the rejecting rule id, or the raw reason."""
        return f"rule {self.rule_id}" if self.rule_id is not None else self.reason

    def to_dict(self) -> dict[str, Any]:
        """Serialise for StudyConfig.skipped_configs."""
        return {
            "raw_config": self.raw_config,
            "reason": self.reason,
            "short_label": self.short_label,
            "errors": self.errors,
            "rule_id": self.rule_id,
        }


_RULE_ID_MARKER = re.compile(r"^\[([a-z0-9_]+)\]")


def _raw_rule_message(err: dict[str, Any]) -> str:
    """Return a value_error entry's message with the ``[rule_id]`` marker leading.

    Prefer the original exception carried in ``ctx`` (its str keeps the marker
    at position 0); fall back to stripping Pydantic's ``Value error, `` prefix
    from ``msg`` when no exception object is present.
    """
    ctx = err.get("ctx")
    if isinstance(ctx, dict):
        original = ctx.get("error")
        if isinstance(original, BaseException):
            return str(original)
    msg = str(err.get("msg", ""))
    prefix = "Value error, "
    return msg[len(prefix) :] if msg.startswith(prefix) else msg


def _extract_rule_id(errors: list[dict[str, Any]]) -> str | None:
    """Return the engine-rule id that rejected a config, or None.

    Engine rules raise ``ValueError(f"[{rule.id}] {message}")`` in
    ``ExperimentConfig._apply_rules``; Pydantic wraps that as a ``value_error``
    entry whose original exception carries the leading ``[rule_id]`` marker.
    Rule ids are snake_case, so the anchored regex stays conservative and never
    matches bracketed prose later in a message. Only one rule fires per config
    (``_apply_rules`` raises on the first error match), so the first marker
    found is authoritative.
    """
    for err in errors:
        if err.get("type") != "value_error":
            continue
        match = _RULE_ID_MARKER.match(_raw_rule_message(err))
        if match is not None:
            return match.group(1)
    return None


# =============================================================================
# Public API
# =============================================================================


def expand_grid(
    raw_study: dict[str, Any],
    study_yaml_path: Path | None = None,
) -> tuple[list[ExperimentConfig], list[SkippedConfig]]:
    """Expand sweep dimensions into a flat list of ExperimentConfig.

    Resolution order:
    1. Load base: file (optional DRY inheritance)
    2. Build fixed dict from non-sweep/non-experiments/non-study_execution/non-base/non-study_name keys
    3. Expand sweep: block into raw config dicts. An empty sweep synthesises one
       inline-model baseline from a top-level task.model, but only when there is
       no explicit experiments: list (otherwise it contributes nothing).
    4. Append explicit experiments: list entries
    5. Pydantic-validate each raw dict, collecting valid + skipped

    Returns (valid_experiments, skipped_configs).
    Raises ConfigError if all configs are invalid or no experiments produced.
    """
    # Step 1: Load base: inheritance
    base_dict = _load_base(raw_study.get("base"), study_yaml_path)

    # Step 2: Fixed dict - experiment-level fields shared across all grid points
    fixed = _extract_fixed(raw_study)
    merged_fixed = {**base_dict, **fixed}  # inline fields override base

    # Step 3: Expand sweep: block into raw config dicts. The inline-model
    # baseline (an empty sweep synthesising one experiment from a top-level
    # task.model) fires only when there is no explicit experiments: list.
    # With explicit entries an empty sweep contributes nothing, so it never
    # adds a phantom default-engine baseline the user never wrote.
    sweep = raw_study.get("sweep", {})
    # `experiments:` present but YAML-null yields None from .get(); treat an
    # explicitly-empty key the same as an absent one rather than TypeError-ing
    # on the iteration below.
    explicit_entries = raw_study.get("experiments") or []
    sweep_raw_configs = _expand_sweep(sweep, merged_fixed, synthesize_baseline=not explicit_entries)

    # Step 4: Append explicit experiments: list entries
    # DEEP-merge each entry onto the fixed config, matching the sweep-axis path
    # (sweep_expansion._expand_sweep also deep_merges) - a shallow {**fixed, **exp}
    # would let an entry that re-declares any part of a nested section (e.g.
    # server.traffic.rate) silently drop the fixed-level siblings of that section
    # (e.g. a fixed server.warmup), which then reads as study-unset and lets the
    # user-config overlay override what the study explicitly declared. Two ways of
    # writing the same study must share merge semantics.
    #
    # Strip non-matching engine sections *inherited from fixed*, but preserve
    # any the user wrote directly in the experiment entry (those are genuine
    # misconfigurations and should fail Pydantic validation).
    explicit_raw_configs = []
    for exp in explicit_entries:
        merged = deep_merge(merged_fixed, exp)
        engine = merged.get("engine", merged_fixed.get("engine", "transformers"))
        for key in ALL_ENGINES:
            if key != engine and key in merged and key not in exp:
                del merged[key]
        explicit_raw_configs.append(merged)

    all_raw_configs = sweep_raw_configs + explicit_raw_configs

    # Guard: no experiments produced at all
    if not all_raw_configs:
        raise ConfigError(
            "Study produced no experiments. "
            "Add a 'sweep:' block, 'experiments:' list, or inline 'model:' field."
        )

    # Step 5: Pydantic-validate each raw dict
    valid: list[ExperimentConfig] = []
    skipped: list[SkippedConfig] = []

    for raw_config in all_raw_configs:
        try:
            valid.append(ExperimentConfig(**raw_config))
        except (ValidationError, TypeError) as exc:
            reason = str(exc)
            errors: list[dict[str, Any]] = []
            if isinstance(exc, ValidationError):
                errors = [dict(e) for e in exc.errors()]
            rule_id = _extract_rule_id(errors)
            # Pydantic's error dicts carry a `ctx` mapping that can hold the raw
            # exception object (non-serialisable). `_extract_rule_id` has read
            # what it needs from it, so drop `ctx` now - SkippedConfig.errors is
            # persisted via json.dumps and must stay serialisable.
            for err in errors:
                err.pop("ctx", None)
            skipped.append(
                SkippedConfig(
                    raw_config=raw_config,
                    reason=reason,
                    errors=errors,
                    rule_id=rule_id,
                )
            )

    total = len(valid) + len(skipped)

    # Guard: all configs invalid
    if len(valid) == 0:
        first_reasons = "; ".join(s.display_reason[:120] for s in skipped[:5])
        raise ConfigError(
            f"nothing to run - all {total} generated config(s) are invalid. "
            f"First failures: {first_reasons}"
        )

    # Warning: >50% skip rate
    if total > 0 and len(skipped) / total > 0.5:
        logger.warning(
            "Most of your sweep is invalid (%d/%d configs skipped). "
            "Check your config combinations.",
            len(skipped),
            total,
        )
        for s in skipped:
            logger.warning("  Skipped (%s): %s", s.short_label, s.display_reason[:200])

    # Combinatorial explosion warnings (tiered). Counts only: this runs during
    # parse, before the execution block is resolved, so the thermal gaps and the
    # effective cycle count are not known here and any runtime figure would be
    # guesswork. ``llem study plan`` reports the gap-only wall-clock lower bound
    # from the RESOLVED study instead.
    n_valid = len(valid)
    exec_cfg = raw_study.get("study_execution", {})
    n_cycles = exec_cfg.get("n_cycles", 1) if isinstance(exec_cfg, dict) else 1
    total_runs = n_valid * n_cycles

    if n_valid > 2000:
        logger.warning(
            "Extremely large study: %d experiments (%d total runs). "
            "Consider reducing sweep dimensions or groups. "
            "Run `llem study plan <file>` for the wall-clock lower bound.",
            n_valid,
            total_runs,
        )
    elif n_valid > 500:
        logger.warning(
            "Very large study: %d experiments (%d total runs with %d cycles). "
            "Consider reducing sweep dimensions or groups. "
            "Run `llem study plan <file>` for the wall-clock lower bound.",
            n_valid,
            total_runs,
            n_cycles,
        )
    elif n_valid > 100:
        logger.info("Large study: %d experiments.", n_valid)

    return valid, skipped


def compute_study_design_hash(experiments: list[ExperimentConfig]) -> str:
    """SHA-256[:16] of the resolved experiment list (execution block excluded).

    Deterministic: uses json.dumps with sort_keys=True. Identical experiment lists
    produce the same hash across calls and interpreter restarts.

    Dumps in ``mode="json"`` so a field typed float but defaulted to an int
    literal serialises identically whether or not the config has been through a
    JSON round-trip (see :func:`compute_declared_config_hash`).
    """
    canonical = json.dumps([exp.model_dump(mode="json") for exp in experiments], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# =============================================================================
# Private helpers
# =============================================================================


def _extract_fixed(raw_study: dict[str, Any]) -> dict[str, Any]:
    """Return experiment-level fields from raw_study (all keys except study-only ones)."""
    return {k: v for k, v in raw_study.items() if k not in _STUDY_ONLY_KEYS}


def _load_base(base_path_str: str | None, study_yaml_path: Path | None) -> dict[str, Any]:
    """Load a base experiment config file, stripping study-only keys.

    Path is resolved relative to the study YAML file's directory.
    Hard error (ConfigError) if the file does not exist.
    """
    if base_path_str is None:
        return {}

    base_path = Path(base_path_str)
    if not base_path.is_absolute() and study_yaml_path is not None:
        base_path = study_yaml_path.parent / base_path

    if not base_path.exists():
        raise ConfigError(
            f"base: file not found: {base_path}. "
            "Path is resolved relative to the study YAML file's directory."
        )

    with base_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    # Strip study-only keys - base: accepts experiment config files only
    return {k: v for k, v in raw.items() if k not in _STUDY_ONLY_KEYS}
