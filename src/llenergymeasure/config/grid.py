"""Sweep grid expansion and cycle ordering for study configurations."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llenergymeasure.config._dict_utils import _unflatten, deep_merge
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import ALL_ENGINES
from llenergymeasure.utils.compat import StrEnum
from llenergymeasure.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Keys that belong to the study YAML structure, not to individual experiments.
# These are stripped from base: files and excluded from the fixed dict.
# "runners" is study-level metadata (per-engine runner config) - not an experiment field.
_STUDY_ONLY_KEYS = frozenset(
    {
        "sweep",
        "experiments",
        "study_execution",
        "base",
        "study_name",
        "version",
        "runners",
        "output",
    }
)


class ExperimentOrder(StrEnum):
    SEQUENTIAL = "sequential"
    INTERLEAVE = "interleave"
    SHUFFLE = "shuffle"
    REVERSE = "reverse"
    LATIN_SQUARE = "latin_square"


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
    3. Expand sweep: block into raw config dicts
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

    # Step 3: Expand sweep: block into raw config dicts
    sweep = raw_study.get("sweep", {})
    sweep_raw_configs = _expand_sweep(sweep, merged_fixed)

    # Step 4: Append explicit experiments: list entries
    # Strip non-matching engine sections *inherited from fixed*, but preserve
    # any the user wrote directly in the experiment entry (those are genuine
    # misconfigurations and should fail Pydantic validation).
    explicit_entries = raw_study.get("experiments", [])
    explicit_raw_configs = []
    for exp in explicit_entries:
        merged = {**merged_fixed, **exp}
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
            skipped.append(
                SkippedConfig(
                    raw_config=raw_config,
                    reason=reason,
                    errors=errors,
                    rule_id=_extract_rule_id(errors),
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

    # Combinatorial explosion warnings (tiered)
    n_valid = len(valid)
    exec_cfg = raw_study.get("study_execution", {})
    n_cycles = exec_cfg.get("n_cycles", 1) if isinstance(exec_cfg, dict) else 1
    total_runs = n_valid * n_cycles
    gap_seconds = (
        exec_cfg.get("experiment_gap_seconds", 0) if isinstance(exec_cfg, dict) else 0
    ) or 0

    if n_valid > 2000:
        min_hours = total_runs * gap_seconds / 3600
        logger.warning(
            "Extremely large study: %d experiments (%d total runs). "
            "Minimum runtime: ~%.0fh (gap time only). "
            "Consider reducing sweep dimensions or groups.",
            n_valid,
            total_runs,
            min_hours,
        )
    elif n_valid > 500:
        min_hours = total_runs * gap_seconds / 3600
        logger.warning(
            "Very large study: %d experiments (%d total runs with %d cycles). "
            "Minimum runtime: ~%.0fh (gap time only). "
            "Consider reducing sweep dimensions or groups.",
            n_valid,
            total_runs,
            n_cycles,
            min_hours,
        )
    elif n_valid > 100:
        logger.info("Large study: %d experiments.", n_valid)

    return valid, skipped


def compute_study_design_hash(experiments: list[ExperimentConfig]) -> str:
    """SHA-256[:16] of the resolved experiment list (execution block excluded).

    Deterministic: uses json.dumps with sort_keys=True. Identical experiment lists
    produce the same hash across calls and interpreter restarts.
    """
    canonical = json.dumps([exp.model_dump() for exp in experiments], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def apply_cycles(
    experiments: list[ExperimentConfig],
    n_cycles: int,
    experiment_order: ExperimentOrder,
    study_design_hash: str,
    shuffle_seed: int | None = None,
) -> list[ExperimentConfig]:
    """Return the ordered execution sequence for n_cycles repetitions.

    sequential:    [A, A, A, B, B, B]  - all cycles of each experiment together
    interleave:    [A, B, A, B, A, B]  - one cycle of each experiment, repeated
    shuffle:       random per-cycle order, seeded from study_design_hash by default
    reverse:       alternating forward/backward per cycle - [A, B, B, A, A, B]
    latin_square:  Williams balanced latin square (counterbalances carryover effects)
    """
    if experiment_order == ExperimentOrder.SEQUENTIAL:
        return [exp for exp in experiments for _ in range(n_cycles)]

    if experiment_order == ExperimentOrder.INTERLEAVE:
        return experiments * n_cycles

    if experiment_order == ExperimentOrder.REVERSE:
        result: list[ExperimentConfig] = []
        for i in range(n_cycles):
            cycle = list(experiments) if i % 2 == 0 else list(reversed(experiments))
            result.extend(cycle)
        return result

    if experiment_order == ExperimentOrder.LATIN_SQUARE:
        return _williams_latin_square(experiments, n_cycles)

    # shuffle
    seed = shuffle_seed if shuffle_seed is not None else int(study_design_hash, 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    result = []
    for _ in range(n_cycles):
        cycle = list(experiments)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result


def _williams_latin_square(
    experiments: list[ExperimentConfig],
    n_cycles: int,
) -> list[ExperimentConfig]:
    """Generate a Williams balanced latin square ordering.

    A Williams design is a latin square where each condition follows every other
    condition exactly once across rows, balancing first-order carryover effects.
    When n_cycles > k (number of experiments), cycles repeat the square rows.
    When n_cycles < k, the first n_cycles rows are used.
    """
    k = len(experiments)
    if k == 0:
        return []

    # Build Williams square rows (works for both even and odd k)
    rows: list[list[int]] = []
    for i in range(k):
        row: list[int] = [0] * k
        for j in range(k):
            if j == 0:
                row[j] = i
            elif j % 2 == 1:
                row[j] = (i + (j + 1) // 2) % k
            else:
                row[j] = (i - j // 2) % k
        rows.append(row)

    result: list[ExperimentConfig] = []
    for cycle_idx in range(n_cycles):
        row = rows[cycle_idx % k]
        result.extend(experiments[idx] for idx in row)
    return result


def count_sweep_structure(raw_sweep: dict[str, Any]) -> tuple[int, int]:
    """Count independent axes and dependent groups in a raw sweep dict.

    An independent axis is a key mapping to a list of scalars (Cartesian product).
    A dependent group is a key mapping to a list of dicts (union of variants).

    Returns (n_axes, n_groups).
    """
    if not raw_sweep:
        return 0, 0

    n_axes = 0
    n_groups = 0

    for _key, values in raw_sweep.items():
        if _is_group(values):
            n_groups += 1
        else:
            n_axes += 1

    return n_axes, n_groups


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


def _strip_other_engine_sections(config_dict: dict[str, Any], engine: str) -> dict[str, Any]:
    """Remove engine-specific sections that don't match *engine*.

    In a multi-engine study, top-level engine sections (e.g. ``tensorrt:``)
    are shared defaults for that engine's experiments.  When the grid expander
    assigns a different engine, those sections must be stripped before Pydantic
    validation - otherwise ``validate_engine_section_match`` rejects the config.
    """
    return {k: v for k, v in config_dict.items() if k not in ALL_ENGINES or k == engine}


# =============================================================================
# Sweep group helpers
# =============================================================================


def _is_group(value: object) -> bool:
    """True if a sweep entry is a group (list of dicts), not an independent axis.

    Disambiguation: a list of scalars is an independent axis (Cartesian product);
    a list of dicts (or containing ``{}``) is a dependent group (union of variants).

    Raises ``ConfigError`` for mixed lists (some dicts, some scalars).
    """
    if not isinstance(value, list) or len(value) == 0:
        return False
    has_dicts = any(isinstance(e, dict) for e in value)
    if not has_dicts:
        return False
    all_dicts = all(isinstance(e, dict) for e in value)
    if not all_dicts:
        raise ConfigError(
            "Sweep entry mixes dicts and scalars. Group entries must all be "
            "dicts; independent axes must all be scalars."
        )
    return True


def _group_engine_scope(group_key: str) -> str | None:
    """Return engine name if a group key is engine-scoped, else None (universal).

    Both engine-native keys (``transformers.engine_params.dtype``) and per-engine
    harness keys (``harness.transformers.batch_size``) are scoped to that engine;
    only the latter carries the ``harness.`` prefix.
    """
    parts = group_key.split(".")
    if len(parts) >= 2 and parts[0] in ALL_ENGINES:
        return parts[0]
    if len(parts) >= 3 and parts[0] == "harness" and parts[1] in ALL_ENGINES:
        return parts[1]
    return None


def _expand_group_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a single group entry into one or more flat dicts (mini-grid).

    Scalar-valued fields pass through unchanged. List-valued fields (list of
    scalars) produce a Cartesian product within the entry. Nested lists like
    ``[[0, 1]]`` are treated as literal list values (not expanded).
    """
    scalar_fields: dict[str, Any] = {}
    grid_keys: list[str] = []
    grid_values: list[list[Any]] = []

    for key, value in entry.items():
        if isinstance(value, list) and len(value) > 0 and not isinstance(value[0], (list, dict)):
            # List of scalars -> mini-grid axis
            grid_keys.append(key)
            grid_values.append(value)
        else:
            scalar_fields[key] = value

    if not grid_keys:
        return [entry]

    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*grid_values):
        variant = dict(scalar_fields)
        for key, value in zip(grid_keys, combo, strict=True):
            variant[key] = value
        expanded.append(variant)
    return expanded


def _expand_group(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand all entries in a group, flattening mini-grids into a union of variants."""
    variants: list[dict[str, Any]] = []
    for entry in entries:
        variants.extend(_expand_group_entry(entry))
    return variants


def _route_key_value(
    config_dict: dict[str, Any],
    key: str,
    value: Any,
) -> dict[str, Any]:
    """Route a single fully-qualified key into *config_dict*.

    Routing rules:
    - Engine-prefixed dotted key (``transformers.engine_params.dtype``) → merge into engine section.
    - Other dotted key (``task.dataset.source``) → unflatten at top level.
    - Simple key → direct assignment.

    Returns the (possibly replaced) config_dict reference.
    """
    if "." in key:
        prefix, param = key.split(".", 1)
        if prefix in ALL_ENGINES:
            engine_dict = config_dict.get(prefix, {})
            nested_update = _unflatten({param: value})
            config_dict[prefix] = deep_merge(engine_dict, nested_update)
        else:
            nested_update = _unflatten({key: value})
            config_dict = deep_merge(config_dict, nested_update)
    else:
        config_dict[key] = value
    return config_dict


def _apply_group_overlay(
    config_dict: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Apply a group variant's fully-qualified keys onto a config dict."""
    for fq_key, value in overlay.items():
        config_dict = _route_key_value(config_dict, fq_key, value)
    return config_dict


def _validate_sweep_groups(
    groups: dict[str, list[dict[str, Any]]],
    axis_keys: set[str],
) -> None:
    """Raise ConfigError if a group name collides with an independent axis key."""
    collisions = set(groups.keys()) & axis_keys
    if collisions:
        raise ConfigError(
            f"Sweep group name(s) collide with independent axis key(s): "
            f"{', '.join(sorted(collisions))}. "
            f"Use abstract names for groups (e.g. 'transformers.compilation' not 'transformers.torch_compile')."
        )


# =============================================================================
# Sweep expansion
# =============================================================================


def _expand_sweep(sweep: dict[str, Any], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a sweep: block into a flat list of raw experiment config dicts.

    Supports two entry types under ``sweep:``:

    - **Independent axes** (list of scalars): Cartesian product across all axes.
    - **Dependent groups** (list of dicts): Union of variant dicts. Groups are
      crossed with each other and with independent axes, but entries *within*
      a group are alternatives (unioned, not crossed).

    Type-based disambiguation: ``list[scalar]`` = axis, ``list[dict]`` = group.
    """
    if not sweep:
        task = fixed.get("task")
        has_model = isinstance(task, dict) and task.get("model")
        if has_model:
            engine = fixed.get("engine", "transformers")
            return [_strip_other_engine_sections(dict(fixed), engine)]
        return []

    # ── Step 1: Partition sweep into axes and groups ──
    universal_dims: dict[str, list[Any]] = {}
    scoped_dims: dict[str, dict[str, list[Any]]] = {}  # {engine: {fq_key: [values]}}
    groups: dict[str, list[dict[str, Any]]] = {}  # {group_name: [variant_dicts]}

    for key, values in sweep.items():
        if _is_group(values):
            groups[key] = _expand_group(values)
            continue

        if not isinstance(values, list):
            values = [values]

        engine_scope = _group_engine_scope(key)
        if engine_scope is not None:
            # Store the full fully-qualified key so routing reconstructs the exact
            # path (engine-native ``transformers.engine_params.dtype`` and
            # harness-scoped ``harness.transformers.batch_size`` both round-trip
            # verbatim).
            scoped_dims.setdefault(engine_scope, {})[key] = values
        else:
            universal_dims[key] = values

    # Derive flat axis key set for collision detection
    axis_keys = set(universal_dims.keys()) | {
        fq_key for params in scoped_dims.values() for fq_key in params
    }
    _validate_sweep_groups(groups, axis_keys)

    # ── Step 2: Separate groups by engine scope ──
    universal_groups: dict[str, list[dict[str, Any]]] = {}
    scoped_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}  # {engine: {name: variants}}

    for group_name, variants in groups.items():
        engine_scope = _group_engine_scope(group_name)
        if engine_scope:
            scoped_groups.setdefault(engine_scope, {})[group_name] = variants
        else:
            universal_groups[group_name] = variants

    # ── Step 3: Determine engines ──
    fixed_engine = fixed.get("engine", "transformers")
    if isinstance(fixed_engine, list):
        engines = list(fixed_engine)
    elif scoped_dims or scoped_groups:
        # Engines implied by scoped axes or scoped groups
        engines = sorted(set(scoped_dims.keys()) | set(scoped_groups.keys()))
    else:
        engines = [fixed_engine]

    # ── Step 4: Per-engine expansion ──
    results: list[dict[str, Any]] = []

    for engine in engines:
        # Collect applicable groups (universal + this engine's scoped)
        applicable_groups: dict[str, list[dict[str, Any]]] = dict(universal_groups)
        applicable_groups.update(scoped_groups.get(engine, {}))

        # Collect applicable axes - reconstruct fully-qualified keys for routing
        engine_scoped = scoped_dims.get(engine, {})
        # scoped_dims stores fully-qualified keys (engine-native and harness-scoped
        # alike), so use them verbatim.
        fq_dim_keys = list(universal_dims.keys()) + list(engine_scoped.keys())
        all_dim_values = list(universal_dims.values()) + list(engine_scoped.values())

        # Cross all group variant lists with each other (lazy - iterated once)
        group_combos: Iterable[tuple[Any, ...]]
        if applicable_groups:
            group_names = list(applicable_groups.keys())
            group_variant_lists = [applicable_groups[n] for n in group_names]
            group_combos = itertools.product(*group_variant_lists)
        else:
            group_combos = [()]  # single empty combo → no group overlays

        if not fq_dim_keys and not applicable_groups:
            # No dimensions or groups for this engine - produce one config
            config_dict: dict[str, Any] = _strip_other_engine_sections(dict(fixed), engine)
            config_dict["engine"] = engine
            results.append(config_dict)
            continue

        # Pre-compute stripped base config once per engine
        base_config = _strip_other_engine_sections(dict(fixed), engine)
        base_config["engine"] = engine

        # axis_combos materialised - reused across group combos
        axis_combos = list(itertools.product(*all_dim_values)) if fq_dim_keys else [()]

        for group_combo in group_combos:
            for axis_combo in axis_combos:
                config_dict = dict(base_config)

                # Apply independent axis values
                for dim_key, value in zip(fq_dim_keys, axis_combo, strict=True):
                    config_dict = _route_key_value(config_dict, dim_key, value)

                # Apply group overlays (each group_combo entry is one variant dict)
                for variant in group_combo:
                    if variant:  # skip empty dicts ({} = baseline, no overlay)
                        config_dict = _apply_group_overlay(config_dict, variant)

                results.append(config_dict)

    return results
