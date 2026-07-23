"""Expansion of a study ``sweep:`` block into raw experiment config dicts.

Turns the two ``sweep:`` entry types - independent axes (``list[scalar]``,
Cartesian product) and dependent groups (``list[dict]``, union of variants) -
into a flat list of raw config dicts, routing fully-qualified keys into the
right engine/harness section along the way. :func:`_expand_sweep` is the entry
point consumed by :func:`llenergymeasure.config.grid.expand_grid`;
:func:`count_sweep_structure` reports the axis/group shape for CLI summaries.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from llenergymeasure.config._dict_utils import _unflatten, deep_merge
from llenergymeasure.config.ssot import ALL_ENGINES
from llenergymeasure.config.sweep_idioms import expand_axis_idiom
from llenergymeasure.utils.exceptions import ConfigError


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

    Both engine-native keys (``transformers.engine_params.dtype``) and the
    llem-owned execution knobs (``transformers.llem_execution.batch_size``) live
    under the engine prefix, so a single leading-segment check covers both.
    """
    parts = group_key.split(".")
    if len(parts) >= 2 and parts[0] in ALL_ENGINES:
        return parts[0]
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


def _expand_sweep(
    sweep: dict[str, Any],
    fixed: dict[str, Any],
    synthesize_baseline: bool = True,
) -> list[dict[str, Any]]:
    """Expand a sweep: block into a flat list of raw experiment config dicts.

    Supports two entry types under ``sweep:``:

    - **Independent axes** (list of scalars): Cartesian product across all axes.
    - **Dependent groups** (list of dicts): Union of variant dicts. Groups are
      crossed with each other and with independent axes, but entries *within*
      a group are alternatives (unioned, not crossed).

    Type-based disambiguation: ``list[scalar]`` = axis, ``list[dict]`` = group.

    An axis value may also be a numeric idiom mapping (see
    :mod:`llenergymeasure.config.sweep_idioms`); it is expanded to a plain
    list here, at load time, so downstream consumers only see lists. Any
    other mapping-valued axis is rejected with ConfigError.

    When ``sweep`` is empty this synthesises a single inline-model baseline from
    ``fixed`` (the no-sweep, no-experiments study form) using ``fixed``'s engine
    or a ``transformers`` default. That synthesis fires only when
    ``synthesize_baseline`` is True; the caller passes False whenever an explicit
    experiments: list is present, so an empty sweep never adds a phantom
    default-engine baseline alongside the user's entries.
    """
    if not sweep:
        if not synthesize_baseline:
            return []
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

        if isinstance(values, dict):
            try:
                values = expand_axis_idiom(values)
            except ValueError as exc:
                raise ConfigError(f"sweep axis '{key}': {exc}") from exc
        elif not isinstance(values, list):
            values = [values]

        engine_scope = _group_engine_scope(key)
        if engine_scope is not None:
            # Store the full fully-qualified key so routing reconstructs the exact
            # path (engine-native ``transformers.engine_params.dtype`` and the
            # execution knob ``transformers.llem_execution.batch_size`` both
            # round-trip verbatim).
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
        # Engines implied by scoped axes or scoped groups. A study may ALSO fix
        # `engine:` explicitly (e.g. `engine: transformers` while sweeping a
        # vllm.* axis) - that engine gets its own baseline run, so union it in
        # rather than letting the scoped axes silently drop it. Only an
        # explicitly-set engine counts; the "transformers" default does not.
        scope_engines = set(scoped_dims.keys()) | set(scoped_groups.keys())
        if "engine" in fixed:
            scope_engines.add(fixed_engine)
        engines = sorted(scope_engines)
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
