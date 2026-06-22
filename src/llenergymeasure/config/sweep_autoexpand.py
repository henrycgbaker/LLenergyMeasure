"""Auto-expansion of sweep values from the curated Pydantic config surface.

Researcher-convenience layer that runs *before* grid expansion. Two entry
points:

- :func:`expand_auto_sweep` rewrites a raw ``sweep:`` block, replacing the
  string sentinel ``auto`` for any engine-field dotted FQN with that field's
  natural value set (introspected from the engine's curated ``Config`` model).
- :func:`build_study_skeleton` emits a complete, commented starting study YAML
  with every auto-expandable curated knob pre-expanded, as a trim-me template.

Because resolution produces concrete value lists, the downstream grid expander
(:func:`llenergymeasure.config.grid.expand_grid`) and the study-layer dedup are
untouched - they only ever see ordinary scalar lists.

The ``auto`` sentinel
=====================

A bare ``auto`` value on an engine-field sweep key of the shape
``<engine>.<section>.<field>`` (e.g. ``transformers.sampling_params.temperature``)
ALWAYS means "expand this field to its natural value set". It does not collide
with ``auto`` used as a real value elsewhere (``energy_sampler: auto``,
``device_map: auto``) because those are not engine-section sweep keys.

Note the one deliberate collision: a few curated fields have a ``Literal`` that
includes the literal string ``"auto"`` (e.g. ``tensorrt.engine_params.dtype``,
``vllm.engine_params.kv_cache_dtype``). For those, bare ``auto`` still means
expand-this-field - it is NOT read as the literal value. To pin such a field to
the literal string ``"auto"`` the researcher writes a one-element list,
``key: ["auto"]``, which passes through untouched. This collision is documented
rather than worked around; the ``: auto`` string UX is intentional.
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from llenergymeasure.config._dict_utils import is_sweep_group
from llenergymeasure.config.introspection import _strip_optional, get_engine_config_model
from llenergymeasure.utils.exceptions import ConfigError

# Sentinel value requesting auto-expansion of a sweep key.
AUTO_SENTINEL = "auto"

# Sweep-key sections that map onto the curated Config model.
_CURATED_SECTIONS = ("engine_params", "sampling_params")


# =============================================================================
# Public API
# =============================================================================


def expand_auto_sweep(sweep: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *sweep* with ``auto`` sentinels resolved to value lists.

    Independent axes (``key: auto``) resolve to the field's natural value set.
    Dependent groups (``key: [ {..}, {..} ]``) have any ``auto`` value *inside*
    a variant dict resolved in place; a group whose variant maps a field to
    ``auto`` expands that variant across the resolved values.

    The engine for each ``auto`` resolution is read from the key's own
    ``<engine>.`` prefix, so a multi-engine study with auto keys for different
    engines resolves each against the right curated Config model.

    Args:
        sweep: The raw ``sweep:`` block (dotted-FQN keys -> values).

    Returns:
        A new sweep dict with no ``auto`` sentinels remaining.

    Raises:
        ConfigError: An ``auto`` sentinel sits on a field that cannot be
            auto-expanded (see :func:`resolve_auto_values`), or a group variant
            uses ``auto`` in a way that cannot be resolved.
    """
    resolved: dict[str, Any] = {}
    for key, value in sweep.items():
        if value == AUTO_SENTINEL:
            resolved[key] = resolve_auto_values(key)
        elif is_sweep_group(value):
            resolved[key] = _resolve_group(key, value)
        else:
            resolved[key] = value
    return resolved


def resolve_auto_values(fqn: str) -> list[Any]:
    """Resolve an engine-field FQN to its natural sweep value set.

    Bucketing policy:
      - ``Literal[...]`` / enum field -> all members, verbatim, in declared order.
      - ``bool`` field                -> ``[False, True]``.
      - numeric field with BOTH a lower (``ge``/``gt``) and upper (``le``/``lt``)
        bound -> three buckets ``[low, mid, high]``:
          * low  = the lower endpoint - ``ge`` used as-is (inclusive), or ``gt``
            nudged inward so the emitted value is in range (exclusive).
          * high = the upper endpoint - ``le`` used as-is (inclusive), or ``lt``
            nudged inward so the emitted value is in range (exclusive).
          * mid  = midpoint ``(low + high) / 2`` of the inclusive span.
        The inward nudge for an exclusive bound is 1 for ``int`` fields and
        ``1e-3 * span`` (a small fraction of the bound gap) for ``float``
        fields, guaranteeing every emitted value satisfies the constraint.
        For ``int`` fields the three buckets are coerced to ints (deduplicated,
        order preserved); for ``float`` fields they stay floats.

    Anything else raises ``ConfigError``:
      - numeric with only one bound, or no bounds (no finite range to bucket),
      - a field whose type we cannot bucket (e.g. ``Any``, ``str``, nested model),
      - an FQN that does not name a field on the curated Config model.

    Args:
        fqn: Dotted engine-field key, e.g. ``transformers.sampling_params.top_p``.
            The engine is read from the key's own ``<engine>.`` prefix.

    Returns:
        The resolved value list.

    Raises:
        ConfigError: The field is not auto-expandable (with an actionable hint).
    """
    field_info = _resolve_field_info(fqn)
    # Strip a leading ``| None`` so ``bool | None`` reduces to ``bool`` etc.; the
    # constraint metadata (``ge``/``gt``/``le``/``lt``) rides on the FieldInfo.
    annotation = _strip_optional(field_info.annotation)
    metadata = list(field_info.metadata)

    if get_origin(annotation) is Literal:
        return list(get_args(annotation))

    if annotation is bool:
        return [False, True]

    if annotation in (int, float):
        return _bucket_numeric(fqn, annotation, metadata)

    raise ConfigError(
        f"Sweep key '{fqn}' cannot be auto-expanded: its curated type "
        f"({_type_label(annotation)}) has no natural value set. "
        f"Supply explicit values, e.g. '{fqn}: [<v1>, <v2>, ...]'."
    )


def build_study_skeleton(engine: str) -> str:
    """Return a commented starting study YAML for *engine* (a trim-me template).

    Every auto-expandable curated knob (across ``engine_params`` and
    ``sampling_params``) is emitted under ``sweep:`` with its resolved value
    set. Non-auto-expandable curated fields (unbounded numeric, ``Any``, nested
    models) are emitted as commented placeholder lines telling the researcher to
    supply explicit values, rather than failing the whole generation.

    A field whose natural set is only *partly* rejected by an engine cross-field
    rule still emits its valid subset as a normal sweep line, with an adjacent
    comment naming the omitted (rejected-in-isolation) values; only a field with
    NO surviving value is fully commented out.

    The emitted (uncommented) portion is guaranteed to parse and round-trip
    through ``load_study_config`` -> ``expand_grid`` with zero invalid configs.
    """
    config_model = _engine_config_model(engine)

    auto_lines: list[str] = []
    placeholder_lines: list[str] = []

    for section in _CURATED_SECTIONS:
        section_model = _section_model(config_model, section)
        if section_model is None:
            continue
        for field_name in section_model.model_fields:
            fqn = f"{engine}.{section}.{field_name}"
            try:
                values = resolve_auto_values(fqn)
            except ConfigError:
                placeholder_lines.append(
                    f"  # {fqn}: []  # supply explicit values (no natural set)"
                )
                continue
            # An auto-expandable type can still carry values an engine cross-field
            # rule rejects in isolation (e.g. transformers early_stopping=True
            # needs num_beams>1). Emit only the surviving values uncommented so the
            # skeleton round-trips with zero invalid configs; if none survive, the
            # whole field becomes a commented placeholder.
            rejected = _rule_rejected_values(engine, section, field_name, values)
            valid = [v for v in values if v not in rejected]
            if not valid:
                placeholder_lines.append(
                    f"  # {fqn}: {_yaml_scalar_list(values)}  "
                    f"# engine rules reject all of {_yaml_scalar_list(rejected)} on their own"
                )
                continue
            line = f"  {fqn}: {_yaml_scalar_list(valid)}"
            if rejected:
                line += (
                    f"  # omitted {_yaml_scalar_list(rejected)} (engine rules reject on their own)"
                )
            auto_lines.append(line)

    return _render_skeleton(engine, auto_lines, placeholder_lines)


def _rule_rejected_values(
    engine: str,
    section: str,
    field_name: str,
    values: list[Any],
) -> list[Any]:
    """Return the subset of *values* a minimal config rejects for this field.

    Validates each value in isolation against ``ExperimentConfig`` (a bare model
    + dataset scaffold), catching engine cross-field rules that the pure
    type-driven resolver cannot see. Used only by the skeleton generator to keep
    its output round-trip-clean.
    """
    # Deferred import: ExperimentConfig lives in the same layer (config/), but
    # importing at module top would create an import cycle (models -> grid ->
    # sweep_autoexpand).
    from llenergymeasure.config.models import ExperimentConfig

    rejected: list[Any] = []
    for value in values:
        candidate: dict[str, Any] = {
            "engine": engine,
            "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
            engine: {section: {field_name: value}},
        }
        try:
            ExperimentConfig(**candidate)
        except Exception:
            rejected.append(value)
    return rejected


# =============================================================================
# FQN -> pydantic FieldInfo mapping
# =============================================================================


def _resolve_field_info(fqn: str) -> FieldInfo:
    """Map an engine-field FQN onto a ``FieldInfo`` on the curated Config model.

    Expects ``<engine>.<section>.<field>`` where section is one of
    ``engine_params`` / ``sampling_params``. The engine is read from the key's
    own prefix. Raises ``ConfigError`` for a malformed FQN, an unknown engine
    prefix, an unknown section, or a field not declared on the curated model.
    """
    parts = fqn.split(".")
    if len(parts) != 3:
        raise ConfigError(
            f"Sweep key '{fqn}' is not an engine-field key. Auto-expansion only "
            f"applies to keys shaped '<engine>.<section>.<field>' "
            f"(section is one of {', '.join(_CURATED_SECTIONS)})."
        )

    engine, section, field_name = parts
    if section not in _CURATED_SECTIONS:
        raise ConfigError(
            f"Sweep key '{fqn}' has unknown section '{section}'. "
            f"Expected one of: {', '.join(_CURATED_SECTIONS)}."
        )

    section_model = _section_model(_engine_config_model(engine), section)
    if section_model is None:
        raise ConfigError(
            f"Sweep key '{fqn}': engine '{engine}' has no '{section}' section "
            f"on its curated Config model."
        )

    field_info = section_model.model_fields.get(field_name)
    if field_info is None:
        raise ConfigError(
            f"Sweep key '{fqn}': field '{field_name}' is not a curated field on "
            f"{engine}.{section}. Supply explicit values, or check the field name "
            f"against the engine's curated Config model."
        )
    return field_info


def _engine_config_model(engine: str) -> type[BaseModel]:
    """Return the curated ``Config`` model for *engine*, as a ``ConfigError``.

    Thin translation over the shared
    :func:`llenergymeasure.config.introspection.get_engine_config_model`
    registry: the auto-expander's contract raises ``ConfigError`` for an unknown
    engine, while the shared helper raises ``ValueError``.
    """
    try:
        return get_engine_config_model(engine)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _section_model(config_model: type[BaseModel], section: str) -> type[BaseModel] | None:
    """Resolve the ``<section>: SubModel | None`` annotation to its BaseModel."""
    field_info = config_model.model_fields.get(section)
    if field_info is None:
        return None
    for candidate in get_args(field_info.annotation) or (field_info.annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


# =============================================================================
# Numeric bucketing
# =============================================================================


def _bucket_numeric(fqn: str, annotation: Any, metadata: list[Any]) -> list[Any]:
    """Bucket a bounded numeric field into ``[low, mid, high]`` (see policy).

    Exclusive bounds (``gt``/``lt``) are nudged inward so every emitted value
    satisfies the constraint (inclusive bounds are used verbatim).
    """
    low_bound = _first_bound(metadata, ("ge", "gt"))
    high_bound = _first_bound(metadata, ("le", "lt"))

    if low_bound is None or high_bound is None:
        raise ConfigError(
            f"Sweep key '{fqn}' is numeric but lacks both a lower and an upper "
            f"bound, so it has no finite range to bucket. Supply explicit values, "
            f"e.g. '{fqn}: [<low>, <mid>, <high>]'."
        )

    bound_low, low_exclusive = low_bound
    bound_high, high_exclusive = high_bound
    span = bound_high - bound_low

    # A zero-span range with an exclusive endpoint admits no valid value: the
    # inward nudge would collapse to 0 and emit a value that violates the bound.
    if span == 0 and (low_exclusive or high_exclusive):
        raise ConfigError(
            f"Sweep key '{fqn}' has a zero-span range (lower == upper) with an "
            f"exclusive bound, so no value satisfies it. Supply explicit values, "
            f"e.g. '{fqn}: [<value>]'."
        )

    nudge = 1 if annotation is int else 1e-3 * span

    low = bound_low + nudge if low_exclusive else bound_low
    high = bound_high - nudge if high_exclusive else bound_high
    mid = (low + high) / 2

    buckets: list[Any]
    if annotation is int:
        buckets = [int(low), int(mid), int(high)]
    else:
        buckets = [float(low), float(mid), float(high)]

    # Deduplicate while preserving order (a tight range can collapse buckets).
    seen: list[Any] = []
    for value in buckets:
        if value not in seen:
            seen.append(value)
    return seen


def _first_bound(metadata: list[Any], names: tuple[str, ...]) -> tuple[float, bool] | None:
    """Return ``(value, is_exclusive)`` for the first present bound in *names*.

    ``is_exclusive`` is True for ``gt``/``lt`` (strict) and False for
    ``ge``/``le`` (inclusive).
    """
    for constraint in metadata:
        for name in names:
            value = getattr(constraint, name, None)
            if value is not None:
                return value, name in ("gt", "lt")
    return None


def _type_label(annotation: Any) -> str:
    """Human-readable type name for error messages."""
    return getattr(annotation, "__name__", str(annotation))


# =============================================================================
# Sweep-group handling
# =============================================================================


def _resolve_group(group_key: str, variants: list[Any]) -> list[Any]:
    """Resolve ``auto`` sentinels inside a dependent group's variant dicts.

    A variant ``{fqn: auto}`` expands into one variant per resolved value.
    Other variant fields pass through untouched. The group key itself is an
    abstract label (not a curated FQN), so it is never resolved.
    """
    resolved: list[Any] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ConfigError(
                f"Sweep group '{group_key}' mixes dicts and scalars; group "
                f"variants must all be dicts."
            )
        resolved.extend(_expand_variant_autos(variant))
    return resolved


def _expand_variant_autos(variant: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand any ``auto`` fields in a single group variant (Cartesian)."""
    auto_keys: list[str] = []
    base: dict[str, Any] = {}
    for k, v in variant.items():
        if v == AUTO_SENTINEL:
            auto_keys.append(k)
        else:
            base[k] = v

    if not auto_keys:
        return [variant]

    expanded = [base]
    for field_fqn in auto_keys:
        values = resolve_auto_values(field_fqn)
        expanded = [{**partial, field_fqn: value} for partial in expanded for value in values]
    return expanded


# =============================================================================
# Skeleton rendering
# =============================================================================


def _yaml_scalar_list(values: list[Any]) -> str:
    """Render a list of scalars as a YAML flow sequence (``[a, b, c]``)."""
    return "[" + ", ".join(_yaml_scalar(v) for v in values) + "]"


def _yaml_scalar(value: Any) -> str:
    """Render a single scalar as YAML (bools lowercase, strings quoted)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _render_skeleton(
    engine: str,
    auto_lines: list[str],
    placeholder_lines: list[str],
) -> str:
    """Assemble the final skeleton YAML text from resolved sweep lines."""
    header = [
        "# Auto-generated study skeleton - TRIM ME.",
        f"# Engine: {engine}. Every auto-expandable curated knob is pre-expanded",
        "# below with its natural value set. This is a STARTING POINT: delete the",
        "# axes you do not want, narrow the value lists, and fill in the task",
        "# placeholders (model + dataset) before running.",
        "#",
        "# Generated by `llem study build-skeleton`.",
    ]

    scaffold = [
        f"study_name: {engine}-skeleton",
        f"engine: {engine}",
        "task:",
        "  model: <MODEL>  # e.g. gpt2 (replace before running)",
        "  dataset:",
        "    source: <DATASET>  # e.g. arc (replace before running)",
        "    n_prompts: 10",
    ]

    sweep_block = ["sweep:"]
    if auto_lines:
        sweep_block.extend(auto_lines)
    else:
        sweep_block.append("  # (no auto-expandable curated knobs for this engine)")
    if placeholder_lines:
        sweep_block.append("  # --- not auto-expandable (supply explicit values) ---")
        sweep_block.extend(placeholder_lines)

    return "\n".join([*header, "", *scaffold, "", *sweep_block, ""])
