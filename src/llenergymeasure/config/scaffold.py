"""Render a study YAML file from the generated per-engine config models.

Backs ``llem study init``. The emitted file is a complete, explicit study
specification: what you read is what runs. All assistance happens here, at
authoring time - the tuning fields, their valid ranges, and their defaults are
read straight off the generated Pydantic config models (the same models
``ExperimentConfig`` validates against), so the scaffold never drifts from what
the engine actually accepts.

Three modes:

- bare (``defaults=False``): the tuning fields are present but commented out,
  each annotated with its type, range, and default. Only ``model`` and
  ``engine`` are uncommented, so the file is immediately runnable and the user
  opts into fields by uncommenting them.
- defaults (``defaults=True``): the same fields are uncommented and pinned to
  their model defaults - a reproducible baseline study.
- bounds (:func:`render_study_bounds`): every field with a derivable finite
  range becomes an explicit value list under a ``sweep:`` block - a maximal
  grid the user prunes before running. Fields with no derivable range stay
  pinned at their defaults. Value derivation (series policies, curated
  overrides, known-rejected filtering) lives in
  :mod:`llenergymeasure.config.series`.

The only difference between bare and defaults is the comment prefix on the
field lines, so an uncommented bare field lands on exactly the value the
defaults mode would emit. All modes honour a curated ``study_default``
override from the engine's optional ``study_overrides.yaml``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from llenergymeasure.config.engine_rules.loader import EngineRulesLoader, Rule
from llenergymeasure.config.introspection import get_engine_config_model
from llenergymeasure.config.series import (
    FieldOverride,
    StudyOverridesError,
    derive_series,
    load_study_overrides,
)
from llenergymeasure.config.ssot import Engine

__all__ = ["all_engine_names", "render_study_bounds", "render_study_scaffold"]

# Sub-sections of every engine's generated Config, in emission order.
_SUB_SECTIONS = ("engine_params", "sampling_params")

# YAML plain-scalar safe set: emit bare, else double-quote via json.dumps.
_PLAIN_SCALAR = re.compile(r"^[A-Za-z0-9_./-]+$")
_YAML_RESERVED = frozenset({"true", "false", "null", "yes", "no", "on", "off", "~"})


def all_engine_names() -> list[str]:
    """Return every engine name in declaration order (transformers, vllm, tensorrt)."""
    return [e.value for e in Engine]


def render_study_scaffold(
    model: str,
    engines: Sequence[str],
    *,
    defaults: bool,
    overrides_root: Path | None = None,
) -> str:
    """Return the study YAML text for a model across one or more engines.

    Args:
        model: HuggingFace model id or local path.
        engines: Engine names to scaffold, in emission order. One entry produces
            a single-engine study; several produce one study with an experiment
            per engine.
        defaults: False emits commented tuning fields (structure only); True
            emits them uncommented at their model defaults (runnable baseline).
        overrides_root: Directory holding per-engine ``study_overrides.yaml``
            files; defaults to the shipped package data. Only ``study_default``
            entries affect this renderer.

    Returns:
        Deterministic YAML text (stable field order, no timestamps), ending in a
        newline. Identical arguments always produce byte-identical output.
    """
    lines: list[str] = []
    lines.extend(_header(engines, defaults=defaults))
    lines.append("")
    lines.append(f"study_name: {_yaml_scalar(_study_name(model, engines))}")
    lines.append("")
    lines.append("# One experiment per engine. task.model is repeated in each so every")
    lines.append("# experiment is a complete, self-contained specification.")
    lines.append("experiments:")
    for engine in engines:
        lines.extend(
            _experiment_block(engine, model, defaults=defaults, overrides_root=overrides_root)
        )
    return "\n".join(lines) + "\n"


def render_study_bounds(
    model: str,
    engines: Sequence[str],
    *,
    overrides_root: Path | None = None,
) -> str:
    """Return the bounds-mode study YAML: pinned fields plus a maximal sweep.

    Every field with a derivable finite range (enum members, bools, numerics
    with both bounds on the model, or a curated ``sweep_values`` override)
    becomes an explicit value list under ``sweep:``, keyed by its full dotted
    path so the grid expander consumes it as-is. All other fields stay pinned
    at their defaults in a top-level engine section. Values the shipped rules
    corpus already knows the engine rejects are dropped before emission.

    Args:
        model: HuggingFace model id or local path.
        engines: Engine names to scaffold, in emission order.
        overrides_root: Directory holding per-engine ``study_overrides.yaml``
            files; defaults to the shipped package data.

    Returns:
        Deterministic YAML text ending in a newline. Identical arguments always
        produce byte-identical output.
    """
    rules_loader = EngineRulesLoader()
    plans = [_engine_bounds_plan(e, rules_loader, overrides_root) for e in engines]
    if len(plans) > 1:
        for plan in plans:
            if not plan.series:
                raise ValueError(
                    f"Engine {plan.engine!r} has no sweepable fields; scaffold it alone "
                    f"with `llem study init -e {plan.engine} --bounds`."
                )

    lines: list[str] = list(_bounds_header(engines))
    lines.append("")
    lines.append(f"study_name: {_yaml_scalar(_study_name(model, engines))}")
    lines.append("")
    lines.append("task:")
    lines.append(f"  model: {_yaml_scalar(model)}")
    if len(engines) == 1:
        lines.append("")
        lines.append(f"engine: {engines[0]}")
    lines.append("")
    lines.append("# Pinned fields - no derivable value range, so each holds one value for")
    lines.append("# every grid point. Edit freely, or move a field into the sweep: block")
    lines.append("# as `<engine>.<section>.<field>: [v1, v2]` to sweep it.")
    for plan in plans:
        lines.append(f"{plan.engine}:")
        for sub_name, pinned_fields in plan.pinned:
            lines.append(f"  {sub_name}:")
            for field_name, field_info, value in pinned_fields:
                lines.append(
                    _field_line(field_name, field_info, defaults=True, value=value, indent="    ")
                )
    lines.append("")
    lines.append("# One list per sweepable field, covering its full valid range (values the")
    lines.append("# engine is known to reject are already dropped). The study is the")
    lines.append("# Cartesian product of these lists - prune each list down to what you")
    lines.append("# want to study before running.")
    lines.append("sweep:")
    for plan in plans:
        for sweep_key, field_info, values in plan.series:
            lines.append(f"  {sweep_key}: {_yaml_scalar(values)}  # {_annotation(field_info)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------


def _header(engines: Sequence[str], *, defaults: bool) -> list[str]:
    """Return the leading comment block explaining the file and how to run it."""
    scope = engines[0] if len(engines) == 1 else "all engines"
    lines = [
        f"# llem study file - {scope}",
        "#",
        "# Generated by `llem study init`. This file is the complete, explicit",
        "# specification of a study: what you read here is exactly what runs.",
        "# Edit it freely, then run:",
        "#",
        "#     llem run <this-file>",
        "#",
    ]
    if defaults:
        lines += [
            "# Every tuning field below is set to its default value, giving a",
            "# reproducible baseline study. Change any value to explore its effect.",
        ]
    else:
        lines += [
            "# The tuning fields below are commented out. Uncomment a line to",
            "# include that field; each line shows its type, valid range, and",
            "# default. Fields left commented use the engine's own default.",
        ]
    return lines


def _bounds_header(engines: Sequence[str]) -> list[str]:
    """Return the leading comment block for the bounds mode."""
    scope = engines[0] if len(engines) == 1 else "all engines"
    return [
        f"# llem study file - {scope} (bounds mode)",
        "#",
        "# Generated by `llem study init --bounds`. This file is the complete,",
        "# explicit specification of a study: what you read here is exactly what",
        "# runs. Edit it freely, then run:",
        "#",
        "#     llem run <this-file>",
        "#",
        "# Every field with a derivable finite range appears under sweep: as an",
        "# explicit value list; everything else stays pinned in the engine",
        "# section. Together the lists form a maximal grid - prune before you run.",
    ]


def _experiment_block(
    engine: str, model: str, *, defaults: bool, overrides_root: Path | None
) -> list[str]:
    """Return one ``experiments:`` list entry for a single engine."""
    overrides = load_study_overrides(engine, overrides_root)
    sub_models = _sub_models(engine)
    _check_override_paths(engine, overrides, sub_models)
    lines = [
        f"  - engine: {engine}",
        "    task:",
        f"      model: {_yaml_scalar(model)}",
        f"    {engine}:",
    ]
    for sub_name in _SUB_SECTIONS:
        sub_model = sub_models.get(sub_name)
        if sub_model is None:
            continue
        lines.append(f"      {sub_name}:")
        for field_name, field_info in sub_model.model_fields.items():
            value = _pinned_value(field_info, overrides.get(f"{sub_name}.{field_name}"))
            lines.append(_field_line(field_name, field_info, defaults=defaults, value=value))
    return lines


def _field_line(
    field_name: str,
    field_info: FieldInfo,
    *,
    defaults: bool,
    value: Any,
    indent: str = "        ",
) -> str:
    """Render one field line: ``<name>: <value>  # <type> <bounds>, default <v>``.

    In bare mode the field is commented out; in defaults mode it is live. Both
    carry the same value and annotation, so uncommenting a bare line yields the
    defaults-mode line verbatim.
    """
    prefix = indent if defaults else f"{indent}# "
    return f"{prefix}{field_name}: {_yaml_scalar(value)}  # {_annotation(field_info)}"


def _pinned_value(field_info: FieldInfo, override: FieldOverride | None) -> Any:
    """The value a non-swept field is pinned to: curated study_default, else model default."""
    if override is not None and override.has_study_default:
        return override.study_default
    return field_info.default


# ---------------------------------------------------------------------------
# Bounds-mode planning (which fields sweep, which stay pinned)
# ---------------------------------------------------------------------------


@dataclass
class _EngineBoundsPlan:
    """Partition of one engine's fields into pinned lines and sweep series."""

    engine: str
    # Per sub-section: (sub_name, [(field_name, field_info, pinned_value), ...])
    pinned: list[tuple[str, list[tuple[str, FieldInfo, Any]]]]
    # (full dotted sweep key, field_info, series values)
    series: list[tuple[str, FieldInfo, list[Any]]]


def _engine_bounds_plan(
    engine: str, rules_loader: EngineRulesLoader, overrides_root: Path | None
) -> _EngineBoundsPlan:
    """Derive the pinned/series partition for one engine."""
    overrides = load_study_overrides(engine, overrides_root)
    sub_models = _sub_models(engine)
    _check_override_paths(engine, overrides, sub_models)
    rules: tuple[Rule, ...]
    try:
        rules = rules_loader.load_rules(engine).rules
    except FileNotFoundError:
        rules = ()
    pinned: list[tuple[str, list[tuple[str, FieldInfo, Any]]]] = []
    series: list[tuple[str, FieldInfo, list[Any]]] = []
    for sub_name in _SUB_SECTIONS:
        sub_model = sub_models.get(sub_name)
        if sub_model is None:
            continue
        sub_pinned: list[tuple[str, FieldInfo, Any]] = []
        for field_name, field_info in sub_model.model_fields.items():
            override = overrides.get(f"{sub_name}.{field_name}")
            sweep_key = f"{engine}.{sub_name}.{field_name}"
            values = derive_series(field_info, field_path=sweep_key, override=override, rules=rules)
            if values is None:
                sub_pinned.append((field_name, field_info, _pinned_value(field_info, override)))
            else:
                series.append((sweep_key, field_info, values))
        pinned.append((sub_name, sub_pinned))
    return _EngineBoundsPlan(engine=engine, pinned=pinned, series=series)


def _check_override_paths(
    engine: str, overrides: dict[str, FieldOverride], sub_models: dict[str, type[BaseModel]]
) -> None:
    """Fail loud when an override names a field path the engine model lacks."""
    known = {
        f"{sub_name}.{field_name}"
        for sub_name, sub_model in sub_models.items()
        for field_name in sub_model.model_fields
    }
    unknown = set(overrides) - known
    if unknown:
        raise StudyOverridesError(
            f"study_overrides.yaml for {engine!r} names unknown field path(s): {sorted(unknown)}."
        )


# ---------------------------------------------------------------------------
# Annotation rendering (type + bounds + default, from the typed model only)
# ---------------------------------------------------------------------------


def _annotation(field_info: FieldInfo) -> str:
    """Return ``<type> <bounds>, default <v>`` derived from the field alone."""
    inner = _unwrap_optional(field_info.annotation)
    bounds = _bounds(field_info)
    type_str = _type_str(inner)
    prefix = f"{type_str} {bounds}" if bounds else type_str
    return f"{prefix}, default {_yaml_scalar(field_info.default)}"


def _type_str(inner: Any) -> str:
    """Human-readable type label for an unwrapped annotation."""
    if get_origin(inner) is Literal:
        members = ", ".join(_yaml_scalar(m) for m in get_args(inner))
        return f"one of [{members}]"
    if inner is Any:
        return "any"
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return "object"
    if inner is bool:
        return "bool"
    if inner is int:
        return "int"
    if inner is float:
        return "float"
    if inner is str:
        return "str"
    return "value"


def _bounds(field_info: FieldInfo) -> str:
    """Return a bounds string like ``>0.0 <=1.0`` from ge/gt/le/lt metadata."""
    parts: list[str] = []
    for op, attr in ((">", "gt"), (">=", "ge"), ("<", "lt"), ("<=", "le")):
        for constraint in field_info.metadata:
            value = getattr(constraint, attr, None)
            if value is not None:
                parts.append(f"{op}{value}")
    return " ".join(parts)


def _unwrap_optional(annotation: Any) -> Any:
    """Strip a ``| None`` wrapper, returning the inner type (or the annotation)."""
    args = get_args(annotation)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if inner:
            return inner[0]
    return annotation


# ---------------------------------------------------------------------------
# Model introspection + scalar rendering
# ---------------------------------------------------------------------------


def _sub_models(engine: str) -> dict[str, type[BaseModel]]:
    """Return an engine's ``engine_params`` / ``sampling_params`` sub-model classes."""
    config = get_engine_config_model(engine)
    out: dict[str, type[BaseModel]] = {}
    for sub_name in _SUB_SECTIONS:
        field = config.model_fields.get(sub_name)
        if field is None:
            continue
        for candidate in get_args(field.annotation) or (field.annotation,):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                out[sub_name] = candidate
                break
    return out


def _yaml_scalar(value: Any) -> str:
    """Render a Python value as a YAML scalar (bool before int; quote unsafe strings)."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        if value and _PLAIN_SCALAR.match(value) and value.lower() not in _YAML_RESERVED:
            return value
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    return repr(value)


def _study_name(model: str, engines: Sequence[str]) -> str:
    """Deterministic study name: model tail slug, plus engine when single-engine."""
    tail = model.split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", tail).strip("-") or "study"
    if len(engines) == 1:
        return f"{slug}-{engines[0]}"
    return slug
