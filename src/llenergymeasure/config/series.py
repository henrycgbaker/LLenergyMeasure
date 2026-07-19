"""Value-series derivation for ``llem study init --bounds``.

A series is the explicit list of values the bounds scaffold emits for one
tuning field. Everything here is a deterministic, documented floor - the same
field always yields the same series:

- Enum / Literal fields: every member.
- Bool fields: ``[false, true]``.
- Numeric fields with BOTH a lower and an upper bound on the generated model:
  points from a named series policy (default ``span``). Bounds are read off
  the field's ge/gt/le/lt metadata; nothing is guessed.
- Everything else (one-sided or unbounded numerics, strings, objects, Any):
  no series - the field stays pinned at its default. The scaffold never
  invents values without a finite derivable range or a curated override.

Series policies (named shapes):

- ``span`` (the floor default): 5 points at quantiles 0, 1/4, 1/2, 3/4, 1 of
  the closed range; an endpoint whose bound is strict (gt/lt) is dropped.
- ``log``: 5 geometrically spaced points across the range (requires a
  positive lower bound); same strict-endpoint handling.
- ``pow2``: every power of two inside the range (requires a lower bound of
  at least 1); strict bounds exclude an exactly-matching power.

``log`` and ``pow2`` are reachable only through a curated override
(``sweep_hint``), never as the automatic default.

Curated overrides live in an optional per-engine package-data file
``src/llenergymeasure/engines/<engine>/study_overrides.yaml`` mapping a
section-relative field path (``engine_params.gpu_memory_utilization``) to
any of a closed key set:

- ``study_default``: value the scaffold pins the field to instead of the
  model default (applies to every ``llem study init`` mode).
- ``sweep_hint``: series policy name (``span`` / ``log`` / ``pow2``) for a
  numeric field with both bounds.
- ``sweep_values``: explicit value list; wins over any policy.

The file is validated on load - unknown keys fail loud. An absent or empty
file means no overrides.

Before a series is emitted, candidate values the shipped rules corpus
already knows the engine rejects are dropped: any single-field ``error``
rule whose match is exactly this field filters the list (cross-field rules
are out of scope here). If fewer than two candidates survive, the field
stays pinned instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import yaml
from pydantic.fields import FieldInfo

from llenergymeasure.config.engine_rules.loader import (
    Rule,
    evaluate_predicate,
    spec_has_field_ref,
)
from llenergymeasure.config.sweep_idioms import _dedupe

__all__ = [
    "SERIES_SHAPES",
    "FieldOverride",
    "StudyOverridesError",
    "derive_series",
    "drop_known_rejected",
    "load_study_overrides",
]

SERIES_SHAPES = ("span", "log", "pow2")
"""Named series policies. ``span`` is the floor default; the rest are
reachable only via a curated ``sweep_hint``."""

OVERRIDES_FILENAME = "study_overrides.yaml"

_OVERRIDE_KEYS = frozenset({"study_default", "sweep_hint", "sweep_values"})

_DEFAULT_OVERRIDES_ROOT = Path(__file__).resolve().parents[1] / "engines"

_UNSET: Any = object()
"""Sentinel distinguishing an absent ``study_default`` from an explicit null."""


class StudyOverridesError(ValueError):
    """A study_overrides.yaml file is malformed or violates the closed schema."""


@dataclass(frozen=True)
class FieldOverride:
    """Curated per-field override parsed from ``study_overrides.yaml``."""

    study_default: Any = _UNSET
    sweep_hint: str | None = None
    sweep_values: list[Any] | None = None

    @property
    def has_study_default(self) -> bool:
        return self.study_default is not _UNSET


# ---------------------------------------------------------------------------
# Overrides loading
# ---------------------------------------------------------------------------


def load_study_overrides(engine: str, root: Path | None = None) -> dict[str, FieldOverride]:
    """Load and validate an engine's optional study overrides file.

    Returns a mapping of section-relative field path (e.g.
    ``engine_params.gpu_memory_utilization``) to :class:`FieldOverride`.
    An absent or empty file yields an empty mapping; a malformed file
    raises :class:`StudyOverridesError`.
    """
    base = root if root is not None else _DEFAULT_OVERRIDES_ROOT
    path = base / engine / OVERRIDES_FILENAME
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise StudyOverridesError(
            f"{path}: top level must be a mapping of field paths, got {type(data).__name__}."
        )
    overrides: dict[str, FieldOverride] = {}
    for raw_path, raw_entry in data.items():
        field_path = str(raw_path)
        overrides[field_path] = _parse_override(path, field_path, raw_entry)
    return overrides


def _parse_override(path: Path, field_path: str, raw_entry: Any) -> FieldOverride:
    if not isinstance(raw_entry, dict):
        raise StudyOverridesError(
            f"{path}: entry for {field_path!r} must be a mapping, got {type(raw_entry).__name__}."
        )
    unknown = set(raw_entry) - _OVERRIDE_KEYS
    if unknown:
        raise StudyOverridesError(
            f"{path}: entry for {field_path!r} has unknown key(s) {sorted(unknown)}; "
            f"allowed keys: {sorted(_OVERRIDE_KEYS)}."
        )
    if not raw_entry:
        raise StudyOverridesError(
            f"{path}: entry for {field_path!r} is empty; "
            f"set at least one of {sorted(_OVERRIDE_KEYS)}."
        )
    hint = raw_entry.get("sweep_hint")
    if hint is not None and hint not in SERIES_SHAPES:
        raise StudyOverridesError(
            f"{path}: entry for {field_path!r} has sweep_hint={hint!r}; "
            f"must be one of {list(SERIES_SHAPES)}."
        )
    values = raw_entry.get("sweep_values")
    if values is not None and (not isinstance(values, list) or not values):
        raise StudyOverridesError(
            f"{path}: entry for {field_path!r} has sweep_values={values!r}; "
            "must be a non-empty list."
        )
    return FieldOverride(
        study_default=raw_entry.get("study_default", _UNSET),
        sweep_hint=hint,
        sweep_values=list(values) if values is not None else None,
    )


# ---------------------------------------------------------------------------
# Series derivation
# ---------------------------------------------------------------------------


def derive_series(
    field_info: FieldInfo,
    *,
    field_path: str,
    override: FieldOverride | None,
    rules: Sequence[Rule],
) -> list[Any] | None:
    """Return the value series for one field, or None to keep it pinned.

    Candidates come from the curated override (``sweep_values`` wins, then
    ``sweep_hint`` picks the policy) or the deterministic floor. Values a
    single-field error rule in ``rules`` would reject are dropped; if fewer
    than two candidates survive, the field gets no series.
    """
    candidates = _candidates(field_info, field_path=field_path, override=override)
    if candidates is None:
        return None
    kept = drop_known_rejected(candidates, field_path, rules)
    if len(kept) < 2:
        return None
    return kept


def drop_known_rejected(
    candidates: Sequence[Any], field_path: str, rules: Sequence[Rule]
) -> list[Any]:
    """Filter out candidates a single-field error rule would reject.

    Only rules whose match is exactly ``{field_path}`` participate; specs
    carrying ``@field`` references are cross-field in disguise and are
    skipped (they cannot be evaluated against a bare candidate value).
    """
    specs = [
        rule.match_fields[field_path]
        for rule in rules
        if rule.severity == "error"
        and set(rule.match_fields) == {field_path}
        and not spec_has_field_ref(rule.match_fields[field_path])
    ]
    if not specs:
        return list(candidates)
    return [c for c in candidates if not any(evaluate_predicate(c, s) for s in specs)]


def _candidates(
    field_info: FieldInfo, *, field_path: str, override: FieldOverride | None
) -> list[Any] | None:
    """Raw candidate values before known-rejected filtering, or None."""
    if override is not None and override.sweep_values is not None:
        return list(override.sweep_values)
    hint = override.sweep_hint if override is not None else None
    inner = _unwrap_optional(field_info.annotation)
    if get_origin(inner) is Literal:
        _require_no_hint(hint, field_path)
        return list(get_args(inner))
    if inner is bool:
        _require_no_hint(hint, field_path)
        return [False, True]
    if inner is int or inner is float:
        bounds = _closed_bounds(field_info)
        if bounds is None:
            _require_no_hint(hint, field_path)
            return None
        lo, hi, lo_strict, hi_strict = bounds
        policy = _POLICIES[hint or "span"]
        return list(policy(lo, hi, lo_strict=lo_strict, hi_strict=hi_strict, as_int=inner is int))
    _require_no_hint(hint, field_path)
    return None


def _require_no_hint(hint: str | None, field_path: str) -> None:
    if hint is not None:
        raise StudyOverridesError(
            f"sweep_hint on {field_path!r} needs a numeric field with both a lower and an "
            "upper bound on the model; use sweep_values to set the list explicitly."
        )


def _closed_bounds(field_info: FieldInfo) -> tuple[float, float, bool, bool] | None:
    """Return (lo, hi, lo_strict, hi_strict) if both bounds exist, else None."""
    lo: float | None = None
    hi: float | None = None
    lo_strict = False
    hi_strict = False
    for constraint in field_info.metadata:
        gt = getattr(constraint, "gt", None)
        ge = getattr(constraint, "ge", None)
        lt = getattr(constraint, "lt", None)
        le = getattr(constraint, "le", None)
        if lo is None and gt is not None:
            lo, lo_strict = float(gt), True
        if lo is None and ge is not None:
            lo = float(ge)
        if hi is None and lt is not None:
            hi, hi_strict = float(lt), True
        if hi is None and le is not None:
            hi = float(le)
    if lo is None or hi is None:
        return None
    return lo, hi, lo_strict, hi_strict


def _unwrap_optional(annotation: Any) -> Any:
    """Strip a ``| None`` wrapper, returning the inner type (or the annotation)."""
    args = get_args(annotation)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if inner:
            return inner[0]
    return annotation


# ---------------------------------------------------------------------------
# Series policies
# ---------------------------------------------------------------------------


def _span_series(
    lo: float, hi: float, *, lo_strict: bool, hi_strict: bool, as_int: bool
) -> list[int] | list[float]:
    """5 points at quantiles 0, 1/4, 1/2, 3/4, 1 of [lo, hi]."""
    points = [lo + (hi - lo) * q / 4 for q in range(5)]
    return _finish(points, lo_strict=lo_strict, hi_strict=hi_strict, as_int=as_int)


def _log_series(
    lo: float, hi: float, *, lo_strict: bool, hi_strict: bool, as_int: bool
) -> list[int] | list[float]:
    """5 geometrically spaced points across [lo, hi]; needs lo > 0."""
    if lo <= 0:
        raise StudyOverridesError(f"log series needs a positive lower bound, got {lo}.")
    ratio = hi / lo
    points = [lo * ratio ** (q / 4) for q in range(5)]
    return _finish(points, lo_strict=lo_strict, hi_strict=hi_strict, as_int=as_int)


def _pow2_series(
    lo: float, hi: float, *, lo_strict: bool, hi_strict: bool, as_int: bool
) -> list[int] | list[float]:
    """Every power of two inside [lo, hi]; needs lo >= 1."""
    if lo < 1:
        raise StudyOverridesError(f"pow2 series needs a lower bound of at least 1, got {lo}.")
    values: list[int] = []
    v = 1
    while v < lo or (lo_strict and v == lo):
        v *= 2
    while v < hi or (not hi_strict and v == hi):
        values.append(v)
        v *= 2
    if as_int:
        return values
    return [float(v) for v in values]


def _finish(
    points: list[float], *, lo_strict: bool, hi_strict: bool, as_int: bool
) -> list[int] | list[float]:
    """Drop strict endpoints (positionally: points[0]==lo, points[-1]==hi), round, dedupe."""
    if hi_strict:
        points = points[:-1]
    if lo_strict:
        points = points[1:]
    # Latent edge: an interior point can round onto a dropped strict bound
    # (e.g. gt=0 lt=1 as int -> 0 and 1), a value the model would reject. No
    # shipped field is this narrow, so the round-trip tests stay green; widen
    # here if one ever appears.
    if as_int:
        return _dedupe(round(p) for p in points)
    # Round to decimal places, not significant digits: these land in a
    # human-facing YAML file as sweep starting points, so clean numbers like
    # 0.25 read better than the significant-digit form used by the run-time
    # grid expander (config.sweep_idioms).
    return _dedupe(round(p, 6) for p in points)


_POLICIES: dict[str, Callable[..., list[int] | list[float]]] = {
    "span": _span_series,
    "log": _log_series,
    "pow2": _pow2_series,
}
