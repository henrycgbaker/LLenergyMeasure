"""Numeric sweep-axis idioms: mapping notations that expand to value lists.

A sweep axis value is normally an explicit YAML list. Three mapping idioms
are accepted as alternative notation for such a list. Each idiom is a
deterministic, versioned notation whose expansion is fixed at load time -
downstream consumers (dedup, rule pruning, counts) only ever see plain lists:

- ``{min: a, max: b, num: n}`` - n evenly spaced values from a to b inclusive.
- ``{log: {min: a, max: b, num: n}}`` - n log-spaced values, a > 0, inclusive
  endpoints.
- ``{pow2: {min: a, max: b}}`` - ascending powers of two within [a, b].

Number typing: values are ints when all bounds are ints and every produced
value is integral; otherwise floats. Float values are rounded to
``_SIGNIFICANT_DIGITS`` significant digits to avoid binary float noise
(0.30000000000000004 -> 0.3). Duplicates after rounding collapse to unique
values preserving order.

Any mapping that is not one of the three idiom shapes raises ``ValueError``.
"""

from __future__ import annotations

import math
from typing import Any

_SIGNIFICANT_DIGITS = 10

_IDIOM_SUMMARY = (
    "valid range shorthands are {min: a, max: b, num: n} (evenly spaced), "
    "{log: {min: a, max: b, num: n}} (log-spaced) and "
    "{pow2: {min: a, max: b}} (powers of two)"
)


def expand_axis_idiom(mapping: dict[str, Any]) -> list[int] | list[float]:
    """Expand a mapping-valued sweep axis into a plain list of numbers.

    Raises ``ValueError`` if the mapping is not exactly one of the three
    idiom shapes, or if an idiom's fields fail validation.
    """
    keys = set(mapping)
    if keys == {"min", "max", "num"}:
        return _expand_linear(mapping)
    if keys == {"log"}:
        return _expand_log(_inner_mapping("log", mapping["log"], {"min", "max", "num"}))
    if keys == {"pow2"}:
        return _expand_pow2(_inner_mapping("pow2", mapping["pow2"], {"min", "max"}))
    raise ValueError(
        f"mapping with keys {sorted(str(k) for k in keys)} is not a recognised range "
        f"shorthand; {_IDIOM_SUMMARY}. Literal mapping values cannot be swept as an axis - "
        "set them in the base config, or sweep them via a group entry "
        "(list of dicts)."
    )


# =============================================================================
# Idiom expansion
# =============================================================================


def _expand_linear(mapping: dict[str, Any]) -> list[int] | list[float]:
    """``{min, max, num}`` -> n evenly spaced values, endpoints inclusive."""
    lo, hi = _bounds("min/max/num", mapping)
    n = _num("min/max/num", mapping["num"])
    # Integer path: exact divisibility check keeps typing deterministic.
    if isinstance(lo, int) and isinstance(hi, int) and (hi - lo) % (n - 1) == 0:
        step = (hi - lo) // (n - 1)
        return _dedupe([lo + step * i for i in range(n)])
    values = [lo + (hi - lo) * i / (n - 1) for i in range(1, n - 1)]
    return _dedupe([float(lo), *(_round_sig(v) for v in values), float(hi)])


def _expand_log(mapping: dict[str, Any]) -> list[int] | list[float]:
    """``{log: {min, max, num}}`` -> n log-spaced values, endpoints inclusive."""
    lo, hi = _bounds("log", mapping)
    if lo <= 0:
        raise ValueError(f"log range shorthand requires min > 0, got min={lo}")
    n = _num("log", mapping["num"])
    ratio = hi / lo
    values = [lo * ratio ** (i / (n - 1)) for i in range(1, n - 1)]
    rounded = [float(lo), *(_round_sig(v) for v in values), float(hi)]
    if isinstance(lo, int) and isinstance(hi, int) and all(v.is_integer() for v in rounded):
        return _dedupe([int(v) for v in rounded])
    return _dedupe(rounded)


def _expand_pow2(mapping: dict[str, Any]) -> list[int] | list[float]:
    """``{pow2: {min, max}}`` -> ascending powers of two within [min, max]."""
    lo, hi = _bounds("pow2", mapping)
    if lo <= 0:
        raise ValueError(f"pow2 range shorthand requires min > 0, got min={lo}")
    # Scan integer exponents with a +-1 safety margin against log2 rounding.
    # Negative exponents (0.5, 0.25, ...) are exact binary floats.
    k_lo = math.floor(math.log2(lo)) - 1
    k_hi = math.ceil(math.log2(hi)) + 1
    powers: list[int | float] = [
        p for k in range(k_lo, k_hi + 1) if lo <= (p := 2**k if k >= 0 else 2.0**k) <= hi
    ]
    if not powers:
        raise ValueError(f"pow2 range shorthand: no power of two lies within [{lo}, {hi}]")
    if all(isinstance(p, int) for p in powers):
        return powers  # type: ignore[return-value]  # narrowed by the all() check
    return [float(p) for p in powers]


# =============================================================================
# Field validation
# =============================================================================


def _inner_mapping(idiom: str, value: Any, expected_keys: set[str]) -> dict[str, Any]:
    """Validate the nested mapping of a wrapped idiom (log, pow2)."""
    keys_desc = ", ".join(sorted(expected_keys))
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(
            f"{idiom} range shorthand requires a nested mapping with exactly the keys "
            f"{{{keys_desc}}}, got {value!r}; {_IDIOM_SUMMARY}"
        )
    return value


def _number(idiom: str, name: str, value: Any) -> int | float:
    """Require a finite int or float (bool excluded)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    raise ValueError(f"{idiom} range shorthand: '{name}' must be a finite number, got {value!r}")


def _bounds(idiom: str, mapping: dict[str, Any]) -> tuple[int | float, int | float]:
    """Require numeric min/max with min <= max."""
    lo = _number(idiom, "min", mapping["min"])
    hi = _number(idiom, "max", mapping["max"])
    if lo > hi:
        raise ValueError(f"{idiom} range shorthand: min ({lo}) must not exceed max ({hi})")
    return lo, hi


def _num(idiom: str, value: Any) -> int:
    """Require an integer point count >= 2."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 2:
        return value
    raise ValueError(f"{idiom} range shorthand: 'num' must be an integer >= 2, got {value!r}")


# =============================================================================
# Rounding and dedup
# =============================================================================


def _round_sig(value: float) -> float:
    """Round to ``_SIGNIFICANT_DIGITS`` significant digits (float-noise guard)."""
    if value == 0.0:
        return 0.0
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def _dedupe(values: list[Any]) -> list[Any]:
    """Collapse duplicates (e.g. after rounding) to unique values, order preserved."""
    return list(dict.fromkeys(values))
