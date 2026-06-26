"""Deterministic Tier-D class-3 candidate enumeration.

The future LLM Tier-D proposer localises constraints on *undeclared-semantic*
fields - "class 3" leaves: parameters that carry a real bound or relationship
stated only in semantics / docstrings, with no machine-readable constraint in
the schema and no mined corpus rule yet. This module computes that candidate
set deterministically from committed artifacts so Tier-D scope is grounded and
reproducible (the design doc's "173" figure has no committed backing).

Inputs (both committed, host-only, no engine import / GPU / LLM):

- ``schema.discovered.json`` - ``engine_params`` + ``sampling_params`` dicts of
  ``{leaf: {type, default, [enum/minimum/...], [$ref]}}`` plus a ``$defs`` map.
- ``rules.proposed.yaml`` - the mined corpus; each invariant's
  ``match.fields`` keys are dotted paths whose tail is a covered leaf name.

The recipe (see the BRIEF / comprehensive-invariant-mining design):

    class-3 candidates = engine_params + sampling_params LEAVES
      MINUS composite-object leaves ($ref to a non-enum $def - a nested config,
            not a scalar semantic leaf)
      MINUS leaves carrying a MACHINE-READABLE CONSTRAINT (class 1/2):
            enum / minimum / maximum / exclusiveMinimum / exclusiveMaximum /
            multipleOf on the leaf shape; OR a $ref to a $defs class that is an
            enum; OR a ``Literal[...]`` in the type string (tensorrt /
            transformers keep enums as Literal in the type, not lifted to keys)
      MINUS leaves already covered by a CORPUS rule (class 4 - the leaf is the
            tail of any match.fields path)
      MINUS the VALUE-TYPE NOISE FILTER: bool leaves (a 2-value field has no
            localizable bound) and free-form str / PathLike / Any / unknown
            leaves (identifiers / paths / opaque - not semantic constraints)
    KEEP numeric (int/float/integer/number) leaves with no machine-readable
    bound - the genuine class-3 surface.

Two type vocabularies coexist and are both handled: vllm sampling types are
JSON-native (number/integer/boolean) while its engine_params are python-spelled
(str/int/float/bool); tensorrt is pydantic-native (python spellings, no lifted
constraint keys); transformers uses python spellings plus the sentinel
``"unknown"`` (treated as opaque -> dropped).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._current import current_outputs_dir  # noqa: E402

# Machine-readable constraint keys that promote a leaf to class 1/2.
_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {"enum", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
)

# Numeric base types across both vocabularies (python-spelled + JSON-native).
_NUMERIC_BASES: frozenset[str] = frozenset({"int", "integer", "float", "number"})

# Boolean base types across both vocabularies.
_BOOL_BASES: frozenset[str] = frozenset({"bool", "boolean"})

_SECTIONS: tuple[str, ...] = ("engine_params", "sampling_params")

_UNION_SPLIT = re.compile(r"\s*\|\s*")
_HEAD_TOKEN = re.compile(r"[\[\s]")


@dataclass(frozen=True)
class Class3Candidate:
    """A single undeclared-semantic (class-3) candidate leaf."""

    engine: str
    section: str  # "engine_params" | "sampling_params"
    leaf: str
    type: str | None
    default: Any
    reason_included: str


@dataclass(frozen=True)
class EngineStageCounts:
    """Per-stage tally of how the leaf surface was whittled to candidates."""

    total_leaves: int
    excluded_composite_ref: int
    excluded_schema_constrained: int
    excluded_corpus_covered: int
    excluded_bool: int
    excluded_freeform: int
    included: int


def _base_types(type_str: str | None) -> set[str]:
    """Return the set of non-None base type tokens in a (possibly union) type.

    ``"int | None"`` -> ``{"int"}``; ``"list[int] | None"`` -> ``{"list"}``;
    ``"str | list[str] | None"`` -> ``{"str", "list"}``. The head token before
    the first ``[`` or whitespace is taken so generics collapse to their
    container and ``Literal[...]`` collapses to ``Literal``.
    """
    if not type_str:
        return set()
    out: set[str] = set()
    for part in _UNION_SPLIT.split(type_str):
        part = part.strip()
        if part in ("None", "NoneType", ""):
            continue
        head = _HEAD_TOKEN.split(part, maxsplit=1)[0]
        out.add(head)
    return out


def _ref_def_name(ref: str | None) -> str | None:
    """Return the ``$defs`` class name a ``$ref`` points at, else None."""
    if not ref:
        return None
    return ref.split("/")[-1]


def _is_composite_ref(shape: dict[str, Any], defs: dict[str, Any]) -> bool:
    """True if the leaf is a ``$ref`` to a non-enum ``$def`` (a nested config).

    Such a leaf is a composite object, not a scalar semantic leaf, so it is not
    a Tier-D constraint target. A ``$ref`` to an enum ``$def`` is *not*
    composite - it is a machine-readable constraint handled separately.
    """
    name = _ref_def_name(shape.get("$ref"))
    if name is None:
        return False
    body = defs.get(name)
    return isinstance(body, dict) and "enum" not in body


def _is_schema_constrained(shape: dict[str, Any], defs: dict[str, Any]) -> bool:
    """True if the leaf already carries a machine-readable constraint (class 1/2).

    Covers: a constraint key on the shape; a ``Literal[...]`` in the type
    string; or a ``$ref`` to an enum ``$def``.
    """
    if any(key in shape for key in _CONSTRAINT_KEYS):
        return True
    type_str = shape.get("type")
    if isinstance(type_str, str) and "Literal[" in type_str:
        return True
    name = _ref_def_name(shape.get("$ref"))
    if name is not None:
        body = defs.get(name)
        if isinstance(body, dict) and "enum" in body:
            return True
    return False


def _corpus_covered_leaves(rules: dict[str, Any]) -> set[str]:
    """Return the set of leaf names covered by any corpus rule.

    Each invariant's ``match.fields`` is a dict whose keys are dotted paths
    (``<engine>.<section>.<leaf>``); the tail segment is the covered leaf.
    """
    covered: set[str] = set()
    for inv in rules.get("invariants", []):
        fields = inv.get("match", {}).get("fields", {})
        if isinstance(fields, dict):
            for path in fields:
                covered.add(str(path).split(".")[-1])
    return covered


def _load_artifacts(engine: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the active-pin schema + proposed rules for *engine*."""
    outputs = current_outputs_dir(engine)
    schema = json.loads((outputs / "schema.discovered.json").read_text())
    rules_text = (outputs / "rules.proposed.yaml").read_text()
    rules = yaml.safe_load(rules_text) or {}
    return schema, rules


def enumerate_class3_candidates(
    engine: str,
) -> tuple[list[Class3Candidate], EngineStageCounts]:
    """Compute the deterministic class-3 candidate set for *engine*.

    Returns the candidate list (sorted by section then leaf) and the per-stage
    exclusion counts. Reads only committed artifacts - no engine import.
    """
    schema, rules = _load_artifacts(engine)
    defs: dict[str, Any] = schema.get("$defs", {}) or {}
    covered = _corpus_covered_leaves(rules)

    candidates: list[Class3Candidate] = []
    total = 0
    n_composite = n_constrained = n_corpus = n_bool = n_freeform = 0

    for section in _SECTIONS:
        leaves: dict[str, Any] = schema.get(section, {}) or {}
        for leaf in sorted(leaves):
            shape = leaves[leaf]
            total += 1

            if _is_composite_ref(shape, defs):
                n_composite += 1
                continue
            if _is_schema_constrained(shape, defs):
                n_constrained += 1
                continue
            if leaf in covered:
                n_corpus += 1
                continue

            bases = _base_types(shape.get("type"))
            if bases and bases <= _BOOL_BASES:
                n_bool += 1
                continue
            if not (bases & _NUMERIC_BASES):
                # No numeric base type and no machine constraint -> opaque /
                # free-form (str, PathLike, Any, unknown, containers): drop.
                n_freeform += 1
                continue

            candidates.append(
                Class3Candidate(
                    engine=engine,
                    section=section,
                    leaf=leaf,
                    type=shape.get("type"),
                    default=shape.get("default"),
                    reason_included="unbounded_numeric",
                )
            )

    counts = EngineStageCounts(
        total_leaves=total,
        excluded_composite_ref=n_composite,
        excluded_schema_constrained=n_constrained,
        excluded_corpus_covered=n_corpus,
        excluded_bool=n_bool,
        excluded_freeform=n_freeform,
        included=len(candidates),
    )
    return candidates, counts


# Engines whose active pin carries a populated outputs/ bundle.
_ENGINES: tuple[str, ...] = ("vllm", "tensorrt", "transformers")


def _print_report(engines: tuple[str, ...], *, as_json: bool) -> None:
    report: dict[str, Any] = {}
    for engine in engines:
        candidates, counts = enumerate_class3_candidates(engine)
        report[engine] = {
            "counts": asdict(counts),
            "candidates": [asdict(c) for c in candidates],
        }

    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return

    for engine in engines:
        counts = report[engine]["counts"]
        cands = report[engine]["candidates"]
        print(f"=== {engine} ===")
        print(f"  total leaves:               {counts['total_leaves']}")
        print(f"  - composite-ref (nested):   {counts['excluded_composite_ref']}")
        print(f"  - schema-constrained (1/2): {counts['excluded_schema_constrained']}")
        print(f"  - corpus-covered (class 4): {counts['excluded_corpus_covered']}")
        print(f"  - bool (noise):             {counts['excluded_bool']}")
        print(f"  - free-form/opaque (noise): {counts['excluded_freeform']}")
        print(f"  = class-3 candidates:       {counts['included']}")
        for c in cands:
            print(f"      [{c['section']}] {c['leaf']}: {c['type']}")
    total = sum(report[e]["counts"]["included"] for e in engines)
    print(f"--- total class-3 candidates across {len(engines)} engine(s): {total} ---")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the deterministic Tier-D class-3 candidate set per engine."
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=_ENGINES,
        help="Restrict to one or more engines (repeatable). Default: all.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")
    args = parser.parse_args(argv)
    engines = tuple(args.engine) if args.engine else _ENGINES
    _print_report(engines, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
