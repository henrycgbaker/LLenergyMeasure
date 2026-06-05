"""Score a trial cell's mined catalogue against the Opus-established ground
truth (GT), via the canonicalising adapter in ``gt_adapter`` + the locked
scorer in ``trial_scoring``.

Why TWO sets of numbers (strict + tolerant)
-------------------------------------------
The GT corpus and the mined cell/baseline catalogues were authored under
DIFFERENT-but-compatible conventions for two things the locked scorer folds
into its identity tuple:

  1. NAMESPACE. GT derives a namespace per its own taxonomy (e.g.
     ``transformers.engine.quantization_config.bnb`` for a bitsandbytes
     field), while the baseline walker emits the flatter
     ``transformers`` for the same field. Same semantic invariant, two
     namespace strings -> disjoint strict identities.
  2. PREDICATE KIND. GT's predicate vocabulary is large and descriptive
     (``range``, ``cross_field_combo``, ``presence_conflict`` ...). The
     scorer's PREDICATE_KINDS is a small closed set. ``gt_adapter`` maps
     GT -> scorer buckets, but the mining tools independently chose their
     OWN bucket for the same field (e.g. GT ``present`` vs cell ``gt`` for
     ``early_stopping``). Same field, two predicate buckets -> disjoint
     strict identities.

The locked scorer (correctly, by design) does an exact set-intersection on
its full identity tuple. Run as-is against GT it therefore UNDER-counts
matches whenever either convention drifts, even though the cell genuinely
surfaced the right field. That under-count is real and worth reporting (it
bounds quality from below), but it is NOT the defensible research headline:
it conflates "the tool missed this invariant" with "the tool found it but
labelled the predicate/namespace differently from the GT author".

So this module reports BOTH:

  * STRICT identity numbers - the unmodified ``trial_scoring`` result with
    canonical GT as reference. A lower bound on recall/precision.
  * TOLERANT numbers - a convention-insensitive re-match:
      - invariants match on ``(leaf_native_field, coarse_predicate_bucket)``
        ONLY (namespace dropped; predicate collapsed to the coarse buckets
        ``type | membership | numeric | exact | presence`` from
        ``gt_adapter.coarse_predicate_bucket``).
      - schema matches on ``(field_name)`` within-namespace-INSENSITIVE
        (the field name is unique enough across a single engine's catalogue
        that namespace drift, not genuine collision, dominates).
    The tolerant numbers are the headline given the known convention drift;
    the strict numbers bound them from below. A large strict<->tolerant gap
    is itself a finding (it quantifies how much of the apparent miss is
    convention noise vs genuine absence).

Caching
-------
Canonicalised GT files are written once to
``findings/ground_truth_canonical/<engine>/<version_slug>/`` (schema JSON +
invariants YAML) so the strict path can hand the locked scorer real file
paths (its ``score_cell`` reads from disk) and re-runs are cheap.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gt_adapter as A
import trial_scoring as S
import yaml

# ground_truth_canonical lives next to ground_truth under findings/.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_FINDINGS_DIR = _SCRIPTS_DIR.parent / "findings"
_GT_DIR = _FINDINGS_DIR / "ground_truth"
_CANON_DIR = _FINDINGS_DIR / "ground_truth_canonical"


# ---------------------------------------------------------------------------
# Canonical-GT caching
# ---------------------------------------------------------------------------


def canonicalise_gt_dir(
    engine: str, version_slug: str, *, force: bool = False
) -> tuple[Path, Path]:
    """Canonicalise a GT dir's schema + invariants, caching to
    ``findings/ground_truth_canonical/<engine>/<version_slug>/``.

    Returns ``(canonical_schema_path, canonical_invariants_path)``.
    """
    src = _GT_DIR / engine / version_slug
    gt_schema_path = src / "schema_ground_truth.json"
    gt_inv_path = src / "invariants_ground_truth.yaml"
    if not gt_schema_path.exists():
        raise FileNotFoundError(f"GT schema missing: {gt_schema_path}")
    if not gt_inv_path.exists():
        raise FileNotFoundError(f"GT invariants missing: {gt_inv_path}")

    out_dir = _CANON_DIR / engine / version_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_schema = out_dir / "schema.canonical.json"
    out_inv = out_dir / "invariants.canonical.yaml"

    if force or not out_schema.exists():
        canon_schema = A.canonicalise_gt_schema(json.loads(gt_schema_path.read_text()))
        out_schema.write_text(json.dumps(canon_schema, indent=2, sort_keys=True))
    if force or not out_inv.exists():
        canon_inv = A.canonicalise_gt_invariants(yaml.safe_load(gt_inv_path.read_text()) or {})
        out_inv.write_text(yaml.safe_dump(canon_inv, sort_keys=False))

    return out_schema, out_inv


# ---------------------------------------------------------------------------
# Tolerant matchers (convention-insensitive)
# ---------------------------------------------------------------------------


def tolerant_invariant_keys(canonical_invariants: dict[str, Any]) -> set[tuple[str, str]]:
    """Return the set of ``(leaf_native_field, coarse_predicate_bucket)``
    tolerant keys for a canonicalised invariants envelope.

    Reads the ``_gt_leaf_field`` / ``_gt_coarse_bucket`` hints that
    ``gt_adapter`` stashes on each entry when available; falls back to
    re-deriving from the synthesised ``match.fields`` + a coarse bucketing
    of the scorer-classified predicate kind (so a mined CELL output - which
    has real ``match`` blocks but no GT hints - is handled too).
    """
    out: set[tuple[str, str]] = set()
    invs = canonical_invariants.get("invariants")
    if not isinstance(invs, list):
        return out
    for inv in invs:
        if not isinstance(inv, dict):
            continue
        leaf = inv.get("_gt_leaf_field")
        bucket = inv.get("_gt_coarse_bucket")
        if leaf is None or bucket is None:
            # Cell/baseline path: derive from the real match block.
            identity = S.invariant_identity(inv)
            _, leaf, pk, _sec = identity
            bucket = _coarse_from_scorer_pk(pk)
        out.add((str(leaf), str(bucket)))
    return out


# Map the scorer's PREDICATE_KINDS to the coarse buckets used for tolerant
# matching, mirroring gt_adapter.coarse_predicate_bucket so both sides land
# in the same coarse space.
_SCORER_PK_TO_COARSE = {
    "type_is_not": "type",
    "not_in": "membership",
    "not_equal": "membership",
    "exact": "exact",
    "gt": "numeric",
    "lt": "numeric",
    "ge": "numeric",
    "le": "numeric",
    "present": "presence",
    "unknown": "presence",
}


def _coarse_from_scorer_pk(pk: str) -> str:
    return _SCORER_PK_TO_COARSE.get(pk, "presence")


def tolerant_schema_names(canonical_schema: dict[str, Any]) -> set[str]:
    """Return the namespace-insensitive set of field NAMES across all of a
    schema's sections (engine_params, sampling_params, $defs.*.properties).
    """
    names: set[str] = set()
    for _ns_name, name in S.schema_field_identities(canonical_schema):
        names.add(name)
    return names


@dataclass
class TolerantScores:
    """Namespace-insensitive recall/precision pair (cell vs GT)."""

    schema_recall: float
    schema_precision: float
    schema_ref_n: int
    schema_cell_n: int
    schema_int_n: int
    invariant_recall: float
    invariant_precision: float
    invariant_ref_n: int
    invariant_cell_n: int
    invariant_int_n: int


def score_tolerant(
    *,
    canonical_gt_schema: dict[str, Any],
    canonical_gt_invariants: dict[str, Any],
    cell_schema: dict[str, Any],
    cell_invariants: dict[str, Any],
) -> TolerantScores:
    """Compute the convention-tolerant recall/precision (cell vs GT)."""
    # Schema: field-name set intersection, namespace dropped.
    gt_names = tolerant_schema_names(canonical_gt_schema)
    cell_names = tolerant_schema_names(cell_schema)
    s_int = gt_names & cell_names
    s_recall = len(s_int) / len(gt_names) if gt_names else 0.0
    s_prec = len(s_int) / len(cell_names) if cell_names else 0.0

    # Invariants: (leaf, coarse_bucket) set intersection, namespace dropped.
    gt_keys = tolerant_invariant_keys(canonical_gt_invariants)
    cell_keys = tolerant_invariant_keys(cell_invariants)
    i_int = gt_keys & cell_keys
    i_recall = len(i_int) / len(gt_keys) if gt_keys else 0.0
    i_prec = len(i_int) / len(cell_keys) if cell_keys else 0.0

    return TolerantScores(
        schema_recall=s_recall,
        schema_precision=s_prec,
        schema_ref_n=len(gt_names),
        schema_cell_n=len(cell_names),
        schema_int_n=len(s_int),
        invariant_recall=i_recall,
        invariant_precision=i_prec,
        invariant_ref_n=len(gt_keys),
        invariant_cell_n=len(cell_keys),
        invariant_int_n=len(i_int),
    )


# ---------------------------------------------------------------------------
# Combined cell-vs-GT scoring
# ---------------------------------------------------------------------------


@dataclass
class CellVsGT:
    """Result of scoring one cell against canonical GT: strict identity
    numbers (from the locked scorer) + convention-tolerant numbers."""

    engine: str
    version_slug: str
    strategy: str
    # strict (locked scorer)
    strict: S.CellScore
    # tolerant
    tolerant: TolerantScores
    # bookkeeping
    cell_schema_path: str = ""
    cell_invariants_path: str = ""


def _load_cell(path: Path | None, *, is_yaml: bool) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        if is_yaml:
            return yaml.safe_load(path.read_text()) or {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def score_cell_vs_gt(
    *,
    strategy: str,
    engine: str,
    version_slug: str,
    bump_distance: str,
    schema_output_path: Path | None,
    invariants_output_path: Path | None,
    wall_clock_sec: float = 0.0,
    energy_wh: float = 0.0,
    config: S.ScoringConfig | None = None,
) -> CellVsGT:
    """Score a cell's mined outputs against the canonical GT for its
    ``(engine, version_slug)``.

    Canonicalises (and caches) the GT dir, runs the locked
    ``trial_scoring.score_cell`` with the canonical GT as reference for the
    STRICT numbers, and computes the TOLERANT numbers in-process.
    """
    canon_schema_path, canon_inv_path = canonicalise_gt_dir(engine, version_slug)

    strict_score, _d1, _d2, _d3 = S.score_cell(
        strategy=strategy,
        engine=engine,
        version_slug=version_slug,
        bump_distance=bump_distance,
        schema_output_path=schema_output_path,
        invariants_output_path=invariants_output_path,
        reference_schema_path=canon_schema_path,
        reference_invariants_path=canon_inv_path,
        wall_clock_sec=wall_clock_sec,
        energy_wh=energy_wh,
        config=config,
    )

    canon_schema = json.loads(canon_schema_path.read_text())
    canon_inv = yaml.safe_load(canon_inv_path.read_text()) or {}
    cell_schema = _load_cell(schema_output_path, is_yaml=False)
    cell_inv = _load_cell(invariants_output_path, is_yaml=True)

    tolerant = score_tolerant(
        canonical_gt_schema=canon_schema,
        canonical_gt_invariants=canon_inv,
        cell_schema=cell_schema,
        cell_invariants=cell_inv,
    )

    return CellVsGT(
        engine=engine,
        version_slug=version_slug,
        strategy=strategy,
        strict=strict_score,
        tolerant=tolerant,
        cell_schema_path=str(schema_output_path) if schema_output_path else "",
        cell_invariants_path=str(invariants_output_path) if invariants_output_path else "",
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke: canonicalise every GT dir and score its engine baseline against
    # it (the prompt's sanity check). Print strict + tolerant numbers.
    pairs = [
        ("transformers", "v4_57_3"),
        ("transformers", "v5_6_2"),
        ("vllm", "v0_7_3"),
        ("vllm", "v0_19_1"),
        ("tensorrt", "v0_21_0"),
        ("tensorrt", "v1_2_1"),
    ]
    repo_root = _SCRIPTS_DIR.parent.parent.parent
    for engine, version in pairs:
        bdir = repo_root / "engine_versions" / engine / version / "outputs"
        bl_schema = bdir / "schema.discovered.json"
        bl_inv = bdir / "invariants.proposed.yaml"
        if not (bl_schema.exists() and bl_inv.exists()):
            print(f"{engine}/{version}: no baseline catalogue on disk (skip)")
            continue
        res = score_cell_vs_gt(
            strategy="baseline",
            engine=engine,
            version_slug=version,
            bump_distance="active",
            schema_output_path=bl_schema,
            invariants_output_path=bl_inv,
        )
        s = res.strict
        t = res.tolerant
        print(f"=== {engine}/{version} (baseline vs GT) ===")
        print(
            f"  STRICT   schema r/p = {s.schema_recall:.3f}/{s.schema_precision:.3f}"
            f"  inv r/p = {s.invariant_recall:.3f}/{s.invariant_precision:.3f}"
        )
        print(
            f"  TOLERANT schema r/p = {t.schema_recall:.3f}/{t.schema_precision:.3f}"
            f"  inv r/p = {t.invariant_recall:.3f}/{t.invariant_precision:.3f}"
        )
    sys.exit(0)
