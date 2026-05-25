"""Per-cell scoring harness for the mining-substrate empirical trial.

Implements the 7-metric rubric + 3 brittleness sub-metrics described in
``.planning/mining-substrate-empirical-trial.md`` (Phase 1 Day 5).

Design contract
---------------
Each cell of the trial is a (strategy, engine, version) triple. The cell
produces two outputs:

* ``schema_output_path``: a JSON file matching the
  ``engine_versions/<engine>/v<v>/outputs/schema.discovered.json`` shape.
* ``invariants_output_path``: a YAML file matching the
  ``engine_versions/<engine>/v<v>/outputs/invariants.proposed.yaml`` shape.

Strategy (a) "pure mining" emits these into the canonical
``engine_versions/`` tree via the existing pipeline. Strategies (b/c/d) emit
parallel artefacts under ``_spike/findings/trial_runs/`` (see
``trial_runner.py`` for the layout decision).

For scoring purposes both shapes ARE the same; this module reads the files,
computes the rubric, and emits a :class:`CellScore`.

Canonical item matching
-----------------------
Two reference identities are used, and ONLY these (no fuzzy match):

* **Schema field identity**: ``(namespace, name)`` where namespace is
  ``"engine_params"`` / ``"sampling_params"`` / ``"$defs.<nested-class>"``.
* **Invariant identity**: ``(namespace, native_field, predicate_kind)``
  where ``namespace`` is the dotted prefix of the match-field key (e.g.
  ``"transformers.sampling"``), ``native_field`` is the leaf field name and
  ``predicate_kind`` is one of ``"exact" | "not_in" | "not_equal" | "gt" |
  "lt" | "ge" | "le" | "type_is_not" | "present" | "unknown"``.

Reuse of validate_invariants
----------------------------
We deliberately do NOT duplicate ``scripts/validate_invariants.py`` logic.
The validation harness answers a different question: "did the library
emit the expected runtime behaviour?". Scoring asks a different question:
"how close is this cell's mined catalogue to the reference catalogue?".

We DO reuse one piece: :data:`scripts._invariant_validation_common.CaseResult`
and friends are imported when ``Phase 3`` runs validated YAML on a cell's
output as a tie-break signal for the failure-mode tag (a cell that emits
"plausible" invariants which then fail runtime-validation is more brittle
than one whose invariants validate clean). Phase 1 stubs this out.

CellScore record schema (versioned)
-----------------------------------
JSON-serialisable dataclass. Format version ``"1.0.0"``. Backward-incompatible
changes bump major; additive bumps minor. Field set:

* Schema metrics: ``schema_recall``, ``schema_precision``,
  ``schema_type_accuracy``, ``schema_failure_mode``.
* Invariant metrics: ``invariant_recall``, ``invariant_precision``,
  ``invariant_severity_accuracy``, ``invariant_failure_mode``.
* Cost: ``wall_clock_sec``, ``energy_wh``.
* Brittleness (filled per cell vs active-version cell of same strategy):
  ``brittleness_pass_through_rate``, ``brittleness_silent_fail_count``,
  ``brittleness_detectable_fail_count``, ``brittleness_patch_cost_loc``.
* Failure-mode tag + observations + raw counts (for downstream stats).

"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Project root sits two parents above _spike/scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


SCORING_FORMAT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Identity types
# ---------------------------------------------------------------------------


SchemaFieldId = tuple[str, str]
"""(namespace, name) - e.g. ("engine_params", "dtype") or
("$defs.CompileConfig", "fullgraph"). Hashable; used for set comparison."""

InvariantId = tuple[str, str, str, str]
"""(namespace, native_field, predicate_kind, secondary_field) - e.g.
("transformers.sampling", "num_beams", "exact", "") for a single-field
invariant or ("transformers.sampling", "do_sample", "exact", "temperature")
for a two-field cross-field invariant (sampling-only-when-greedy on
temperature).

The ``secondary_field`` slot was added in Phase 2.5 to disambiguate
multi-field invariants where the primary key + predicate_kind would
otherwise collapse several logical invariants into one identity (e.g.
all 7 sampling-only-when-greedy dormancies share
("transformers.sampling", "do_sample", "exact") so without
secondary_field they score as 1 identity, not 7).

Single-field invariants emit ``""`` (empty string, NOT None) in the
secondary slot - hashable tuples need string-typed slots throughout.

The ``severity`` is deliberately NOT in the identity tuple - severity
accuracy is scored as a separate metric (the rubric calls for "matching
severity" as a per-invariant accuracy measurement, not as an identity
criterion)."""

# Recognised predicate kinds. ``unknown`` is the catch-all for shapes the
# normaliser cannot classify; tracked so we don't silently lose entries.
PREDICATE_KINDS = frozenset(
    {
        "exact",
        "not_in",
        "not_equal",
        "gt",
        "lt",
        "ge",
        "le",
        "type_is_not",
        "present",
        "unknown",
    }
)


# ---------------------------------------------------------------------------
# Failure-mode taxonomy
# ---------------------------------------------------------------------------


class FailureMode(str, Enum):
    """Categorical failure-mode tag, paralleling the trial rubric.

    A cell can carry multiple failure modes (e.g. a CRASH on schema +
    SILENT miss on invariants); the :class:`CellScore` ``failure_modes``
    field is a list.
    """

    NONE = "none"
    """Cell completed; output present and parsable; recall + precision computed."""

    CRASH = "crash"
    """Cell raised before producing output (timeout, exception, no file written)."""

    DETECTABLE = "detectable"
    """Cell produced output but with visible defects: parse error, malformed
    envelope, empty arrays, marker fields like ``error: ...``. Downstream
    tooling can see the failure."""

    SILENT = "silent"
    """Cell produced PARSABLE output that is materially incorrect: e.g.
    sampling-params present by name but with type spec stripped (Bake-off D's
    transformers v5.9 finding); high precision but very low recall with no
    visible signal of degradation. The dangerous case. Computed as a tag
    when recall < silent_threshold while ``parsable == True`` AND no
    detectable signals are present."""

    PARTIAL = "partial"
    """Cell produced PARTIAL output - e.g. schema parsed but invariants
    didn't (Bake-off B's markdown-fence wrapping). One of the two artefacts
    is present and parsable; the other is missing or unparsable."""


# ---------------------------------------------------------------------------
# Score dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemDiff:
    """A structured per-item diff entry used for the per-cell artefacts.

    Cells emit ``recall_misses.yaml``, ``precision_spurious.yaml`` and
    ``type_mismatches.yaml`` next to their score file so a human reviewer
    can see WHICH items diverged, not just how many.
    """

    identity: tuple[str, ...]
    """The identity tuple of the item (length-2 for schema, length-3 for
    invariants). Stored as a tuple of strings for JSON round-trip."""

    kind: str
    """One of ``"recall_miss"``, ``"precision_spurious"``,
    ``"type_mismatch"``, ``"severity_mismatch"``."""

    reference_value: Any | None = None
    """The reference catalogue's value at this identity (e.g. type spec,
    severity, full invariant body). ``None`` if not applicable."""

    cell_value: Any | None = None
    """The cell's value at this identity. ``None`` if absent (recall miss
    case) or not applicable."""

    note: str = ""
    """Free-text annotation surfaced to the human reviewer."""


@dataclass
class CellScore:
    """Per-cell score record. JSON-serialisable; written to
    ``_spike/findings/trial_scores/<strategy>__<engine>__<version_slug>.json``.

    The 10 numeric fields + failure-mode tag + observations + raw counts
    constitute the canonical per-cell record. Aggregator (trial_aggregate.py)
    consumes these directly.
    """

    # ---- Identity ----
    strategy: str
    """One of ``"a" | "b" | "c" | "d-ab" | "d-ac"`` (extensible via the
    trial's hybrid-space variants)."""

    engine: str
    """One of ``"transformers" | "vllm" | "tensorrt"``."""

    version_slug: str
    """e.g. ``"v4_57_3"`` - the filesystem-safe version identifier."""

    bump_distance: str
    """One of ``"v-2" | "v-1" | "active" | "v+1" | "v+major"`` - the cell's
    position relative to the strategy's active baseline. Used by the
    aggregator's brittleness rollup."""

    # ---- 7-metric rubric ----
    # Schema metrics
    schema_recall: float = 0.0
    schema_precision: float = 0.0
    schema_type_accuracy: float = 0.0

    # Invariants metrics
    invariant_recall: float = 0.0
    invariant_precision: float = 0.0
    invariant_severity_accuracy: float = 0.0

    # Cost
    wall_clock_sec: float = 0.0
    energy_wh: float = 0.0

    # ---- Failure-mode tag (per artefact) ----
    schema_failure_mode: str = FailureMode.NONE.value
    invariant_failure_mode: str = FailureMode.NONE.value
    failure_modes: list[str] = field(default_factory=list)
    """Aggregate list - typically ``[schema_failure_mode,
    invariant_failure_mode]`` deduped, but room to carry additional tags
    surfaced by the runner (e.g. ``"timeout"``, ``"ollama_500"``)."""

    # ---- Brittleness sub-metrics (filled by aggregator, NOT runner) ----
    brittleness_pass_through_rate: float | None = None
    """% of reference items the strategy STILL surfaces at this version,
    normalised by what it surfaced correctly at the active version. None
    when the cell IS the active version, or when the active-version cell
    is unavailable."""

    brittleness_silent_fail_count: int | None = None
    """Number of items present-by-identity at the active-version cell that
    became silent failures (present but materially wrong) at this bump."""

    brittleness_detectable_fail_count: int | None = None
    """Number of items present-by-identity at the active-version cell that
    became detectable failures at this bump."""

    brittleness_patch_cost_loc: int | None = None
    """Estimated LoC delta to restore active-version quality at this bump.
    Auto-estimated for strategy (a) via a simple file-diff heuristic;
    flagged as ``None`` (human estimate needed) for (b)/(c)/(d) prompt-
    iteration cases."""

    # ---- Raw counts (for downstream reproducibility) ----
    schema_reference_count: int = 0
    schema_cell_count: int = 0
    schema_intersection_count: int = 0
    invariant_reference_count: int = 0
    invariant_cell_count: int = 0
    invariant_intersection_count: int = 0

    # ---- Free-text and structured observations ----
    observations: list[str] = field(default_factory=list)
    """Adjacent observations not covered by the rubric (per the trial's
    epistemic framing § 'Capture failure modes explicitly'). Used by the
    aggregator's adjacent-observations rollup."""

    # ---- Metadata ----
    scoring_format_version: str = SCORING_FORMAT_VERSION
    scored_at: str = ""
    """ISO-8601 timestamp at scoring time. Set by the runner; left empty by
    the scorer itself."""

    reference_path: str = ""
    """Absolute path to the reference catalogue used for scoring. Stored so
    a re-run can be reproduced."""

    cell_schema_path: str = ""
    cell_invariants_path: str = ""

    def to_json(self) -> str:
        """JSON-serialise the score record (sort_keys=True for deterministic
        on-disk byte form; aggregator diffs run-to-run against this)."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


@dataclass(frozen=True)
class ScoringConfig:
    """Knobs that affect how scoring classifies edges.

    Defaults are conservative (low silent_threshold; strict severity).
    Phase 2 may tune these once we have a baseline distribution of cell
    scores.
    """

    silent_threshold: float = 0.20
    """If recall < silent_threshold AND parsable AND no detectable signals,
    the cell is tagged ``FailureMode.SILENT``. Default 0.20 - anything below
    20% recall in a parsable envelope is treated as silent failure."""

    fuzzy_field_match: bool = False
    """Always False in v1.0. Reserved for a future setting that permits
    e.g. case-insensitive field-name matching. The trial explicitly forbids
    fuzzy matching for the comparison axis to stay clean."""

    severity_exact: bool = True
    """If True, severity must match exactly (``"error" == "error"``).
    If False, ``"warning"`` and ``"warn"`` are coalesced. Default True; the
    trial's epistemic framing prefers strictness."""

    score_only_intersection_for_severity: bool = True
    """If True (default), severity accuracy is computed over the intersection
    of identities (mirroring schema type_accuracy). If False, severity
    misses count as severity mismatches (penalises strategies that miss
    items twice - once on recall and once on severity)."""


# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------


def schema_field_identities(schema_obj: dict[str, Any]) -> set[SchemaFieldId]:
    """Extract the set of (namespace, name) identities from a schema dict.

    Walks ``engine_params``, ``sampling_params`` and the ``$defs.<class>.properties``
    keys (mirroring the canonical envelope shape).
    """
    identities: set[SchemaFieldId] = set()
    for namespace in ("engine_params", "sampling_params"):
        section = schema_obj.get(namespace, {})
        if isinstance(section, dict):
            for name in section:
                identities.add((namespace, name))
    defs = schema_obj.get("$defs", {})
    if isinstance(defs, dict):
        for class_name, class_def in defs.items():
            properties = (class_def or {}).get("properties", {})
            if isinstance(properties, dict):
                ns = f"$defs.{class_name}"
                for name in properties:
                    identities.add((ns, name))
    return identities


def schema_type_at(schema_obj: dict[str, Any], identity: SchemaFieldId) -> Any:
    """Return the ``type`` value (or anyOf-summary) at a given identity, or None.

    Used by ``type_accuracy`` scoring. The returned value is the raw
    dict entry's ``type`` slot - comparison is exact-string for primitive
    types and shape-aware for anyOf cases (Phase 2 implements the anyOf
    matcher).
    """
    namespace, name = identity
    if namespace in ("engine_params", "sampling_params"):
        entry = (schema_obj.get(namespace, {}) or {}).get(name)
    elif namespace.startswith("$defs."):
        class_name = namespace.removeprefix("$defs.")
        entry = (
            (schema_obj.get("$defs", {}) or {}).get(class_name, {}).get("properties", {}).get(name)
        )
    else:
        return None
    if not isinstance(entry, dict):
        return None
    if "type" in entry:
        return entry["type"]
    if "anyOf" in entry:
        return ("anyOf", tuple(sorted(_anyof_summary(entry["anyOf"]))))
    return None


def _anyof_summary(any_of: list[dict[str, Any]]) -> Iterable[str]:
    """Tuple-of-types summary for an anyOf entry; ignores ``description``."""
    for entry in any_of or []:
        t = entry.get("type")
        if t is None:
            yield "object"
        elif isinstance(t, list):
            yield ",".join(sorted(str(x) for x in t))
        else:
            yield str(t)


def invariant_identity(inv: dict[str, Any]) -> InvariantId:
    """Compute the canonical 4-tuple identity for an invariant.

    Returns ``(namespace, native_field, predicate_kind, secondary_field)``.
    The ``secondary_field`` slot is the LEAF NAME of the second match-field
    key (if any), enabling disambiguation of multi-field cross-checks
    where the primary key + predicate_kind would otherwise collapse
    several logical invariants into one identity.

    Phase 2.5 rubric fix:
    - For single-field invariants: secondary_field = ``""`` (empty string).
    - For multi-field invariants: secondary_field = the leaf name of the
      second key (e.g. ``"temperature"`` for the
      ``do_sample=False AND temperature is not None`` dormancy).
    - Field ordering is preserved from the match.fields dict (Python
      3.7+ dict insertion order is stable, and producers emit fields
      in a consistent order).

    Falls back to ``("", "<id-or-unknown>", "unknown", "")`` when the
    match block is missing - strategies that emit malformed invariants
    land in a distinct bucket that the aggregator can spot.
    """
    match_fields = (inv.get("match") or {}).get("fields") or {}
    if not isinstance(match_fields, dict) or not match_fields:
        return ("", str(inv.get("id", "")) or "<unknown>", "unknown", "")
    # First field key is the PRIMARY; second (if present) is the SECONDARY.
    keys = list(match_fields.keys())
    primary_key = keys[0]
    primary_value = match_fields[primary_key]
    namespace, _, native_field = primary_key.rpartition(".")
    predicate_kind = _predicate_kind_for(primary_value)
    if len(keys) >= 2:
        secondary_key = keys[1]
        _, _, secondary_field = secondary_key.rpartition(".")
    else:
        secondary_field = ""
    return (namespace, native_field, predicate_kind, secondary_field)


def _predicate_kind_for(value: Any) -> str:
    """Classify a match-field value into the predicate-kind taxonomy."""
    if not isinstance(value, dict):
        return "exact"
    keys = set(value.keys())
    if "not_in" in keys:
        return "not_in"
    if "not_equal" in keys:
        return "not_equal"
    if ">" in keys:
        return "gt"
    if "<" in keys:
        return "lt"
    if ">=" in keys:
        return "ge"
    if "<=" in keys:
        return "le"
    if "type_is_not" in keys:
        return "type_is_not"
    if "present" in keys:
        return "present"
    return "unknown"


def invariant_identities(invariants_obj: dict[str, Any]) -> set[InvariantId]:
    """Extract the set of invariant identities from a proposed.yaml dict."""
    inv_list = invariants_obj.get("invariants") if isinstance(invariants_obj, dict) else None
    if not isinstance(inv_list, list):
        return set()
    return {invariant_identity(inv) for inv in inv_list if isinstance(inv, dict)}


def invariants_by_identity(
    invariants_obj: dict[str, Any],
) -> dict[InvariantId, dict[str, Any]]:
    """Index invariants by canonical identity. Last-write-wins on collisions
    (strategies producing dup entries get logged as adjacent observations,
    not silently dropped).
    """
    inv_list = invariants_obj.get("invariants") if isinstance(invariants_obj, dict) else None
    out: dict[InvariantId, dict[str, Any]] = {}
    if not isinstance(inv_list, list):
        return out
    for inv in inv_list:
        if not isinstance(inv, dict):
            continue
        out[invariant_identity(inv)] = inv
    return out


# ---------------------------------------------------------------------------
# Severity comparison
# ---------------------------------------------------------------------------


def normalise_severity(severity: str | None, *, exact: bool) -> str:
    """Map a raw severity string to a comparable form.

    When ``exact=True``, the raw lowercase string is returned. When
    ``exact=False``, ``"warning"`` and ``"warn"`` coalesce; ``"err"`` -> ``"error"``.
    """
    if severity is None:
        return ""
    s = str(severity).strip().lower()
    if exact:
        return s
    if s in {"warn", "warning"}:
        return "warning"
    if s in {"err", "error"}:
        return "error"
    return s


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    return yaml.safe_load(text) or {}


def score_schema(
    *,
    reference_schema: dict[str, Any],
    cell_schema: dict[str, Any],
    config: ScoringConfig,
) -> tuple[
    float, float, float, FailureMode, int, int, int, list[ItemDiff], list[ItemDiff], list[ItemDiff]
]:
    """Score a cell's schema output against the reference.

    Returns ``(recall, precision, type_accuracy, failure_mode, ref_count,
    cell_count, intersection_count, recall_misses, precision_spurious,
    type_mismatches)``.
    """
    identities_ref = schema_field_identities(reference_schema)
    identities_cell = schema_field_identities(cell_schema)
    intersection = identities_ref & identities_cell

    ref_count = len(identities_ref)
    cell_count = len(identities_cell)
    int_count = len(intersection)

    recall = int_count / ref_count if ref_count else 0.0
    precision = int_count / cell_count if cell_count else 0.0

    # Type accuracy on the intersection
    type_matches = 0
    type_mismatches: list[ItemDiff] = []
    for ident in intersection:
        ref_type = schema_type_at(reference_schema, ident)
        cell_type = schema_type_at(cell_schema, ident)
        if ref_type == cell_type:
            type_matches += 1
        else:
            type_mismatches.append(
                ItemDiff(
                    identity=tuple(str(x) for x in ident),
                    kind="type_mismatch",
                    reference_value=_jsonable_type(ref_type),
                    cell_value=_jsonable_type(cell_type),
                    note="schema type spec divergence",
                )
            )
    type_accuracy = type_matches / int_count if int_count else 0.0

    # Failure-mode classification
    # Detectable: envelope marker (e.g. cell has top-level "error" key) OR both
    # sections empty when reference has populated sections.
    failure_mode = FailureMode.NONE
    if ("error" in cell_schema and cell_schema.get("error")) or cell_count == 0:
        failure_mode = FailureMode.DETECTABLE
    elif recall < config.silent_threshold:
        # Parsable + low recall + no detectable signal = silent.
        failure_mode = FailureMode.SILENT

    # Build per-item diffs
    recall_misses: list[ItemDiff] = []
    for ident in sorted(identities_ref - identities_cell):
        recall_misses.append(
            ItemDiff(
                identity=tuple(str(x) for x in ident),
                kind="recall_miss",
                reference_value=_jsonable_type(schema_type_at(reference_schema, ident)),
                cell_value=None,
                note="reference field absent in cell output",
            )
        )
    precision_spurious: list[ItemDiff] = []
    for ident in sorted(identities_cell - identities_ref):
        precision_spurious.append(
            ItemDiff(
                identity=tuple(str(x) for x in ident),
                kind="precision_spurious",
                reference_value=None,
                cell_value=_jsonable_type(schema_type_at(cell_schema, ident)),
                note="cell field absent in reference",
            )
        )

    return (
        recall,
        precision,
        type_accuracy,
        failure_mode,
        ref_count,
        cell_count,
        int_count,
        recall_misses,
        precision_spurious,
        type_mismatches,
    )


def _jsonable_type(value: Any) -> Any:
    """Convert schema_type_at outputs into JSON-friendly forms.

    Tuples (used for anyOf summaries) become lists.
    """
    if isinstance(value, tuple):
        return [list(v) if isinstance(v, tuple) else v for v in value]
    return value


def score_invariants(
    *,
    reference_invariants: dict[str, Any],
    cell_invariants: dict[str, Any],
    config: ScoringConfig,
) -> tuple[
    float, float, float, FailureMode, int, int, int, list[ItemDiff], list[ItemDiff], list[ItemDiff]
]:
    """Score a cell's invariant output against the reference.

    Returns ``(recall, precision, severity_accuracy, failure_mode, ref_count,
    cell_count, intersection_count, recall_misses, precision_spurious,
    severity_mismatches)``.
    """
    identities_ref = invariant_identities(reference_invariants)
    identities_cell = invariant_identities(cell_invariants)
    intersection = identities_ref & identities_cell

    ref_count = len(identities_ref)
    cell_count = len(identities_cell)
    int_count = len(intersection)

    recall = int_count / ref_count if ref_count else 0.0
    precision = int_count / cell_count if cell_count else 0.0

    ref_index = invariants_by_identity(reference_invariants)
    cell_index = invariants_by_identity(cell_invariants)

    # Severity accuracy
    severity_matches = 0
    severity_mismatches: list[ItemDiff] = []
    severity_denominator = int_count if config.score_only_intersection_for_severity else ref_count
    for ident in intersection:
        ref_sev = normalise_severity(ref_index[ident].get("severity"), exact=config.severity_exact)
        cell_sev = normalise_severity(
            cell_index[ident].get("severity"), exact=config.severity_exact
        )
        if ref_sev == cell_sev:
            severity_matches += 1
        else:
            severity_mismatches.append(
                ItemDiff(
                    identity=tuple(str(x) for x in ident),
                    kind="severity_mismatch",
                    reference_value=ref_sev,
                    cell_value=cell_sev,
                    note="severity divergence",
                )
            )
    severity_accuracy = severity_matches / severity_denominator if severity_denominator else 0.0

    # Failure-mode classification
    failure_mode = FailureMode.NONE
    inv_list = cell_invariants.get("invariants") if isinstance(cell_invariants, dict) else None
    if cell_invariants.get("error") or inv_list is None or (isinstance(inv_list, list) and len(inv_list) == 0):
        failure_mode = FailureMode.DETECTABLE
    elif recall < config.silent_threshold:
        failure_mode = FailureMode.SILENT

    # Build per-item diffs
    recall_misses: list[ItemDiff] = []
    for ident in sorted(identities_ref - identities_cell):
        ref_inv = ref_index[ident]
        recall_misses.append(
            ItemDiff(
                identity=tuple(str(x) for x in ident),
                kind="recall_miss",
                reference_value={
                    "id": ref_inv.get("id"),
                    "severity": ref_inv.get("severity"),
                },
                cell_value=None,
                note="reference invariant absent in cell output",
            )
        )
    precision_spurious: list[ItemDiff] = []
    for ident in sorted(identities_cell - identities_ref):
        cell_inv = cell_index[ident]
        precision_spurious.append(
            ItemDiff(
                identity=tuple(str(x) for x in ident),
                kind="precision_spurious",
                reference_value=None,
                cell_value={
                    "id": cell_inv.get("id"),
                    "severity": cell_inv.get("severity"),
                },
                note="cell invariant absent in reference",
            )
        )

    return (
        recall,
        precision,
        severity_accuracy,
        failure_mode,
        ref_count,
        cell_count,
        int_count,
        recall_misses,
        precision_spurious,
        severity_mismatches,
    )


def score_cell(
    *,
    strategy: str,
    engine: str,
    version_slug: str,
    bump_distance: str,
    schema_output_path: Path | None,
    invariants_output_path: Path | None,
    reference_schema_path: Path,
    reference_invariants_path: Path,
    wall_clock_sec: float,
    energy_wh: float,
    config: ScoringConfig | None = None,
    extra_observations: list[str] | None = None,
    extra_failure_modes: list[str] | None = None,
) -> tuple[CellScore, list[ItemDiff], list[ItemDiff], list[ItemDiff]]:
    """Top-level entry: score one cell's output and return the score record.

    Returns ``(score, schema_diffs, invariant_diffs, type_or_severity_diffs)``
    so the caller can emit the per-cell side-artefacts
    (recall_misses.yaml + precision_spurious.yaml + type_mismatches.yaml).
    """
    cfg = config or ScoringConfig()
    observations = list(extra_observations or [])
    failure_modes: list[str] = list(extra_failure_modes or [])

    # Reference is always required.
    if not reference_schema_path.exists():
        raise FileNotFoundError(f"reference schema missing: {reference_schema_path}")
    if not reference_invariants_path.exists():
        raise FileNotFoundError(f"reference invariants missing: {reference_invariants_path}")
    ref_schema = _read_json(reference_schema_path)
    ref_invs = _read_yaml(reference_invariants_path)

    # Read cell outputs robustly.
    schema_partial = False
    invariants_partial = False
    cell_schema: dict[str, Any] = {}
    cell_invs: dict[str, Any] = {}

    if schema_output_path is None:
        schema_partial = True
        observations.append("cell schema output path is None (CRASH or skipped)")
    elif not schema_output_path.exists():
        schema_partial = True
        observations.append(f"cell schema output file missing: {schema_output_path}")
    else:
        try:
            cell_schema = _read_json(schema_output_path)
        except Exception as exc:
            schema_partial = True
            observations.append(f"cell schema unreadable: {type(exc).__name__}: {exc}")

    if invariants_output_path is None:
        invariants_partial = True
        observations.append("cell invariants output path is None (CRASH or skipped)")
    elif not invariants_output_path.exists():
        invariants_partial = True
        observations.append(f"cell invariants output file missing: {invariants_output_path}")
    else:
        try:
            cell_invs = _read_yaml(invariants_output_path)
        except Exception as exc:
            invariants_partial = True
            observations.append(f"cell invariants unreadable: {type(exc).__name__}: {exc}")

    if schema_partial:
        schema_recall = schema_precision = schema_type_accuracy = 0.0
        schema_mode = FailureMode.PARTIAL if not invariants_partial else FailureMode.CRASH
        s_ref_n = len(schema_field_identities(ref_schema))
        s_cell_n = 0
        s_int_n = 0
        schema_miss: list[ItemDiff] = []
        schema_spur: list[ItemDiff] = []
        schema_type_diffs: list[ItemDiff] = []
    else:
        (
            schema_recall,
            schema_precision,
            schema_type_accuracy,
            schema_mode,
            s_ref_n,
            s_cell_n,
            s_int_n,
            schema_miss,
            schema_spur,
            schema_type_diffs,
        ) = score_schema(reference_schema=ref_schema, cell_schema=cell_schema, config=cfg)

    if invariants_partial:
        inv_recall = inv_precision = inv_severity_accuracy = 0.0
        inv_mode = FailureMode.PARTIAL if not schema_partial else FailureMode.CRASH
        i_ref_n = len(invariant_identities(ref_invs))
        i_cell_n = 0
        i_int_n = 0
        inv_miss: list[ItemDiff] = []
        inv_spur: list[ItemDiff] = []
        inv_sev_diffs: list[ItemDiff] = []
    else:
        (
            inv_recall,
            inv_precision,
            inv_severity_accuracy,
            inv_mode,
            i_ref_n,
            i_cell_n,
            i_int_n,
            inv_miss,
            inv_spur,
            inv_sev_diffs,
        ) = score_invariants(reference_invariants=ref_invs, cell_invariants=cell_invs, config=cfg)

    if schema_partial and invariants_partial:
        # Both missing -> crash
        schema_mode = FailureMode.CRASH
        inv_mode = FailureMode.CRASH

    # Aggregate failure_modes (deduped)
    all_modes = failure_modes + [schema_mode.value, inv_mode.value]
    seen: set[str] = set()
    deduped: list[str] = []
    for m in all_modes:
        if m not in seen:
            deduped.append(m)
            seen.add(m)

    from datetime import datetime, timezone

    score = CellScore(
        strategy=strategy,
        engine=engine,
        version_slug=version_slug,
        bump_distance=bump_distance,
        schema_recall=schema_recall,
        schema_precision=schema_precision,
        schema_type_accuracy=schema_type_accuracy,
        invariant_recall=inv_recall,
        invariant_precision=inv_precision,
        invariant_severity_accuracy=inv_severity_accuracy,
        wall_clock_sec=wall_clock_sec,
        energy_wh=energy_wh,
        schema_failure_mode=schema_mode.value,
        invariant_failure_mode=inv_mode.value,
        failure_modes=deduped,
        schema_reference_count=s_ref_n,
        schema_cell_count=s_cell_n,
        schema_intersection_count=s_int_n,
        invariant_reference_count=i_ref_n,
        invariant_cell_count=i_cell_n,
        invariant_intersection_count=i_int_n,
        observations=observations,
        scored_at=datetime.now(timezone.utc).isoformat(),
        reference_path=str(reference_schema_path),
        cell_schema_path=str(schema_output_path) if schema_output_path else "",
        cell_invariants_path=str(invariants_output_path) if invariants_output_path else "",
    )
    return score, schema_miss + schema_spur, inv_miss + inv_spur, schema_type_diffs + inv_sev_diffs


# ---------------------------------------------------------------------------
# Brittleness post-processing
# ---------------------------------------------------------------------------


def compute_brittleness(
    *,
    cell: CellScore,
    active_cell: CellScore | None,
) -> CellScore:
    """Annotate a cell with brittleness sub-metrics by comparing against
    the same-strategy active-version cell.

    No-op if ``cell.bump_distance == "active"`` or ``active_cell is None``.
    Otherwise computes pass-through rate, silent/detectable counts, and a
    naive patch-cost estimate (file-diff for strategy ``"a"``; ``None`` =
    "needs human estimate" for the LLM strategies).

    NotImplementedError for stub Phase 1. Phase 3 fills this in.
    """
    raise NotImplementedError(
        "Phase 1 stub. Phase 3 implements:\n"
        "  if cell.bump_distance == 'active' or active_cell is None: return cell\n"
        "  pt_rate = compute_pass_through(cell, active_cell)\n"
        "  silent, detectable = silent_vs_detectable_breakdown(cell, active_cell)\n"
        "  patch_cost = auto_patch_cost(cell, active_cell) if cell.strategy == 'a' else None\n"
        "  return dataclass.replace(cell, brittleness_pass_through_rate=pt_rate, ...)"
    )


# ---------------------------------------------------------------------------
# Sidecar artefact writers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runtime-validation feedback (Phase 2.5; optional / not folded into precision)
# ---------------------------------------------------------------------------


@dataclass
class RuntimeValidation:
    """Per-invariant runtime-validation outcome for the Phase 2.5 "could"
    metric ``runtime_validated_count``.

    Plan recommended this as a SEPARATE metric (not folded into
    precision); a cell that emits invariants which DO trigger live
    library behaviour is more trustworthy than one whose invariants
    don't.

    Phase 2.5 ships scaffolding only - the actual replay against the
    live library (via ``scripts._invariant_validation_common.run_case``)
    requires the engine's runtime imports, which the project venv has
    only for transformers v4_57_3 active. The full matrix's
    runtime-validation pass is deferred to Phase 4 post-hoc analysis.
    """

    invariant_id: str
    """Invariant identity (the ``id`` field, not the 4-tuple)."""

    positive_confirmed: bool = False
    """True if the live library raised/warned/dormanted on
    ``kwargs_positive`` (the case that SHOULD trigger)."""

    negative_confirmed: bool = False
    """True if the live library did NOT raise/warn on ``kwargs_negative``
    (the case that should NOT trigger)."""

    observed_outcome: str = ""
    """The captured outcome label (error / dormant_announced / warn / no_op)."""

    error: str | None = None
    """If the validation infrastructure crashed (e.g. wrong native_type,
    can't instantiate), the error message lives here. The invariant is
    NOT counted as validated in this case."""


def runtime_validate_invariants(
    invariants_path: Path,
    *,
    engine: str = "transformers",
    only_ids: set[str] | None = None,
) -> list[RuntimeValidation]:
    """Replay each cell-emitted invariant's positive/negative cases against
    the live library.

    Phase 2.5 scope: transformers only (the only engine importable from
    the project venv). Other engines raise NotImplementedError; Phase 4
    runs them inside the per-engine canonical containers.

    Returns one ``RuntimeValidation`` per invariant. The aggregator can
    compute:

    - ``runtime_validated_positive_count`` = number with positive_confirmed.
    - ``runtime_validated_negative_count`` = number with negative_confirmed.
    - ``runtime_validated_both_count`` = number with both.

    These are NOT folded into precision; they're separate signals the
    coordinator can use to compare cells.
    """
    if engine != "transformers":
        raise NotImplementedError(
            f"Runtime validation for engine={engine!r} requires its canonical "
            f"container; Phase 2.5 ships transformers-only scaffolding."
        )

    inv_data = yaml.safe_load(invariants_path.read_text()) or {}
    inv_list = inv_data.get("invariants") or []
    out: list[RuntimeValidation] = []

    try:
        from scripts._invariant_validation_common import run_case  # type: ignore[import]
    except ImportError as exc:
        # Project's validation infrastructure unavailable; record the gap.
        for inv in inv_list:
            inv_id = inv.get("id", "") if isinstance(inv, dict) else ""
            out.append(
                RuntimeValidation(
                    invariant_id=inv_id,
                    error=f"validation infra unavailable: {exc}",
                )
            )
        return out

    for inv in inv_list:
        if not isinstance(inv, dict):
            continue
        inv_id = inv.get("id", "")
        if only_ids is not None and inv_id not in only_ids:
            continue
        rv = _runtime_validate_one_transformers(inv, run_case=run_case)
        out.append(rv)
    return out


def _runtime_validate_one_transformers(
    inv: dict[str, Any],
    *,
    run_case: Any,
) -> RuntimeValidation:
    """Run one transformers invariant's positive + negative cases.

    Returns a ``RuntimeValidation`` with positive/negative confirmations.
    Catches ALL exceptions in the validation infrastructure - a crash here
    is metadata, not a hard error.
    """
    inv_id = inv.get("id", "")
    native_type = inv.get("native_type", "")
    kwargs_positive = inv.get("kwargs_positive") or {}
    kwargs_negative = inv.get("kwargs_negative") or {}

    if not isinstance(kwargs_positive, dict) or not isinstance(kwargs_negative, dict):
        return RuntimeValidation(invariant_id=inv_id, error="kwargs not dicts")

    try:
        if "GenerationConfig" in native_type:
            from transformers import GenerationConfig  # type: ignore[import]

            def make_pos() -> Any:
                return GenerationConfig(**kwargs_positive)

            def make_neg() -> Any:
                return GenerationConfig(**kwargs_negative)
        elif "BitsAndBytesConfig" in native_type:
            from transformers import BitsAndBytesConfig  # type: ignore[import]

            def make_pos() -> Any:
                return BitsAndBytesConfig(**kwargs_positive)

            def make_neg() -> Any:
                return BitsAndBytesConfig(**kwargs_negative)
        else:
            return RuntimeValidation(
                invariant_id=inv_id, error=f"unsupported native_type: {native_type!r}"
            )
    except Exception as exc:
        return RuntimeValidation(invariant_id=inv_id, error=f"setup failure: {exc}")

    try:
        pos_capture = run_case(make_pos, logger_names=("transformers",))
    except Exception as exc:
        return RuntimeValidation(invariant_id=inv_id, error=f"positive case crash: {exc}")
    try:
        neg_capture = run_case(make_neg, logger_names=("transformers",))
    except Exception as exc:
        return RuntimeValidation(
            invariant_id=inv_id,
            error=f"negative case crash: {exc}",
            positive_confirmed=bool(pos_capture.exception_type or pos_capture.logger_messages),
        )

    # Positive: should trigger SOMETHING (exception, warning, or logger msg)
    pos_triggered = bool(
        pos_capture.exception_type or pos_capture.warnings_captured or pos_capture.logger_messages
    )
    # Negative: should NOT trigger (clean construction, no exception/warnings)
    neg_clean = (
        (
            pos_capture.exception_type is None
            and not neg_capture.exception_type
            and not neg_capture.warnings_captured
            and not neg_capture.logger_messages
        )
        if pos_triggered
        else False
    )
    # Actually, neg_clean refers to neg_capture only:
    neg_clean = (
        not neg_capture.exception_type
        and not neg_capture.warnings_captured
        and not neg_capture.logger_messages
    )

    # Determine outcome label
    if pos_capture.exception_type:
        outcome = "error"
    elif pos_capture.warnings_captured:
        outcome = "warn"
    elif pos_capture.logger_messages:
        outcome = "dormant_announced"
    else:
        outcome = "no_op"

    return RuntimeValidation(
        invariant_id=inv_id,
        positive_confirmed=pos_triggered,
        negative_confirmed=neg_clean,
        observed_outcome=outcome,
    )


def write_diff_artefact(
    *,
    diffs: list[ItemDiff],
    out_path: Path,
    kind_label: str,
) -> None:
    """Write a list of ItemDiff entries to YAML for human review.

    ``kind_label`` lands as the envelope's top-level key (e.g.
    ``"recall_misses"``); each entry is a dict with the ItemDiff fields.
    Stable sort by identity tuple so re-runs produce byte-identical files
    (the aggregator can diff them across runs).
    """
    sorted_diffs = sorted(diffs, key=lambda d: d.identity)
    envelope = {
        "scoring_format_version": SCORING_FORMAT_VERSION,
        kind_label: [
            {
                "identity": list(d.identity),
                "kind": d.kind,
                "reference_value": d.reference_value,
                "cell_value": d.cell_value,
                "note": d.note,
            }
            for d in sorted_diffs
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(envelope, sort_keys=False, default_flow_style=False))


# ---------------------------------------------------------------------------
# Test helpers (used by test_trial_scoring.py)
# ---------------------------------------------------------------------------


def _smoke_self_score_reference(
    *,
    reference_schema_path: Path,
    reference_invariants_path: Path,
) -> CellScore:
    """Score the reference against itself. Expected: recall=precision=1.0
    on both axes; type_accuracy and severity_accuracy=1.0; failure_modes
    all NONE.

    Used by ``test_trial_scoring.py`` as the foundation smoke check.

    NotImplementedError for stub Phase 1; the test asserts NotImplementedError
    is raised by ``score_cell`` until Phase 2 wires up the real scorer.
    """
    return score_cell(  # type: ignore[return-value]
        strategy="a",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="active",
        schema_output_path=reference_schema_path,
        invariants_output_path=reference_invariants_path,
        reference_schema_path=reference_schema_path,
        reference_invariants_path=reference_invariants_path,
        wall_clock_sec=0.0,
        energy_wh=0.0,
    )
