"""``llem report-gaps`` - feedback-loop proposer for the rules corpus.

Reads ``runtime_observations.jsonl`` emitted by
:mod:`llenergymeasure.study.runtime_observations`, groups captured warnings
and log records by their normalised message template, partitions configs
into *collision_configs* (A) and *not collision_configs* (B), and proposes one
corpus rule per observed template.

Design:

- **No corpus mutation.** We emit YAML *fragments* to ``--out PATH``; the
  live ``src/llenergymeasure/engines/{engine}/rules.yaml`` is never touched.
- **Severity is mechanical.** ``dormant`` for log-channel emissions (the
  engine silently ignored a declared field); ``error`` when
  ``include_exceptions=True`` and the record is an exception.
- **Emission channels.** ``EmissionChannel`` is this module's own runtime
  taxonomy (which channel a captured record came through); it is not part of
  the shipped rules schema.
- **Sentinel filtering.** ``subprocess_died`` / ``exception`` records
  don't prove "rule didn't fire" - excluded from B always; excluded from
  A unless ``include_exceptions=True``.
- **Emission channels.** This module produces ``warnings_warn``,
  ``logger_warning``, ``logger_warning_once``, and ``runtime_exception``
  only - a documented subset of :data:`EmissionChannel`.

Source of config kwargs: per-experiment ``_resolution.json`` sidecars,
flattened into ``dict[str, Any]`` per ``config_hash``. Located via
``manifest.json`` when present (keyed on full hash), with an 8-char
prefix-scan fallback for manifest-less studies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import yaml

from llenergymeasure.config.ssot import Engine
from llenergymeasure.domain.bundle_artefacts import MANIFEST_FILENAME, RESOLUTION_FILENAME
from llenergymeasure.study.runtime_observations import RUNTIME_OBSERVATIONS_FILENAME
from llenergymeasure.utils.io import load_json

__all__ = [
    "GapProposal",
    "ReportGapsError",
    "find_runtime_gaps",
    "render_yaml_fragment",
]

logger = logging.getLogger(__name__)

EmissionChannel = Literal[
    "warnings_warn",
    "logger_warning",
    "logger_warning_once",
    "minor_issues_dict",
    "none",
    "runtime_exception",
]
"""How an engine user-visibly signals an issue in a captured runtime record.

This is the gap-proposer's own runtime taxonomy, not part of the shipped
rules schema (the closed severity enum lives in
:mod:`llenergymeasure.config.engine_rules`). It classifies the channel a
captured warning/log/exception came through so a proposed fragment can note
it for the human reviewer.
"""

#: Eight-hex-char prefix used as the fallback key when ``manifest.json``
#: is missing. Collision risk ~10^-9 per pair at study scale; the manifest
#: path avoids it entirely when available.
_HASH_PREFIX_LEN = 8


class ReportGapsError(Exception):
    """Raised when ``find_runtime_gaps`` fails with a user-actionable cause."""


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapProposal:
    """One proposed corpus rule for a single unmatched normalised template."""

    normalised_template: str
    source_channel: EmissionChannel
    engine: Engine
    library_version: str
    match_fields: dict[str, Any] | None
    evidence_field_value_distribution: dict[str, dict[str, list[str]]]
    collision_count: int
    contrast_count: int
    representative_message: str
    needs_generalisation_review: bool
    severity: Literal["dormant", "error"]


# ---------------------------------------------------------------------------
# Internal record shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Emission:
    template: str
    representative_message: str
    channel: EmissionChannel
    config_hash: str
    engine: Engine
    library_version: str


@dataclass(frozen=True)
class _Record:
    config_hash: str
    engine: Engine
    library_version: str
    outcome: str
    emissions: list[_Emission]
    kwargs: dict[str, Any] | None


_ALLOWED_ENGINES: frozenset[str] = frozenset(e.value for e in Engine)


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------


def find_runtime_gaps(
    study_dirs: list[Path],
    engine: Engine | str | None = None,
    include_exceptions: bool = False,
) -> list[GapProposal]:
    """Scan one or more study directories and return one proposal per template.

    Every distinct ``(engine, normalised_template)`` observed at runtime
    yields a proposal; output sorts by that key. ``engine`` accepts an
    :class:`Engine` enum or its string value and filters to a single engine.
    """
    if not study_dirs:
        raise ReportGapsError("No study directories provided. Pass at least one --study-dir.")

    engine_filter: str | None = str(engine) if engine is not None else None
    if engine_filter is not None and engine_filter not in _ALLOWED_ENGINES:
        raise ReportGapsError(
            f"Unsupported engine filter: {engine_filter!r}. "
            f"Expected one of: {sorted(_ALLOWED_ENGINES)}."
        )

    # Fail loud on a missing / non-directory study dir. A typo'd path would
    # otherwise yield "No gaps found" with exit 0 - a false all-clear on a
    # corpus-gating command.
    for study_dir in study_dirs:
        path = Path(study_dir)
        if not path.exists():
            raise ReportGapsError(f"Study directory does not exist: {path}")
        if not path.is_dir():
            raise ReportGapsError(f"Study path is not a directory: {path}")

    records: list[_Record] = []
    sidecar_failures = 0
    for study_dir in study_dirs:
        study_records, study_failures = _load_study(
            Path(study_dir), engine_filter, include_exceptions
        )
        records.extend(study_records)
        sidecar_failures += study_failures
    if sidecar_failures:
        logger.warning(
            "Skipped %d malformed _resolution.json sidecar(s); see prior WARNING log lines.",
            sidecar_failures,
        )

    if not records:
        return []

    # Build unique (engine, template) → emissions map. Key on engine too
    # because the same string emitted by two engines is conceptually two
    # different rules (different native types, different severity).
    emissions_by_key: dict[tuple[str, str], list[_Emission]] = {}
    representative_by_key: dict[tuple[str, str], _Emission] = {}
    for rec in records:
        for emission in rec.emissions:
            key = (emission.engine.value, emission.template)
            emissions_by_key.setdefault(key, []).append(emission)
            representative_by_key.setdefault(key, emission)

    records_by_engine: dict[str, list[_Record]] = {}
    for rec in records:
        records_by_engine.setdefault(rec.engine.value, []).append(rec)

    proposals: list[GapProposal] = []
    for (eng, template), emissions in sorted(emissions_by_key.items()):
        fired_hashes = {e.config_hash for e in emissions}
        collision_configs: list[dict[str, Any]] = []
        contrast_configs: list[dict[str, Any]] = []
        for rec in records_by_engine.get(eng, []):
            if rec.kwargs is None:
                continue
            if rec.config_hash in fired_hashes:
                collision_configs.append(rec.kwargs)
            elif rec.outcome not in {"subprocess_died", "exception"}:
                contrast_configs.append(rec.kwargs)

        match_fields = _infer_predicate(collision_configs, contrast_configs)
        evidence = _field_value_distribution(collision_configs, contrast_configs)
        rep = representative_by_key[(eng, template)]
        severity: Literal["dormant", "error"] = (
            "error" if rep.channel == "runtime_exception" else "dormant"
        )
        needs_review = match_fields is None or len(match_fields) >= 2

        proposals.append(
            GapProposal(
                normalised_template=template,
                source_channel=rep.channel,
                engine=rep.engine,
                library_version=rep.library_version,
                match_fields=match_fields,
                evidence_field_value_distribution=evidence,
                collision_count=len(collision_configs),
                contrast_count=len(contrast_configs),
                representative_message=rep.representative_message,
                needs_generalisation_review=needs_review,
                severity=severity,
            )
        )
    return proposals


# ---------------------------------------------------------------------------
# JSONL loading + kwargs resolution
# ---------------------------------------------------------------------------


# Table driving the two record-level list emission sources. Each entry is
# ``(jsonl_list_key, emission_channel, template_field)``. Exception records
# are a single-dict shape and get their own append block below.
_EMISSION_SOURCES: tuple[tuple[str, EmissionChannel, str], ...] = (
    ("warnings", "warnings_warn", "message_template"),
    ("logger_records", "logger_warning", "message_template"),
)


def _append_emission(
    emissions: list[_Emission],
    item: dict[str, Any],
    channel: EmissionChannel,
    template_field: str,
    *,
    config_hash: str,
    engine: Engine,
    library_version: str,
) -> None:
    """Append one :class:`_Emission` if ``item`` carries a non-empty template."""
    template = str(item.get(template_field) or "")
    if not template:
        return
    emissions.append(
        _Emission(
            template=template,
            representative_message=str(item.get("message") or template),
            channel=channel,
            config_hash=config_hash,
            engine=engine,
            library_version=library_version,
        )
    )


def _load_study(
    study_dir: Path,
    engine_filter: str | None,
    include_exceptions: bool,
) -> tuple[list[_Record], int]:
    """Return (records, sidecar_failure_count) for one study dir."""
    jsonl_path = study_dir / RUNTIME_OBSERVATIONS_FILENAME
    if not jsonl_path.exists():
        logger.info("No %s in %s; skipping.", RUNTIME_OBSERVATIONS_FILENAME, study_dir)
        return [], 0

    kwargs_by_hash, sidecar_failures = _load_kwargs_by_hash(study_dir)

    records: list[_Record] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s.", line_no, jsonl_path)
                continue
            rec = _parse_record(raw, engine_filter, include_exceptions, kwargs_by_hash)
            if rec is not None:
                records.append(rec)
    return records, sidecar_failures


def _parse_record(
    raw: dict[str, Any],
    engine_filter: str | None,
    include_exceptions: bool,
    kwargs_by_hash: dict[str, dict[str, Any]],
) -> _Record | None:
    engine_str = str(raw.get("engine", ""))
    if engine_str not in _ALLOWED_ENGINES:
        return None
    if engine_filter is not None and engine_str != engine_filter:
        return None
    engine_enum = Engine(engine_str)
    outcome = str(raw.get("outcome", ""))
    config_hash = str(raw.get("config_hash", ""))
    library_version = str(raw.get("library_version", ""))

    emissions: list[_Emission] = []
    for list_key, channel, template_field in _EMISSION_SOURCES:
        for item in raw.get(list_key) or []:
            _append_emission(
                emissions,
                item,
                channel,
                template_field,
                config_hash=config_hash,
                engine=engine_enum,
                library_version=library_version,
            )
    if include_exceptions and outcome == "exception":
        _append_emission(
            emissions,
            raw.get("exception") or {},
            "runtime_exception",
            "message_template",
            config_hash=config_hash,
            engine=engine_enum,
            library_version=library_version,
        )

    return _Record(
        config_hash=config_hash,
        engine=engine_enum,
        library_version=library_version,
        outcome=outcome,
        emissions=emissions,
        kwargs=kwargs_by_hash.get(config_hash),
    )


def _load_kwargs_by_hash(study_dir: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Return ``({full_config_hash: flat_kwargs_dict}, sidecar_failure_count)``.

    Preferred path: read ``manifest.json`` (written by
    :class:`llenergymeasure.study.manifest.ManifestWriter`) which keys
    entries by full ``config_hash`` and records the ``result_file``
    relative path - ``_resolution.json`` sits in the same directory.

    Fallback path: scan experiment subdirs and key on the 8-char hex
    suffix. Logged as a warning so operators know the preferred lookup
    is unavailable.
    """
    if not study_dir.exists() or not study_dir.is_dir():
        return {}, 0
    manifest_path = study_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return _load_kwargs_via_manifest(study_dir, manifest_path)
    logger.warning(
        "manifest.json not found in %s; falling back to prefix-scan (collision risk ~1e-9).",
        study_dir,
    )
    return _load_kwargs_via_prefix_scan(study_dir)


def _load_kwargs_via_manifest(
    study_dir: Path, manifest_path: Path
) -> tuple[dict[str, dict[str, Any]], int]:
    try:
        manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read manifest.json at %s: %s", manifest_path, exc)
        return {}, 0
    by_hash: dict[str, dict[str, Any]] = {}
    failures = 0
    for entry in manifest.get("experiments") or []:
        if not isinstance(entry, dict):
            continue
        config_hash = str(entry.get("config_hash") or "")
        result_file = entry.get("result_file")
        if not config_hash or not isinstance(result_file, str) or not result_file:
            continue
        if config_hash in by_hash:
            # Same config can repeat across cycles; one flat dict suffices.
            continue
        resolution_path = study_dir / Path(result_file).parent / RESOLUTION_FILENAME
        if not resolution_path.exists():
            continue
        flat, failed = _read_resolution_sidecar(resolution_path)
        if failed:
            failures += 1
            continue
        if flat is not None:
            by_hash[config_hash] = flat
    return by_hash, failures


def _load_kwargs_via_prefix_scan(
    study_dir: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Fallback: walk subdirs with ``*_{hash8}`` suffix; key on full hash."""
    by_prefix: dict[str, dict[str, Any]] = {}
    failures = 0
    for entry in sorted(study_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Dir format: [{NNN}_]c{cycle}_{slug}_{hash8}
        parts = entry.name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        hash_prefix = parts[1]
        if len(hash_prefix) != _HASH_PREFIX_LEN or not all(
            c in "0123456789abcdef" for c in hash_prefix
        ):
            continue
        resolution_path = entry / RESOLUTION_FILENAME
        if not resolution_path.exists():
            continue
        flat, failed = _read_resolution_sidecar(resolution_path)
        if failed:
            failures += 1
            continue
        if flat is not None:
            by_prefix.setdefault(hash_prefix, flat)
    return _PrefixHashLookup(by_prefix), failures


def _read_resolution_sidecar(
    path: Path,
) -> tuple[dict[str, Any] | None, bool]:
    """Parse ``_resolution.json`` at ``path``; return (flat_dict or None, failed)."""
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse _resolution.json at %s: %s", path, exc)
        return None, True
    overrides = payload.get("overrides") or {}
    if not isinstance(overrides, dict):
        return None, False
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, dict) and "effective" in value:
            flat[str(key)] = value["effective"]
    return flat, False


class _PrefixHashLookup(dict):  # type: ignore[type-arg]
    """Prefix-scan fallback: ``.get(full_hash)`` resolves via first 8 hex chars.

    Subclasses ``dict`` so the call site's type annotation
    (``dict[str, dict[str, Any]]``) still holds. Only ``.get`` is
    consulted by callers.
    """

    def __init__(self, prefix_map: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self._prefix_map = prefix_map

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if not isinstance(key, str) or len(key) < _HASH_PREFIX_LEN:
            return default
        return self._prefix_map.get(key[:_HASH_PREFIX_LEN], default)


# ---------------------------------------------------------------------------
# Predicate inference (arity 1 → 2 → 3 + present:true fallback)
# ---------------------------------------------------------------------------


def _infer_predicate(
    collision_configs: list[dict[str, Any]],
    contrast_configs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the smallest field-value dict that distinguishes A from B, or None.

    Tries arity 1, 2, 3; the first arity whose tuple is uniformly shared
    across A and absent in B wins. ``present:true`` fallback catches "any
    non-None value triggers" shapes that equality-on-value never sees.
    """
    if not collision_configs:
        return None
    fields = sorted({k for c in collision_configs + contrast_configs for k in c})
    if not fields:
        return None

    for arity in range(1, 4):
        if arity > len(fields):
            break
        for fset in combinations(fields, arity):
            a_tuples = {tuple(c.get(f) for f in fset) for c in collision_configs}
            if len(a_tuples) != 1:
                continue
            value_tuple = next(iter(a_tuples))
            if any(
                all(b.get(f) == v for f, v in zip(fset, value_tuple, strict=False))
                for b in contrast_configs
            ):
                continue
            return dict(zip(fset, value_tuple, strict=False))

    for fname in fields:
        if all(c.get(fname) is not None for c in collision_configs) and all(
            b.get(fname) is None for b in contrast_configs
        ):
            return {fname: {"present": True}}
    return None


def _field_value_distribution(
    collision_configs: list[dict[str, Any]],
    contrast_configs: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Return per-field string-ified value sets for the A and B partitions."""
    out: dict[str, dict[str, list[str]]] = {}
    fields = sorted({k for c in collision_configs + contrast_configs for k in c})
    for f in fields:
        out[f] = {
            "collision_configs": sorted({str(c.get(f)) for c in collision_configs}),
            "contrast_configs": sorted({str(c.get(f)) for c in contrast_configs}),
        }
    return out


# ---------------------------------------------------------------------------
# YAML renderer
# ---------------------------------------------------------------------------


_BANNER = (
    "# Rule fragment proposed by 'llem report-gaps'. Review and APPEND to\n"
    "# src/llenergymeasure/engines/{engine}/rules.yaml under the 'rules:' key.\n"
    "# ----------------------------------------------------------------------\n"
    "# provenance.source is a placeholder ('manual'); confirm before merging.\n"
    "# The evidence block below is a reviewer aid - delete it before appending.\n"
)


def render_yaml_fragment(proposal: GapProposal) -> str:
    """Render one :class:`GapProposal` as a schema-valid Rule fragment.

    The document parses through
    :func:`llenergymeasure.config.engine_rules.loader._parse_rule` - every
    key is a real Rule field with enum-valid placeholders, so the round-trip
    test passes. A trailing comment block carries the runtime evidence
    (collision/contrast counts, per-field value distribution) for reviewers.
    """
    template = proposal.normalised_template
    match_fields: dict[str, Any] = dict(proposal.match_fields) if proposal.match_fields else {}
    engine_str = proposal.engine.value

    proposal_doc: dict[str, Any] = {
        "id": _proposed_rule_id(proposal),
        "engine": engine_str,
        "severity": proposal.severity,
        "match": {"fields": match_fields},
        "message_template": template,
        "provenance": {
            "source": "manual",
            "verified": "runtime",
            "engine_version": proposal.library_version or "<TODO: human - library version>",
            "citation": "<TODO: human - runtime-derived; no source citation>",
            "date": "<TODO: human - YYYY-MM-DD>",
        },
        "normalised_fields": [],
    }

    body = yaml.safe_dump(proposal_doc, sort_keys=False, default_flow_style=False, width=100)
    return _BANNER + body + _render_evidence_comment(proposal)


def _render_evidence_comment(proposal: GapProposal) -> str:
    """Return a ``#``-prefixed reviewer evidence block (ignored by the loader)."""
    lines = [
        "# --- reviewer evidence (delete before appending) ---",
        f"# source_channel: {proposal.source_channel}",
        f"# needs_generalisation_review: {proposal.needs_generalisation_review}",
        (
            f"# observed in {proposal.collision_count} config(s); "
            f"absent in {proposal.contrast_count} config(s)"
        ),
        f"# representative message: {proposal.representative_message!r}",
    ]
    for field_name, dist in sorted(proposal.evidence_field_value_distribution.items()):
        lines.append(
            f"#   {field_name}: fired={dist['collision_configs']} silent={dist['contrast_configs']}"
        )
    return "\n".join(lines) + "\n"


def _proposed_rule_id(proposal: GapProposal) -> str:
    """Return ``{engine}_runtime_{slug}`` for a reviewer-friendly rule id."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", proposal.normalised_template.lower()).strip("_")
    slug = cleaned[:40] or "unnamed"
    return f"{proposal.engine.value}_runtime_{slug}"
