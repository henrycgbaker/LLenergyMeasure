"""Equivalence-groups sidecar - pre-run resolved groups + post-run observed detection.

Design: ``.product/designs/config-deduplication-dormancy/sweep-dedup.md`` §6.

Written alongside the study's results bundle. ``pre_run_groups`` is populated
at sweep-expansion time by :func:`resolve_library_effective` and serialised immediately;
``observed_collision_groups`` is populated after the study completes by scanning
sidecars for shared observed-config-hash values.

The observed-config-hash collision invariant (§4.1) guarantees that in a post-resolved-config-hash dedup run set,
any group with ``len(member_resolved_config_hashes) >= 2`` is a **proven library-resolution mechanism gap**.
The file is a post-hoc analysis artifact; nothing reads it back at runtime
(``llem report-gaps`` consumes ``runtime_observations.jsonl``, not this file).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from llenergymeasure.results.persistence import _atomic_write

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreRunGroup:
    """Pre-run equivalence group recorded at sweep-expansion time."""

    resolved_config_hash: str
    canonical_config_excerpt: dict[str, Any]
    member_experiment_ids: tuple[str, ...]
    member_count: int
    representative_experiment_id: str
    would_dedup: bool
    deduplicated: bool


@dataclass(frozen=True)
class ObservedCollisionGroup:
    """Post-run observed-config-hash collision group - a library-resolution mechanism gap if member count >= 2."""

    observed_config_hash: str
    engine: str
    engine_version: str
    member_resolved_config_hashes: tuple[str, ...]
    member_experiment_ids: tuple[str, ...]
    gap_detected: bool


@dataclass
class EquivalenceGroups:
    """Top-level equivalence-groups record written as ``equivalence_groups.json``."""

    study_id: str
    study_name: str
    dedup_mode: Literal["resolved", "off"]
    validated_rules_version: str = ""
    groups: list[PreRunGroup] = field(default_factory=list)
    observed_collision_groups: list[ObservedCollisionGroup] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Post-run observed-config-hash grouping - scan sidecars after study completes
# ---------------------------------------------------------------------------


def find_observed_collisions(sidecars: list[dict[str, Any]]) -> list[ObservedCollisionGroup]:
    """Group sidecars by ``(engine, engine_version, observed_config_hash)``.

    Any group with size >= 2 AND distinct ``resolved_config_hash`` across its members is
    flagged as a proven library-resolution mechanism gap - per sweep-dedup.md §4.1.

    Each sidecar dict must carry at minimum ``engine``, ``engine_version``,
    ``resolved_config_hash``, ``observed_config_hash``, and ``experiment_id`` keys. Sidecars missing any
    of these are silently skipped (pre-50.3a data, or runs with dedup_mode=off
    for which observed-config-hash may be partial).
    """
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sc in sidecars:
        obs_hash = sc.get("observed_config_hash")
        if not obs_hash:
            continue
        engine = str(sc.get("engine", ""))
        version = str(sc.get("engine_version", ""))
        buckets[(engine, version, obs_hash)].append(sc)

    groups: list[ObservedCollisionGroup] = []
    for (engine, version, obs_hash), members in buckets.items():
        if len(members) < 2:
            continue
        resolved_config_hashes = tuple(str(m.get("resolved_config_hash", "")) for m in members)
        exp_ids = tuple(str(m.get("experiment_id", "")) for m in members)
        # Gap only if the resolved_config_hashes differ - matching resolved-config means the
        # library-resolution mechanism already collapsed them.
        gap_detected = len(set(resolved_config_hashes)) > 1
        groups.append(
            ObservedCollisionGroup(
                observed_config_hash=obs_hash,
                engine=engine,
                engine_version=version,
                member_resolved_config_hashes=resolved_config_hashes,
                member_experiment_ids=exp_ids,
                gap_detected=gap_detected,
            )
        )
    return groups


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


def write_equivalence_groups(groups: EquivalenceGroups, path: Path) -> None:
    """Write :class:`EquivalenceGroups` to ``path`` atomically as JSON.

    Each group dataclass is converted to a dict via ``asdict`` before dumping;
    the top-level scalar fields are added by hand.
    """
    payload = {
        "study_id": groups.study_id,
        "study_name": groups.study_name,
        "dedup_mode": groups.dedup_mode,
        "validated_rules_version": groups.validated_rules_version,
        "groups": [asdict(g) for g in groups.groups],
        "observed_collision_groups": [asdict(g) for g in groups.observed_collision_groups],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json.dumps(payload, indent=2, default=str), path)
