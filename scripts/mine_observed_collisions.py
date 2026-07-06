#!/usr/bin/env python3
"""Observed-collision miner (proposer S3).

Scans a completed study results directory and proposes dormant-rule CANDIDATES
from observed collisions: experiments whose declared configs DIFFER yet whose
engine-effective (observed) state is identical.

Per experiment ``config.json`` sidecar: group by
``(engine, library_version, observed_config_hash)`` (a shared observed hash
means the engine resolved every member to the same effective config); within
any group whose DECLARED configs differ, diff them field-by-field to name the
fields whose variation left the observed state identical. Each such field is a
dormancy candidate. When several fields vary together (a multi-field diff),
each candidate lists the others as ``co_occurring_fields`` - the miner cannot
attribute dormancy to one field alone; the verification ladder settles that.

Candidates go to the VERSION-SCOPED pool
(``<pool-root>/<engine>/v<version>/candidates/observed_collisions.yaml``) as
UNVERIFIED proposals. This script NEVER writes to the shipped
``src/llenergymeasure/engines/<engine>/rules.yaml`` corpora, which are frozen.

Usage: ``python scripts/mine_observed_collisions.py <study_dir> [--pool-root DIR]``
Exit codes: 0 = scan completed; 2 = study directory does not exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("mine_observed_collisions")

SCHEMA_VERSION = "1.0.0"

# Sentinel for "field absent in this declared config" - distinct from a declared
# null so absence and an explicit None never compare equal.
_MISSING = object()


@dataclass
class _Sidecar:
    """The fields of one experiment ``config.json`` the miner needs."""

    experiment_dir: str
    engine: str
    library_version: str
    observed_config_hash: str
    declared_config: dict[str, Any]


@dataclass
class ScanStats:
    """Counters surfaced in the run summary (never raise, always report)."""

    experiments_scanned: int = 0
    skipped_missing_declared: int = 0
    skipped_missing_observed_hash: int = 0
    skipped_unreadable: int = 0
    collision_groups: int = 0


# ---------------------------------------------------------------------------
# Declared-config diffing
# ---------------------------------------------------------------------------


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested declared config to dotted leaf paths.

    ``{"vllm": {"engine_params": {"enforce_eager": True}}}`` becomes
    ``{"vllm.engine_params.enforce_eager": True}``. Lists and scalars are
    leaves; only dicts are descended.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _value_key(value: Any) -> str:
    """Stable comparison key for a declared value (handles lists/dicts)."""
    if value is _MISSING:
        return "\x00__missing__"
    return json.dumps(value, sort_keys=True, default=str)


def _distinct_values(field_path: str, flats: list[dict[str, Any]]) -> list[Any]:
    """Distinct declared values for ``field_path`` across a group, first-seen order."""
    seen: set[str] = set()
    out: list[Any] = []
    for flat in flats:
        value = flat.get(field_path, _MISSING)
        vkey = _value_key(value)
        if vkey not in seen:
            seen.add(vkey)
            out.append("<absent>" if value is _MISSING else value)
    return out


def _differing_fields(flats: list[dict[str, Any]]) -> list[str]:
    """Leaf paths whose value is not identical across every config in the group."""
    all_paths = set().union(*(flat.keys() for flat in flats)) if flats else set()
    differing: list[str] = []
    for path in sorted(all_paths):
        values = {_value_key(flat.get(path, _MISSING)) for flat in flats}
        if len(values) > 1:
            differing.append(path)
    return differing


def _slug(field_path: str) -> str:
    """Filesystem/id-safe slug for a dotted field path."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", field_path)).strip("_")


# ---------------------------------------------------------------------------
# Loading + mining
# ---------------------------------------------------------------------------


def load_sidecars(study_dir: Path) -> tuple[list[_Sidecar], ScanStats]:
    """Read every ``config.json`` under ``study_dir`` into sidecar records.

    Experiments whose sidecar predates the declared-config field, or that lack
    an observed hash, are SKIPPED with a counted warning - never a crash.
    """
    stats = ScanStats()
    sidecars: list[_Sidecar] = []
    for config_path in sorted(study_dir.rglob("config.json")):
        stats.experiments_scanned += 1
        rel_dir = str(config_path.parent.relative_to(study_dir))
        try:
            payload = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            stats.skipped_unreadable += 1
            logger.warning("skipping %s: unreadable config.json", rel_dir)
            continue
        declared = payload.get("declared_config")
        if not isinstance(declared, dict):
            stats.skipped_missing_declared += 1
            logger.warning("skipping %s: legacy sidecar without declared_config", rel_dir)
            continue
        observed_hash = payload.get("observed_config_hash")
        if not observed_hash:
            stats.skipped_missing_observed_hash += 1
            logger.warning("skipping %s: no observed_config_hash", rel_dir)
            continue
        sidecars.append(
            _Sidecar(
                experiment_dir=rel_dir,
                engine=str(payload.get("engine", "unknown")),
                library_version=str(payload.get("library_version", "unknown")),
                observed_config_hash=str(observed_hash),
                declared_config=declared,
            )
        )
    return sidecars, stats


def _make_candidate(
    field_path: str,
    co_occurring: list[str],
    members: list[_Sidecar],
    flats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one dormant-rule candidate for a single differing field.

    Shape is the unified candidate schema the verification ladder consumes:
    the constrained field lives under ``match.fields`` with a ``present`` spec
    (the identity probe reads the recorded ``declared_values`` from provenance),
    and the (unverified) ``verdict`` block is filled in by the probe kernel. No
    ``citation`` key: observed-collision candidates cite runtime evidence, not
    source, so ``check_citations.py`` skips rather than fails them.
    """
    engine = members[0].engine
    version = members[0].library_version
    observed_hash = members[0].observed_config_hash
    experiment_dirs = sorted(m.experiment_dir for m in members)
    return {
        "id": f"{engine}_collision_dormant_{_slug(field_path)}_{observed_hash[:8]}",
        "engine": engine,
        "engine_version": version,
        "severity": "dormant",
        "match": {"fields": {field_path: {"present": True}}},
        "provenance": {
            "source": "observed_collision",
            "engine": engine,
            "engine_version": version,
            "observed_config_hash": observed_hash,
            "colliding_experiments": experiment_dirs,
            "declared_values": _distinct_values(field_path, flats),
            "co_occurring_fields": co_occurring,
        },
        "verdict": {"status": "unverified", "tier": None, "date": None, "evidence": None},
    }


def mine_collisions(sidecars: list[_Sidecar], stats: ScanStats) -> list[dict[str, Any]]:
    """Group by observed identity and emit dormant-rule candidates for diffs."""
    groups: dict[tuple[str, str, str], list[_Sidecar]] = defaultdict(list)
    for sidecar in sidecars:
        key = (sidecar.engine, sidecar.library_version, sidecar.observed_config_hash)
        groups[key].append(sidecar)

    candidates: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        flats = [_flatten(m.declared_config) for m in members]
        differing = _differing_fields(flats)
        if not differing:
            # Declared configs identical (e.g. repeat cycles) - dedup, not a collision.
            continue
        stats.collision_groups += 1
        for field_path in differing:
            co_occurring = [other for other in differing if other != field_path]
            candidates.append(_make_candidate(field_path, co_occurring, members, flats))
    return candidates


# ---------------------------------------------------------------------------
# Writing to the version-scoped candidate pool
# ---------------------------------------------------------------------------

_POOL_HEADER = (
    "# Observed-collision candidate rules (proposer S3).\n"
    "# Emitted by scripts/mine_observed_collisions.py from a completed study.\n"
    "# UNVERIFIED candidates for the verification ladder. Do NOT hand-copy into\n"
    "# the shipped src/llenergymeasure/engines/<engine>/rules.yaml corpora.\n"
)


def _pool_path(pool_root: Path, engine: str, version: str) -> Path:
    version_dir = "v" + re.sub(r"[^0-9a-zA-Z]+", "_", version).strip("_")
    return pool_root / engine / version_dir / "candidates" / "observed_collisions.yaml"


def write_candidates(
    candidates: list[dict[str, Any]],
    pool_root: Path,
    *,
    generated_at: str | None = None,
) -> list[Path]:
    """Write candidates to per-(engine, version) files under the pool root."""
    generated_at = generated_at or date.today().isoformat()
    by_pool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_pool[(candidate["engine"], candidate["engine_version"])].append(candidate)

    written: list[Path] = []
    for (engine, version), group in sorted(by_pool.items()):
        document = {
            "schema_version": SCHEMA_VERSION,
            "source": "observed_collision",
            "engine": engine,
            "engine_version": version,
            "generated_at": generated_at,
            "candidates": group,
        }
        path = _pool_path(pool_root, engine, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
        path.write_text(_POOL_HEADER + body)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(stats: ScanStats, candidates: list[dict[str, Any]], written: list[Path]) -> None:
    lines = [
        "observed-collision miner summary:",
        f"  experiments scanned      : {stats.experiments_scanned}",
        f"  skipped (no declared cfg): {stats.skipped_missing_declared}",
        f"  skipped (no observed hash): {stats.skipped_missing_observed_hash}",
        f"  skipped (unreadable)     : {stats.skipped_unreadable}",
        f"  collision groups         : {stats.collision_groups}",
        f"  candidates emitted       : {len(candidates)}",
    ]
    for path in written:
        lines.append(f"  wrote {path}")
    print("\n".join(lines), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, help="Completed study results directory")
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=Path("engine_versions"),
        help="Root of the version-scoped candidate pool (default: engine_versions)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.study_dir.is_dir():
        print(f"error: study directory not found: {args.study_dir}", file=sys.stderr)
        return 2

    sidecars, stats = load_sidecars(args.study_dir)
    candidates = mine_collisions(sidecars, stats)
    written = write_candidates(candidates, args.pool_root)
    _print_summary(stats, candidates, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
