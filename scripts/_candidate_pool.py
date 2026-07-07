"""Shared candidate-pool contract for the engine-rule proposers.

Both proposers - the analyst cold read (S2, ``scripts/analyst_cold_read.py``)
and the observed-collision miner (S3, ``scripts/mine_observed_collisions.py``) -
write UNVERIFIED candidates into the same version-scoped pool, in one unified
schema the verification ladder consumes unchanged. This module owns the shared
pieces of that contract so the two proposers cannot drift apart:

- :func:`pool_path` - the on-disk layout
  ``<pool_root>/<engine>/v<mangled-version>/candidates/<filename>``.
- :func:`write_pool` - the envelope writer (``schema_version`` / ``source`` /
  ``engine`` / ``engine_version`` / ``generated_at`` / ``candidates`` plus any
  proposer-specific header keys) with the leading header comment.
- :func:`slug` - the filesystem/id-safe slug for a field path or free text.
- :func:`unverified_verdict` - the verdict stub every fresh candidate carries.

Neither proposer writes the shipped ``src/llenergymeasure/engines/<engine>/rules.yaml``
corpora; those are frozen and the ladder promotes into them separately.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.0.0"


def slug(text: str) -> str:
    """Filesystem/id-safe slug for a dotted field path or free text."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", text)).strip("_")


def unverified_verdict() -> dict[str, Any]:
    """The verdict stub every freshly-proposed candidate carries."""
    return {"status": "unverified", "tier": None, "date": None, "evidence": None}


def pool_path(pool_root: Path, engine: str, version: str, filename: str) -> Path:
    """Version-scoped pool path ``<pool_root>/<engine>/v<mangled>/candidates/<filename>``."""
    version_dir = "v" + re.sub(r"[^0-9a-zA-Z]+", "_", version).strip("_")
    return pool_root / engine / version_dir / "candidates" / filename


def write_pool(
    path: Path,
    *,
    source: str,
    engine: str,
    version: str,
    generated_at: str,
    candidates: list[dict[str, Any]],
    header: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write one candidate-pool file with the shared envelope and header comment.

    ``extra`` carries proposer-specific header keys (e.g. the analyst's ``model``
    and ``samples``); they slot in after ``engine_version`` and before
    ``generated_at`` so both proposers' envelopes share a stable key order.
    """
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "engine": engine,
        "engine_version": version,
    }
    if extra:
        document.update(extra)
    document["generated_at"] = generated_at
    document["candidates"] = candidates
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    path.write_text(header + body)
