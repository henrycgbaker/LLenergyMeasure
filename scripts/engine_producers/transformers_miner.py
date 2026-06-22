"""Transformers validation-invariants miner - landmark-verified orchestrator.

Composes two extraction paths to emit a deterministic invariants corpus:

1. **GenerationConfig invariants via library-API introspection** - delegated to
   :mod:`scripts.engine_producers.transformers_dynamic_invariant_miner`. Every dormancy
   invariant is discovered by probing ``GenerationConfig.validate(strict=True)``
   against a synthesised per-type probe value; every error-class invariant's
   message is lifted from the library's own ``ValueError``. Invariants emitted:
   ``added_by: dynamic_miner``.

2. **BitsAndBytesConfig type-check invariants** - hand-curated here. BNB import
   triggers a CUDA context on GPU-bearing hosts, which would make the miner
   unsafe to run CPU-only in CI. Keeping BNB invariants as landmark-verified
   manual seed preserves CPU-safety; the justification is the CUDA-context
   risk, not editorial preference. Invariants emitted: ``added_by: manual_seed``.

Landmark verification: imports ``transformers`` and confirms
``GenerationConfig``, ``BitsAndBytesConfig``, ``validate`` and ``post_init``
exist. Missing landmark → :class:`MinerLandmarkMissingError`. Source paths
and line numbers are derived via :func:`inspect.getsourcefile` and a text
grep - informational only, not used for invariant matching.

Flash-attention validation (``validate_transformers_flash_attn_dtype``) is
out of scope: its check lives in ``PreTrainedModel._autoset_attn_implementation``,
not ``GenerationConfig.validate`` or ``BitsAndBytesConfig.post_init``, and is
not reachable via this miner's landmarks. Stays hand-written in
``config/models.py`` pending a PreTrainedModel-level miner.

Usage::

    python -m scripts.engine_producers.transformers_miner \\
        --out src/llenergymeasure/engines/transformers/rules.proposed.yaml

With ``LLENERGY_MINER_FROZEN_AT`` set for byte-stable reproducibility.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# NOTE: the miners package is a sibling; when run via ``python -m
# scripts.engine_producers.transformers_miner`` the implicit ``scripts``
# package makes this work. When run as a script directly, we need the
# project root on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._base import (  # noqa: E402  (late import after sys.path)
    InvariantCandidate,
    MinerLandmarkMissingError,
    MinerSource,
)
from scripts.engine_producers._current import load_current  # noqa: E402
from scripts.engine_producers.transformers_dynamic_invariant_miner import (  # noqa: E402
    walk_generation_config_invariants,
)

# Symbols this orchestrator (and its delegated dynamic miner) relies on
# inside the live ``transformers`` package. Read by ``scripts._drift``
# before mining runs; a missing landmark flips the probe verdict to
# ``fail`` and skips the downstream stages without re-importing here.
#
# Sourced from the version-pinned archive at
# ``llenergymeasure._engine_archive.transformers.<safe_version>.producers.static_invariant_miner``.
# PEP 562 ``__getattr__`` defers the archive import + SSOT parse until the
# probe (or in-module code) accesses ``LANDMARKS``; module import stays
# cheap so unrelated test collection does not drag in the engine machinery.


def _get_landmarks() -> tuple[str, ...]:
    """Resolve LANDMARKS for the current SSOT ``library.current_version``.

    Caches in module ``globals()`` after first call so subsequent in-module
    references (and the probe's ``getattr(module, "LANDMARKS")``) read
    directly from the module dict without re-dispatching.
    """
    cached = globals().get("LANDMARKS")
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    from engine_versions._dispatcher import load_producer

    ssot = load_current("transformers")
    library = ssot.get("library")
    if not isinstance(library, dict) or "current_version" not in library:
        raise ValueError(
            "engine_versions/transformers.yaml is missing library.current_version; "
            "cannot resolve archived LANDMARKS."
        )
    landmarks = load_producer(
        engine="transformers",
        version=str(library["current_version"]),
        producer="static_invariant_miner",
    ).LANDMARKS
    globals()["LANDMARKS"] = landmarks
    return landmarks  # type: ignore[no-any-return]


def __getattr__(name: str) -> object:
    """PEP 562 hook: lazy LANDMARKS export from the per-version archive."""
    if name == "LANDMARKS":
        return _get_landmarks()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# BitsAndBytesConfig type-check invariants (kept hand-curated - CPU-safe)
# ---------------------------------------------------------------------------
#
# These invariants surface BNB's ``isinstance``-checking ``post_init`` TypeErrors
# before BNB is actually constructed. BNB's ``import`` triggers a CUDA
# context on GPU hosts; the miner stays CPU-safe by not importing it.
# Predicate uses ``type_is_not`` - fires only when the field is set AND has
# the wrong concrete type; a valid value (``True`` for a bool field) does
# not match.
#
# Provenance: ``manual_seed``. Re-auditing on BNB library bumps is a
# maintainer task - until a BNB-side introspection path is wired up (e.g.
# inside a CUDA-bearing container at CI time), the partition stays as-is.

_BNB_TYPE_RULES: tuple[tuple[str, str, Any, Any], ...] = (
    # (field, expected_type_label, positive_value, negative_value)
    # type_label matches ``type(value).__name__`` (strict class-name match
    # used by the loader's ``type_is_not`` predicate). ``torch.dtype``
    # instances have ``type(v).__name__ == "dtype"`` - not "torch.dtype".
    ("load_in_4bit", "bool", "yes", False),
    ("load_in_8bit", "bool", 1, False),
    ("llm_int8_threshold", "float", "6.0", 6.0),
    ("llm_int8_skip_modules", "list", "head", ["head"]),
    ("llm_int8_enable_fp32_cpu_offload", "bool", "yes", False),
    ("llm_int8_has_fp16_weight", "bool", 0, False),
    ("bnb_4bit_compute_dtype", "dtype", "float16", None),
    ("bnb_4bit_quant_type", "str", 7, "nf4"),
    ("bnb_4bit_use_double_quant", "bool", 1, False),
)


# ---------------------------------------------------------------------------
# Landmark + source-path utilities
# ---------------------------------------------------------------------------


def _check_landmarks() -> tuple[str, str]:
    """Verify the archived LANDMARKS resolve under the live ``transformers`` package.

    Returns ``(installed_version, generation_config_source_path)``.

    Iterates :data:`LANDMARKS` (lazy-loaded from
    ``llenergymeasure._engine_archive``) and resolves each via the same
    ``_resolve_landmark`` helper the probe uses, so the miner's runtime
    landmark check cannot drift from the probe's static check. A missing
    landmark raises :class:`MinerLandmarkMissingError`.
    """
    from scripts._drift import _resolve_landmark

    for landmark in _get_landmarks():
        try:
            _resolve_landmark(landmark)
        except (AttributeError, ImportError) as exc:
            raise MinerLandmarkMissingError(landmark, detail=str(exc)) from exc

    import transformers  # type: ignore
    from transformers import GenerationConfig  # type: ignore

    source_path = inspect.getsourcefile(GenerationConfig) or "<unknown>"
    return transformers.__version__, source_path


def _relative_source_path(abs_path: str) -> str:
    """Strip host-specific prefixes so the corpus is reproducible across machines.

    ``/home/alice/.local/lib/python3.10/site-packages/transformers/...``
    → ``transformers/...`` - rooted at ``site-packages/``.
    """
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


def _today() -> str:
    return dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# BNB invariant factory
# ---------------------------------------------------------------------------


def _make_bnb_type_invariant(
    field: str,
    type_label: str,
    positive: Any,
    negative: Any,
    source_path: str,
    today: str,
) -> InvariantCandidate:
    """Factory for ``BitsAndBytesConfig`` type-check invariants."""
    return InvariantCandidate(
        id=f"transformers_bnb_{field}_type",
        engine="transformers",
        library="bitsandbytes",
        invariant_under_test=(
            f"BitsAndBytesConfig.post_init() rejects non-{type_label} values for `{field}`"
        ),
        severity="error",
        native_type="transformers.BitsAndBytesConfig",
        miner_source=MinerSource(
            path=source_path,
            method="post_init",
            line_at_scan=0,
        ),
        match_fields={
            f"transformers.quant.{field}": {
                "present": True,
                "type_is_not": type_label,
            },
        },
        kwargs_positive={field: positive},
        kwargs_negative={field: negative},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=(f"`{field}` must be a {type_label}, got {{declared_value}}."),
        references=[
            "transformers.utils.quantization_config.BitsAndBytesConfig.post_init() "
            "- manually audited type-check raises"
        ],
        added_by="manual_seed",
        added_at=today,
    )


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def _candidate_to_dict(c: InvariantCandidate) -> dict[str, Any]:
    """Render a :class:`InvariantCandidate` into the YAML corpus entry shape."""
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "invariant_under_test": c.invariant_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "miner_source": asdict(c.miner_source),
        "match": {
            "engine": c.engine,
            "fields": c.match_fields,
        },
        "kwargs_positive": c.kwargs_positive,
        "kwargs_negative": c.kwargs_negative,
        "expected_outcome": c.expected_outcome,
        "message_template": c.message_template,
        "references": c.references,
        "added_by": c.added_by,
        "added_at": c.added_at,
    }


# Env var that pins the envelope ``mined_at`` for byte-stable reproducibility
# (CI sets it; tests reference this constant rather than the literal string).
FROZEN_AT_ENV = "LLENERGY_MINER_FROZEN_AT"


def walk() -> tuple[list[InvariantCandidate], dict[str, Any]]:
    """Return ``(candidates, envelope_metadata)`` - full corpus for this engine.

    Composes the introspection-derived GenerationConfig invariants with the
    hand-curated BNB invariants. Invariants in the returned list carry their own
    ``added_by`` tag; the envelope captures the engine version.
    """
    installed_version, abs_source_path = _check_landmarks()

    # Corpus paths are relative to site-packages so the committed YAML is
    # reproducible across checkouts with different ``~/.local`` roots.
    source_path = _relative_source_path(abs_source_path)

    today = _today()
    candidates: list[InvariantCandidate] = []

    candidates.extend(
        walk_generation_config_invariants(
            abs_source_path=abs_source_path,
            rel_source_path=source_path,
            today=today,
        )
    )

    # BitsAndBytesConfig invariants - source path is the quantization_config module.
    # Cheaply locate it without importing bnb (which may touch CUDA).
    bnb_source_path = source_path
    try:
        import transformers.utils.quantization_config as _qcfg  # type: ignore
    except ImportError:  # pragma: no cover - landmark missing, handled upstream
        pass
    else:
        abs_bnb = inspect.getsourcefile(_qcfg)
        if abs_bnb:
            bnb_source_path = _relative_source_path(abs_bnb)
    for field, type_label, pos, neg in _BNB_TYPE_RULES:
        candidates.append(
            _make_bnb_type_invariant(field, type_label, pos, neg, bnb_source_path, today)
        )

    # Allow tests / reproducibility checks to pin the timestamp.
    frozen = os.environ.get(FROZEN_AT_ENV)
    mined_at = (
        frozen
        if frozen
        else (dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    )
    envelope = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": installed_version,
        "mined_at": mined_at,
    }
    return candidates, envelope


def emit_yaml(candidates: list[InvariantCandidate], envelope: dict[str, Any]) -> str:
    """Serialise candidates + envelope to a deterministic YAML string.

    Key order is fixed (not alphabetical) for readability: envelope first,
    then candidates sorted by ``(miner_source.method, id)``.
    """
    import yaml

    sorted_candidates = sorted(candidates, key=lambda c: (c.miner_source.method, c.id))
    doc = {
        "schema_version": envelope["schema_version"],
        "engine": envelope["engine"],
        "engine_version": envelope["engine_version"],
        "mined_at": envelope["mined_at"],
        "invariants": [_candidate_to_dict(c) for c in sorted_candidates],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write extracted YAML to this path.",
    )
    args = parser.parse_args(argv)

    candidates, envelope = walk()
    text = emit_yaml(candidates, envelope)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} transformers invariants to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
