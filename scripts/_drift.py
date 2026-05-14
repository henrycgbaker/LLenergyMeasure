"""Drift tool - bidirectional landmark drift detection for per-engine producers.

The drift tool is the first step of each per-concern workflow (engine-invariants
+ engine-schemas). Phase A covers the "removed" direction only, which is a
drop-in replacement for the former ``_probe.py`` non-regression gate:

    pass : every landmark in ``producer.LANDMARKS`` resolves to a live
           class / method / module attribute under ``import {library}``.
    fail : at least one landmark is missing, OR the producer module
           itself fails to import after one retry.

Phase B adds the "added" direction - validators that exist in the live library
but are not declared in the per-version archive LANDMARKS. This is the additive
blind spot coverage check: when Renovate bumps a library to a version with new
validators, Phase B surfaces them so the maintainer knows to extend LANDMARKS +
walker logic. Phase B is warning-only: ``verdict`` stays ``pass`` even when
``landmarks_added`` is non-empty.

Phase C adds CI gating, EXCLUSIONS.yaml per archive version, and sticky PR comments.

EXCLUSIONS.yaml (``engine_versions/<engine>/v<safe>/EXCLUSIONS.yaml``) is a
maintainer-managed allow-list for ``landmarks_added`` entries the maintainer has
explicitly chosen not to walk. Entries are keyed by stable fingerprint (sha256 of
the canonical dotted path). Expired entries (``expires:`` date in the past) cause
the tool to emit an error and CI to fail, forcing periodic re-review.

``--gate {none,delta,absolute}`` controls CI gating behaviour:

    none     : Phase B behaviour - warning only, exit 0 always (default).
    delta    : exit 1 when Δ_high_added > 0 (new high-confidence gaps since
               last green main). Baseline read from current.yaml
               ``last_probe.added_baseline``.
    absolute : exit 1 when any high-confidence landmark_added present after
               exclusions.

``--pr-number N`` enables sticky PR comment generation via
``scripts/ci/upsert_pr_comment.sh`` (marker ``llem-drift-coverage``).

``direction`` field semantics:

    "removed" : one or more declared landmarks are absent from the live library.
    "added"   : one or more live validators are absent from LANDMARKS.
    "mixed"   : both removed AND added gaps found simultaneously.
    "stable"  : all declared landmarks resolve; no undeclared validators found.

Diagnostic fields (``version_inside_envelope``, ``fingerprint_drift``,
``landmarks_added``, ``landmarks_added_low_confidence``) ride along on every
report. Only ``landmarks_added`` (high-confidence) is surfaced in CI output by
default. ``landmarks_added_low_confidence`` is present in the JSON for
completeness.

Producer-module discovery uses a per-engine convention table (see
``_PRODUCER_MODULES``); each producer module exposes a
``LANDMARKS: tuple[str, ...]`` constant of dotted attribute paths
(e.g. ``"transformers.GenerationConfig"``,
``"vllm.config.parallel.ParallelConfig.__post_init__"``).

The CLI producer kind tokens ("invariants", "schemas") are the probe-contract
names, NOT the producer-file names. The ``_PRODUCER_MODULES`` map translates
them to the underlying module paths. CLI callers keep the same interface as the
former ``_probe.py``.

Usage::

    python -m scripts._drift --engine transformers --producer invariants

Emits a JSON ``DriftReport`` to stdout. Exit 0 on either verdict - the
binary verdict travels in the JSON, and downstream workflow steps gate
on it. Exit 2 only on infrastructure failure (current.yaml missing, producer
module unimportable, current.yaml malformed).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import enum
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Literal

import yaml
from packaging.specifiers import SpecifierSet

# Make the top-level ``scripts`` package importable when invoked as a
# plain script (``python scripts/_drift.py``) as well as via
# ``python -m scripts._drift``.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._base import MinerLandmarkMissingError  # noqa: E402
from scripts.engine_producers._current import current_path, safe_version  # noqa: E402

ProducerKind = Literal["invariants", "schemas"]
DirectionKind = Literal["removed", "added", "mixed", "stable"]

# Per-engine producer module map. The drift tool lives one layer above the
# producers; this table is the single seam where (engine, producer) cells
# resolve to a Python module path. Keep it exhaustive - adding a new
# engine means adding a row here.
_PRODUCER_MODULES: dict[tuple[str, ProducerKind], str] = {
    ("transformers", "invariants"): "scripts.engine_producers.transformers_miner",
    ("vllm", "invariants"): "scripts.engine_producers.vllm_static_invariant_miner",
    ("tensorrt", "invariants"): "scripts.engine_producers.tensorrt_static_invariant_miner",
    ("transformers", "schemas"): "scripts.engine_producers.transformers_schema_introspector",
    ("vllm", "schemas"): "scripts.engine_producers.vllm_schema_introspector",
    ("tensorrt", "schemas"): "scripts.engine_producers.tensorrt_schema_introspector",
}

# current.yaml ``miner_pins.*`` keys are typed by extraction strategy
# (``static | dynamic | discovery``); the drift tool's user-facing producer
# kinds are typed by concern (``invariants | schemas``). One layer of
# translation lives here.
_SSOT_PIN_FOR_PRODUCER: dict[ProducerKind, str] = {
    "invariants": "static",
    "schemas": "discovery",
}


# ---------------------------------------------------------------------------
# Phase C: EXCLUSIONS.yaml support
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExclusionEntry:
    """One parsed entry from EXCLUSIONS.yaml.

    ``fingerprint`` is the sha256 of the canonical dotted path of the symbol
    (the same hash produced by ``_landmark_fingerprint``). ``qualname`` is
    advisory (human-readable); it is NOT used for matching - fingerprint only.

    ``expires`` is optional. When set and < today, the exclusion is INVALID
    and causes the drift report to record an error, flipping CI to fail.
    """

    fingerprint: str
    qualname: str
    reason: str
    added: str
    expires: str | None = None


def _landmark_fingerprint(dotted_path: str) -> str:
    """Stable sha256 of a canonical dotted landmark path.

    Used both for generating the fingerprint that goes INTO an EXCLUSIONS.yaml
    entry and for matching incoming ``landmarks_added`` entries against the
    allow-list. The hash is over the UTF-8-encoded dotted path only (no line
    numbers, no file paths - those churn; the canonical name doesn't).
    """
    return hashlib.sha256(dotted_path.encode()).hexdigest()


def load_exclusions(engine: str, version: str) -> dict[str, ExclusionEntry]:
    """Load ``engine_versions/<engine>/v<safe>/EXCLUSIONS.yaml`` if it exists.

    Returns a dict keyed by fingerprint. Invalid entries (wrong shape, unknown
    keys) are skipped with a warning to stderr. The file's absence is not an
    error - most version archives won't have an EXCLUSIONS.yaml.

    ``version`` is the raw library version string (e.g. ``"0.7.3"``); this
    function computes the ``safe_version`` (``v0_7_3``) internally.
    """
    exclusions_path = (
        _PROJECT_ROOT / "engine_versions" / engine / safe_version(version) / "EXCLUSIONS.yaml"
    )
    if not exclusions_path.exists():
        return {}

    try:
        raw = yaml.safe_load(exclusions_path.read_text())
    except yaml.YAMLError as exc:
        print(
            f"Warning: could not parse {exclusions_path}: {exc}. Exclusions file skipped.",
            file=sys.stderr,
        )
        return {}

    if not isinstance(raw, list):
        print(
            f"Warning: {exclusions_path} did not parse to a list. Exclusions file skipped.",
            file=sys.stderr,
        )
        return {}

    result: dict[str, ExclusionEntry] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            print(
                f"Warning: {exclusions_path} entry {idx} is not a mapping; skipped.",
                file=sys.stderr,
            )
            continue
        required_keys = {"fingerprint", "qualname", "reason", "added"}
        missing = required_keys - item.keys()
        if missing:
            print(
                f"Warning: {exclusions_path} entry {idx} missing keys {sorted(missing)}; skipped.",
                file=sys.stderr,
            )
            continue
        entry = ExclusionEntry(
            fingerprint=str(item["fingerprint"]),
            qualname=str(item["qualname"]),
            reason=str(item["reason"]),
            added=str(item["added"]),
            expires=str(item["expires"]) if item.get("expires") is not None else None,
        )
        result[entry.fingerprint] = entry

    return result


def _check_exclusion_expiry(
    entry: ExclusionEntry,
) -> bool:
    """Return True iff the exclusion entry is expired (``expires`` < today).

    Returns False when ``expires`` is absent or cannot be parsed. The
    caller is responsible for converting this into an error in the report.
    """
    if entry.expires is None:
        return False
    try:
        expiry = datetime.date.fromisoformat(entry.expires)
    except ValueError:
        # Unparseable expiry date - treat as non-expired so it doesn't silently
        # pass; caller should still log it as a warning.
        return False
    return expiry < datetime.datetime.now(tz=datetime.timezone.utc).date()


GateKind = Literal["none", "delta", "absolute"]


@dataclass(frozen=True)
class DriftReport:
    """Structured report of one drift-tool run.

    Serialised to stdout JSON and to ``engine_versions/{engine}.compat.json``
    for cross-run cache + fingerprint-drift detection.

    Phase A fields
    --------------
    ``direction`` = "removed" when ``landmarks_missing`` is non-empty; "stable" otherwise.
    ``landmarks_added`` and ``landmarks_added_low_confidence`` are empty in a Phase A run
    (no introspection roots configured for the engine).

    Phase B fields
    --------------
    ``landmarks_added`` : high-confidence live validators absent from LANDMARKS. Surfaced
    in CI workflow comments. Warning-only - does NOT flip verdict to fail.
    ``landmarks_added_low_confidence`` : medium/low-confidence candidates. Present in JSON
    for completeness; not surfaced in CI output by default.
    ``direction`` gains two new values: "added" (only live gaps found) and "mixed"
    (both removed and added gaps present simultaneously).

    Diagnostic fields (``version_inside_envelope``, ``fingerprint_drift``) ride along on
    every report and steer human attention on pass-but-suspicious bumps. They NEVER
    affect verdict.

    Phase C fields
    --------------
    ``exclusions_applied`` : dotted paths from ``landmarks_added`` that were suppressed
    by a matching EXCLUSIONS.yaml entry. These do NOT appear in ``landmarks_added``.
    ``expired_exclusions`` : qualnames of EXCLUSIONS.yaml entries whose ``expires``
    date has passed. When non-empty, verdict flips to ``fail`` regardless of gate mode.
    ``added_baseline`` : the set of high-confidence added gaps as of the last green
    main commit, read from ``current.yaml last_probe.added_baseline``. Used by
    ``--gate delta`` to compute Δ.
    ``landmarks_added_delta`` : ``landmarks_added`` entries NOT in ``added_baseline``
    (new gaps introduced by this PR). Only populated when ``--gate delta`` is active.
    ``gate`` : the gate mode that was active during this run (from ``--gate`` flag).
    """

    engine: str
    producer: ProducerKind
    direction: DirectionKind
    schema_version: int
    current_version: str
    verdict: Literal["pass", "fail"]
    fingerprint: str
    fingerprint_drift: list[str] = field(default_factory=list)
    landmarks_missing: list[str] = field(default_factory=list)
    landmarks_added: list[str] = field(default_factory=list)
    landmarks_added_low_confidence: list[str] = field(default_factory=list)
    version_inside_envelope: bool = True
    # Phase C fields
    exclusions_applied: list[str] = field(default_factory=list)
    expired_exclusions: list[str] = field(default_factory=list)
    landmarks_added_delta: list[str] = field(default_factory=list)
    gate: GateKind = "none"


# ---------------------------------------------------------------------------
# Landmark resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedLandmark:
    """One successfully-resolved landmark and its source coordinates.

    The fingerprint is a sorted hash of these tuples; ``filename`` +
    ``lineno`` are advisory but track real refactors (a method moving
    from line 142 to 187 across a patch release).
    """

    landmark: str
    qualname: str
    filename: str | None
    lineno: int | None


def _resolve_landmark(landmark: str) -> _ResolvedLandmark:
    """Resolve a dotted landmark like ``"vllm.config.parallel.ParallelConfig.__post_init__"``.

    Strategy: try progressively-shorter module prefixes; whatever doesn't
    import is treated as attribute access on the deepest importable
    module. Raises :class:`AttributeError` / :class:`ImportError` on
    miss; the caller catches and folds the failure into ``landmarks_missing``.
    """
    parts = landmark.split(".")
    module: ModuleType | None = None
    module_idx = 0
    # Find the longest importable prefix.
    for split in range(len(parts), 0, -1):
        module_path = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_path)
            module_idx = split
            break
        except ImportError:
            continue
    if module is None:
        raise ImportError(f"No importable prefix in landmark {landmark!r}")

    obj: object = module
    for attr in parts[module_idx:]:
        obj = getattr(obj, attr)

    try:
        filename = inspect.getsourcefile(obj)  # type: ignore[arg-type]
    except TypeError:
        filename = None
    try:
        _, lineno = inspect.getsourcelines(obj)  # type: ignore[arg-type]
    except (TypeError, OSError):
        lineno = None

    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", landmark)
    return _ResolvedLandmark(
        landmark=landmark,
        qualname=str(qualname),
        filename=filename,
        lineno=lineno,
    )


def _fingerprint(resolved: list[_ResolvedLandmark]) -> str:
    """Stable hash of the resolved landmarks' (qualname, filename, lineno) tuples."""
    parts = sorted(
        f"{r.landmark}|{r.qualname}|{r.filename or ''}|{r.lineno or 0}" for r in resolved
    )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Phase B: live library introspection
# ---------------------------------------------------------------------------

# Engine-specific module roots to walk when discovering live validators.
# Each value is either a tuple of module paths (import and walk all classes
# found in each) or a tuple of fully-qualified class paths (import the
# class directly - used when the module is too broad to walk exhaustively).
#
# "transformers" uses class-level targeting: walking all of ``transformers``
# would pull in thousands of classes and trigger CUDA-sensitive imports
# (e.g. BitsAndBytesConfig). Instead we enumerate only the 4 classes
# declared in LANDMARKS.
#
# "vllm" and "tensorrt" use module-level targeting: their config modules are
# narrow enough to walk fully. For tensorrt, the NGC entrypoint sets up CUDA
# forward-compat; the drift tool is invoked inside the container so imports
# succeed. vllm's config subpackage refactored from a flat module (0.7.x) to
# a per-concern subpackage (0.16+); both layouts resolve via importlib.
_ENGINE_INTROSPECTION_ROOTS: dict[str, tuple[str, ...]] = {
    "transformers": (
        # Class-level targets - not module-level. Avoids pulling in the full
        # transformers import graph (CUDA-bearing BNB, flash-attn, etc.).
        "transformers.GenerationConfig",
        "transformers.BitsAndBytesConfig",
        "transformers.AutoModelForCausalLM",
        "transformers.PreTrainedModel",
    ),
    "vllm": (
        "vllm.config",
        "vllm.sampling_params",
    ),
    "tensorrt": (
        "tensorrt_llm.llmapi.llm_args",
        "tensorrt_llm.builder",
    ),
}

# Confidence tier labels (vulture-style: kind-derived, not learned scores).
_CONF_HIGH = "high"
_CONF_MEDIUM = "medium"
_CONF_LOW = "low"


def _is_pydantic_v2_model(cls: type) -> bool:
    """True if ``cls`` is a pydantic v2 BaseModel or pydantic-dataclass."""
    try:
        import pydantic  # type: ignore[import]

        return isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)
    except ImportError:
        return False


def _is_pydantic_v1_model(cls: type) -> bool:
    """True if ``cls`` appears to be a pydantic v1 model (pre-v2 API)."""
    # v1 models have __fields__ and __validators__ but NOT __pydantic_decorators__.
    return (
        hasattr(cls, "__fields__")
        and hasattr(cls, "__validators__")
        and not hasattr(cls, "__pydantic_decorators__")
        and not hasattr(cls, "model_fields")
    )


def _discover_pydantic_v2(cls: type, module_prefix: str) -> list[tuple[str, str]]:
    """Discover validator methods from a pydantic v2 BaseModel / pydantic-dataclass.

    Reads ``cls.__pydantic_decorators__`` (a ``DecoratorInfos`` dataclass).
    Pydantic v2 flattens inherited validators into this attribute - do NOT
    walk MRO manually; that would double-count.

    Returns ``[(confidence, dotted_path), ...]``.

    ``__pydantic_decorators__`` is documented-internal but the de-facto contract
    consumed by autodoc-pydantic, LangChain, and others. Not wrapped in
    try/except so callers see real bugs.
    """
    dec = cls.__pydantic_decorators__
    results: list[tuple[str, str]] = []
    qualname = _dotted_qualname(cls, module_prefix)

    # Buckets that indicate genuine validator surface.
    high_buckets = ("field_validators", "model_validators", "root_validators", "validators")
    # Serializer/computed buckets are validator-adjacent but lower signal.
    medium_buckets = ("field_serializers", "model_serializers", "computed_fields")

    for bucket_name in high_buckets:
        bucket = getattr(dec, bucket_name, {})
        for method_name in bucket:
            results.append((_CONF_HIGH, f"{qualname}.{method_name}"))

    for bucket_name in medium_buckets:
        bucket = getattr(dec, bucket_name, {})
        for method_name in bucket:
            results.append((_CONF_MEDIUM, f"{qualname}.{method_name}"))

    # model_fields are declared fields - high confidence surface.
    model_fields = getattr(cls, "model_fields", {})
    for field_name in model_fields:
        results.append((_CONF_HIGH, f"{qualname}.{field_name}"))

    return results


def _discover_pydantic_v1(cls: type, module_prefix: str) -> list[tuple[str, str]]:
    """Discover validator methods from a pydantic v1 model."""
    results: list[tuple[str, str]] = []
    qualname = _dotted_qualname(cls, module_prefix)

    for field_name in getattr(cls, "__fields__", {}):
        results.append((_CONF_HIGH, f"{qualname}.{field_name}"))
    for validator_name in getattr(cls, "__validators__", {}):
        results.append((_CONF_HIGH, f"{qualname}.{validator_name}"))
    for method_name in getattr(cls, "__pre_root_validators__", []):
        results.append((_CONF_HIGH, f"{qualname}.{method_name}"))
    for method_name in getattr(cls, "__post_root_validators__", []):
        results.append((_CONF_HIGH, f"{qualname}.{method_name}"))
    return results


def _discover_dataclass(cls: type, module_prefix: str) -> list[tuple[str, str]]:
    """Discover fields and __post_init__ from a stdlib ``@dataclass``."""
    results: list[tuple[str, str]] = []
    qualname = _dotted_qualname(cls, module_prefix)

    try:
        for f in dataclasses.fields(cls):
            if not f.name.startswith("_"):
                results.append((_CONF_HIGH, f"{qualname}.{f.name}"))
    except TypeError:
        # Not a dataclass (dataclasses.fields raises TypeError on non-dataclasses).
        pass

    if hasattr(cls, "__post_init__"):
        results.append((_CONF_HIGH, f"{qualname}.__post_init__"))
    return results


def _discover_plain_class(cls: type, module_prefix: str) -> list[tuple[str, str]]:
    """Discover validator-like callables on a plain class via name heuristics.

    Uses ``vars(cls)`` (not ``dir(cls)`` / ``inspect.getmembers``) to avoid
    triggering descriptors and conflating inherited members.

    Confidence tiers:
    - ``high``: none from plain classes (no decorator evidence).
    - ``medium``: public ``validate_*`` / ``check_*`` / ``verify_*`` callables
      in ``vars(cls)`` (might be a manual validator, might be a passthrough).
    - ``low``: private ``_validate_*`` / ``_verify_*`` helpers.
    """
    results: list[tuple[str, str]] = []
    qualname = _dotted_qualname(cls, module_prefix)

    for name, obj in vars(cls).items():
        if not callable(obj):
            continue
        lower = name.lower()
        # Public validate_* / check_* / verify_* (no leading underscore)
        if not name.startswith("_") and any(
            lower.startswith(prefix) for prefix in ("validate", "check", "verify")
        ):
            results.append((_CONF_MEDIUM, f"{qualname}.{name}"))
            continue
        # Private _validate_* / _verify_*
        if (
            name.startswith("_")
            and not name.startswith("__")
            and any(lower.lstrip("_").startswith(prefix) for prefix in ("validate", "verify"))
        ):
            results.append((_CONF_LOW, f"{qualname}.{name}"))
    return results


def _dotted_qualname(cls: type, module_prefix: str) -> str:
    """Canonical dotted path for ``cls`` relative to its defining module.

    Prefers the module's ``__name__`` attribute for the class's own
    ``__module__`` + ``__qualname__`` combination. Falls back to
    ``module_prefix.cls.__qualname__`` when the class's ``__module__``
    attribute is missing.
    """
    own_module = getattr(cls, "__module__", None) or module_prefix
    qualname = getattr(cls, "__qualname__", None) or cls.__name__
    return f"{own_module}.{qualname}"


def _class_validator_surface(cls: type, module_prefix: str) -> list[tuple[str, str]]:
    """Layered dispatch: discover validator surface of a single class.

    Dispatch order (first match wins):
    1. Pydantic v2 BaseModel / pydantic-dataclass - use ``__pydantic_decorators__``
    2. Pydantic v1 - use ``__fields__`` + ``__validators__``
    3. stdlib ``@dataclass`` - use ``dataclasses.fields`` + ``__post_init__``
    4. Enum (StrEnum / IntEnum) - skip, no validator concept
    5. Plain class - name-heuristic on ``vars(cls)``

    Note: pydantic-dataclasses ARE pydantic v2 BaseModel subclasses after
    decoration, so step 1 catches them before step 3.
    """
    if _is_pydantic_v2_model(cls):
        return _discover_pydantic_v2(cls, module_prefix)
    if _is_pydantic_v1_model(cls):
        return _discover_pydantic_v1(cls, module_prefix)
    # Enum check before dataclass: StrEnum is also a dataclass in Python 3.11+
    # when using the functional API, but enum members have no validator surface.
    if issubclass(cls, enum.Enum):
        return []
    if dataclasses.is_dataclass(cls):
        return _discover_dataclass(cls, module_prefix)
    return _discover_plain_class(cls, module_prefix)


def _import_class_from_path(dotted_path: str) -> type | None:
    """Import and return a class from a dotted path like ``transformers.GenerationConfig``.

    Returns ``None`` if the import fails (engine not installed in the current
    Python environment - expected for Docker-only engines on the host).
    Handles partial-import failures gracefully (one broken class does not
    abort the whole walk).

    Unlike ``_resolve_landmark`` (which raises on failure and returns a
    ``_ResolvedLandmark``), this helper returns the raw type or ``None`` - the
    difference in return contract warrants keeping them separate.
    """
    parts = dotted_path.split(".")
    module: ModuleType | None = None
    module_idx = 0
    for split in range(len(parts), 0, -1):
        try:
            candidate = importlib.import_module(".".join(parts[:split]))
            module = candidate
            module_idx = split
            break
        except ImportError:
            continue
    if module is None:
        return None
    obj: object = module
    for attr in parts[module_idx:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    if not isinstance(obj, type):
        return None
    return obj


def _walk_module_classes(module_path: str) -> list[tuple[type, str]]:
    """Import a module and return ``[(class, module_path), ...]`` for all classes defined there.

    Only returns classes whose ``__module__`` matches ``module_path`` exactly
    (avoids picking up re-exported classes from other modules). On ImportError
    returns an empty list - engine not installed in this environment.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return []
    results: list[tuple[type, str]] = []
    for _name, obj in vars(module).items():
        if isinstance(obj, type) and getattr(obj, "__module__", None) == module_path:
            results.append((obj, module_path))
    return results


def discover_live_landmarks(engine: str) -> tuple[set[str], set[str]]:
    """Walk the live library surface for ``engine`` and return discovered paths.

    Returns ``(high_confidence, low_confidence)`` where each is a ``set[str]``
    of canonical dotted paths like ``"vllm.config.ParallelConfig.validate_X"``.

    High-confidence paths include:
    - Pydantic v2 field_validators / model_validators / declared model_fields
    - Pydantic v1 __fields__ + __validators__
    - Stdlib @dataclass fields + __post_init__
    - Plain class public ``validate_*`` / ``check_*`` / ``verify_*`` callables

    Low-confidence paths include:
    - Plain class private ``_validate_*`` / ``_verify_*`` helpers
    - Pydantic serializer / computed_field buckets (medium tier)

    Returns empty sets when the engine library is not installed in the current
    Python environment (expected for Docker-only engines running on the host).

    The function is graceful on partial failures: one broken class does not
    abort the walk. Per-class errors are silently skipped so callers always
    get a (possibly empty) result.
    """
    roots = _ENGINE_INTROSPECTION_ROOTS.get(engine, ())
    high: set[str] = set()
    low: set[str] = set()

    def _bucket(cls: type, mod_prefix: str) -> None:
        """Surface-discover ``cls`` and partition results into ``high`` / ``low``."""
        try:
            for conf, path in _class_validator_surface(cls, mod_prefix):
                if conf == _CONF_HIGH:
                    high.add(path)
                else:
                    low.add(path)
        except Exception:  # per-class failure is non-fatal
            pass

    for root in roots:
        # Determine whether the root is a class path or a module path.
        # Heuristic: if the last component starts with an uppercase letter,
        # treat it as a class; otherwise treat as a module to walk.
        parts = root.split(".")
        if parts[-1][0].isupper():
            # Class-level targeting (e.g. ``transformers.GenerationConfig``).
            cls = _import_class_from_path(root)
            if cls is not None:
                _bucket(cls, ".".join(parts[:-1]))
        else:
            # Module-level targeting (e.g. ``vllm.config``).
            for cls, mod_path in _walk_module_classes(root):
                _bucket(cls, mod_path)

    return high, low


# ---------------------------------------------------------------------------
# current.yaml helpers
# ---------------------------------------------------------------------------


def _load_current(engine: str) -> dict[str, object]:
    """Read + parse ``engine_versions/{engine}/current.yaml``.

    Raises :class:`FileNotFoundError` if missing. The drift tool treats
    current.yaml absence as an infrastructure error (exit code 2) - every
    supported engine must have a current.yaml before the tool is wired up.
    """
    path = current_path(engine)
    text = path.read_text()  # FileNotFoundError surfaces here
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"current.yaml at {path} did not parse to a mapping.")
    return data


def _check_envelope(ssot: dict[str, object], producer: ProducerKind, current_version: str) -> bool:
    """True iff ``current_version`` falls inside ``miner_pins.{ssot_key}``."""
    pins = ssot.get("miner_pins") or {}
    if not isinstance(pins, dict):
        return False
    raw = pins.get(_SSOT_PIN_FOR_PRODUCER[producer])
    if raw is None:
        return False
    spec = SpecifierSet(str(raw))
    return spec.contains(current_version, prereleases=True)


def _compat_path(engine: str) -> Path:
    """Return the path to the cached fingerprint file for ``engine``.

    The compat cache lives at ``engine_versions/{engine}.compat.json`` -
    sibling of the per-engine sub-directory (not inside it). Resolved via
    ``current_path`` parent so tests can monkeypatch the location.
    """
    # current_path(engine) -> engine_versions/{engine}/current.yaml
    # .parent          -> engine_versions/{engine}/
    # .parent          -> engine_versions/
    return current_path(engine).parent.parent / f"{engine}.compat.json"


def _read_cached_fingerprint(
    engine: str, producer: ProducerKind, current_version: str
) -> str | None:
    """Return the previous-run fingerprint for this (engine, producer), or ``None``.

    Returns ``None`` when:

    - The cache file does not exist or is malformed.
    - The cached entry is missing a ``current_version`` field.
    - The cached entry's ``current_version`` differs from the current
      library version. This is the cross-version skip: bumping
      ``library.current_version`` invalidates the prior fingerprint by
      definition (it was computed against a different installed library),
      so reporting "every landmark drifted" would be noise. Returning
      ``None`` here causes ``run()`` to treat this as a first-run
      fingerprint and emit empty ``fingerprint_drift``; the next run at
      the bumped version writes a fresh cache entry.
    """
    path = _compat_path(engine)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        cache = json.loads(text)
    except json.JSONDecodeError:
        return None
    entry = (cache.get(producer) if isinstance(cache, dict) else None) or {}
    if not isinstance(entry, dict):
        return None
    # Accept both the new field name ("current_version") and the legacy
    # "library_version" written by older cache entries from _probe.py.
    cached_version = entry.get("current_version") or entry.get("library_version")
    if cached_version != current_version:
        # Cache was written against a different library version; treat as miss.
        return None
    fp = entry.get("fingerprint")
    return str(fp) if isinstance(fp, str) else None


def _write_cached_report(engine: str, report: DriftReport) -> None:
    """Atomically merge ``report`` into ``engine_versions/{engine}.compat.json``."""
    path = _compat_path(engine)
    cache: dict[str, object] = {}
    try:
        existing = json.loads(path.read_text())
        if isinstance(existing, dict):
            cache = existing
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    cache[report.producer] = asdict(report)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Producer module loading
# ---------------------------------------------------------------------------


def _import_producer(engine: str, producer: ProducerKind) -> ModuleType:
    """Import the producer module for ``(engine, producer)``.

    Retries once on :class:`MinerLandmarkMissingError` raised at import
    time (per design D-8) - a flaky import is treated as transient and
    retried before the verdict is finalised. A second failure escalates
    to caller; the CLI maps it to exit code 2 (infrastructure failure).
    """
    key = (engine, producer)
    if key not in _PRODUCER_MODULES:
        raise KeyError(f"Unsupported (engine, producer) pair: {key}")
    module_path = _PRODUCER_MODULES[key]
    try:
        return importlib.import_module(module_path)
    except MinerLandmarkMissingError:
        return importlib.import_module(module_path)


def _read_landmarks(module: ModuleType) -> tuple[str, ...]:
    """Read ``LANDMARKS`` from ``module``; raises :class:`AttributeError` if absent."""
    landmarks = module.LANDMARKS  # type: ignore[attr-defined]
    if not isinstance(landmarks, tuple) or not all(isinstance(x, str) for x in landmarks):
        raise TypeError(
            f"{module.__name__}.LANDMARKS must be a tuple[str, ...]; got {type(landmarks).__name__}."
        )
    return landmarks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _read_added_baseline(current: dict[str, object]) -> list[str] | None:
    """Read ``last_probe.added_baseline`` from a parsed current.yaml mapping.

    Returns ``None`` when the field is absent (engines that haven't had
    a Phase C writeback yet, or first-run on a new engine) so callers can
    distinguish "no baseline written yet" from "baseline written but empty"
    (the latter means "no high-confidence gaps as of last green main"; the
    former means "no comparison possible yet — gate should not fail").
    """
    last_probe = current.get("last_probe")
    if not isinstance(last_probe, dict):
        return None
    raw = last_probe.get("added_baseline")
    if not isinstance(raw, list):
        return None
    return [str(x) for x in raw if isinstance(x, str)]


def run(*, engine: str, producer: ProducerKind, gate: GateKind = "none") -> DriftReport:
    """Run the drift check for one ``(engine, producer)`` cell.

    Phase A + B + C: "removed" and "added" directions, with optional CI gating
    and EXCLUSIONS.yaml filtering.

    Returns a :class:`DriftReport` with verdict ``pass`` when every declared
    landmark resolves AND gate conditions are satisfied. Verdict ``fail`` when:
    - At least one declared landmark raises AttributeError/ImportError (Phase A).
    - An EXCLUSIONS.yaml entry has an expired ``expires`` date (Phase C).
    - ``gate="delta"`` and Δ_high_added > 0 (new high-confidence gaps since
      last green main, after exclusions).
    - ``gate="absolute"`` and any high-confidence added gaps remain after exclusions.

    Diagnostic fields (envelope check, fingerprint drift) are computed
    regardless of verdict.

    Direction values:
    - ``"removed"`` : one or more declared landmarks absent from live library.
    - ``"added"``   : live validators absent from LANDMARKS (Phase B signal).
    - ``"mixed"``   : both removed and added gaps found simultaneously.
    - ``"stable"``  : all declared landmarks resolve; no undeclared validators found.

    Phase B introspection is best-effort: when the engine library is not
    installed (Docker-only engines on the host), ``discover_live_landmarks``
    returns empty sets and ``landmarks_added`` stays empty.

    Phase C EXCLUSIONS.yaml is loaded from
    ``engine_versions/<engine>/v<safe>/EXCLUSIONS.yaml``. Entries with
    valid (non-expired) fingerprints matching a ``landmarks_added`` entry
    are moved to ``exclusions_applied`` and omitted from ``landmarks_added``.
    Expired entries populate ``expired_exclusions`` and force verdict=fail.
    """
    current = _load_current(engine)  # FileNotFoundError -> caller handles as infra error
    library = current.get("library")
    if not isinstance(library, dict) or "current_version" not in library:
        raise ValueError(f"current.yaml for {engine} missing library.current_version.")
    current_version = str(library["current_version"])

    producer_module = _import_producer(engine, producer)
    landmarks = _read_landmarks(producer_module)

    resolved: list[_ResolvedLandmark] = []
    missing: list[str] = []
    for landmark in landmarks:
        try:
            resolved.append(_resolve_landmark(landmark))
        except (AttributeError, ImportError):
            missing.append(landmark)

    fingerprint = _fingerprint(resolved)
    cached = _read_cached_fingerprint(engine, producer, current_version)
    drift: list[str] = []
    if cached is not None and cached != fingerprint:
        # Per-landmark drift would require caching per-landmark hashes;
        # the binary "fingerprint shifted at all" signal is what the
        # writeback comment surfaces, so the list captures the affected
        # landmarks, not the cause.
        drift = [r.landmark for r in resolved]

    # Phase B: added-direction discovery.
    # Only surface for "invariants" producer - schemas use a different surface.
    # For other producer kinds, skip introspection (no false-positive noise).
    added_high_raw: list[str] = []
    added_low: list[str] = []
    if producer == "invariants":
        live_high, live_low = discover_live_landmarks(engine)
        declared = set(landmarks)
        # "added" = live but not in declared LANDMARKS (the additive blind spot).
        added_high_raw = sorted(live_high - declared)
        added_low = sorted(live_low - declared)

    # Phase C: apply EXCLUSIONS.yaml filtering.
    exclusions = load_exclusions(engine, current_version)
    exclusions_applied: list[str] = []
    expired_exclusions: list[str] = []

    # Check all loaded exclusion entries for expiry, regardless of whether they
    # match an added gap - expired entries are a hard error.
    for excl in exclusions.values():
        if _check_exclusion_expiry(excl):
            expired_exclusions.append(excl.qualname)

    # Filter added_high_raw through exclusions.
    added_high: list[str] = []
    for path in added_high_raw:
        fp = _landmark_fingerprint(path)
        if fp in exclusions:
            excl = exclusions[fp]
            if not _check_exclusion_expiry(excl):
                # Valid (non-expired) exclusion - suppress from landmarks_added.
                exclusions_applied.append(path)
                continue
            # Expired exclusion - path stays in added_high, expiry already recorded above.
        added_high.append(path)

    # Phase C: delta computation.
    # When gate=delta but no baseline has been written yet (first run for this
    # engine), treat as warn-only (no comparison possible). Once a green main
    # commit lands and update_last_probe.py writes the baseline, subsequent
    # PRs will gate normally.
    baseline_raw = _read_added_baseline(current) if gate == "delta" else None
    baseline_absent = gate == "delta" and baseline_raw is None
    baseline_set = set(baseline_raw or [])
    landmarks_added_delta: list[str] = []
    if gate == "delta" and not baseline_absent:
        landmarks_added_delta = sorted(set(added_high) - baseline_set)

    # Verdict: fail when:
    # 1. Declared landmarks are missing (Phase A - always fail).
    # 2. Any exclusion entry has expired (Phase C - force review).
    # 3. Gate=delta with a baseline present and Δ_high > 0.
    # 4. Gate=absolute and any high-confidence gaps remain.
    verdict: Literal["pass", "fail"] = "pass"
    if missing:
        verdict = "fail"
    if expired_exclusions:
        verdict = "fail"
    if (gate == "delta" and not baseline_absent and landmarks_added_delta) or (
        gate == "absolute" and added_high
    ):
        verdict = "fail"

    # Direction: reflect the combined state of both directions.
    has_missing = bool(missing)
    has_added = bool(added_high)
    if has_missing and has_added:
        direction: DirectionKind = "mixed"
    elif has_missing:
        direction = "removed"
    elif has_added:
        direction = "added"
    else:
        direction = "stable"

    return DriftReport(
        engine=engine,
        producer=producer,
        direction=direction,
        schema_version=1,
        current_version=current_version,
        verdict=verdict,
        fingerprint=fingerprint,
        fingerprint_drift=drift,
        landmarks_missing=missing,
        landmarks_added=added_high,
        landmarks_added_low_confidence=added_low,
        version_inside_envelope=_check_envelope(current, producer, current_version),
        exclusions_applied=exclusions_applied,
        expired_exclusions=expired_exclusions,
        landmarks_added_delta=landmarks_added_delta,
        gate=gate,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scripts._drift", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine", required=True, help="Engine name (transformers / vllm / tensorrt)."
    )
    parser.add_argument(
        "--producer",
        required=True,
        choices=("invariants", "schemas"),
        help="Concern produced by the underlying module.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Skip updating engine_versions/{engine}.compat.json (useful for CI dry-runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the DriftReport JSON. When set, JSON goes to this file "
            "atomically (write to .tmp, fsync, rename) and stdout stays clean. When "
            "absent, JSON is written to stdout (legacy behaviour). Use --output in "
            "CI when the engine library may write to stdout at import time (e.g. "
            "tensorrt-llm's logger), which would otherwise pollute the captured JSON."
        ),
    )
    # Phase C flags
    parser.add_argument(
        "--gate",
        choices=("none", "delta", "absolute"),
        default="none",
        help=(
            "CI gating mode. "
            "'none' (default): warning-only, exit 0 always (Phase B behaviour). "
            "'delta': fail when new high-confidence gaps introduced since last green main. "
            "'absolute': fail when any high-confidence gaps remain after exclusions."
        ),
    )
    parser.add_argument(
        "--pr-number",
        default=None,
        help=(
            "PR number to post a sticky coverage comment to. When set, the drift tool "
            "generates a markdown comment and posts it via "
            "scripts/ci/upsert_pr_comment.sh (marker: llem-drift-coverage). "
            "Requires GH_TOKEN and REPO env vars (same as the upsert helper). "
            "Ignored when --comment-output is set."
        ),
    )
    parser.add_argument(
        "--comment-output",
        type=Path,
        default=None,
        help=(
            "Write the rendered comment body to PATH and skip the in-process gh post. "
            "For CI cells whose engine container lacks gh."
        ),
    )
    return parser.parse_args(argv)


def _write_report_to_file(path: Path, report: DriftReport) -> None:
    """Atomically write *report* as JSON to *path*.

    Uses ``tempfile.mkstemp`` for a collision-free temp file in the parent
    directory, fsyncs before ``os.replace`` so the rename only commits a
    durable file (matters on power loss / hard reboot), and cleans up the
    temp file on any failure so retries never trip over orphaned ``.tmp``
    files. Parent directory is created if missing.

    The output is chmod'd to 0644. ``tempfile.mkstemp`` defaults to 0600
    (owner-only) for security, but CI invokes the drift tool inside a Docker
    container running as root and reads the result on the host via a bind
    mount. Without 0644 the non-root host runner user gets ``Permission
    denied`` when opening the JSON.
    """
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


_UPSERT_COMMENT_SCRIPT = _PROJECT_ROOT / "scripts" / "ci" / "upsert_pr_comment.sh"
_DRIFT_COMMENT_MARKER_PREFIX = "llem-drift-coverage"


def _drift_comment_marker(engine: str, producer: ProducerKind) -> str:
    """Per-(engine, producer) marker so invariants and schemas comments coexist
    on a PR instead of overwriting each other."""
    return f"{_DRIFT_COMMENT_MARKER_PREFIX}-{engine}-{producer}"


def _render_drift_comment(report: DriftReport) -> str:
    """Render a markdown PR comment body for the drift report.

    Comment shape:

        <!-- llem-drift-coverage-<engine>-<producer> -->
        ### Coverage drift - {engine} v{ver} ({producer})

        {summary line}

        <details><summary>New gaps (N)</summary>
        - `qualified.path` (high)
        ...
        </details>
    """
    safe_ver = safe_version(report.current_version)
    delta = report.landmarks_added_delta
    delta_set = set(delta)
    pre_existing = [p for p in report.landmarks_added if p not in delta_set]
    n_new = len(delta)
    n_preexisting = len(pre_existing)
    marker = _drift_comment_marker(report.engine, report.producer)

    lines: list[str] = [
        f"<!-- {marker} -->",
        f"### Coverage drift - {report.engine} {safe_ver} ({report.producer})",
        "",
    ]

    if report.expired_exclusions:
        lines.append(
            "> **Expired exclusions:** "
            + ", ".join(f"`{q}`" for q in sorted(report.expired_exclusions))
            + " - re-review required."
        )
        lines.append("")

    if report.gate == "delta":
        lines.append(
            f"Delta +{n_new} added gaps since main, "
            f"{n_preexisting} pre-existing, "
            f"{len(report.exclusions_applied)} excluded"
        )
    else:
        total = len(report.landmarks_added)
        lines.append(
            f"{total} added gaps (after {len(report.exclusions_applied)} exclusions applied)"
        )

    lines.append("")

    if delta:
        lines.append(f"<details><summary>New gaps ({n_new})</summary>")
        lines.append("")
        for path in sorted(delta):
            lines.append(f"- `{path}` (high)")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if pre_existing:
        lines.append(f"<details><summary>Pre-existing gaps ({n_preexisting})</summary>")
        lines.append("")
        for path in sorted(pre_existing):
            lines.append(f"- `{path}` (high)")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if report.exclusions_applied:
        lines.append(
            f"<details><summary>Exclusions applied ({len(report.exclusions_applied)})</summary>"
        )
        lines.append("")
        for path in sorted(report.exclusions_applied):
            lines.append(f"- `{path}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def _post_drift_comment(pr_number: str, comment_body: str, marker: str) -> None:
    """Post (or update) the sticky drift comment via upsert_pr_comment.sh.

    Failures log to stderr but do not abort: a comment failure should not
    fail CI when the gate itself passed.
    """
    script = _UPSERT_COMMENT_SCRIPT
    env = dict(os.environ)
    env["PR"] = pr_number
    env["MARKER"] = marker
    # REPO must already be set in the environment (set by the workflow).
    # GH_TOKEN must already be set.
    try:
        result = subprocess.run(
            ["bash", str(script)],
            input=comment_body.encode(),
            env=env,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(
                f"Warning: upsert_pr_comment.sh exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace')}",
                file=sys.stderr,
            )
        else:
            print(result.stdout.decode(errors="replace").strip(), file=sys.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Warning: could not post drift comment: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gate: GateKind = args.gate  # type: ignore[assignment]
    try:
        report = run(engine=args.engine, producer=args.producer, gate=gate)
    except (FileNotFoundError, KeyError, ImportError, AttributeError, TypeError, ValueError) as exc:
        # Infrastructure failure: current.yaml missing / malformed / producer
        # module unimportable / LANDMARKS missing or malformed. Distinct
        # from a fail-verdict probe (which writes JSON to stdout/file).
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2

    if not args.no_write_cache:
        try:
            _write_cached_report(args.engine, report)
        except OSError as exc:
            print(
                json.dumps({"error": "CacheWriteFailed", "message": str(exc)}),
                file=sys.stderr,
            )
            return 2

    if args.output is not None:
        try:
            _write_report_to_file(args.output, report)
        except OSError as exc:
            print(
                json.dumps({"error": "OutputWriteFailed", "message": str(exc)}),
                file=sys.stderr,
            )
            return 2
    else:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))

    # Phase C: sticky PR comment.
    # Two paths:
    #   --comment-output PATH: write the rendered body to disk; caller posts
    #     it (used in cell workflows where the engine container lacks gh).
    #   --pr-number alone: post in-process via upsert_pr_comment.sh (legacy
    #     CLI-from-host path; requires gh on PATH).
    if args.comment_output is not None:
        comment = _render_drift_comment(report)
        try:
            args.comment_output.parent.mkdir(parents=True, exist_ok=True)
            args.comment_output.write_text(comment, encoding="utf-8")
            os.chmod(args.comment_output, 0o644)
        except OSError as exc:
            print(
                json.dumps({"error": "CommentWriteFailed", "message": str(exc)}),
                file=sys.stderr,
            )
            return 2
    elif args.pr_number:
        comment = _render_drift_comment(report)
        _post_drift_comment(
            args.pr_number,
            comment,
            marker=_drift_comment_marker(args.engine, args.producer),
        )

    # Exit code: 1 when verdict=fail AND the failure was caused by a gate (not
    # infra). The probe-fail gate in the cell workflow reads verdict from JSON,
    # so the exit code here is the CI gate signal for Phase C.
    # Exit code 0 for Phase A/B probe-fail is preserved: those are surfaced via
    # the JSON verdict field and the cell workflow's existing probe-fail gate step.
    if report.verdict == "fail" and gate != "none":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
