"""Drift tool - bidirectional landmark drift detection for per-engine producers.

The drift tool is the first step of each per-concern workflow (engine-invariants
+ engine-schemas). Phase A covers the "removed" direction only, which is a
drop-in replacement for the former ``_probe.py`` non-regression gate:

    pass : every landmark in ``producer.LANDMARKS`` resolves to a live
           class / method / module attribute under ``import {library}``.
    fail : at least one landmark is missing, OR the producer module
           itself fails to import after one retry.

Phase B (follow-up) will add the "added" direction - validators that exist
in the live library but are not declared in the per-version archive.
Phase C will add exclusions and CI gating.

``direction`` field semantics:

    "removed" : one or more declared landmarks are absent from the live library.
    "added"   : (Phase B) one or more live validators are absent from LANDMARKS.
    "stable"  : all declared landmarks resolve; no undeclared validators found.

In Phase A, each ``DriftReport`` carries ``direction`` = "removed" when
``landmarks_missing`` is non-empty, and ``direction`` = "stable" otherwise.
``landmarks_added`` is always empty in Phase A.

Diagnostic fields (``version_inside_envelope``, ``fingerprint_drift``)
ride along on every report and are surfaced in workflow comments. They
NEVER affect the verdict - those signals steer the human's attention on
``pass``-but-suspicious bumps without gating the pipeline.

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
import hashlib
import importlib
import inspect
import json
import os
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
from scripts.engine_producers._current import current_path  # noqa: E402

ProducerKind = Literal["invariants", "schemas"]

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


@dataclass(frozen=True)
class DriftReport:
    """Structured report of one drift-tool run.

    Serialised to stdout JSON and to ``engine_versions/{engine}.compat.json``
    for cross-run cache + fingerprint-drift detection.

    Phase A emits ``direction`` = "removed" when ``landmarks_missing`` is
    non-empty; "stable" otherwise. ``landmarks_added`` is always empty in
    Phase A - Phase B will fill it by walking the live library surface.
    """

    engine: str
    producer: ProducerKind
    direction: Literal["removed", "added", "stable"]
    schema_version: int
    current_version: str
    verdict: Literal["pass", "fail"]
    fingerprint: str
    fingerprint_drift: list[str] = field(default_factory=list)
    landmarks_missing: list[str] = field(default_factory=list)
    landmarks_added: list[str] = field(default_factory=list)
    version_inside_envelope: bool = True


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


def run(*, engine: str, producer: ProducerKind) -> DriftReport:
    """Run the drift check for one ``(engine, producer)`` cell.

    Phase A: "removed" direction only.

    Returns a :class:`DriftReport` with verdict ``pass`` iff every
    landmark resolves; ``fail`` iff one or more landmarks raise
    :class:`AttributeError` / :class:`ImportError` during resolution.

    Diagnostic fields (envelope check, fingerprint drift) are computed
    regardless of verdict.

    ``direction`` is ``"removed"`` when ``landmarks_missing`` is non-empty;
    ``"stable"`` otherwise. ``landmarks_added`` is always ``[]`` in Phase A.
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

    verdict: Literal["pass", "fail"] = "fail" if missing else "pass"
    direction: Literal["removed", "added", "stable"] = "removed" if missing else "stable"

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
        landmarks_added=[],
        version_inside_envelope=_check_envelope(current, producer, current_version),
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run(engine=args.engine, producer=args.producer)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
