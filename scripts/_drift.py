"""Drift tool - reachability check for per-engine producer landmarks.

For each landmark declared in a producer's ``LANDMARKS`` tuple, try
``importlib.import_module`` on progressively-shorter prefixes plus ``getattr``
on the remainder. A landmark that doesn't resolve (raises AttributeError /
ImportError) joins ``landmarks_missing``; a non-empty ``landmarks_missing``
flips the verdict to ``fail``.

The verdict is the only signal CI gates on. Diagnostic fields ride along on
every report and are informational - they NEVER affect verdict:

    ``fingerprint``             : sha256 of (qualname, filename, lineno)
                                  tuples; shifts on real refactors.
    ``fingerprint_drift``       : landmarks whose coordinates moved since
                                  the cached fingerprint was written.
    ``landmarks_aliased``       : landmarks whose declared path resolves
                                  through a package re-export rather than
                                  at the canonical home (e.g. upstream
                                  refactored a flat module into a subpackage
                                  and kept the old import paths working
                                  via ``from X.Y import Z`` in ``__init__``).
                                  A maintainer-facing hint that the canonical
                                  module path has moved; future producer cuts
                                  should declare against the canonical path.

Producer-module discovery uses a per-engine convention table (see
``_PRODUCER_MODULES``); each producer module exposes a
``LANDMARKS: tuple[str, ...]`` constant of dotted attribute paths
(e.g. ``"transformers.GenerationConfig"``,
``"vllm.config.parallel.ParallelConfig.__post_init__"``).

The CLI producer kind tokens ("invariants", "schemas") are the probe-contract
names, NOT the producer-file names. ``_PRODUCER_MODULES`` translates them to
the underlying module paths.

Usage::

    python -m scripts._drift --engine vllm --producer invariants

Emits a JSON ``DriftReport`` to stdout (or to ``--output PATH``). Exit 0 on
either verdict - the binary verdict travels in the JSON's verdict field, and
downstream workflow steps gate on it. Exit 2 only on infrastructure failure
(current.yaml missing, producer module unimportable, current.yaml malformed).
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

# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftReport:
    """Structured report of one drift-tool run.

    Serialised to stdout JSON (or ``--output PATH``) and to
    ``engine_versions/{engine}.compat.json`` for cross-run cache +
    fingerprint-drift detection.

    ``verdict`` is ``fail`` iff ``landmarks_missing`` is non-empty.

    Diagnostic fields (``fingerprint_drift``, ``landmarks_aliased``) ride
    along on every report and steer human attention on pass-but-suspicious
    bumps. They NEVER affect verdict.
    """

    engine: str
    producer: ProducerKind
    schema_version: int
    current_version: str
    verdict: Literal["pass", "fail"]
    fingerprint: str
    fingerprint_drift: list[str] = field(default_factory=list)
    landmarks_missing: list[str] = field(default_factory=list)
    landmarks_aliased: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Landmark resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedLandmark:
    """One successfully-resolved landmark and its source coordinates.

    The fingerprint is a sorted hash of (landmark, qualname, filename, lineno);
    ``filename`` + ``lineno`` are advisory but track real refactors (a method
    moving from line 142 to 187 across a patch release).

    ``aliased`` is True when ``obj.__module__`` differs from the longest
    importable prefix of ``landmark`` - i.e. the landmark resolved through a
    package re-export shim (the upstream library moved the symbol's canonical
    home but kept the old import path working). Objects without ``__module__``
    (modules themselves, certain C extension types) are not aliased.
    """

    landmark: str
    qualname: str
    filename: str | None
    lineno: int | None
    aliased: bool


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

    declared_module = ".".join(parts[:module_idx])
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
    resolved_module = getattr(obj, "__module__", None)
    aliased = resolved_module is not None and resolved_module != declared_module
    return _ResolvedLandmark(
        landmark=landmark,
        qualname=str(qualname),
        filename=filename,
        lineno=lineno,
        aliased=aliased,
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

    Indirects through ``current_path`` from this module's namespace so tests
    can monkeypatch ``_drift.current_path`` for hermetic fixtures.
    Equivalent in shape to ``scripts.engine_producers._current.load_current``
    but kept private here for the test-surface reason above.
    """
    path = current_path(engine)
    text = path.read_text()  # FileNotFoundError surfaces here
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"current.yaml at {path} did not parse to a mapping.")
    return data


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
    time - a flaky import is treated as transient and retried before the
    verdict is finalised. A second failure escalates to caller; the CLI
    maps it to exit code 2 (infrastructure failure).
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

    Returns a :class:`DriftReport` with verdict ``pass`` when every declared
    landmark resolves under the current library, ``fail`` when at least one
    is absent.

    Diagnostic fields (fingerprint drift, landmarks aliased) are computed
    regardless of verdict and ride along on the report. They never affect
    verdict.
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

    aliased = [r.landmark for r in resolved if r.aliased]

    verdict: Literal["pass", "fail"] = "fail" if missing else "pass"

    return DriftReport(
        engine=engine,
        producer=producer,
        schema_version=1,
        current_version=current_version,
        verdict=verdict,
        fingerprint=fingerprint,
        fingerprint_drift=drift,
        landmarks_missing=missing,
        landmarks_aliased=aliased,
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
