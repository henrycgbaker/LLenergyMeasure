"""Probe primitive — binary reusability check for per-engine producers.

The probe is the first step of each per-concern workflow (engine-invariants
+ engine-schemas). It answers one question: do the landmarks the producer
relies on still resolve under the currently-installed library?

Verdict semantics (binary, per the engine-coupling design doc §3):

    pass : every landmark in ``producer.LANDMARKS`` resolves to a live
           class / method / module attribute under ``import {library}``.
    fail : at least one landmark is missing, OR the producer module
           itself fails to import after one retry.

Diagnostic fields (``version_inside_envelope``, ``fingerprint_drift``)
ride along on every report and are surfaced in workflow comments. They
NEVER affect the verdict — those signals steer the human's attention on
``pass``-but-suspicious bumps without gating the pipeline.

Producer-module discovery uses a per-engine convention table (see
``_PRODUCER_MODULES``); each producer module exposes a
``LANDMARKS: tuple[str, ...]`` constant of dotted attribute paths
(e.g. ``"transformers.GenerationConfig"``,
``"vllm.config.parallel.ParallelConfig.__post_init__"``).

Usage::

    python -m scripts._probe --engine transformers --producer invariants

Emits a JSON ``ProbeReport`` to stdout. Exit 0 on either verdict — the
binary verdict travels in the JSON, and downstream workflow steps gate
on it. Exit 2 only on infrastructure failure (SSOT missing, producer
module unimportable, SSOT malformed).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
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
# plain script (``python scripts/_probe.py``) as well as via
# ``python -m scripts._probe``.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_miners._base import MinerLandmarkMissingError  # noqa: E402
from scripts.engine_miners._ssot import ssot_path  # noqa: E402

ProducerKind = Literal["invariants", "schemas"]

# Per-engine producer module map. The probe lives one layer above the
# producers; this table is the single seam where (engine, producer) cells
# resolve to a Python module path. Keep it exhaustive — adding a new
# engine means adding a row here.
_PRODUCER_MODULES: dict[tuple[str, ProducerKind], str] = {
    ("transformers", "invariants"): "scripts.engine_miners.transformers_miner",
    ("vllm", "invariants"): "scripts.engine_miners.vllm_static_miner",
    ("tensorrt", "invariants"): "scripts.engine_miners.tensorrt_static_miner",
    ("transformers", "schemas"): "scripts.engine_introspectors.transformers_introspector",
    ("vllm", "schemas"): "scripts.engine_introspectors.vllm_introspector",
    ("tensorrt", "schemas"): "scripts.engine_introspectors.tensorrt_introspector",
}

# SSOT ``miner_pins.*`` keys are typed by extraction strategy
# (``static | dynamic | discovery``); the probe's user-facing producer
# kinds are typed by concern (``invariants | schemas``). One layer of
# translation lives here.
_SSOT_PIN_FOR_PRODUCER: dict[ProducerKind, str] = {
    "invariants": "static",
    "schemas": "discovery",
}


@dataclass(frozen=True)
class ProbeReport:
    """Structured report of one probe run.

    Serialised to stdout JSON and to ``engine_versions/{engine}.compat.json``
    for cross-run cache + fingerprint-drift detection.
    """

    engine: str
    producer: ProducerKind
    verdict: Literal["pass", "fail"]
    library_version: str
    version_inside_envelope: bool
    fingerprint: str
    fingerprint_drift: list[str] = field(default_factory=list)
    landmarks_missing: list[str] = field(default_factory=list)
    ran_at: str = ""


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
# SSOT helpers
# ---------------------------------------------------------------------------


def _load_ssot(engine: str) -> dict[str, object]:
    """Read + parse ``engine_versions/{engine}.yaml``.

    Raises :class:`FileNotFoundError` if missing. The probe treats SSOT
    absence as an infrastructure error (exit code 2) — every supported
    engine must have a SSOT before the probe is wired up for it.
    """
    path = ssot_path(engine)
    text = path.read_text()  # FileNotFoundError surfaces here
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"SSOT at {path} did not parse to a mapping.")
    return data


def _check_envelope(ssot: dict[str, object], producer: ProducerKind, library_version: str) -> bool:
    """True iff ``library_version`` falls inside ``miner_pins.{ssot_key}``."""
    pins = ssot.get("miner_pins") or {}
    if not isinstance(pins, dict):
        return False
    raw = pins.get(_SSOT_PIN_FOR_PRODUCER[producer])
    if raw is None:
        return False
    spec = SpecifierSet(str(raw))
    return spec.contains(library_version, prereleases=True)


def _compat_path(engine: str) -> Path:
    """Return the path to the cached fingerprint file for ``engine``."""
    return ssot_path(engine).parent / f"{engine}.compat.json"


def _read_cached_fingerprint(engine: str, producer: ProducerKind) -> str | None:
    """Return the previous-run fingerprint for this (engine, producer), or ``None``."""
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
    fp = entry.get("fingerprint") if isinstance(entry, dict) else None
    return str(fp) if isinstance(fp, str) else None


def _write_cached_report(engine: str, report: ProbeReport) -> None:
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
    time (per design D-8) — a flaky import is treated as transient and
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


def probe(*, engine: str, producer: ProducerKind) -> ProbeReport:
    """Run the probe for one ``(engine, producer)`` cell.

    Returns a :class:`ProbeReport` with verdict ``pass`` iff every
    landmark resolves; ``fail`` iff one or more landmarks raise
    :class:`AttributeError` / :class:`ImportError` during resolution.

    Diagnostic fields (envelope check, fingerprint drift) are computed
    regardless of verdict.
    """
    ssot = _load_ssot(engine)  # FileNotFoundError -> caller handles as infra error
    library = ssot.get("library")
    if not isinstance(library, dict) or "current_version" not in library:
        raise ValueError(f"SSOT for {engine} missing library.current_version.")
    library_version = str(library["current_version"])

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
    cached = _read_cached_fingerprint(engine, producer)
    drift: list[str] = []
    if cached is not None and cached != fingerprint:
        # Per-landmark drift would require caching per-landmark hashes;
        # the binary "fingerprint shifted at all" signal is what the
        # writeback comment surfaces, so the list captures the affected
        # landmarks, not the cause.
        drift = [r.landmark for r in resolved]

    verdict: Literal["pass", "fail"] = "fail" if missing else "pass"

    return ProbeReport(
        engine=engine,
        producer=producer,
        verdict=verdict,
        library_version=library_version,
        version_inside_envelope=_check_envelope(ssot, producer, library_version),
        fingerprint=fingerprint,
        fingerprint_drift=drift,
        landmarks_missing=missing,
        ran_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scripts._probe", description=__doc__.splitlines()[0])
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
            "Path to write the ProbeReport JSON. When set, JSON goes to this file "
            "atomically (write to .tmp, fsync, rename) and stdout stays clean. When "
            "absent, JSON is written to stdout (legacy behaviour). Use --output in "
            "CI when the engine library may write to stdout at import time (e.g. "
            "tensorrt-llm's logger), which would otherwise pollute the captured JSON."
        ),
    )
    return parser.parse_args(argv)


def _write_report_to_file(path: Path, report: ProbeReport) -> None:
    """Atomically write *report* as JSON to *path*.

    Uses ``tempfile.mkstemp`` for a collision-free temp file in the parent
    directory, fsyncs before ``os.replace`` so the rename only commits a
    durable file (matters on power loss / hard reboot), and cleans up the
    temp file on any failure so retries never trip over orphaned ``.tmp``
    files. Parent directory is created if missing.
    """
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = probe(engine=args.engine, producer=args.producer)
    except (FileNotFoundError, KeyError, ImportError, AttributeError, TypeError, ValueError) as exc:
        # Infrastructure failure: SSOT missing / malformed / producer
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
