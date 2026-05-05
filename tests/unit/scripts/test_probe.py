"""Tests for :mod:`scripts._probe` — the probe primitive.

The probe primitive answers a binary question ("do all landmarks resolve
under the live library?") and emits a richly-diagnosed report. These
tests verify pass/fail verdict derivation, fingerprint stability + drift,
SSOT envelope checks, and round-tripping through the cache file.

LANDMARKS are stdlib symbols (``json.JSONDecodeError`` etc.) rather than
real engine symbols: the probe imports modules via ``importlib`` and
introspects with ``getattr``, so any module path works. Engine libraries
are not installed on host (engines run only inside their Docker images
— see ``docs/development.md``); using stdlib symbols keeps the probe's
resolution logic exercised host-side without an engine dependency.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import _probe  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _install_synthetic_producer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str,
    producer: _probe.ProducerKind,
    landmarks: tuple[str, ...],
    module_name: str = "scripts._probe_synthetic_producer",
) -> types.ModuleType:
    """Register a fake producer module + retarget the ``_PRODUCER_MODULES`` map.

    The probe loads producer modules by ``importlib.import_module``;
    placing one in ``sys.modules`` is sufficient. Tests use this to
    decouple LANDMARKS contents from the real miner / introspector
    bodies, which live elsewhere on the repo's release cadence.
    """
    module = types.ModuleType(module_name)
    module.LANDMARKS = landmarks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(_probe._PRODUCER_MODULES, (engine, producer), module_name)
    return module


def _redirect_compat_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: str = "transformers"
) -> Path:
    """Point both SSOT path and compat cache writes at ``tmp_path``.

    Copies the real engine SSOT into ``tmp_path`` so envelope checks can
    still read it; redirects ``ssot_path`` to the temporary copy.
    """
    real_ssot = _PROJECT_ROOT / "engine_versions" / f"{engine}.yaml"
    fake_ssot = tmp_path / f"{engine}.yaml"
    fake_ssot.write_text(real_ssot.read_text())

    def _fake_path(name: str) -> Path:
        return tmp_path / f"{name}.yaml"

    monkeypatch.setattr(_probe, "ssot_path", _fake_path)
    return fake_ssot


# ---------------------------------------------------------------------------
# Verdict cases
# ---------------------------------------------------------------------------


def test_probe_pass_when_all_landmarks_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All landmarks resolve → verdict ``pass``, ``landmarks_missing`` empty."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=(
            "json.JSONDecodeError",
            "json.JSONEncoder",
        ),
    )

    report = _probe.probe(engine="transformers", producer="invariants")

    assert report.verdict == "pass"
    assert report.landmarks_missing == []
    assert report.engine == "transformers"
    assert report.producer == "invariants"
    assert report.fingerprint  # non-empty hex digest


def test_probe_fail_when_landmark_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One missing landmark → verdict ``fail`` with the offender enumerated."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=(
            "json.JSONDecodeError",
            "json.NonExistentSymbolXYZ",
        ),
    )

    report = _probe.probe(engine="transformers", producer="invariants")

    assert report.verdict == "fail"
    assert report.landmarks_missing == ["json.NonExistentSymbolXYZ"]


# ---------------------------------------------------------------------------
# Fingerprint behaviour
# ---------------------------------------------------------------------------


def test_probe_fingerprint_stable_across_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same library, same landmarks → identical fingerprint across runs."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    first = _probe.probe(engine="transformers", producer="invariants")
    second = _probe.probe(engine="transformers", producer="invariants")

    assert first.fingerprint == second.fingerprint


def test_probe_fingerprint_drift_listed_on_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fingerprint shift across runs surfaces in ``fingerprint_drift``."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    first = _probe.probe(engine="transformers", producer="invariants")
    assert first.fingerprint_drift == []  # no cache yet
    _probe._write_cached_report("transformers", first)

    # Swap the LANDMARKS so the fingerprint shifts; the cached value is
    # the previous one. Drift should be reported.
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONEncoder",),
    )
    second = _probe.probe(engine="transformers", producer="invariants")

    assert second.fingerprint != first.fingerprint
    assert second.fingerprint_drift  # non-empty


# ---------------------------------------------------------------------------
# Diagnostics + envelope
# ---------------------------------------------------------------------------


def test_probe_version_inside_envelope_matches_ssot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``version_inside_envelope`` reflects the SSOT ``miner_pins.static`` range."""
    fake_ssot = _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    # Rewrite the SSOT so the current version sits OUTSIDE the static pin —
    # version_inside_envelope must flip to False without touching verdict.
    fake_ssot.write_text(
        "schema_version: 1\n"
        "engine: transformers\n"
        "library:\n"
        "  pep503_name: transformers\n"
        '  current_version: "0.0.1"\n'
        "miner_pins:\n"
        '  static: ">=99.0,<100.0"\n'
        '  dynamic: ">=99.0,<100.0"\n'
        '  discovery: ">=99.0,<100.0"\n'
    )

    report = _probe.probe(engine="transformers", producer="invariants")

    assert report.verdict == "pass"  # landmarks still resolve
    assert report.version_inside_envelope is False
    assert report.library_version == "0.0.1"


def test_probe_writes_compat_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``main`` writes a round-trippable ``{engine}.compat.json`` cache file."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    rc = _probe.main(["--engine", "transformers", "--producer", "invariants"])
    assert rc == 0

    cache_path = tmp_path / "transformers.compat.json"
    assert cache_path.is_file()
    cache = json.loads(cache_path.read_text())
    assert "invariants" in cache
    assert cache["invariants"]["verdict"] == "pass"
    assert cache["invariants"]["fingerprint"]


def test_probe_output_flag_writes_to_file_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--output PATH`` writes the JSON to a file and prints nothing to stdout.

    Required for engines whose libraries log to stdout at import time
    (tensorrt-llm's custom logger), which would otherwise pollute the
    captured JSON when CI redirects stdout to a file.
    """
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    out = tmp_path / "probe-out" / "report.json"
    rc = _probe.main(["--engine", "transformers", "--producer", "invariants", "--output", str(out)])
    assert rc == 0

    assert out.is_file(), "Probe should write JSON to --output path"
    captured = capsys.readouterr()
    assert captured.out == "", "stdout should be empty when --output is set"

    # Mode 0644 — readable by the host runner user when the probe is invoked
    # from inside a Docker container running as root via a bind mount.
    # `tempfile.mkstemp` defaults to 0600 which would block host reads.
    assert out.stat().st_mode & 0o777 == 0o644, "output must be 0644 for host bind-mount reads"

    payload = json.loads(out.read_text())
    assert payload["engine"] == "transformers"
    assert payload["verdict"] == "pass"


def test_probe_atomic_output_rename_failure_leaves_destination_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crash at the ``os.replace`` step must NOT leave a partial JSON at
    the destination AND must clean up the temp file (no orphan ``.tmp``)."""
    import os as _os

    out = tmp_path / "report.json"
    out.write_text("STALE", encoding="utf-8")  # pre-existing content

    # Force the rename step to fail. The temp file should still be cleaned
    # up (no orphan ``.tmp`` left in the directory) and the destination
    # should still contain its original content.
    def boom(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(_os, "replace", boom)

    report = _probe.ProbeReport(
        engine="transformers",
        producer="invariants",
        verdict="pass",
        library_version="9.9.9",
        version_inside_envelope=True,
        fingerprint="deadbeef",
        fingerprint_drift=[],
        landmarks_missing=[],
    )
    with pytest.raises(OSError, match="simulated rename failure"):
        _probe._write_report_to_file(out, report)

    # Destination unchanged (the rename never happened)
    assert out.read_text() == "STALE"
    # No orphan ``.tmp`` files left behind in the parent directory
    assert not list(tmp_path.glob("*.tmp")), "tempfile cleanup must remove .tmp on failure"


def test_probe_ssot_missing_returns_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No SSOT for the engine → CLI exits 2 with a stderr error envelope."""
    monkeypatch.setattr(_probe, "ssot_path", lambda name: tmp_path / f"{name}.yaml")
    _install_synthetic_producer(
        monkeypatch,
        engine="ghost_engine",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    rc = _probe.main(["--engine", "ghost_engine", "--producer", "invariants"])
    assert rc == 2


def test_probe_unsupported_engine_producer_pair_returns_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unknown ``(engine, producer)`` mapping → exit 2, no JSON on stdout."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(_probe, "_PRODUCER_MODULES", {})  # empty map
    rc = _probe.main(["--engine", "transformers", "--producer", "invariants"])
    assert rc == 2
