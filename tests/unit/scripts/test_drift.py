"""Tests for :mod:`scripts._drift` - reachability check for producer landmarks.

The drift tool answers a binary question: do all landmarks declared in a
producer's ``LANDMARKS`` tuple resolve under the live library? Emits a
:class:`DriftReport` whose ``verdict`` is ``"fail"`` iff
``landmarks_missing`` is non-empty.

These tests verify pass/fail verdict derivation, fingerprint stability +
drift, current.yaml envelope checks, and round-tripping through the cache
file.

LANDMARKS are stdlib symbols (``json.JSONDecodeError`` etc.) rather than
real engine symbols: the drift tool imports modules via ``importlib`` and
introspects with ``getattr``, so any module path works. Engine libraries
are not installed on host (engines run only inside their Docker images
- see ``docs/development.md``); using stdlib symbols keeps the resolution
logic exercised host-side without an engine dependency.
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

from scripts import _drift  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _install_synthetic_producer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str,
    producer: _drift.ProducerKind,
    landmarks: tuple[str, ...],
    module_name: str = "scripts._drift_synthetic_producer",
    uncovered_probe: tuple[list[str], list[str]] | None = None,
) -> types.ModuleType:
    """Register a fake producer module + retarget the ``_PRODUCER_MODULES`` map.

    The drift tool loads producer modules by ``importlib.import_module``;
    placing one in ``sys.modules`` is sufficient. Tests use this to
    decouple LANDMARKS contents from the real miner / introspector
    bodies, which live elsewhere on the repo's release cadence.

    When ``uncovered_probe`` is given, the module also exposes a
    ``probe_uncovered_validators`` callable returning that
    ``(uncovered, unanalyzable)`` tuple - the optional drift-completeness
    prober ``run()`` invokes for the "invariants" producer.
    """
    module = types.ModuleType(module_name)
    module.LANDMARKS = landmarks  # type: ignore[attr-defined]
    if uncovered_probe is not None:

        def _probe() -> tuple[list[str], list[str]]:
            return uncovered_probe

        module.probe_uncovered_validators = _probe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(_drift._PRODUCER_MODULES, (engine, producer), module_name)
    return module


def _redirect_compat_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: str = "transformers"
) -> Path:
    """Point both current.yaml path and compat cache writes at ``tmp_path``.

    Copies the real engine current.yaml into a per-engine subdirectory
    under ``tmp_path`` so envelope checks can still read it; redirects
    ``current_path`` to the temporary copy.

    The compat.json cache ends up at ``tmp_path/{engine}.compat.json``
    because ``_compat_path`` walks two parents up from the fake current.yaml
    (``tmp_path/{engine}/current.yaml`` -> ``.parent.parent`` = ``tmp_path``).
    """
    real_current = _PROJECT_ROOT / "engine_versions" / engine / "current.yaml"
    fake_engine_dir = tmp_path / engine
    fake_engine_dir.mkdir(parents=True, exist_ok=True)
    fake_current = fake_engine_dir / "current.yaml"
    fake_current.write_text(real_current.read_text())

    def _fake_path(name: str) -> Path:
        return tmp_path / name / "current.yaml"

    monkeypatch.setattr(_drift, "current_path", _fake_path)
    return fake_current


# ---------------------------------------------------------------------------
# Verdict cases
# ---------------------------------------------------------------------------


def test_drift_pass_when_all_landmarks_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All landmarks resolve -> verdict ``pass``, ``landmarks_missing`` empty."""
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

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.verdict == "pass"
    assert report.landmarks_missing == []
    assert report.engine == "transformers"
    assert report.producer == "invariants"
    assert report.fingerprint  # non-empty hex digest


def test_drift_fail_when_landmark_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One missing landmark -> verdict ``fail`` with the offender enumerated."""
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

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.verdict == "fail"
    assert report.landmarks_missing == ["json.NonExistentSymbolXYZ"]


def test_drift_schema_version_is_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``schema_version`` is 2 (bumped when the validator-completeness fields landed)."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.schema_version == 2


# ---------------------------------------------------------------------------
# Fingerprint behaviour
# ---------------------------------------------------------------------------


def test_drift_fingerprint_stable_across_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same library, same landmarks -> identical fingerprint across runs."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    first = _drift.run(engine="transformers", producer="invariants")
    second = _drift.run(engine="transformers", producer="invariants")

    assert first.fingerprint == second.fingerprint


def test_drift_fingerprint_drift_listed_on_change(
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

    first = _drift.run(engine="transformers", producer="invariants")
    assert first.fingerprint_drift == []  # no cache yet
    _drift._write_cached_report("transformers", first)

    # Swap the LANDMARKS so the fingerprint shifts; the cached value is
    # the previous one. Drift should be reported.
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONEncoder",),
    )
    second = _drift.run(engine="transformers", producer="invariants")

    assert second.fingerprint != first.fingerprint
    assert second.fingerprint_drift  # non-empty


def test_drift_names_only_the_moved_landmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fingerprint_drift names the specific moved landmark, not every landmark.

    Regression for Bug 5: when the aggregate fingerprint shifted, the report
    listed ALL resolved landmarks, so the CI human-attention comment could never
    point at the field that actually moved. With two landmarks where only one
    changes source coordinates, only that one must appear in fingerprint_drift.
    """
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("pkg.Stable", "pkg.Moved"),
    )

    # Coordinates per landmark; the second run bumps only "pkg.Moved"'s lineno.
    coords = {"pkg.Stable": 10, "pkg.Moved": 20}

    def _fake_resolve(landmark: str) -> _drift._ResolvedLandmark:
        return _drift._ResolvedLandmark(
            landmark=landmark,
            qualname=landmark.rsplit(".", 1)[-1],
            filename="pkg.py",
            lineno=coords[landmark],
            aliased=False,
        )

    monkeypatch.setattr(_drift, "_resolve_landmark", _fake_resolve)

    first = _drift.run(engine="transformers", producer="invariants")
    assert first.fingerprint_drift == []  # no cache yet
    assert set(first.landmark_fingerprints) == {"pkg.Stable", "pkg.Moved"}
    _drift._write_cached_report("transformers", first)

    # Move only pkg.Moved (lineno 20 -> 99); pkg.Stable is untouched.
    coords["pkg.Moved"] = 99
    second = _drift.run(engine="transformers", producer="invariants")

    assert second.fingerprint != first.fingerprint
    assert second.fingerprint_drift == ["pkg.Moved"], (
        "only the landmark whose coordinates moved must be named"
    )


# ---------------------------------------------------------------------------
# Cache + diagnostics
# ---------------------------------------------------------------------------


def test_drift_writes_compat_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``main`` writes a round-trippable ``{engine}.compat.json`` cache file."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    rc = _drift.main(["--engine", "transformers", "--producer", "invariants"])
    assert rc == 0

    cache_path = tmp_path / "transformers.compat.json"
    assert cache_path.is_file()
    cache = json.loads(cache_path.read_text())
    assert "invariants" in cache
    assert cache["invariants"]["verdict"] == "pass"
    assert cache["invariants"]["fingerprint"]
    assert cache["invariants"]["schema_version"] == 2
    assert cache["invariants"]["current_version"]


def test_drift_output_flag_writes_to_file_keeps_stdout_clean(
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

    out = tmp_path / "drift-out" / "report.json"
    rc = _drift.main(["--engine", "transformers", "--producer", "invariants", "--output", str(out)])
    assert rc == 0

    assert out.is_file(), "Drift tool should write JSON to --output path"
    captured = capsys.readouterr()
    assert captured.out == "", "stdout should be empty when --output is set"

    # Mode 0644 - readable by the host runner user when the tool is invoked
    # from inside a Docker container running as root via a bind mount.
    # ``tempfile.mkstemp`` defaults to 0600 which would block host reads.
    assert out.stat().st_mode & 0o777 == 0o644, "output must be 0644 for host bind-mount reads"

    payload = json.loads(out.read_text())
    assert payload["engine"] == "transformers"
    assert payload["verdict"] == "pass"
    assert payload["schema_version"] == 2
    assert "current_version" in payload
    assert "landmarks_missing" in payload


def test_drift_atomic_output_rename_failure_leaves_destination_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crash at the ``os.replace`` step must NOT leave a partial JSON at
    the destination AND must clean up the temp file (no orphan ``.tmp``)."""
    import os as _os

    out = tmp_path / "report.json"
    out.write_text("STALE", encoding="utf-8")  # pre-existing content

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(_os, "replace", boom)

    report = _drift.DriftReport(
        engine="transformers",
        producer="invariants",
        schema_version=2,
        current_version="9.9.9",
        verdict="pass",
        fingerprint="deadbeef",
        fingerprint_drift=[],
        landmarks_missing=[],
    )
    with pytest.raises(OSError, match="simulated rename failure"):
        _drift._write_report_to_file(out, report)

    # Destination unchanged (the rename never happened)
    assert out.read_text() == "STALE"
    # No orphan ``.tmp`` files left behind in the parent directory
    assert not list(tmp_path.glob("*.tmp")), "tempfile cleanup must remove .tmp on failure"


def test_drift_ssot_missing_returns_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No current.yaml for the engine -> CLI exits 2 with a stderr error envelope."""
    monkeypatch.setattr(_drift, "current_path", lambda name: tmp_path / f"{name}.yaml")
    _install_synthetic_producer(
        monkeypatch,
        engine="ghost_engine",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
    )

    rc = _drift.main(["--engine", "ghost_engine", "--producer", "invariants"])
    assert rc == 2


def test_drift_unsupported_engine_producer_pair_returns_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unknown ``(engine, producer)`` mapping -> exit 2, no JSON on stdout."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(_drift, "_PRODUCER_MODULES", {})  # empty map
    rc = _drift.main(["--engine", "transformers", "--producer", "invariants"])
    assert rc == 2


# ---------------------------------------------------------------------------
# T2 alias detection
# ---------------------------------------------------------------------------


def _install_class_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    canonical_module: str,
    alias_parent: str | None,
    class_name: str = "Foo",
) -> type:
    """Install a synthetic class in ``canonical_module`` with optional re-export.

    When ``alias_parent`` is set, also installs a parent-level module that
    re-exports the class (sets ``alias_parent.<class_name> = cls``). The
    class's ``__module__`` always points at the canonical module, so an
    alias-path resolution will surface as aliased.
    """
    canonical = types.ModuleType(canonical_module)
    cls = type(class_name, (), {})
    cls.__module__ = canonical_module
    setattr(canonical, class_name, cls)
    monkeypatch.setitem(sys.modules, canonical_module, canonical)

    if alias_parent is not None:
        alias = types.ModuleType(alias_parent)
        setattr(alias, class_name, cls)  # re-export
        monkeypatch.setitem(sys.modules, alias_parent, alias)

    return cls


def test_resolved_landmark_carries_aliased_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_resolve_landmark`` sets ``aliased`` from declared vs resolved module comparison."""
    _install_class_module(
        monkeypatch,
        canonical_module="t2_resolver_canonical_pkg",
        alias_parent=None,
        class_name="Bar",
    )
    _install_class_module(
        monkeypatch,
        canonical_module="t2_resolver_alias_pkg.canonical",
        alias_parent="t2_resolver_alias_pkg",
        class_name="Baz",
    )

    canonical = _drift._resolve_landmark("t2_resolver_canonical_pkg.Bar")
    aliased = _drift._resolve_landmark("t2_resolver_alias_pkg.Baz")

    assert canonical.aliased is False
    assert aliased.aliased is True


def test_aliased_landmark_surfaces_in_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A landmark whose declared path goes through a re-export surfaces in ``landmarks_aliased``."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_class_module(
        monkeypatch,
        canonical_module="t2_alias_test_pkg.canonical",
        alias_parent="t2_alias_test_pkg",
        class_name="Foo",
    )
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("t2_alias_test_pkg.Foo",),  # aliased: parent re-exports from .canonical
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.verdict == "pass"
    assert report.landmarks_aliased == ["t2_alias_test_pkg.Foo"]
    assert report.landmarks_missing == []


def test_canonical_landmark_does_not_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A landmark resolving at its canonical home is NOT flagged as aliased."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_class_module(
        monkeypatch,
        canonical_module="t2_canonical_landmark_pkg",
        alias_parent=None,
        class_name="Foo",
    )
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("t2_canonical_landmark_pkg.Foo",),
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.verdict == "pass"
    assert report.landmarks_aliased == []


def test_aliased_never_affects_verdict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``landmarks_aliased`` is informational; verdict stays ``pass`` while ``landmarks_missing`` is empty."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_class_module(
        monkeypatch,
        canonical_module="t2_verdict_pkg.canonical",
        alias_parent="t2_verdict_pkg",
        class_name="Foo",
    )
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("t2_verdict_pkg.Foo",),
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.verdict == "pass"  # not flipped by alias
    assert report.landmarks_aliased  # but informational signal present
    assert report.landmarks_missing == []


# ---------------------------------------------------------------------------
# Validator drift-completeness diagnostic
# ---------------------------------------------------------------------------


def test_uncovered_validators_surface_in_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optional prober's ``(uncovered, unanalyzable)`` ride along on the report."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
        uncovered_probe=(["Foo.validate_after"], ["Bar.inherited_validator"]),
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.validators_uncovered == ["Foo.validate_after"]
    assert report.validators_unanalyzable == ["Bar.inherited_validator"]


def test_uncovered_validators_never_affect_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-empty ``validators_uncovered`` is informational; verdict stays ``pass``."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),  # all resolve -> landmarks_missing empty
        uncovered_probe=(["Foo.relocated_check", "Baz.new_validator"], []),
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.validators_uncovered  # non-empty completeness signal
    assert report.landmarks_missing == []
    assert report.verdict == "pass"  # not flipped by the uncovered set


def test_uncovered_validators_not_probed_for_schemas_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """For the "schemas" producer the prober is NOT called - the fields stay empty.

    Even when a schema producer module happens to expose
    ``probe_uncovered_validators``, ``run()`` only invokes it for "invariants".
    """
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="schemas",
        landmarks=("json.JSONDecodeError",),
        uncovered_probe=(["Should.not_appear"], ["Should.not_appear_either"]),
    )

    report = _drift.run(engine="transformers", producer="schemas")

    assert report.validators_uncovered == []
    assert report.validators_unanalyzable == []


def test_uncovered_validators_empty_when_prober_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A producer WITHOUT ``probe_uncovered_validators`` yields empty lists (permits-zero)."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
        # no uncovered_probe -> module has no probe_uncovered_validators attr
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.validators_uncovered == []
    assert report.validators_unanalyzable == []


def test_uncovered_validators_schema_version_is_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The completeness-field report carries ``schema_version`` 2."""
    _redirect_compat_dir(monkeypatch, tmp_path)
    _install_synthetic_producer(
        monkeypatch,
        engine="transformers",
        producer="invariants",
        landmarks=("json.JSONDecodeError",),
        uncovered_probe=(["Foo.validate_after"], []),
    )

    report = _drift.run(engine="transformers", producer="invariants")

    assert report.schema_version == 2
