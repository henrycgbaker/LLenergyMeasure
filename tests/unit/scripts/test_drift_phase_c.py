"""Tests for :mod:`scripts._drift` Phase C (EXCLUSIONS.yaml + CI gating + sticky comments).

Phase C adds three concerns on top of Phase B:

1. EXCLUSIONS.yaml per archive version - fingerprint-keyed allow-list for
   ``landmarks_added`` entries the maintainer has chosen not to walk. Expired
   entries (``expires`` < today) force verdict=fail.

2. Delta-only CI gating - ``--gate delta`` fails CI only when new high-confidence
   gaps have been introduced since last green main (Δ > 0). Baseline read from
   ``current.yaml last_probe.added_baseline``.

3. Sticky PR comment - ``_render_drift_comment()`` produces a markdown body
   with the stable HTML marker ``llem-drift-coverage`` for dedup.

Tests in this module operate purely on Python-land functions and use synthetic
types / monkeypatching - no engine library installation required.
"""

from __future__ import annotations

import datetime
import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import (  # noqa: E402
    _drift,
    update_last_probe,
)

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_drift_phase_b.py - kept self-contained)
# ---------------------------------------------------------------------------


def _install_synthetic_producer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str,
    producer: _drift.ProducerKind,
    landmarks: tuple[str, ...],
    module_name: str = "scripts._drift_synthetic_producer_phase_c",
) -> types.ModuleType:
    """Register a fake producer module and retarget ``_PRODUCER_MODULES``."""
    module = types.ModuleType(module_name)
    module.LANDMARKS = landmarks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(_drift._PRODUCER_MODULES, (engine, producer), module_name)
    return module


def _redirect_compat_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: str = "transformers"
) -> Path:
    """Point current.yaml path and compat cache writes at ``tmp_path``."""
    real_current = _PROJECT_ROOT / "engine_versions" / engine / "current.yaml"
    fake_engine_dir = tmp_path / engine
    fake_engine_dir.mkdir(parents=True, exist_ok=True)
    fake_current = fake_engine_dir / "current.yaml"
    fake_current.write_text(real_current.read_text())

    def _fake_path(name: str) -> Path:
        return tmp_path / name / "current.yaml"

    monkeypatch.setattr(_drift, "current_path", _fake_path)
    return fake_current


def _make_exclusions_yaml(entries: list[dict[str, Any]]) -> str:
    """Serialise a list of exclusion dicts to YAML text."""
    return yaml.dump(entries, default_flow_style=False, allow_unicode=True)


def _fp(path: str) -> str:
    """Compute the landmark fingerprint for a dotted path (mirrors _landmark_fingerprint)."""
    return hashlib.sha256(path.encode()).hexdigest()


# ---------------------------------------------------------------------------
# _landmark_fingerprint
# ---------------------------------------------------------------------------


class TestLandmarkFingerprint:
    """The fingerprint is a stable sha256 of the dotted path."""

    def test_deterministic(self) -> None:
        """Same path always produces same fingerprint."""
        path = "vllm.config.SchedulerConfig._validate_dtype"
        assert _drift._landmark_fingerprint(path) == _drift._landmark_fingerprint(path)

    def test_different_paths_differ(self) -> None:
        """Different paths produce different fingerprints."""
        a = _drift._landmark_fingerprint("vllm.config.A.validate_x")
        b = _drift._landmark_fingerprint("vllm.config.B.validate_x")
        assert a != b

    def test_known_value(self) -> None:
        """Fingerprint matches expected sha256 of the encoded path."""
        path = "vllm.config.SchedulerConfig._test"
        expected = hashlib.sha256(path.encode()).hexdigest()
        assert _drift._landmark_fingerprint(path) == expected


# ---------------------------------------------------------------------------
# load_exclusions
# ---------------------------------------------------------------------------


class TestLoadExclusions:
    """load_exclusions() parses EXCLUSIONS.yaml and returns fingerprint-keyed dict."""

    def _make_archive_dir(self, tmp_path: Path, engine: str, version: str) -> Path:
        """Create the archive directory structure and return it."""
        from scripts.engine_producers._current import safe_version

        archive_dir = tmp_path / "engine_versions" / engine / safe_version(version)
        archive_dir.mkdir(parents=True, exist_ok=True)
        return archive_dir

    def _patch_project_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Redirect load_exclusions to look in tmp_path by patching _PROJECT_ROOT."""
        monkeypatch.setattr(_drift, "_PROJECT_ROOT", tmp_path)

    def test_missing_file_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When EXCLUSIONS.yaml does not exist, returns empty dict."""
        self._patch_project_root(monkeypatch, tmp_path)
        self._make_archive_dir(tmp_path, "vllm", "0.7.3")
        # No EXCLUSIONS.yaml created
        result = _drift.load_exclusions("vllm", "0.7.3")
        assert result == {}

    def test_valid_entry_parsed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A valid EXCLUSIONS.yaml entry is loaded keyed by fingerprint."""
        self._patch_project_root(monkeypatch, tmp_path)
        archive_dir = self._make_archive_dir(tmp_path, "vllm", "0.7.3")
        path = "vllm.config.SchedulerConfig._test_helper"
        fp = _fp(path)
        entries = [
            {
                "fingerprint": fp,
                "qualname": path,
                "reason": "private helper, not a validator",
                "added": "2026-05-14",
            }
        ]
        (archive_dir / "EXCLUSIONS.yaml").write_text(_make_exclusions_yaml(entries))

        result = _drift.load_exclusions("vllm", "0.7.3")
        assert fp in result
        entry = result[fp]
        assert entry.qualname == path
        assert entry.reason == "private helper, not a validator"
        assert entry.expires is None

    def test_entry_with_expires_parsed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Entry with optional ``expires`` field is parsed correctly."""
        self._patch_project_root(monkeypatch, tmp_path)
        archive_dir = self._make_archive_dir(tmp_path, "vllm", "0.7.3")
        path = "vllm.config.SomeConfig.validate_x"
        fp = _fp(path)
        entries = [
            {
                "fingerprint": fp,
                "qualname": path,
                "reason": "temp exclusion",
                "added": "2026-05-14",
                "expires": "2026-08-14",
            }
        ]
        (archive_dir / "EXCLUSIONS.yaml").write_text(_make_exclusions_yaml(entries))

        result = _drift.load_exclusions("vllm", "0.7.3")
        assert fp in result
        assert result[fp].expires == "2026-08-14"

    def test_invalid_entry_missing_keys_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Entries missing required keys are skipped; valid entries remain."""
        self._patch_project_root(monkeypatch, tmp_path)
        archive_dir = self._make_archive_dir(tmp_path, "vllm", "0.7.3")
        valid_path = "vllm.config.A.validate_x"
        valid_fp = _fp(valid_path)
        entries = [
            # Invalid: missing 'reason' and 'added'
            {"fingerprint": "deadbeef", "qualname": "some.path"},
            # Valid
            {
                "fingerprint": valid_fp,
                "qualname": valid_path,
                "reason": "ok",
                "added": "2026-05-14",
            },
        ]
        (archive_dir / "EXCLUSIONS.yaml").write_text(_make_exclusions_yaml(entries))

        result = _drift.load_exclusions("vllm", "0.7.3")
        assert len(result) == 1
        assert valid_fp in result

    def test_non_list_yaml_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A YAML file that isn't a list is rejected (returns empty, warns to stderr)."""
        self._patch_project_root(monkeypatch, tmp_path)
        archive_dir = self._make_archive_dir(tmp_path, "vllm", "0.7.3")
        (archive_dir / "EXCLUSIONS.yaml").write_text("not_a_list: true\n")
        result = _drift.load_exclusions("vllm", "0.7.3")
        assert result == {}

    def test_version_with_dots_maps_to_safe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Version string with dots is converted to safe form (v0_7_3)."""
        self._patch_project_root(monkeypatch, tmp_path)
        archive_dir = self._make_archive_dir(tmp_path, "vllm", "0.7.3")
        path = "vllm.config.X.validate"
        fp = _fp(path)
        entries = [{"fingerprint": fp, "qualname": path, "reason": "r", "added": "2026-05-14"}]
        (archive_dir / "EXCLUSIONS.yaml").write_text(_make_exclusions_yaml(entries))

        result = _drift.load_exclusions("vllm", "0.7.3")
        assert fp in result


# ---------------------------------------------------------------------------
# _check_exclusion_expiry
# ---------------------------------------------------------------------------


class TestCheckExclusionExpiry:
    """_check_exclusion_expiry() returns True only for entries whose expires < today."""

    def _make_entry(self, expires: str | None) -> _drift.ExclusionEntry:
        return _drift.ExclusionEntry(
            fingerprint="abc",
            qualname="some.path",
            reason="test",
            added="2026-05-14",
            expires=expires,
        )

    def test_no_expires_not_expired(self) -> None:
        """Entry without expires is never expired."""
        entry = self._make_entry(None)
        assert _drift._check_exclusion_expiry(entry) is False

    def test_future_expires_not_expired(self) -> None:
        """Entry with expires in the future is not expired."""
        future = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
        entry = self._make_entry(future)
        assert _drift._check_exclusion_expiry(entry) is False

    def test_past_expires_is_expired(self) -> None:
        """Entry with expires in the past is expired."""
        past = "2020-01-01"
        entry = self._make_entry(past)
        assert _drift._check_exclusion_expiry(entry) is True

    def test_today_expires_is_expired(self) -> None:
        """Entry with expires = today is expired (expires < today is strict less-than)."""
        # today < today is False, so today is NOT expired.
        today = datetime.date.today().isoformat()
        entry = self._make_entry(today)
        # today is NOT < today, so not expired
        assert _drift._check_exclusion_expiry(entry) is False

    def test_yesterday_expires_is_expired(self) -> None:
        """Entry with expires = yesterday is expired."""
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        entry = self._make_entry(yesterday)
        assert _drift._check_exclusion_expiry(entry) is True

    def test_unparseable_expires_not_expired(self) -> None:
        """An unparseable expires value is treated as not-expired (defensive)."""
        entry = self._make_entry("not-a-date")
        assert _drift._check_exclusion_expiry(entry) is False


# ---------------------------------------------------------------------------
# Exclusion filtering in run()
# ---------------------------------------------------------------------------


class TestExclusionFiltering:
    """Verify that load_exclusions is applied correctly inside run()."""

    def _run_with_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        declared_landmarks: tuple[str, ...],
        live_high: set[str],
        live_low: set[str],
        exclusions: dict[str, _drift.ExclusionEntry],
        gate: _drift.GateKind = "none",
        added_baseline: list[str] | None = None,
    ) -> _drift.DriftReport:
        """Wire synthetic producer + fake discover + fake exclusions, run()."""
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=declared_landmarks,
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda engine: (live_high, live_low))
        monkeypatch.setattr(_drift, "load_exclusions", lambda engine, version: exclusions)
        # Inject baseline into current.yaml if requested.
        if added_baseline is not None:
            _inject_baseline(tmp_path, "transformers", added_baseline)
        return _drift.run(engine="transformers", producer="invariants", gate=gate)

    def test_valid_exclusion_removes_from_landmarks_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A valid exclusion entry removes the symbol from landmarks_added."""
        path = "json.JSONEncoder"
        fp = _fp(path)
        exclusion = _drift.ExclusionEntry(
            fingerprint=fp,
            qualname=path,
            reason="not a validator",
            added="2026-05-14",
        )
        report = self._run_with_overrides(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError", path},
            live_low=set(),
            exclusions={fp: exclusion},
        )
        assert path not in report.landmarks_added
        assert path in report.exclusions_applied

    def test_excluded_symbol_not_in_landmarks_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The excluded symbol must not appear in landmarks_added at all."""
        path = "json.JSONEncoder"
        fp = _fp(path)
        exclusion = _drift.ExclusionEntry(
            fingerprint=fp, qualname=path, reason="ok", added="2026-05-14"
        )
        report = self._run_with_overrides(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={path},
            live_low=set(),
            exclusions={fp: exclusion},
        )
        assert report.landmarks_added == []
        assert path in report.exclusions_applied

    def test_exclusion_by_fingerprint_not_qualname(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exclusion lookup is fingerprint-based. A wrong fingerprint doesn't suppress."""
        path = "json.JSONEncoder"
        wrong_fp = _fp("json.SomethingElse")
        exclusion = _drift.ExclusionEntry(
            fingerprint=wrong_fp, qualname="json.SomethingElse", reason="ok", added="2026-05-14"
        )
        report = self._run_with_overrides(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={path},
            live_low=set(),
            exclusions={wrong_fp: exclusion},
        )
        # path is NOT suppressed because its fingerprint doesn't match.
        assert path in report.landmarks_added
        assert report.exclusions_applied == []

    def test_expired_exclusion_forces_fail_verdict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An expired exclusion entry forces verdict=fail regardless of gate."""
        path = "json.JSONEncoder"
        fp = _fp(path)
        exclusion = _drift.ExclusionEntry(
            fingerprint=fp,
            qualname=path,
            reason="was ok",
            added="2026-01-01",
            expires="2020-01-01",  # expired
        )
        report = self._run_with_overrides(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError"},
            live_low=set(),
            exclusions={fp: exclusion},
            gate="none",
        )
        assert report.verdict == "fail"
        assert path in report.expired_exclusions

    def test_expired_exclusion_path_stays_in_landmarks_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An expired exclusion does NOT suppress the path from landmarks_added."""
        path = "json.JSONEncoder"
        fp = _fp(path)
        exclusion = _drift.ExclusionEntry(
            fingerprint=fp,
            qualname=path,
            reason="was ok",
            added="2026-01-01",
            expires="2020-01-01",
        )
        report = self._run_with_overrides(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={path},
            live_low=set(),
            exclusions={fp: exclusion},
        )
        # expired exclusion -> path stays in landmarks_added, appears in expired_exclusions
        assert path in report.landmarks_added
        assert path in report.expired_exclusions

    def test_no_exclusions_behaviour_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When no exclusions exist, landmarks_added is unaffected (Phase B compat)."""
        path = "json.JSONEncoder"
        report = self._run_with_overrides(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={path},
            live_low=set(),
            exclusions={},
        )
        assert path in report.landmarks_added
        assert report.exclusions_applied == []
        assert report.expired_exclusions == []
        # Phase B: verdict still pass with gate=none
        assert report.verdict == "pass"


# ---------------------------------------------------------------------------
# Delta gating logic
# ---------------------------------------------------------------------------


def _inject_baseline(tmp_path: Path, engine: str, baseline: list[str]) -> None:
    """Append an ``added_baseline`` line to the fake current.yaml's last_probe block."""
    current_yaml_path = tmp_path / engine / "current.yaml"
    text = current_yaml_path.read_text()
    # Append added_baseline at the end of the last_probe: block.
    # Find the last_probe: block and append the field.
    lines = text.splitlines()
    out_lines = []
    in_block = False
    inserted = False
    for line in lines:
        if line.startswith("last_probe:"):
            in_block = True
        elif in_block and not inserted:
            stripped = line.strip()
            # Insert before the end of block (first non-last_probe key or EOF)
            if stripped and not line.startswith("  "):
                # We've left the block; insert just before this line.
                rendered = yaml.dump(baseline, default_flow_style=True).strip()
                out_lines.append(f"  added_baseline: {rendered}")
                in_block = False
                inserted = True
        out_lines.append(line)
    if in_block and not inserted:
        # Block goes to EOF; append at end.
        rendered = yaml.dump(baseline, default_flow_style=True).strip()
        out_lines.append(f"  added_baseline: {rendered}")
    current_yaml_path.write_text("\n".join(out_lines) + "\n")


class TestDeltaGating:
    """Verify delta gating: fail only when new gaps appear since last green main."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        declared_landmarks: tuple[str, ...],
        live_high: set[str],
        live_low: set[str],
        gate: _drift.GateKind,
        baseline: list[str],
    ) -> _drift.DriftReport:
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=declared_landmarks,
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda engine: (live_high, live_low))
        monkeypatch.setattr(_drift, "load_exclusions", lambda engine, version: {})
        _inject_baseline(tmp_path, "transformers", baseline)
        return _drift.run(engine="transformers", producer="invariants", gate=gate)

    def test_gate_none_never_fails_on_additions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=none: verdict always pass on additions (Phase B compat)."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={"json.JSONEncoder", "json.JSONDecoder"},
            live_low=set(),
            gate="none",
            baseline=[],
        )
        assert report.verdict == "pass"
        assert report.landmarks_added_delta == []

    def test_gate_delta_no_new_gaps_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=delta: verdict pass when all gaps are pre-existing (in baseline)."""
        pre_existing = "json.JSONEncoder"
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={pre_existing},
            live_low=set(),
            gate="delta",
            baseline=[pre_existing],
        )
        assert report.verdict == "pass"
        assert report.landmarks_added_delta == []

    def test_gate_delta_new_gap_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=delta: verdict fail when a new high-confidence gap appears."""
        pre_existing = "json.JSONEncoder"
        new_gap = "json.JSONDecoder"
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={pre_existing, new_gap},
            live_low=set(),
            gate="delta",
            baseline=[pre_existing],
        )
        assert report.verdict == "fail"
        assert new_gap in report.landmarks_added_delta
        assert pre_existing not in report.landmarks_added_delta

    def test_gate_delta_empty_baseline_all_gaps_new(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=delta with empty baseline: all current gaps are "new"."""
        gap = "json.JSONEncoder"
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={gap},
            live_low=set(),
            gate="delta",
            baseline=[],
        )
        assert report.verdict == "fail"
        assert gap in report.landmarks_added_delta

    def test_gate_delta_no_additions_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=delta: no additions -> verdict pass."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError"},
            live_low=set(),
            gate="delta",
            baseline=[],
        )
        assert report.verdict == "pass"
        assert report.landmarks_added_delta == []

    def test_gate_absolute_any_gap_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=absolute: any high-confidence gap -> fail, regardless of baseline."""
        gap = "json.JSONEncoder"
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={gap},
            live_low=set(),
            gate="absolute",
            baseline=[gap],  # even if in baseline, absolute still fails
        )
        assert report.verdict == "fail"

    def test_gate_absolute_no_gaps_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=absolute with no gaps -> pass."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError"},
            live_low=set(),
            gate="absolute",
            baseline=[],
        )
        assert report.verdict == "pass"

    def test_delta_field_only_populated_with_delta_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """landmarks_added_delta is only populated when gate=delta."""
        gap = "json.JSONEncoder"
        report_none = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={gap},
            live_low=set(),
            gate="none",
            baseline=[],
        )
        assert report_none.landmarks_added_delta == [], (
            "gate=none must not populate landmarks_added_delta"
        )

    def test_gate_field_preserved_in_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The gate mode is reflected in DriftReport.gate."""
        for gate_mode in ("none", "delta", "absolute"):
            _redirect_compat_dir(monkeypatch, tmp_path, "transformers")
            _install_synthetic_producer(
                monkeypatch,
                engine="transformers",
                producer="invariants",
                landmarks=("json.JSONDecodeError",),
            )
            monkeypatch.setattr(_drift, "discover_live_landmarks", lambda engine: (set(), set()))
            monkeypatch.setattr(_drift, "load_exclusions", lambda engine, version: {})
            report = _drift.run(
                engine="transformers",
                producer="invariants",
                gate=gate_mode,  # type: ignore[arg-type]
            )
            assert report.gate == gate_mode

    def test_missing_declared_landmark_fails_regardless_of_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Phase A probe-fail (missing declared landmark) overrides delta gate."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.NonExistentXYZ",),
            live_high=set(),
            live_low=set(),
            gate="none",
            baseline=[],
        )
        assert report.verdict == "fail"
        assert "json.NonExistentXYZ" in report.landmarks_missing


# ---------------------------------------------------------------------------
# _render_drift_comment
# ---------------------------------------------------------------------------


class TestRenderDriftComment:
    """Verify the markdown comment body shape."""

    def _base_report(self, **kwargs: Any) -> _drift.DriftReport:
        defaults: dict[str, Any] = {
            "engine": "vllm",
            "producer": "invariants",
            "direction": "stable",
            "schema_version": 1,
            "current_version": "0.7.3",
            "verdict": "pass",
            "fingerprint": "abc123",
            "fingerprint_drift": [],
            "landmarks_missing": [],
            "landmarks_added": [],
            "landmarks_added_low_confidence": [],
            "version_inside_envelope": True,
            "exclusions_applied": [],
            "expired_exclusions": [],
            "landmarks_added_delta": [],
            "gate": "none",
        }
        defaults.update(kwargs)
        return _drift.DriftReport(**defaults)

    def test_comment_contains_marker(self) -> None:
        """Comment body contains the stable HTML marker."""
        report = self._base_report()
        body = _drift._render_drift_comment(report)
        assert "<!-- llem-drift-coverage -->" in body

    def test_comment_contains_engine_and_version(self) -> None:
        """Comment body mentions engine name and safe version."""
        report = self._base_report(engine="vllm", current_version="0.7.3")
        body = _drift._render_drift_comment(report)
        assert "vllm" in body
        assert "v0_7_3" in body

    def test_comment_contains_producer(self) -> None:
        """Comment body mentions the producer kind."""
        report = self._base_report(producer="invariants")
        body = _drift._render_drift_comment(report)
        assert "invariants" in body

    def test_new_gaps_in_details_block(self) -> None:
        """New gaps (in landmarks_added_delta) appear inside a <details> block."""
        new_gap = "json.JSONEncoder"
        report = self._base_report(
            landmarks_added=[new_gap],
            landmarks_added_delta=[new_gap],
            gate="delta",
        )
        body = _drift._render_drift_comment(report)
        assert "<details>" in body
        assert new_gap in body

    def test_preexisting_gaps_in_details_block(self) -> None:
        """Pre-existing gaps (in landmarks_added but not delta) appear in their own block."""
        preexisting = "json.JSONDecoder"
        report = self._base_report(
            landmarks_added=[preexisting],
            landmarks_added_delta=[],
            gate="delta",
        )
        body = _drift._render_drift_comment(report)
        assert "Pre-existing" in body
        assert preexisting in body

    def test_exclusions_applied_in_details_block(self) -> None:
        """Applied exclusions appear in a dedicated <details> block."""
        excluded = "json.JSONDecoder"
        report = self._base_report(
            exclusions_applied=[excluded],
        )
        body = _drift._render_drift_comment(report)
        assert "Exclusions applied" in body
        assert excluded in body

    def test_expired_exclusions_in_warning(self) -> None:
        """Expired exclusions are surfaced prominently."""
        report = self._base_report(
            expired_exclusions=["some.Expired.qualname"],
        )
        body = _drift._render_drift_comment(report)
        assert "Expired exclusions" in body

    def test_empty_report_no_details_blocks(self) -> None:
        """A stable report with no gaps has no <details> blocks."""
        report = self._base_report()
        body = _drift._render_drift_comment(report)
        assert "<details>" not in body

    def test_delta_summary_line_when_gate_delta(self) -> None:
        """gate=delta produces a delta-style summary line."""
        report = self._base_report(
            landmarks_added=["json.JSONEncoder"],
            landmarks_added_delta=["json.JSONEncoder"],
            gate="delta",
        )
        body = _drift._render_drift_comment(report)
        assert "Delta" in body or "delta" in body or "added gaps since main" in body


# ---------------------------------------------------------------------------
# update_last_probe: added_baseline written on pass
# ---------------------------------------------------------------------------


class TestUpdateLastProbeBaseline:
    """Verify update_last_probe writes added_baseline when DriftReport carries landmarks_added."""

    def _minimal_report(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "verdict": "pass",
            "version_inside_envelope": True,
            "fingerprint": "aabbcc",
            "fingerprint_drift": [],
            "current_version": "4.57.3",
            "library_version": "4.57.3",
            "landmarks_added": [],
            "landmarks_added_low_confidence": [],
            "landmarks_missing": [],
        }
        base.update(overrides)
        return base

    def _make_current_yaml(self, tmp_path: Path) -> Path:
        # Synthetic content; not copied from the real file to avoid cross-test
        # contamination if another test mutates engine_versions/transformers/current.yaml.
        current_path = tmp_path / "current.yaml"
        current_path.write_text(
            "schema_version: 1\n"
            "engine: transformers\n"
            "library:\n"
            "  pep503_name: transformers\n"
            '  current_version: "4.57.3"\n'
            "  current_commit_sha: null\n"
            "image:\n"
            '  base_image_ref: "llenergymeasure:transformers-4.57.3"\n'
            "  llem_image_ref: null\n"
            "miner_pins:\n"
            '  static: ">=4.57,<4.58"\n'
            '  dynamic: ">=4.57,<4.58"\n'
            '  discovery: ">=4.57,<4.58"\n'
            "artefact_paths:\n"
            "  engine_invariants: src/llenergymeasure/engines/transformers/invariants.proposed.yaml\n"
            "  validated_invariants: src/llenergymeasure/engines/transformers/invariants.validated.yaml\n"
            "  discovered_schemas: src/llenergymeasure/engines/transformers/schema.discovered.json\n"
            "last_probe:\n"
            "  verdict: pass\n"
            "  version_inside_envelope: true\n"
            '  fingerprint: "synthetic"\n'
            "  fingerprint_drift: []\n"
        )
        return current_path

    def test_baseline_written_on_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When verdict=pass and landmarks_added is non-empty, added_baseline is written."""

        current_yaml = self._make_current_yaml(tmp_path)
        monkeypatch.setattr("scripts.update_last_probe.current_path", lambda engine: current_yaml)
        report = self._minimal_report(
            landmarks_added=["transformers.GenerationConfig.some_new_field"]
        )
        rc = update_last_probe.update(engine="transformers", report=report)
        assert rc == 0

        text = current_yaml.read_text()
        assert "added_baseline" in text

    def test_baseline_not_written_on_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When verdict=fail, added_baseline is NOT written."""
        current_yaml = self._make_current_yaml(tmp_path)
        monkeypatch.setattr("scripts.update_last_probe.current_path", lambda engine: current_yaml)
        report = self._minimal_report(
            verdict="fail",
            landmarks_added=["transformers.GenerationConfig.some_field"],
        )
        rc = update_last_probe.update(engine="transformers", report=report)
        assert rc == 0

        text = current_yaml.read_text()
        assert "added_baseline" not in text

    def test_baseline_empty_list_when_no_gaps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When verdict=pass and landmarks_added=[], baseline is written as empty list."""
        current_yaml = self._make_current_yaml(tmp_path)
        monkeypatch.setattr("scripts.update_last_probe.current_path", lambda engine: current_yaml)
        report = self._minimal_report(landmarks_added=[])
        rc = update_last_probe.update(engine="transformers", report=report)
        assert rc == 0

        text = current_yaml.read_text()
        assert "added_baseline" in text

    def test_baseline_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Running update twice with the same report produces identical output."""
        current_yaml = self._make_current_yaml(tmp_path)
        monkeypatch.setattr("scripts.update_last_probe.current_path", lambda engine: current_yaml)
        report = self._minimal_report(landmarks_added=["transformers.GenerationConfig.field_x"])
        update_last_probe.update(engine="transformers", report=report)
        text_after_first = current_yaml.read_text()

        update_last_probe.update(engine="transformers", report=report)
        text_after_second = current_yaml.read_text()

        assert text_after_first == text_after_second

    def test_old_report_without_landmarks_added_skips_baseline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legacy ProbeReport without landmarks_added does not cause KeyError."""
        current_yaml = self._make_current_yaml(tmp_path)
        monkeypatch.setattr("scripts.update_last_probe.current_path", lambda engine: current_yaml)
        # Old-style report without landmarks_added
        old_report: dict[str, Any] = {
            "verdict": "pass",
            "version_inside_envelope": True,
            "fingerprint": "aabbcc",
            "fingerprint_drift": [],
            "library_version": "4.57.3",
        }
        rc = update_last_probe.update(engine="transformers", report=old_report)
        assert rc == 0
        text = current_yaml.read_text()
        assert "added_baseline" not in text


# ---------------------------------------------------------------------------
# CLI integration: --gate flag wires through to run()
# ---------------------------------------------------------------------------


class TestCliGateFlag:
    """Verify the --gate CLI flag affects exit code and report."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, live_high: set[str]) -> None:
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=(),
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda e: (live_high, set()))
        monkeypatch.setattr(_drift, "load_exclusions", lambda e, v: {})

    def test_gate_none_exit_0_even_with_gaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--gate none: exit 0 even when landmarks_added is non-empty."""
        self._setup(monkeypatch, tmp_path, {"json.JSONEncoder"})
        rc = _drift.main(
            [
                "--engine",
                "transformers",
                "--producer",
                "invariants",
                "--gate",
                "none",
                "--no-write-cache",
            ]
        )
        assert rc == 0

    def test_gate_delta_exit_1_when_new_gap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--gate delta: exit 1 when new high-confidence gap (baseline present, empty)."""
        self._setup(monkeypatch, tmp_path, {"json.JSONEncoder"})
        # Explicit empty baseline (signals "no gaps as of last green main";
        # any new gap is therefore drift). Absent baseline = warn-only per
        # the first-run-on-fresh-engine semantics.
        _inject_baseline(tmp_path, "transformers", [])
        rc = _drift.main(
            [
                "--engine",
                "transformers",
                "--producer",
                "invariants",
                "--gate",
                "delta",
                "--no-write-cache",
            ]
        )
        assert rc == 1

    def test_gate_delta_exit_0_when_no_baseline_warn_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--gate delta with no baseline written: warn-only (exit 0)."""
        self._setup(monkeypatch, tmp_path, {"json.JSONEncoder"})
        # Real current.yaml ships with `added_baseline: []` once a green main
        # has run; strip it here to simulate the first-run-on-fresh-engine
        # state where no baseline has ever been written.
        fake_current = tmp_path / "transformers" / "current.yaml"
        fake_current.write_text(
            "\n".join(
                line
                for line in fake_current.read_text().splitlines()
                if not line.strip().startswith("added_baseline")
            )
            + "\n"
        )
        rc = _drift.main(
            [
                "--engine",
                "transformers",
                "--producer",
                "invariants",
                "--gate",
                "delta",
                "--no-write-cache",
            ]
        )
        assert rc == 0

    def test_gate_delta_exit_0_when_no_new_gap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--gate delta: exit 0 when gap is in baseline (pre-existing)."""
        preexisting = "json.JSONEncoder"
        _redirect_compat_dir(monkeypatch, tmp_path)
        _inject_baseline(tmp_path, "transformers", [preexisting])
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=(),
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda e: ({preexisting}, set()))
        monkeypatch.setattr(_drift, "load_exclusions", lambda e, v: {})
        rc = _drift.main(
            [
                "--engine",
                "transformers",
                "--producer",
                "invariants",
                "--gate",
                "delta",
                "--no-write-cache",
            ]
        )
        assert rc == 0

    def test_gate_absolute_exit_1_with_any_gap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--gate absolute: exit 1 when any gap present."""
        self._setup(monkeypatch, tmp_path, {"json.JSONEncoder"})
        rc = _drift.main(
            [
                "--engine",
                "transformers",
                "--producer",
                "invariants",
                "--gate",
                "absolute",
                "--no-write-cache",
            ]
        )
        assert rc == 1

    def test_gate_none_exit_0_with_probe_fail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """gate=none: probe-fail (missing landmark) still exits 0 (verdict in JSON)."""
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=("json.NonExistentXYZ",),
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda e: (set(), set()))
        monkeypatch.setattr(_drift, "load_exclusions", lambda e, v: {})
        rc = _drift.main(
            [
                "--engine",
                "transformers",
                "--producer",
                "invariants",
                "--gate",
                "none",
                "--no-write-cache",
            ]
        )
        # Exit 0 even on probe-fail when gate=none (verdict carried in JSON).
        assert rc == 0
