"""Tests for :mod:`scripts.update_last_probe` — the SSOT last_probe writer.

The helper is invoked from the engine-coupling probe-writeback workflow:
it reads a ``ProbeReport`` JSON on stdin and rewrites the four mutable
fields under ``last_probe:`` in ``engine_versions/{engine}.yaml``. These
tests pin the determinism contract (idempotent re-runs, byte-identical
file when nothing changed, line-surgical edit preserving comments and
quoting) and the GitHub Actions output surface.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import update_last_probe  # noqa: E402

_SEED_SSOT = """\
# Per-engine version-bundle SSOT for transformers.
#
# Header comment block must survive every probe-writeback.
schema_version: 1
engine: transformers
library:
  pep503_name: transformers
  current_version: "4.57.3"
  current_commit_sha: null
image:
  base_image_ref: "llenergymeasure:transformers-4.57.3"
  llem_image_ref: null
miner_pins:
  static: ">=4.56,<4.57"
  dynamic: ">=4.56,<4.57"
  discovery: ">=4.56,<4.57"
artefact_paths:
  engine_invariants: src/llenergymeasure/engines/transformers/invariants.proposed.yaml
  validated_invariants: src/llenergymeasure/engines/transformers/invariants.validated.yaml
  discovered_schemas: src/llenergymeasure/engines/transformers/schema.discovered.json
last_probe:
  verdict: unrun
  version_inside_envelope: null
  fingerprint: null
  fingerprint_drift: []
"""


def _seed_ssot(tmp_path: Path, engine: str, body: str = _SEED_SSOT) -> Path:
    """Write ``body`` to ``tmp_path/{engine}.yaml`` and return its path."""
    target = tmp_path / f"{engine}.yaml"
    target.write_text(body)
    return target


@pytest.fixture()
def fake_ssot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``update_last_probe.ssot_path`` to a temporary SSOT copy."""
    target = _seed_ssot(tmp_path, "transformers")

    def _fake_path(engine: str) -> Path:
        return tmp_path / f"{engine}.yaml"

    monkeypatch.setattr(update_last_probe, "ssot_path", _fake_path)
    return target


def _pass_report(
    *,
    fingerprint: str = "deadbeefcafebabe1234567890abcdef",
    drift: list[str] | None = None,
    inside_envelope: bool = True,
    verdict: str = "pass",
) -> dict[str, object]:
    """Build a minimal ProbeReport-shaped dict for stdin / direct calls."""
    return {
        "engine": "transformers",
        "producer": "invariants",
        "verdict": verdict,
        "library_version": "4.57.3",
        "version_inside_envelope": inside_envelope,
        "fingerprint": fingerprint,
        "fingerprint_drift": drift or [],
        "landmarks_missing": [],
    }


# ---------------------------------------------------------------------------
# Mutation behaviour
# ---------------------------------------------------------------------------


def test_updates_last_probe_block_from_pass_report(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh ``unrun`` SSOT + ``pass`` report -> all four fields rewritten."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = update_last_probe.update(engine="transformers", report=_pass_report())
    assert rc == 0
    text = fake_ssot.read_text()
    assert "  verdict: pass\n" in text
    assert "  version_inside_envelope: true\n" in text
    assert '  fingerprint: "deadbeefcafebabe1234567890abcdef"\n' in text
    assert "  fingerprint_drift: []\n" in text


def test_updates_last_probe_block_from_fail_report(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fail`` verdict + drift list -> rendered as bare word + flow-style list."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    report = _pass_report(
        verdict="fail",
        drift=["json.JSONDecodeError", "json.JSONEncoder"],
        inside_envelope=False,
    )
    rc = update_last_probe.update(engine="transformers", report=report)
    assert rc == 0
    text = fake_ssot.read_text()
    assert "  verdict: fail\n" in text
    assert "  version_inside_envelope: false\n" in text
    assert '  fingerprint_drift: ["json.JSONDecodeError", "json.JSONEncoder"]\n' in text


# ---------------------------------------------------------------------------
# Determinism contract
# ---------------------------------------------------------------------------


def test_rerun_with_same_input_is_byte_identical_noop(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-running with the same ProbeReport produces zero file mutation."""
    output_file = tmp_path / "gh-output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    report = _pass_report()
    assert update_last_probe.update(engine="transformers", report=report) == 0
    after_first = fake_ssot.read_text()

    # Truncate GITHUB_OUTPUT between runs so the second run's flag is unambiguous.
    output_file.write_text("")
    assert update_last_probe.update(engine="transformers", report=report) == 0
    after_second = fake_ssot.read_text()

    assert after_first == after_second
    second_output = output_file.read_text()
    assert "changed=false" in second_output
    assert "verdict=pass" in second_output


def test_first_run_emits_changed_true(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When mutating from ``unrun`` seed, ``changed=true`` flows to outputs."""
    output_file = tmp_path / "gh-output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    rc = update_last_probe.update(engine="transformers", report=_pass_report())
    assert rc == 0
    output = output_file.read_text()
    assert "changed=true" in output
    assert "verdict=pass" in output


def test_line_surgical_edit_preserves_comments_and_unrelated_lines(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Header comments, key order, and unrelated lines all survive the edit."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    update_last_probe.update(engine="transformers", report=_pass_report())
    text = fake_ssot.read_text()
    # Header comments preserved.
    assert text.startswith("# Per-engine version-bundle SSOT for transformers.\n")
    assert "# Header comment block must survive every probe-writeback." in text
    # Unrelated quoted scalar untouched.
    assert 'current_version: "4.57.3"' in text
    # Key order in last_probe block preserved.
    verdict_idx = text.index("  verdict:")
    envelope_idx = text.index("  version_inside_envelope:")
    fingerprint_idx = text.index("  fingerprint:")
    drift_idx = text.index("  fingerprint_drift:")
    assert verdict_idx < envelope_idx < fingerprint_idx < drift_idx
    # miner_pins block untouched.
    assert 'static: ">=4.56,<4.57"' in text
    assert 'dynamic: ">=4.56,<4.57"' in text
    assert 'discovery: ">=4.56,<4.57"' in text


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_ssot_returns_infra_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No SSOT for the engine -> exit 2, stderr error envelope."""
    monkeypatch.setattr(update_last_probe, "ssot_path", lambda name: tmp_path / f"{name}.yaml")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = update_last_probe.update(engine="transformers", report=_pass_report())
    assert rc == 2


def test_report_missing_required_field_returns_infra_error(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ProbeReport missing one of the mutable fields -> exit 2."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    bad = _pass_report()
    del bad["fingerprint"]
    rc = update_last_probe.update(engine="transformers", report=bad)
    assert rc == 2


def test_ssot_without_last_probe_block_returns_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SSOT missing ``last_probe:`` block entirely -> exit 2."""
    body = "schema_version: 1\nengine: transformers\n"
    target = _seed_ssot(tmp_path, "transformers", body)
    monkeypatch.setattr(update_last_probe, "ssot_path", lambda _e: target)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = update_last_probe.update(engine="transformers", report=_pass_report())
    assert rc == 2


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_reads_json_from_stdin(fake_ssot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main(['--engine', 'transformers'])`` consumes stdin JSON and writes."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_pass_report())))
    rc = update_last_probe.main(["--engine", "transformers"])
    assert rc == 0
    assert "  verdict: pass\n" in fake_ssot.read_text()


def test_main_empty_stdin_returns_infra_error(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No JSON on stdin -> exit 2 with stderr error envelope."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr("sys.stdin", StringIO(""))
    rc = update_last_probe.main(["--engine", "transformers"])
    assert rc == 2


def test_main_invalid_json_stdin_returns_infra_error(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed JSON on stdin -> exit 2."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr("sys.stdin", StringIO("not-json{["))
    rc = update_last_probe.main(["--engine", "transformers"])
    assert rc == 2
