"""Tests for :mod:`scripts.widen_miner_pin` — the SSOT pin-widener helper.

The helper backs the ``/approve-reuse`` slash command. These tests pin
the determinism contract (idempotency, line-surgical edits, no-op when
the version is already covered) and the producer-kind translation
(``invariants`` -> ``static``, ``schemas`` -> ``discovery``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import widen_miner_pin  # noqa: E402

_BASE_SSOT = """\
# Header comment block must survive every widening.
#
# Multi-line context block.
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


def _seed_ssot(tmp_path: Path, engine: str, body: str) -> Path:
    """Write ``body`` to ``tmp_path/{engine}.yaml`` and return its path."""
    target = tmp_path / f"{engine}.yaml"
    target.write_text(body)
    return target


@pytest.fixture()
def fake_ssot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``widen_miner_pin.ssot_path`` to a temporary SSOT copy."""
    target = _seed_ssot(tmp_path, "transformers", _BASE_SSOT)

    def _fake_path(engine: str) -> Path:
        return tmp_path / f"{engine}.yaml"

    monkeypatch.setattr(widen_miner_pin, "ssot_path", _fake_path)
    return target


def test_invariants_maps_to_static_miner_pin(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``invariants`` widens the ``static`` SSOT key, leaves ``discovery`` alone."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = widen_miner_pin.widen(engine="transformers", producer="invariants")
    assert rc == 0
    text = fake_ssot.read_text()
    assert 'static: ">=4.56,<4.58"' in text
    assert 'discovery: ">=4.56,<4.57"' in text  # untouched


def test_schemas_maps_to_discovery_miner_pin(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``schemas`` widens the ``discovery`` SSOT key, leaves ``static`` alone."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = widen_miner_pin.widen(engine="transformers", producer="schemas")
    assert rc == 0
    text = fake_ssot.read_text()
    assert 'static: ">=4.56,<4.57"' in text  # untouched
    assert 'discovery: ">=4.56,<4.58"' in text


def test_version_already_inside_envelope_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bumped version inside the existing range -> no file mutation."""
    body = _BASE_SSOT.replace('static: ">=4.56,<4.57"', 'static: ">=4.56,<4.99"')
    target = _seed_ssot(tmp_path, "transformers", body)
    monkeypatch.setattr(widen_miner_pin, "ssot_path", lambda _e: target)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    before = target.read_text()
    rc = widen_miner_pin.widen(engine="transformers", producer="invariants")
    after = target.read_text()

    assert rc == 0
    assert before == after


def test_widening_is_idempotent(fake_ssot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the widener twice in a row produces zero diff after the first."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert widen_miner_pin.widen(engine="transformers", producer="invariants") == 0
    after_first = fake_ssot.read_text()
    assert widen_miner_pin.widen(engine="transformers", producer="invariants") == 0
    after_second = fake_ssot.read_text()
    assert after_first == after_second


def test_line_surgical_edit_preserves_comments_and_quoting(fake_ssot: Path) -> None:
    """Header comments, key order, and string quoting all survive the edit."""
    widen_miner_pin.widen(engine="transformers", producer="invariants")
    text = fake_ssot.read_text()
    assert text.startswith("# Header comment block must survive every widening.\n")
    assert 'current_version: "4.57.3"' in text  # quoting preserved
    # Key order in miner_pins block preserved.
    static_idx = text.index("  static:")
    dynamic_idx = text.index("  dynamic:")
    discovery_idx = text.index("  discovery:")
    assert static_idx < dynamic_idx < discovery_idx


def test_emits_github_outputs(
    fake_ssot: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``$GITHUB_OUTPUT`` is set, emits ``changed``, ``old_range``, ``new_range``."""
    output_file = tmp_path / "gh-output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    rc = widen_miner_pin.widen(engine="transformers", producer="invariants")
    assert rc == 0
    output = output_file.read_text()
    assert "changed=true" in output
    assert "old_range=>=4.56,<4.57" in output
    assert "new_range=>=4.56,<4.58" in output


def test_emits_changed_false_when_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No-op widening still emits ``changed=false`` for the workflow to read."""
    body = _BASE_SSOT.replace('static: ">=4.56,<4.57"', 'static: ">=4.56,<4.99"')
    target = _seed_ssot(tmp_path, "transformers", body)
    monkeypatch.setattr(widen_miner_pin, "ssot_path", lambda _e: target)

    output_file = tmp_path / "gh-output"
    output_file.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    widen_miner_pin.widen(engine="transformers", producer="invariants")
    output = output_file.read_text()
    assert "changed=false" in output


def test_missing_miner_pin_key_fails_loud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Missing SSOT ``miner_pins.<key>`` -> non-zero exit, no mutation."""
    body = _BASE_SSOT.replace('  static: ">=4.56,<4.57"\n', "")
    target = _seed_ssot(tmp_path, "transformers", body)
    monkeypatch.setattr(widen_miner_pin, "ssot_path", lambda _e: target)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    rc = widen_miner_pin.widen(engine="transformers", producer="invariants")
    assert rc == 2


def test_widening_below_lower_bound_refuses_to_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bumped version below the lower bound -> refuse rather than emit a wrong widening."""
    body = _BASE_SSOT.replace('current_version: "4.57.3"', 'current_version: "4.50.0"')
    target = _seed_ssot(tmp_path, "transformers", body)
    monkeypatch.setattr(widen_miner_pin, "ssot_path", lambda _e: target)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    before = target.read_text()
    rc = widen_miner_pin.widen(engine="transformers", producer="invariants")
    after = target.read_text()

    assert rc == 2
    assert before == after


def test_widen_range_string_recognised_shape() -> None:
    """``>=A,<B`` widens to ``>=A,<{next_minor}`` with lower bound preserved."""
    from packaging.version import Version

    result = widen_miner_pin._widen_range_string(">=4.56,<4.57", Version("4.57.3"))
    assert result == ">=4.56,<4.58"


def test_widen_range_string_fallback_for_unrecognised_shape() -> None:
    """Multi-clause / equality ranges -> append ``<=current_version`` ceiling."""
    from packaging.version import Version

    result = widen_miner_pin._widen_range_string(">=4.56,<4.57,!=4.56.5", Version("4.57.3"))
    assert "<=4.57.3" in result
    # Original constraints all preserved.
    for clause in (">=4.56", "<4.57", "!=4.56.5"):
        assert clause in result
