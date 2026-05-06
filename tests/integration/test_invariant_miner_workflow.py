"""Smoke test for the invariant-miner (validation gate) workflow glue.

Runs the validate + diff + fixpoint pipeline end-to-end on a fixture corpus.
Does NOT spin up Docker or require transformers — the validation step is driven
through a patched synthetic runner. This test is the "trust the glue" check
that complements the unit tests for each individual script.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import _invariant_validation_common, validate_invariants  # noqa: E402
from scripts import diff_engine_invariants as diff_invariants  # noqa: E402
from scripts._invariant_validation_common import run_case  # noqa: E402
from scripts.engine_miners._fixpoint_test import fixpoint_test_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic corpus + runner
# ---------------------------------------------------------------------------


_FIXTURE_CORPUS = """
schema_version: 1.0.0
engine: transformers
engine_version: test-1.0.0
miner_pinned_range: test-range
mined_at: 2026-04-23T00:00:00Z
invariants:
  - id: synthetic_error
    engine: transformers
    library: transformers
    invariant_under_test: Synthetic raises on bad input
    severity: error
    native_type: fixture.raises
    miner_source: {path: fixture.py, method: __init__, line_at_scan: 0}
    match:
      engine: transformers
      fields:
        x: {present: true}
    kwargs_positive: {x: 1}
    kwargs_negative: {x: 0}
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: 'x must not be {x}'
    references: []
    added_by: manual_seed
    added_at: '2026-04-23'
  - id: synthetic_dormant
    engine: transformers
    library: transformers
    invariant_under_test: Synthetic silent strip
    severity: dormant
    native_type: fixture.normalises
    miner_source: {path: fixture.py, method: __init__, line_at_scan: 0}
    match:
      engine: transformers
      fields:
        do_sample: false
        temperature: {present: true, not_equal: 1.0}
    kwargs_positive: {do_sample: false, temperature: 0.9}
    kwargs_negative: {do_sample: true, temperature: 0.9}
    expected_outcome:
      outcome: dormant_silent
      emission_channel: none
      normalised_fields: ['temperature']
    message_template: 'temperature stripped when do_sample=False'
    references: []
    added_by: manual_seed
    added_at: '2026-04-23'
"""


class _Normaliser:
    def __init__(self, **kwargs: Any) -> None:
        self.do_sample = kwargs.get("do_sample", True)
        self.temperature = kwargs.get("temperature", 1.0)
        if not self.do_sample and self.temperature != 1.0:
            self.temperature = 1.0


def _synthetic_runner(
    native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
) -> _invariant_validation_common.CaptureBuffers:
    if native_type == "fixture.raises":
        if kwargs.get("x", 0) > 0:

            def _boom() -> None:
                raise ValueError(f"x must not be {kwargs['x']}")

            return run_case(_boom)
        return run_case(lambda: type("Empty", (), {})())
    if native_type == "fixture.normalises":
        return run_case(lambda: _Normaliser(**kwargs))
    raise AssertionError(f"unexpected native_type {native_type}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "transformers.yaml"
    path.write_text(_FIXTURE_CORPUS)
    return path


@pytest.fixture
def patched_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(validate_invariants._ENGINE_RUNNERS, "transformers", _synthetic_runner)
    monkeypatch.setattr(validate_invariants, "_resolve_engine_version", lambda _e: "test-1.0.0")
    return _synthetic_runner


# ---------------------------------------------------------------------------
# Glue tests
# ---------------------------------------------------------------------------


class TestWorkflowGlue:
    def test_validate_writes_envelope_and_no_divergence(
        self, fixture_corpus_path: Path, tmp_path: Path, patched_runner: Any
    ) -> None:
        out = tmp_path / "transformers.validated.yaml"
        _envelope, divergences = validate_invariants.validate_engine(
            engine="transformers",
            corpus_path=fixture_corpus_path,
            out_path=out,
        )
        assert out.exists()
        written = yaml.safe_load(out.read_text())
        assert written["schema_version"] == "1.0.0"
        assert len(written["cases"]) == 2
        assert divergences == []

    def test_diff_classifier_marks_added_rule_safe(
        self, fixture_corpus_path: Path, tmp_path: Path, patched_runner: Any
    ) -> None:
        out1 = tmp_path / "first.validated.yaml"
        envelope1, _ = validate_invariants.validate_engine(
            engine="transformers",
            corpus_path=fixture_corpus_path,
            out_path=out1,
        )

        # Simulate a new invariant being added to the envelope.
        envelope2 = dict(envelope1)
        envelope2["cases"] = [
            *envelope1["cases"],
            {
                "id": "new_rule",
                "outcome": "warn",
                "emission_channel": "warnings_warn",
                "observed_messages": [],
                "observed_silent_normalisations": {},
                "positive_confirmed": True,
                "negative_confirmed": True,
                "duration_ms": 1,
            },
        ]

        result = diff_invariants.diff_invariants(envelope1, envelope2)
        assert not result.is_breaking
        assert any(c.kind == "added_invariant" for c in result.safe)

    def test_fixpoint_test_runs_on_fixture_corpus(self, fixture_corpus_path: Path) -> None:
        corpus = yaml.safe_load(fixture_corpus_path.read_text())
        # Should not raise.
        fixpoint_test_corpus(corpus)

    def test_markdown_comment_body_well_formed(
        self, fixture_corpus_path: Path, tmp_path: Path, patched_runner: Any
    ) -> None:
        out = tmp_path / "validation.validated.yaml"
        envelope, _ = validate_invariants.validate_engine(
            engine="transformers",
            corpus_path=fixture_corpus_path,
            out_path=out,
        )
        # Diff against empty baseline to get a full "added_rule" summary.
        empty_envelope = dict(envelope)
        empty_envelope["cases"] = []
        md_out = tmp_path / "comment.md"
        result = diff_invariants.diff_invariants(empty_envelope, envelope)
        md = diff_invariants.render_markdown(result, title="Smoke")
        md_out.write_text(md)
        assert md_out.exists()
        assert "## Smoke" in md
        assert "added_invariant" in md
