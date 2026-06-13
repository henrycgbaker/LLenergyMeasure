"""Unit tests for the in-container diagnose gate driver (CR2).

The diagnose gate (``scripts/diagnose_gate_in_container.py``) must confirm on
exactly the SAME bar as the two production gate paths in ``validate_rules.py``:
``compute_gate_soundness_divergences`` must come back empty AND
``warm_up_engine_observation`` must run once before the loop. These tests stub
the construct+observe captures and exercise the real production soundness check
to prove a proposal whose positive raises for the wrong reason (a type-coercion
artefact) is downgraded from confirmed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import diagnose_gate_in_container as gate  # noqa: E402
from scripts._rules_validation_common import CaptureBuffers, CaseResult, ErrorDetail  # noqa: E402


def _capture(
    *,
    exception_type: str | None = None,
    exception_message: str | None = None,
    error_details: tuple[ErrorDetail, ...] = (),
    observed_state: dict[str, Any] | None = None,
) -> CaptureBuffers:
    return CaptureBuffers(
        exception_type=exception_type,
        exception_message=exception_message,
        warnings_captured=(),
        logger_messages=(),
        observed_state=observed_state,
        duration_ms=0,
        error_details=error_details,
    )


def _stub_captures(
    monkeypatch: pytest.MonkeyPatch, *, case: CaseResult, pos: CaptureBuffers, neg: CaptureBuffers
) -> None:
    def _stub(engine: str, inv: dict[str, Any]):
        return case, pos, neg

    monkeypatch.setattr(gate.V, "_validate_invariant_with_captures", _stub)


def test_coercion_artefact_positive_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive that raises a parsing/coercion error (not the claimed value
    rule) is rejected by the production soundness check, so the diagnose gate
    downgrades confirmed -> not_confirmed even though pos/neg confirmed."""
    # case reports pos+neg confirmed (the weaker bar the gate used to trust).
    case = CaseResult(
        id="transformers_max_new_tokens_positive",
        outcome="error",
        emission_channel="none",
        positive_confirmed=True,
        negative_confirmed=True,
    )
    # ...but the positive raised an int_parsing coercion artefact on the claimed
    # field, which compute_gate_soundness_divergences rejects.
    pos = _capture(
        exception_type="ValidationError",
        exception_message="Input should be a valid integer",
        error_details=(ErrorDetail(loc=("max_new_tokens",), error_type="int_parsing"),),
    )
    neg = _capture()
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    proposal = {
        "rule_id": "transformers_max_new_tokens_positive",
        "native_type": "transformers.GenerationConfig",
        "severity": "error",
        "kwargs_positive": {"max_new_tokens": "not-an-int"},
        "kwargs_negative": {"max_new_tokens": 16},
        "match": {"fields": {"max_new_tokens": {"<=": 0}}},
    }
    out = gate.gate_one("transformers", proposal)
    assert out["verdict"] == "not_confirmed"
    assert "type_coercion_artifact" in out["soundness_failed"]


def test_clean_positive_is_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive that raises on the claimed field with no coercion artefact
    passes the soundness check and is confirmed."""
    case = CaseResult(
        id="transformers_real_rule",
        outcome="error",
        emission_channel="none",
        positive_confirmed=True,
        negative_confirmed=True,
    )
    pos = _capture(
        exception_type="ValueError",
        exception_message="max_new_tokens must be greater than 0",
        error_details=(ErrorDetail(loc=("max_new_tokens",), error_type="greater_than"),),
    )
    neg = _capture()
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    proposal = {
        "rule_id": "transformers_real_rule",
        "native_type": "transformers.GenerationConfig",
        "severity": "error",
        "message_template": "max_new_tokens must be greater than 0",
        "kwargs_positive": {"max_new_tokens": -1},
        "kwargs_negative": {"max_new_tokens": 16},
        "match": {"fields": {"max_new_tokens": {"<=": 0}}},
    }
    out = gate.gate_one("transformers", proposal)
    assert out["verdict"] == "confirmed"
    assert "soundness_failed" not in out


def test_negative_that_fires_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative probe that itself raises is a dead negative case; the
    production soundness check rejects it."""
    case = CaseResult(
        id="transformers_dead_negative",
        outcome="error",
        emission_channel="none",
        positive_confirmed=True,
        negative_confirmed=True,
    )
    pos = _capture(
        exception_type="ValueError",
        exception_message="num_beams must be > 0",
        error_details=(ErrorDetail(loc=("num_beams",), error_type="greater_than"),),
    )
    neg = _capture(
        exception_type="ValueError",
        exception_message="num_beams must be > 0",
        error_details=(ErrorDetail(loc=("num_beams",), error_type="greater_than"),),
    )
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    proposal = {
        "rule_id": "transformers_dead_negative",
        "native_type": "transformers.GenerationConfig",
        "severity": "error",
        "message_template": "num_beams must be > 0",
        "kwargs_positive": {"num_beams": -1},
        "kwargs_negative": {"num_beams": -2},
        "match": {"fields": {"num_beams": {"<=": 0}}},
    }
    out = gate.gate_one("transformers", proposal)
    assert out["verdict"] == "not_confirmed"
    assert "negative_does_not_raise" in out["soundness_failed"]


def test_main_warms_up_engine_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """main fires warm_up_engine_observation once before the gate loop (the
    determinism fix the production paths apply)."""
    calls: list[str] = []
    monkeypatch.setattr(gate.V, "warm_up_engine_observation", lambda e: calls.append(e))
    monkeypatch.setattr(gate, "gate_one", lambda engine, p: {"rule_id": p["rule_id"]})

    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    in_path.write_text(json.dumps([{"rule_id": "a"}, {"rule_id": "b"}]))
    monkeypatch.setattr(sys, "argv", ["prog", "vllm", str(in_path), str(out_path)])

    assert gate.main() == 0
    assert calls == ["vllm"]  # exactly once, before the per-proposal loop
    assert len(json.loads(out_path.read_text())) == 2
