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


# ---------------------------------------------------------------------------
# Tier-D construction-violation gate
# ---------------------------------------------------------------------------


def _tier_d_proposal(**over: Any) -> dict[str, Any]:
    base = {
        "rule_id": "vllm_tier_d_engine_params_max_logprobs",
        "tier_d": True,
        "native_type": "vllm.config.ModelConfig",
        "severity": "warn",
        "kwargs_positive": {"max_logprobs": -5},
        "kwargs_negative": {"max_logprobs": 5},
        "match": {"fields": {"vllm.engine_params.max_logprobs": {"<": 0}}},
    }
    base.update(over)
    return base


def test_tier_d_illegal_raises_and_legal_constructs_is_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The construction-violation path confirms when the illegal value raises on
    the claimed field and the legal value constructs cleanly."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    pos = _capture(
        exception_type="ValueError",
        exception_message="max_logprobs must be >= 0",
        error_details=(ErrorDetail(loc=("max_logprobs",), error_type="greater_than_equal"),),
    )
    neg = _capture()
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    out = gate.gate_one_tier_d("vllm", _tier_d_proposal())
    assert out["verdict"] == "confirmed"
    assert out["severity"] == "warn" and out["tier_d"] is True
    assert out["illegal_raises"] is True and out["constructs_legal"] is True
    assert "soundness_failed" not in out


def test_tier_d_illegal_constructs_cleanly_is_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unbounded field whose 'illegal' value constructs without raising is NOT
    confirmed under the strict default (no empirical enforcement)."""
    case = CaseResult(id="x", outcome="no_op", emission_channel="none")
    pos = _capture()  # illegal value constructed cleanly -> no raise
    neg = _capture()
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    out = gate.gate_one_tier_d("vllm", _tier_d_proposal())
    assert out["verdict"] == "not_confirmed"
    assert out["illegal_raises"] is False and out["constructs_legal"] is True


@pytest.mark.parametrize("sentinel", [-1, 0])
def test_tier_d_sentinel_flagged_when_illegal_value_is_a_sentinel(
    monkeypatch: pytest.MonkeyPatch, sentinel: int
) -> None:
    """A -1 (unlimited) or 0 (disabled/auto) 'illegal' value that constructs
    cleanly is reported as a sentinel (the max_logprobs=-1-is-legal class of
    hallucination), and not confirmed."""
    case = CaseResult(id="x", outcome="no_op", emission_channel="none")
    _stub_captures(monkeypatch, case=case, pos=_capture(), neg=_capture())

    out = gate.gate_one_tier_d("vllm", _tier_d_proposal(kwargs_positive={"max_logprobs": sentinel}))
    assert out["illegal_is_sentinel"] is True
    assert out["verdict"] == "not_confirmed"


def test_tier_d_unresolved_native_type_is_infra_not_a_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NativeTypeResolutionError on the illegal probe is INFRA, never counted
    as an empirical bound raise (the live tensorrt mine surfaced this)."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    unresolved = _capture(
        exception_type="NativeTypeResolutionError",
        exception_message="could not resolve class 'SamplingParams'",
    )
    _stub_captures(monkeypatch, case=case, pos=unresolved, neg=unresolved)

    out = gate.gate_one_tier_d("tensorrt", _tier_d_proposal(native_type="tensorrt_llm.Nope"))
    assert out["verdict"] == "infra_error"
    assert "native_type unresolved" in out["error"]


def test_tier_d_negative_construction_drift_is_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the legal probe will not even construct (a TypeError/import drift), the
    advisory cannot be adjudicated - infra_error, not a confirmation."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    pos = _capture(
        exception_type="ValueError",
        exception_message="bad",
        error_details=(ErrorDetail(loc=("max_tokens",), error_type="greater_than"),),
    )
    neg = _capture(exception_type="TypeError", exception_message="missing required arg")
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    out = gate.gate_one_tier_d("tensorrt", _tier_d_proposal())
    assert out["verdict"] == "infra_error"


def test_tier_d_empty_native_type_is_infra_error_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A citation that did not resolve leaves native_type empty; the gate cannot
    construct it, so the proposal is an infra_error - never a silent confirm."""

    def _unresolvable(engine: str, inv: dict[str, Any]):
        assert inv["native_type"] == ""  # the unresolved-citation case
        raise ValueError("native_type unresolved: ''")

    monkeypatch.setattr(gate.V, "_validate_invariant_with_captures", _unresolvable)
    out = gate.gate_one_tier_d("vllm", _tier_d_proposal(native_type=""))
    assert out["verdict"] == "infra_error"


def test_tier_d_coercion_artefact_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive that raises only via a type-coercion artefact (not the claimed
    bound) is rejected by the production soundness check, same as gate_one."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    pos = _capture(
        exception_type="ValidationError",
        exception_message="Input should be a valid integer",
        error_details=(ErrorDetail(loc=("max_logprobs",), error_type="int_parsing"),),
    )
    neg = _capture()
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    out = gate.gate_one_tier_d("vllm", _tier_d_proposal(kwargs_positive={"max_logprobs": "bad"}))
    assert out["verdict"] == "not_confirmed"
    assert "type_coercion_artifact" in out["soundness_failed"]


def test_tier_d_dead_negative_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the legal value itself raises, the advisory cannot be confirmed."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    exc = _capture(
        exception_type="ValueError",
        exception_message="max_logprobs must be >= 0",
        error_details=(ErrorDetail(loc=("max_logprobs",), error_type="greater_than_equal"),),
    )
    _stub_captures(monkeypatch, case=case, pos=exc, neg=exc)

    out = gate.gate_one_tier_d("vllm", _tier_d_proposal())
    assert out["verdict"] == "not_confirmed"
    assert out["constructs_legal"] is False


def test_tier_d_env_dependent_raise_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raise attributable to the box's device/topology (GPU count, world size),
    not the field value, is box-dependent and must NOT confirm - even though the
    illegal value raises and the legal value constructs. Guards gate determinism
    (the prefill_context_parallel_size 'World size > available GPUs' false positive)."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    pos = _capture(
        exception_type="ValueError",
        exception_message=(
            "World size (8) is larger than the number of available GPUs (0) in this node."
        ),
    )
    neg = _capture()  # legal value constructs cleanly
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    out = gate.gate_one_tier_d(
        "vllm",
        _tier_d_proposal(
            rule_id="vllm_tier_d_engine_params_prefill_context_parallel_size",
            kwargs_positive={"prefill_context_parallel_size": 8},
            kwargs_negative={"prefill_context_parallel_size": 1},
            match={"fields": {"vllm.engine_params.prefill_context_parallel_size": {">": 1}}},
        ),
    )
    assert out["verdict"] == "not_confirmed"
    assert out["illegal_raises"] is True and out["constructs_legal"] is True
    assert out["env_dependent"] is True


def test_tier_d_field_value_raise_with_empty_loc_still_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine field-value bound that raises with NO recoverable locus (a plain
    ValueError from _verify_args, like vllm logprobs) must still confirm - the
    env-dependence guard keys on the message, not the absence of a locus, so it
    does not regress real empty-loc bounds."""
    case = CaseResult(id="x", outcome="error", emission_channel="none")
    pos = _capture(
        exception_type="ValueError",
        exception_message="logprobs must be non-negative or -1",
    )  # no error_details -> empty loc, as _verify_args raises produce
    neg = _capture()
    _stub_captures(monkeypatch, case=case, pos=pos, neg=neg)

    out = gate.gate_one_tier_d(
        "vllm",
        _tier_d_proposal(
            rule_id="vllm_tier_d_sampling_params_logprobs",
            kwargs_positive={"logprobs": -2},
            kwargs_negative={"logprobs": 5},
            match={"fields": {"vllm.sampling_params.logprobs": {"<": -1}}},
        ),
    )
    assert out["verdict"] == "confirmed"
    assert out["env_dependent"] is False


def test_tier_d_infra_error_when_construction_blows_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(engine: str, inv: dict[str, Any]):
        raise RuntimeError("native_type unresolved")

    monkeypatch.setattr(gate.V, "_validate_invariant_with_captures", _boom)
    out = gate.gate_one_tier_d("vllm", _tier_d_proposal())
    assert out["verdict"] == "infra_error"
    assert "native_type unresolved" in out["error"]


def test_dispatch_routes_tier_d_proposals(monkeypatch: pytest.MonkeyPatch) -> None:
    """main()'s per-proposal dispatch sends tier_d proposals to the Tier-D path
    and others to the firing-confirmation path."""
    monkeypatch.setattr(
        gate, "gate_one_tier_d", lambda e, p: {"rule_id": p["rule_id"], "path": "d"}
    )
    monkeypatch.setattr(gate, "gate_one", lambda e, p: {"rule_id": p["rule_id"], "path": "fire"})
    assert gate._gate_one("vllm", {"rule_id": "a", "tier_d": True})["path"] == "d"
    assert gate._gate_one("vllm", {"rule_id": "b"})["path"] == "fire"


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
