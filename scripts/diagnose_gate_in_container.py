"""In-container gate driver for the Stage-1 LLM diagnose proposer.

Runs INSIDE the engine cache image. Reads a JSON list of diagnose proposals
(each: rule_id, native_type, severity, kwargs_positive, kwargs_negative, plus
the carried ``match`` so the gate-soundness locus check can run), drives each
through the REAL gate (``scripts.validate_rules._validate_invariant_with_captures``),
and writes a verdict JSON. This reuses the production construct+observe path
VERBATIM - the same code that adjudicates the live decay alarm - so
"gate-confirmed" here means exactly what it means in production. No gate logic
is reimplemented.

A proposal is GATE-CONFIRMED iff positive_confirmed AND negative_confirmed AND
the production gate-soundness checks pass: the positive probe reproduces the
claimed behaviour (raise for severity=error; emit / normalise for
severity=dormant), the negative probe does not, and the raise is attributable
to the claimed rule (no type-coercion artefact, correct error locus, template
match). This is the SAME bar as the two production gate paths
(``validate_rules.py``): ``compute_gate_soundness_divergences`` must come back
empty, and ``warm_up_engine_observation`` is fired once before the loop so the
diagnose gate observes the engine in the same warmed state as production.

Silent-dormancy claims (severity=dormant where the positive constructs as a
no-op) CANNOT positive-confirm as an emission at construction grain - that is
correct by design (silent dormancy is an equivalence, not a construction
emission). Such a proposal is reported as ``not_construction_confirmable``, a
distinct verdict from ``not_confirmed``, so the caller can surface it as a cited
proposal for review rather than counting it a failure.

A proposal tagged ``tier_d`` takes the CONSTRUCTION-VIOLATION path
(``gate_one_tier_d``): the claimed-illegal positive must RAISE and the legal
negative must construct, with the same coercion/locus soundness guards - the
empirical sentinel guard for the Tier-D pure-inference advisories. The advisory
is always EMITTED at warn (the universal safety gate); the gate only decides
whether the bound is real, and reports rich per-probe fields so the host can
apply a stricter or more permissive keep-policy without re-running the gate. The
verdict records the GRAIN it confirmed at. Some engines enforce generation
bounds at generate()-time, not config construction (transformers ``num_beams>=1``
raises only inside ``model.generate()``); for those, a GENERATE-grain fallback
(``_generate_grain_confirms`` on a tiny CPU model) runs when construction did not
confirm, so the gate is not blind to the bounds the LLM is uniquely good at
finding.

Invoked by :class:`llenergymeasure.api.diagnose.ContainerGateRunner`:

    python3 scripts/diagnose_gate_in_container.py <engine> <in.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/repo")

import scripts.validate_rules as V


def _severity_for(proposal: dict[str, Any]) -> str:
    """Gate severity for routing strictness.

    The host (``diagnose._proposal_from_diagnosis``) ALWAYS populates a valid
    ``severity`` (carried, or derived from the classification when there is no
    carried entry), so the container trusts it. The fallback to ``error`` (the
    strictest routing) only fires defensively if the field is genuinely absent -
    it does NOT re-derive the host's classification routing (that drifts).
    """
    sev = str(proposal.get("severity", "")).lower()
    return sev if sev in {"error", "dormant", "warn"} else "error"


def gate_one(engine: str, proposal: dict[str, Any]) -> dict[str, Any]:
    severity = _severity_for(proposal)
    inv = {
        "id": proposal["rule_id"],
        "native_type": proposal.get("native_type") or "",
        "severity": severity,
        "kwargs_positive": dict(proposal.get("kwargs_positive") or {}),
        "kwargs_negative": dict(proposal.get("kwargs_negative") or {}),
        "expected_outcome": proposal.get("expected_outcome")
        or {"outcome": "error" if severity == "error" else "dormant_announced"},
        "match": proposal.get("match") or {"fields": {}},
    }
    out: dict[str, Any] = {"rule_id": proposal["rule_id"], "severity": severity}
    try:
        case, pos, neg = V._validate_invariant_with_captures(engine, inv)
    except Exception as exc:  # construction blew up before observe
        out.update({"verdict": "infra_error", "error": f"{type(exc).__name__}: {exc}"})
        return out

    # Production gate-soundness checks (validate_rules.py:712,1129): a positive
    # that raises for the wrong reason (type-coercion artefact, sibling-Literal
    # locus, template mismatch) or a negative that fires is rejected by the
    # production alarm. Run the SAME checks here so the diagnose gate cannot
    # confirm what production would reject; record the failed check names.
    soundness = V.compute_gate_soundness_divergences(inv, pos, neg)
    soundness_failed = [d.check_failed for d in soundness if d.check_failed is not None]

    confirmed = bool(case.positive_confirmed and case.negative_confirmed) and not soundness
    # Silent-dormancy claim that constructs as a no-op: cannot positive-confirm at
    # construction grain. Distinguish from a real failure so the caller routes it
    # to review rather than discard.
    silent_dormancy_noop = (
        not case.positive_confirmed
        and severity == "dormant"
        and proposal.get("classification") == "dormancy_now_silent"
        and case.outcome in {"pass", "no_op", "dormant_silent"}
    )
    if confirmed:
        verdict = "confirmed"
    elif silent_dormancy_noop:
        verdict = "not_construction_confirmable"
    else:
        verdict = "not_confirmed"
    out.update(
        {
            "outcome": case.outcome,
            "positive_confirmed": bool(case.positive_confirmed),
            "negative_confirmed": bool(case.negative_confirmed),
            "pos_exception": pos.exception_type,
            "pos_message": (pos.exception_message or "")[:160],
            "verdict": verdict,
        }
    )
    if soundness_failed:
        out["soundness_failed"] = soundness_failed
    return out


def _is_sentinel_value(value: Any) -> bool:
    """True for the integer sentinels a claimed lower bound commonly mis-flags.

    ``-1`` (unlimited) and ``0`` (disabled/auto) are routinely accepted by
    inference libraries even where a docstring says ``>0``; a Tier-D positive
    probe that picks one and DOESN'T raise is the `max_logprobs=-1`-is-legal
    hallucination the sentinel guard exists to catch.
    """
    return value in (-1, 0) and not isinstance(value, bool)


# Engines whose generation bounds are enforced at generate()-time, not at config
# construction, so the construction gate is structurally blind to them and a
# generate-grain probe is needed (e.g. transformers num_beams>=1).
_GENERATE_GRAIN_ENGINES = {"transformers"}
_TINY_LM: Any = None


def _tiny_causal_lm() -> Any:
    """A tiny random CPU model for the generate()-grain probe (built once).

    From config only - no checkpoint, no download, no GPU. Generation bounds like
    ``num_beams>=1`` raise inside ``model.generate()`` (beam-tensor allocation),
    never at ``GenerationConfig`` construction, so a one-token generate on this
    model is the only grain that surfaces them.
    """
    global _TINY_LM
    if _TINY_LM is None:
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.for_model(
            "gpt2", n_layer=1, n_head=1, n_embd=8, vocab_size=64, n_positions=64
        )
        _TINY_LM = AutoModelForCausalLM.from_config(cfg).eval()
    return _TINY_LM


def _generate_grain_confirms(proposal: dict[str, Any]) -> tuple[bool, str | None]:
    """Confirm a transformers bound at GENERATE()-time.

    Returns ``(confirmed, illegal_exception_name)``: confirmed iff the illegal
    kwargs make ``model.generate()`` raise and the legal kwargs do not.
    """
    import torch
    from transformers import GenerationConfig

    model = _tiny_causal_lm()
    ids = torch.tensor([[1, 2, 3]])

    def _raises(kwargs: dict[str, Any]) -> str | None:
        try:
            with torch.no_grad():
                model.generate(
                    ids,
                    generation_config=GenerationConfig(max_new_tokens=2, pad_token_id=0, **kwargs),
                )
            return None
        except Exception as exc:
            return type(exc).__name__

    illegal_exc = _raises(dict(proposal.get("kwargs_positive") or {}))
    legal_exc = _raises(dict(proposal.get("kwargs_negative") or {}))
    return (illegal_exc is not None and legal_exc is None), illegal_exc


def gate_one_tier_d(engine: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Construction-violation gate for a Tier-D pure-inference advisory.

    The confirmation grain is CONSTRUCTION, not firing/emission: the
    claimed-illegal ``kwargs_positive`` must actually RAISE and the legal
    ``kwargs_negative`` must construct cleanly, with the SAME coercion/locus
    soundness guards production uses - the empirical sentinel guard that kills the
    `max_logprobs=-1`-is-legal class of hallucination. The advisory is EMITTED at
    warn regardless (the universal safety gate); the gate only decides whether the
    bound is real. The rich ``constructs_legal`` / ``illegal_raises`` /
    ``coercion_artifact`` / ``illegal_is_sentinel`` fields let the host apply a
    more permissive keep-policy without re-running the gate.
    """
    rule_id = proposal["rule_id"]
    # severity=error routes the strict positive-MUST-raise soundness branch.
    # message_template is intentionally omitted: a Tier-D constraint is human
    # prose, not the library's exception string, so the template-match soundness
    # check would spuriously fail on it.
    inv = {
        "id": rule_id,
        "native_type": proposal.get("native_type") or "",
        "severity": "error",
        "kwargs_positive": dict(proposal.get("kwargs_positive") or {}),
        "kwargs_negative": dict(proposal.get("kwargs_negative") or {}),
        "expected_outcome": {"outcome": "error"},
        "match": proposal.get("match") or {"fields": {}},
    }
    illegal_values = list((proposal.get("kwargs_positive") or {}).values())
    out: dict[str, Any] = {
        "rule_id": rule_id,
        "severity": "warn",
        "tier_d": True,
        "illegal_is_sentinel": any(_is_sentinel_value(v) for v in illegal_values),
    }
    try:
        _case, pos, neg = V._validate_invariant_with_captures(engine, inv)
    except Exception as exc:  # construction blew up before observe
        out.update({"verdict": "infra_error", "error": f"{type(exc).__name__}: {exc}"})
        return out

    # Mirror the production gate's infra guards (validate_rules._carried_verdict):
    # an unresolved native_type on the positive probe, or a construction-drift
    # error on the negative probe, is an INFRA failure - NOT an empirical bound
    # raise. Without this a NativeTypeResolutionError on the illegal probe would
    # be mis-counted as illegal_raises=True (the live tensorrt mine surfaced it).
    if pos.exception_type == "NativeTypeResolutionError":
        out.update(
            {
                "verdict": "infra_error",
                "error": f"native_type unresolved: {pos.exception_message or ''}",
            }
        )
        return out
    if neg.exception_type in V._CONSTRUCTION_DRIFT_EXC_NAMES:
        out.update(
            {
                "verdict": "infra_error",
                "error": f"negative probe will not construct ({neg.exception_type}): "
                f"{neg.exception_message or ''}",
            }
        )
        return out

    soundness = V.compute_gate_soundness_divergences(inv, pos, neg)
    soundness_failed = [d.check_failed for d in soundness if d.check_failed is not None]
    constructs_legal = neg.exception_type is None
    illegal_raises = pos.exception_type is not None
    coercion_artifact = V.CHECK_TYPE_COERCION_ARTIFACT in soundness_failed
    # STRICT default: the bound is real iff the illegal value raises, the legal
    # value constructs, and the raise is attributable to the claimed field (no
    # coercion artefact / locus mismatch / dead negative). soundness empty implies
    # all of those for a severity=error, no-template inv.
    confirmed = illegal_raises and constructs_legal and not soundness
    out.update(
        {
            "constructs_legal": constructs_legal,
            "illegal_raises": illegal_raises,
            "illegal_exc_type": pos.exception_type,
            "illegal_message": (pos.exception_message or "")[:160],
            "coercion_artifact": coercion_artifact,
            "verdict": "confirmed" if confirmed else "not_confirmed",
            "grain": "construction" if confirmed else None,
        }
    )
    if soundness_failed:
        out["soundness_failed"] = soundness_failed

    # Generate-grain fallback: a bound enforced at generate()-time (transformers
    # num_beams>=1) is invisible to the construction gate. If construction did not
    # confirm but the legal probe at least constructs, try the generate grain for
    # engines that enforce there. The advisory stays warn; only the grain differs.
    if not confirmed and constructs_legal and engine in _GENERATE_GRAIN_ENGINES:
        try:
            gen_confirmed, gen_illegal_exc = _generate_grain_confirms(proposal)
        except Exception as exc:  # model build / generate infra failure
            out["generate_grain_error"] = f"{type(exc).__name__}: {exc}"
        else:
            out["generate_illegal_exc"] = gen_illegal_exc
            if gen_confirmed:
                out.update({"verdict": "confirmed", "grain": "generate"})
    return out


def _gate_one(engine: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a proposal to the Tier-D or the firing-confirmation gate path."""
    if proposal.get("tier_d"):
        return gate_one_tier_d(engine, proposal)
    return gate_one(engine, proposal)


def main() -> int:
    engine = sys.argv[1]
    in_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    proposals = json.loads(in_path.read_text())
    # Flush process-once import-side-effect logging before the gate loop, exactly
    # as the production paths do (validate_rules.py:754,1088). Without it the first
    # proposal absorbs vLLM's one-time platform-detection burst and misclassifies.
    V.warm_up_engine_observation(engine)
    verdicts = [_gate_one(engine, p) for p in proposals]
    out_path.write_text(json.dumps(verdicts, indent=2))
    confirmed = sum(1 for v in verdicts if v.get("verdict") == "confirmed")
    print(f"gated {len(verdicts)} proposals: {confirmed} confirmed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
