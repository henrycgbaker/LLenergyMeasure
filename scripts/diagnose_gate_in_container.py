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


def main() -> int:
    engine = sys.argv[1]
    in_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    proposals = json.loads(in_path.read_text())
    # Flush process-once import-side-effect logging before the gate loop, exactly
    # as the production paths do (validate_rules.py:754,1088). Without it the first
    # proposal absorbs vLLM's one-time platform-detection burst and misclassifies.
    V.warm_up_engine_observation(engine)
    verdicts = [gate_one(engine, p) for p in proposals]
    out_path.write_text(json.dumps(verdicts, indent=2))
    confirmed = sum(1 for v in verdicts if v.get("verdict") == "confirmed")
    print(f"gated {len(verdicts)} proposals: {confirmed} confirmed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
