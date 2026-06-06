"""Study Round 0 pilot - union + gate the tensorrt-1.2.1 invariant GT.

Unions the four GT sources for the tensorrt v1_2_1 invariant task by
TOLERANT invariant identity, runtime-gates every kwargs-bearing candidate
inside the engine's canonical container (the production
scripts/validate_invariants.py, invoked via the trial dispatch), and writes
the gate-confirmed pilot GT plus a report.

Sources:
  passA  Opus entry-point / call-graph walk      (GT schema; kwargs on a subset)
  passB  Opus class-hierarchy / type-tree walk   (GT schema; kwargs on a subset)
  mech   mechanical improved-det-v2              (proposed schema; no kwargs -> identity-only)
  poc    existing PoC GT                          (GT schema; no native_type -> mostly identity-only)

Tolerant identity = (leaf_native_field, coarse_predicate_bucket), reusing
gt_scoring's tolerant keying. A tolerant group is gate-CONFIRMED when at least
one of its kwargs-bearing members constructs as declared (positive fires +
negative does not, per the production gate). Groups with no runtime-reachable
member are labelled unverified.

This is the pilot before fan-out: it validates the union+gate pipeline
end-to-end on a single cell and quantifies whether the gate-validated union
is materially richer / more trustworthy than the PoC N=1 GT.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TRIAL_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _TRIAL_SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gt_adapter  # noqa: E402
import gt_scoring  # noqa: E402
import trial_scoring as S  # noqa: E402
import yaml  # noqa: E402

ENGINE = "tensorrt"
VERSION_SLUG = "v1_2_1"

_FINDINGS = _TRIAL_SCRIPTS_DIR.parent / "findings"
_OUT_DIR = _FINDINGS / "study" / "ground_truth" / "tensorrt" / "v1_2_1" / "invariants"

# source-key -> (path, is_gt_schema). GT-schema sources are canonicalised
# through gt_adapter (gaining a synthesised match block + tolerant hints while
# preserving native_type / kwargs); the mechanical source is already in the
# proposed schema with a real match block.
SOURCES: dict[str, tuple[Path, bool]] = {
    "passA": (_OUT_DIR / "passA_entrypoint.yaml", True),
    "passB": (_OUT_DIR / "passB_classtree.yaml", True),
    "mech": (
        _FINDINGS
        / "trial_runs/wave2/w2-a-improved-det-v2/tensorrt/v1_2_1/invariants.proposed.yaml",
        False,
    ),
    "poc": (
        _FINDINGS / "ground_truth/tensorrt/v1_2_1/invariants_ground_truth.yaml",
        True,
    ),
}


@dataclass
class Candidate:
    source: str
    orig_id: str
    inv: dict[str, Any]
    tkey: tuple[str, str]
    gateable: bool
    # filled in after gating
    verdict: str = "ungated"  # confirmed | failed | skipped | infra_error | ungated
    observed_outcome: str = ""
    error: str = ""


@dataclass
class Group:
    tkey: tuple[str, str]
    sources: set[str] = field(default_factory=set)
    members: list[Candidate] = field(default_factory=list)

    @property
    def gateable_members(self) -> list[Candidate]:
        return [c for c in self.members if c.gateable]

    @property
    def confirmed_members(self) -> list[Candidate]:
        return [c for c in self.members if c.verdict == "confirmed"]

    @property
    def status(self) -> str:
        verdicts = {c.verdict for c in self.gateable_members}
        if "confirmed" in verdicts:
            return "confirmed"
        if "failed" in verdicts:
            return "failed"
        if "skipped" in verdicts:
            return "skipped"
        if "infra_error" in verdicts:
            return "infra_error"
        return "unreachable"  # no gateable member at all


def _tolerant_key(inv: dict[str, Any]) -> tuple[str, str]:
    """Per-entry tolerant key (leaf_native_field, coarse_predicate_bucket).

    Mirrors gt_scoring.tolerant_invariant_keys: prefer the canonicalisation
    hints, else fall back to the locked invariant_identity on the match block.
    """
    leaf = inv.get("_gt_leaf_field")
    bucket = inv.get("_gt_coarse_bucket")
    if leaf is None or bucket is None:
        _, leaf, pk, _sec = S.invariant_identity(inv)
        bucket = gt_scoring._coarse_from_scorer_pk(pk)
    return (str(leaf), str(bucket))


def _gateable(inv: dict[str, Any]) -> bool:
    """Stageable to the gate: needs a native_type to construct. The gate
    synthesises kwargs from the declared predicate when the entry carries no
    hand-authored kwargs_positive/negative, so kwargs are no longer required
    here - only a constructible type is."""
    return bool(inv.get("native_type"))


def _has_authored_kwargs(inv: dict[str, Any]) -> bool:
    return isinstance(inv.get("kwargs_positive"), dict) and isinstance(
        inv.get("kwargs_negative"), dict
    )


def load_candidates() -> list[Candidate]:
    cands: list[Candidate] = []
    for source, (path, is_gt_schema) in SOURCES.items():
        if not path.exists():
            print(f"[pilot] WARNING: source {source} missing at {path}", flush=True)
            continue
        data = yaml.safe_load(path.read_text()) or {}
        if is_gt_schema:
            invs = gt_adapter.canonicalise_gt_invariants(data).get("invariants") or []
        else:
            invs = data.get("invariants") or []
        for inv in invs:
            if not isinstance(inv, dict):
                continue
            cands.append(
                Candidate(
                    source=source,
                    orig_id=str(inv.get("id", "")),
                    inv=inv,
                    tkey=_tolerant_key(inv),
                    gateable=_gateable(inv),
                )
            )
    return cands


def gate(cands: list[Candidate], *, timeout_sec: int = 1800) -> None:
    """Runtime-gate every kwargs-bearing candidate in ONE container run.

    Mutates each gateable Candidate's verdict / observed_outcome / error in
    place. Non-gateable candidates keep verdict='ungated'.
    """
    gateables = [c for c in cands if c.gateable]
    if not gateables:
        print("[pilot] no gateable candidates; skipping container run", flush=True)
        return

    staged: list[dict[str, Any]] = []
    idmap: dict[str, Candidate] = {}
    for i, c in enumerate(gateables):
        sid = f"{c.source}__{i}__{c.orig_id}"
        idmap[sid] = c
        inv = dict(c.inv)
        inv["id"] = sid
        staged.append(inv)

    envelope = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version_slug": VERSION_SLUG,
        "miner": "study_gt_pilot",
        "invariants": staged,
    }
    with tempfile.NamedTemporaryFile(suffix=".proposed.yaml", delete=False, mode="w") as tmp:
        spath = Path(tmp.name)
    spath.write_text(yaml.safe_dump(envelope, sort_keys=False))
    print(
        f"[pilot] gating {len(gateables)} kwargs-bearing candidates in the "
        f"{ENGINE}/{VERSION_SLUG} container (this loads the TRT-LLM image)...",
        flush=True,
    )
    try:
        validations = S.runtime_validate_invariants_dispatch(
            spath,
            engine=ENGINE,
            version_slug=VERSION_SLUG,
            timeout_sec=timeout_sec,
        )
    finally:
        spath.unlink(missing_ok=True)

    by_id = {v.invariant_id: v for v in validations}
    for sid, c in idmap.items():
        rv = by_id.get(sid)
        if rv is None:
            c.verdict = "infra_error"
            c.error = "no validation result returned"
            continue
        if rv.error:
            c.verdict = "infra_error"
            c.error = rv.error
            c.observed_outcome = rv.observed_outcome
            continue
        c.observed_outcome = rv.observed_outcome
        if rv.positive_confirmed and rv.negative_confirmed:
            c.verdict = "confirmed"
        elif rv.observed_outcome.startswith("skipped"):
            c.verdict = "skipped"
        else:
            c.verdict = "failed"


def build_groups(cands: list[Candidate]) -> dict[tuple[str, str], Group]:
    groups: dict[tuple[str, str], Group] = {}
    for c in cands:
        g = groups.setdefault(c.tkey, Group(tkey=c.tkey))
        g.sources.add(c.source)
        g.members.append(c)
    return groups


def _gt_entry(c: Candidate, runtime_label: str) -> dict[str, Any]:
    inv = c.inv
    entry: dict[str, Any] = {
        "id": c.orig_id,
        "source": c.source,
        "tolerant_key": list(c.tkey),
        "native_type": inv.get("native_type"),
        "native_field": inv.get("native_field"),
        "predicate_kind": inv.get("predicate_kind"),
        "predicate_value": inv.get("predicate_value"),
        "severity": inv.get("severity"),
        "runtime": runtime_label,
    }
    if inv.get("kwargs_positive") is not None:
        entry["kwargs_positive"] = inv.get("kwargs_positive")
        entry["kwargs_negative"] = inv.get("kwargs_negative")
    if c.observed_outcome:
        entry["observed_outcome"] = c.observed_outcome
    return entry


def write_pilot_gt(groups: dict[tuple[str, str], Group]) -> Path:
    confirmed: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for g in groups.values():
        if g.status == "confirmed":
            rep = g.confirmed_members[0]
            entry = _gt_entry(rep, "confirmed")
            entry["contributing_sources"] = sorted(g.sources)
            confirmed.append(entry)
        else:
            # representative = a gateable member if any, else first member
            rep = g.gateable_members[0] if g.gateable_members else g.members[0]
            label = "unverified" if g.status in {"skipped", "unreachable"} else g.status
            entry = _gt_entry(rep, label)
            entry["contributing_sources"] = sorted(g.sources)
            entry["gate_status"] = g.status
            unverified.append(entry)

    confirmed.sort(key=lambda e: tuple(e["tolerant_key"]))
    unverified.sort(key=lambda e: tuple(e["tolerant_key"]))

    out = {
        "schema_version": "study_pilot_gt/1.0.0",
        "engine": ENGINE,
        "engine_version": "1.2.1",
        "task": "invariants",
        "round": "R0_pilot",
        "gate": "scripts/validate_invariants.py (in nvcr.io/nvidia/tensorrt-llm/release:1.2.1)",
        "identity": "tolerant: (leaf_native_field, coarse_predicate_bucket)",
        "n_confirmed": len(confirmed),
        "n_unverified": len(unverified),
        "confirmed": confirmed,
        "unverified": unverified,
    }
    out_path = _OUT_DIR / "PILOT_GT.yaml"
    out_path.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
    return out_path


def write_report(
    cands: list[Candidate],
    groups: dict[tuple[str, str], Group],
) -> tuple[Path, dict[str, Any]]:
    by_source: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_source[c.source].append(c)

    source_keys: dict[str, set[tuple[str, str]]] = {
        src: {c.tkey for c in cs} for src, cs in by_source.items()
    }
    union_keys = set(groups.keys())
    confirmed_keys = {k for k, g in groups.items() if g.status == "confirmed"}
    poc_keys = source_keys.get("poc", set())

    # GT-growth: gate-confirmed tolerant identities absent from the PoC GT.
    growth_keys = confirmed_keys - poc_keys

    # Per-source unique contribution: tolerant identities only this source has.
    unique_by_source: dict[str, int] = {}
    for src in SOURCES:
        others: set[tuple[str, str]] = set()
        for other_src, keys in source_keys.items():
            if other_src != src:
                others |= keys
        unique_by_source[src] = len(source_keys.get(src, set()) - others)

    # Gate rejections: probed candidates that RAN and were not confirmed.
    rejected = [c for c in cands if c.verdict == "failed"]
    infra = [c for c in cands if c.verdict == "infra_error"]
    skipped = [c for c in cands if c.verdict == "skipped"]
    confirmed_cands = [c for c in cands if c.verdict == "confirmed"]
    # Split confirmations by probe provenance: synthesised by the gate vs
    # hand-authored kwargs in the source.
    synth_confirmed = [c for c in confirmed_cands if not _has_authored_kwargs(c.inv)]
    authored_confirmed = [c for c in confirmed_cands if _has_authored_kwargs(c.inv)]

    # The warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid).
    warn_cands = [c for c in cands if "warn_on_unstable_feature_usage" in c.orig_id]

    metrics = {
        "engine": ENGINE,
        "version": "1.2.1",
        "per_source_candidate_counts": {s: len(cs) for s, cs in by_source.items()},
        "per_source_tolerant_key_counts": {s: len(k) for s, k in source_keys.items()},
        "union_size_tolerant": len(union_keys),
        "n_gate_confirmed_keys": len(confirmed_keys),
        "n_probed_candidates": sum(1 for c in cands if c.gateable),
        "confirmed_via_synthesis": len(synth_confirmed),
        "confirmed_via_authored_kwargs": len(authored_confirmed),
        "verdict_counts": {
            "confirmed": len(confirmed_cands),
            "failed": len(rejected),
            "skipped": len(skipped),
            "infra_error": len(infra),
        },
        "gt_growth_keys_vs_poc": len(growth_keys),
        "poc_tolerant_keys": len(poc_keys),
        "per_source_unique_contribution": unique_by_source,
        "group_status_counts": _count(g.status for g in groups.values()),
    }

    def fmt_keys(keys: set[tuple[str, str]], limit: int = 40) -> list[str]:
        rows = sorted(f"{leaf} [{bucket}]" for leaf, bucket in keys)
        return rows[:limit]

    L: list[str] = []
    L.append("# Pilot GT report - tensorrt 1.2.1 invariants (union + gate)")
    L.append("")
    L.append(
        "Round 0 pilot: union the 4 GT sources by tolerant identity "
        "(leaf_native_field, coarse_predicate_bucket), runtime-gate every "
        "kwargs-bearing candidate in the production validator inside "
        "nvcr.io/nvidia/tensorrt-llm/release:1.2.1, keep gate-confirmed as GT."
    )
    L.append("")
    L.append("## Per-source candidate counts")
    L.append("")
    L.append("| source | raw candidates | tolerant keys | gateable | unique tolerant keys |")
    L.append("|---|---|---|---|---|")
    for src in SOURCES:
        cs = by_source.get(src, [])
        L.append(
            f"| {src} | {len(cs)} | {len(source_keys.get(src, set()))} | "
            f"{sum(1 for c in cs if c.gateable)} | {unique_by_source.get(src, 0)} |"
        )
    L.append("")
    L.append("## Union + gate")
    L.append("")
    L.append(f"- Union size (distinct tolerant identities across 4 sources): **{len(union_keys)}**")
    L.append(f"- Gate-confirmed tolerant identities: **{len(confirmed_keys)}**")
    L.append(
        f"- Probed candidates (native_type present, kwargs authored or synthesised): "
        f"**{metrics['n_probed_candidates']}** "
        f"(confirmed={len(confirmed_cands)}, failed={len(rejected)}, "
        f"skipped={len(skipped)}, infra_error={len(infra)})"
    )
    L.append(
        f"- Confirmations by probe provenance: "
        f"**{len(synth_confirmed)} synthesised** by the gate, "
        f"{len(authored_confirmed)} from hand-authored kwargs"
    )
    L.append("")
    L.append("Group status breakdown (per tolerant identity):")
    L.append("")
    for status, n in sorted(metrics["group_status_counts"].items()):
        L.append(f"- {status}: {n}")
    L.append("")
    L.append("## GT-growth vs PoC N=1 GT")
    L.append("")
    L.append(
        f"PoC GT contributed **{len(poc_keys)}** tolerant identities. The "
        f"gate-confirmed union grows GT by **{len(growth_keys)}** confirmed "
        f"identities the PoC GT lacked:"
    )
    L.append("")
    for row in fmt_keys(growth_keys):
        L.append(f"- {row}")
    if len(growth_keys) > 40:
        L.append(f"- ... (+{len(growth_keys) - 40} more)")
    L.append("")
    L.append("## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)")
    L.append("")
    if rejected:
        L.append("| id | source | native_type | observed_outcome |")
        L.append("|---|---|---|---|")
        for c in sorted(rejected, key=lambda c: (c.source, c.orig_id)):
            L.append(
                f"| {c.orig_id} | {c.source} | {c.inv.get('native_type')} | {c.observed_outcome} |"
            )
    else:
        L.append("_None - every gateable candidate that ran was confirmed._")
    L.append("")
    L.append("### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)")
    L.append("")
    if warn_cands:
        for c in sorted(warn_cands, key=lambda c: (c.source, c.orig_id)):
            L.append(
                f"- `{c.orig_id}` (source={c.source}, gateable={c.gateable}, "
                f"verdict={c.verdict}, observed={c.observed_outcome or 'n/a'})"
            )
    else:
        L.append("_No candidate id matched warn_on_unstable_feature_usage._")
    L.append("")
    if infra:
        L.append("## Infra errors (could not run in container)")
        L.append("")
        for c in sorted(infra, key=lambda c: (c.source, c.orig_id))[:30]:
            L.append(f"- `{c.orig_id}` ({c.source}): {c.error[:160]}")
        if len(infra) > 30:
            L.append(f"- ... (+{len(infra) - 30} more)")
        L.append("")

    report_path = _OUT_DIR / "PILOT_REPORT.md"
    report_path.write_text("\n".join(L))
    (_OUT_DIR / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return report_path, metrics


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for it in items:
        out[str(it)] += 1
    return dict(out)


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = load_candidates()
    print(f"[pilot] loaded {len(cands)} candidates across {len(SOURCES)} sources", flush=True)
    gate(cands)
    groups = build_groups(cands)
    gt_path = write_pilot_gt(groups)
    report_path, metrics = write_report(cands, groups)

    print("\n=== PILOT SUMMARY ===", flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"\n[pilot] wrote {gt_path}", flush=True)
    print(f"[pilot] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
