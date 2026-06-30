"""Deterministic Tier-D oracle: the construction-confirmable single-field FLOOR.

This is the DETERMINISTIC FLOOR of the Tier-D hybrid architecture. For each
class-3 candidate
(:func:`scripts.engine_producers.tier_d_candidates.enumerate_class3_candidates`)
it probes a small ladder of canonical illegal values through the SAME
construction-violation gate the LLM mine uses
(:class:`llenergymeasure.api.diagnose.ContainerGateRunner` ->
``scripts/diagnose_gate_in_container.py``), with a legal negative. A candidate
has a construction-confirmable lower bound iff some canonical illegal value
raises while the legal value constructs - exactly the gate's bar, so the oracle
never confirms anything the gate would reject.

The oracle is single-field only: cross-field relationships are the LLM general
reader's job, not the deterministic floor's. Its bounds are deterministically
RE-DERIVABLE (re-run the probe ladder against the engine container at any pin),
so when later wired into the corpus they must NOT be frozen / carried like the
LLM advisories - re-mining recomputes them.

The illegal ladder is deliberately small and conservative:

- ints: ``[-2, -1, 0]`` - the lower-bound region. ``2**31`` is excluded: a huge
  value risks integer-overflow / ``ZeroDivisionError``-as-bound / device-topology
  raises that the gate's env-dependent guard or a crash would mis-handle.
- floats: ``[-1.0, -0.5, 0.0]`` - the same lower-bound region for ``float`` /
  ``number`` leaves.

The legal negative is a small in-range value (``1`` for ints, ``0.5`` for
floats) that should construct cleanly for a genuine lower-bound field.

    python3 scripts/engine_producers/tier_d_oracle.py \
        --engine vllm --image llenergymeasure:vllm-0.19.1 \
        --source-root /tmp/vllm-src \
        --surface-file vllm/config/__init__.py \
        --surface-file vllm/sampling_params.py \
        --docker-flag --gpus --docker-flag none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.api.diagnose import (  # noqa: E402
    ContainerGateRunner,
    GateRunner,
    tier_d_native_types,
)
from scripts.engine_producers.tier_d_candidates import (  # noqa: E402
    Class3Candidate,
    enumerate_class3_candidates,
)

_ENGINES = ("vllm", "tensorrt", "transformers")

# Canonical illegal lower-bound ladders. No 2**31: a huge probe risks overflow /
# ZeroDivisionError-as-bound / device-topology raises the gate would mis-handle.
INT_ILLEGAL: tuple[int, ...] = (-2, -1, 0)
FLOAT_ILLEGAL: tuple[float, ...] = (-1.0, -0.5, 0.0)
# Legal negatives: a small in-range value a genuine lower-bound field accepts.
INT_LEGAL = 1
FLOAT_LEGAL = 0.5


def _is_float(type_str: str | None) -> bool:
    """True if the candidate's declared type is float-ish (``float`` / ``number``)."""
    return bool(type_str) and ("float" in type_str or "number" in type_str)


def _oracle_proposal(
    engine: str,
    candidate: Class3Candidate,
    native_type: str,
    illegal_value: float,
    legal_value: float,
) -> dict[str, object]:
    """Build a single canonical Tier-D gate proposal for one illegal probe.

    Shaped exactly like the LLM mine's Tier-D proposal
    (:func:`llenergymeasure.api.diagnose._tier_d_proposal`): ``tier_d: True`` so
    the gate routes the construction-violation path, a single-field ``match``
    keyed on the candidate's project-config path (so the gate's locus check can
    attribute the raise), and the canonical illegal/legal kwargs.
    """
    path = f"{engine}.{candidate.section}.{candidate.leaf}"
    return {
        "rule_id": f"oracle::{candidate.section}::{candidate.leaf}::{illegal_value}",
        "tier_d": True,
        "native_type": native_type,
        "severity": "warn",
        "match": {"engine": engine, "fields": {path: {"<": legal_value}}},
        "expected_outcome": {"outcome": "warn", "emission_channel": "none"},
        "kwargs_positive": {candidate.leaf: illegal_value},
        "kwargs_negative": {candidate.leaf: legal_value},
        "invariant_under_test": f"deterministic oracle: {candidate.leaf} lower bound",
        "references": [f"oracle: canonical illegal probe {illegal_value}"],
    }


def build_proposals(
    engine: str,
    candidates: list[Class3Candidate],
    native_types: dict[str, str],
) -> list[dict[str, object]]:
    """Build the canonical illegal-probe proposals for every resolvable candidate.

    A candidate whose owning class native_type does not resolve from the source
    surface is skipped (the gate could not construct it). One proposal per
    (candidate, illegal value) on the type-appropriate ladder.
    """
    proposals: list[dict[str, object]] = []
    for candidate in candidates:
        native_type = native_types.get(candidate.leaf)
        if not native_type:
            continue
        if _is_float(candidate.type):
            illegal_values: tuple[float, ...] = FLOAT_ILLEGAL
            legal_value: float = FLOAT_LEGAL
        else:
            illegal_values = INT_ILLEGAL
            legal_value = INT_LEGAL
        for illegal_value in illegal_values:
            proposals.append(
                _oracle_proposal(engine, candidate, native_type, illegal_value, legal_value)
            )
    return proposals


def _confirmable_map(
    candidates: list[Class3Candidate],
    verdicts: list[dict[str, object]],
) -> dict[tuple[str, str], list[float]]:
    """Collapse confirmed verdicts into ``{(section, leaf): [illegal_values]}``.

    The ``(section, leaf)`` key is recovered from the candidate set keyed by leaf
    (the rule_id encodes section::leaf::value, but the candidate set is the
    authoritative owner). Only ``verdict == "confirmed"`` rows contribute; a leaf
    with no confirmed probe is absent.
    """
    by_leaf = {(c.section, c.leaf): c for c in candidates}
    confirmable: dict[tuple[str, str], list[float]] = {}
    for verdict in verdicts:
        if verdict.get("verdict") != "confirmed":
            continue
        parts = str(verdict.get("rule_id", "")).split("::")
        if len(parts) != 4:
            continue
        _, section, leaf, illegal = parts
        if (section, leaf) not in by_leaf:
            continue
        value: float = float(illegal) if "." in illegal else int(illegal)
        confirmable.setdefault((section, leaf), []).append(value)
    return confirmable


def run_oracle(
    *,
    engine: str,
    source_root: Path,
    surface_files: list[str],
    gate_runner: GateRunner,
) -> dict[tuple[str, str], list[float]]:
    """Run the deterministic oracle and return the confirmable-bound map.

    Enumerates class-3 candidates, resolves each owning class native_type from the
    source surface, builds canonical illegal-probe proposals, and runs them
    through ``gate_runner`` (the SAME gate the LLM mine uses; injected so the
    orchestration is unit-testable without docker). Returns
    ``{(section, leaf): [illegal_values that construction-confirm the bound]}``.
    """
    candidates, _counts = enumerate_class3_candidates(engine)
    if not candidates:
        raise SystemExit(f"no class-3 candidates for {engine}; nothing to probe")
    native_types = tier_d_native_types(
        source_root=source_root,
        config_surface_files=surface_files,
        candidate_leaves=[c.leaf for c in candidates],
    )
    proposals = build_proposals(engine, candidates, native_types)
    if not proposals:
        raise SystemExit(f"no candidate native_type resolved from the source surface for {engine}")
    verdicts = gate_runner(engine, proposals)
    return _confirmable_map(candidates, verdicts)


def _report(engine: str, confirmable: dict[tuple[str, str], list[float]]) -> None:
    print(f"=== {engine}: deterministic floor = {len(confirmable)} confirmable candidate(s) ===")
    for (section, leaf), illegals in sorted(confirmable.items()):
        print(f"  {section}.{leaf}: raises at illegal={sorted(illegals)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic Tier-D oracle: the construction-confirmable single-field floor."
    )
    parser.add_argument("--engine", required=True, choices=_ENGINES)
    parser.add_argument("--image", required=True, help="Engine cache image to run the gate inside.")
    parser.add_argument("--source-root", required=True, type=Path, help="Engine source tree root.")
    parser.add_argument(
        "--surface-file",
        action="append",
        required=True,
        dest="surface_files",
        help="Config-surface source file relative to --source-root (repeatable).",
    )
    parser.add_argument(
        "--docker-flag",
        action="append",
        default=[],
        dest="docker_flags",
        help="Extra docker run flag for the gate container, e.g. --gpus / none (repeatable).",
    )
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/llem-tier-d-oracle"))
    args = parser.parse_args(argv)

    gate_runner = ContainerGateRunner(
        image=args.image,
        repo=Path.cwd(),
        workdir=args.workdir,
        docker_flags=tuple(args.docker_flags),
    )
    confirmable = run_oracle(
        engine=args.engine,
        source_root=args.source_root,
        surface_files=args.surface_files,
        gate_runner=gate_runner,
    )
    _report(args.engine, confirmable)
    return 0


__all__ = ["build_proposals", "main", "run_oracle"]


if __name__ == "__main__":
    raise SystemExit(main())
