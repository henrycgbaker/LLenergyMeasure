"""Unit tests for the deterministic Tier-D oracle.

No GPU and no docker: the container gate is injected as a stub returning canned
verdicts. The test proves the oracle computes the REAL class-3 candidate set,
builds the canonical illegal-probe proposals (single-field, tier_d, no 2**31),
and collapses confirmed verdicts into the ``(section, leaf) -> illegal_values``
map.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import tier_d_oracle as oracle  # noqa: E402
from scripts.engine_producers.tier_d_candidates import (  # noqa: E402
    enumerate_class3_candidates,
)


def _surface(path: Path, leaves: list[str]) -> None:
    lines = ["class ModelConfig:"]
    lines += [f"    {leaf}: int = 0" for leaf in leaves]
    path.write_text("\n".join(lines) + "\n")


def test_run_oracle_builds_canonical_proposals_and_collapses(tmp_path: Path) -> None:
    candidates, _ = enumerate_class3_candidates("vllm")
    assert candidates, "vllm must have a non-empty class-3 candidate set"
    first = candidates[0]
    _surface(tmp_path / "config.py", [c.leaf for c in candidates])

    captured: dict[str, Any] = {}

    def _gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        captured["engine"] = engine
        captured["proposals"] = proposals
        # Confirm only the proposals targeting the first candidate's leaf.
        out = []
        for p in proposals:
            confirmed = p["kwargs_positive"].get(first.leaf) is not None
            out.append(
                {
                    "rule_id": p["rule_id"],
                    "verdict": "confirmed" if confirmed else "not_confirmed",
                }
            )
        return out

    result = oracle.run_oracle(
        engine="vllm",
        source_root=tmp_path,
        surface_files=["config.py"],
        gate_runner=_gate,
    )

    # The gate was called for vllm with at least one proposal per candidate.
    assert captured["engine"] == "vllm"
    proposals = captured["proposals"]
    assert proposals, "oracle must build canonical proposals"

    # Every proposal is a single-field tier_d construction-violation probe.
    for p in proposals:
        assert p["tier_d"] is True
        assert p["severity"] == "warn"
        assert p["native_type"], "native_type must resolve from the surface"
        assert len(p["kwargs_positive"]) == 1
        assert len(p["kwargs_negative"]) == 1
        fields = p["match"]["fields"]
        assert len(fields) == 1
        (path,) = fields
        assert path.startswith("vllm.")
        # No 2**31 / huge probe in the illegal ladder.
        (illegal,) = p["kwargs_positive"].values()
        assert illegal in (-2, -1, 0, -1.0, -0.5, 0.0)

    # The int candidate's probes use the int illegal ladder against a legal 1.
    first_probes = [p for p in proposals if first.leaf in p["kwargs_positive"]]
    assert {p["kwargs_positive"][first.leaf] for p in first_probes} == {-2, -1, 0}
    assert all(p["kwargs_negative"][first.leaf] == 1 for p in first_probes)

    # Confirmed verdicts collapse to exactly the first candidate's (section, leaf).
    assert (first.section, first.leaf) in result
    assert sorted(result[(first.section, first.leaf)]) == [-2, -1, 0]
    assert all(key == (first.section, first.leaf) for key in result)


def test_build_proposals_skips_unresolved_native_type() -> None:
    candidates, _ = enumerate_class3_candidates("vllm")
    # Empty native_type map -> no candidate resolves -> no proposals built.
    proposals = oracle.build_proposals("vllm", candidates, {})
    assert proposals == []


def test_run_oracle_raises_when_no_native_type_resolves(tmp_path: Path) -> None:
    # A surface with no matching config class resolves no native_types.
    (tmp_path / "config.py").write_text("x = 1\n")

    def _gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise AssertionError("gate must not run when no proposal is built")

    import pytest

    with pytest.raises(SystemExit):
        oracle.run_oracle(
            engine="vllm",
            source_root=tmp_path,
            surface_files=["config.py"],
            gate_runner=_gate,
        )
