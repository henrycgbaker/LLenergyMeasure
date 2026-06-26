"""Unit tests for the Tier-D advisory mine driver.

No GPU and no ollama: the proposer model and the container gate are injected as
stubs. The test proves the driver computes the REAL deterministic class-3
candidate set for an engine, threads it through the Tier-D proposer, and applies
the scripts-level match-path validator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import diagnose_tier_d_mine as driver  # noqa: E402
from scripts.engine_producers._section_classifier import (  # noqa: E402
    UnresolvableMatchPathError,
)
from scripts.engine_producers.tier_d_candidates import (  # noqa: E402
    enumerate_class3_candidates,
)


class _StubModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def _surface(path: Path, leaves: list[str]) -> None:
    lines = ["class ModelConfig:"]
    lines += [f"    {leaf}: int = 0" for leaf in leaves]
    path.write_text("\n".join(lines) + "\n")


def test_driver_computes_real_candidates_and_runs(tmp_path: Path, monkeypatch) -> None:
    # Use the real vllm candidate set; window a surface owning the first candidate.
    candidates, _ = enumerate_class3_candidates("vllm")
    assert candidates, "vllm must have a non-empty class-3 candidate set"
    first = candidates[0]
    _surface(tmp_path / "config.py", [c.leaf for c in candidates])

    # Stub the proposer to flag the first candidate as a real bound, citing the
    # surface class so native_type resolves; gate confirms it.
    import json

    rule_id = f"vllm_tier_d_{first.section}_{first.leaf}"
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": rule_id,
                    "classification": "bounded",
                    "reason": "must be >= 0",
                    "citation": "config.py:2",
                    "constraint": "must be >= 0",
                    "fires_when": {"<": 0},
                    "kwargs_positive": {first.leaf: -1},
                    "kwargs_negative": {first.leaf: 1},
                }
            ]
        }
    )

    def _gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"rule_id": p["rule_id"], "verdict": "confirmed"} for p in proposals]

    # Skip the real Config path-resolve check (the synthetic surface is not the
    # generated Config); that path is covered by diagnose's own tests.
    monkeypatch.setattr(driver, "_path_validator", lambda engine: lambda path: None)

    result = driver.run_tier_d_mine(
        engine="vllm",
        new_version="0.19.1",
        source_root=tmp_path,
        surface_files=["config.py"],
        model=_StubModel(raw),
        gate_runner=_gate,
    )
    assert result.engine == "vllm" and result.mode == "tier_d"
    assert [e["id"] for e in result.confirmed] == [rule_id]
    assert result.confirmed[0]["added_by"] == "llm_advisory"


def test_driver_path_validator_uses_assert_path_resolves() -> None:
    # The validator wraps the scripts-level resolver and raises on a bad path.
    check = driver._path_validator("vllm")
    with pytest.raises(UnresolvableMatchPathError):
        check("vllm.engine_params.this_leaf_does_not_exist_anywhere_xyz")
