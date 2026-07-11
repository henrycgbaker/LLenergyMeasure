"""Tests for scripts/engine_producers/_runtime_literals.py.

Concerns:

1. Expressibility truth table (which static types already cover a string value).
2. Candidate sources: corpus (comparands + message tokens), source-text scan,
   docstring block scoping, LLM proposals reader.
3. The two-leg construction probe matrix with an injected fake constructor.
4. Merge determinism (shuffled order -> byte-identical) + entry shape.
5. Auto-narrow surfacing.
6. run_stage end-to-end on a tmp repo layout with an injected constructor.
7. The standing census check against the shipped artifacts (pending regen).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine_versions import _outputs  # noqa: E402
from scripts.engine_producers import _runtime_literals as rl  # noqa: E402

# ---------------------------------------------------------------------------
# Expressibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        ({"type": "bool"}, "never", False),  # string stranded on bool -> inexpressible
        ({"type": "str"}, "foo", True),  # string on str -> already expressible
        ({"type": "int"}, "x", False),  # string on int -> inexpressible
        ({"type": "str | int"}, "foo", True),  # union containing str
        ({"type": "boolean"}, "never", False),  # JSON-native spelling of bool
        ({"type": "unknown"}, "anything", True),  # unknown sentinel -> permissive
        ({"type": None}, "anything", True),  # no type -> permissive
        ({}, "anything", True),  # empty spec (Any) -> permissive
        ({"$ref": "#/$defs/X"}, "anything", True),  # $ref blob -> permissive
        ({"enum": ["a", "b"]}, "a", True),  # enum member
        ({"enum": ["a", "b"]}, "c", False),  # enum non-member -> inexpressible
        # An unmappable non-None token (an engine class) -> permissive.
        ({"type": "SomeClass | None"}, "x", True),
        # A value already recorded as a runtime literal is expressible.
        ({"type": "bool", "runtime_literals": [{"value": "never"}]}, "never", True),
    ],
)
def test_expressible(spec: dict, value: str, expected: bool) -> None:
    assert rl._expressible(spec, value) is expected


# ---------------------------------------------------------------------------
# Corpus extraction
# ---------------------------------------------------------------------------


def test_corpus_candidates_extraction() -> None:
    fields: rl.FieldSpecs = {
        "early_stopping": ("engine_params", {"type": "bool"}),
        "cache_impl": ("engine_params", {"type": "str"}),
    }
    rules = {
        "rules": [
            # not_in with 'never' on a bool field -> candidate; bool comparands
            # (skipped, non-str) and the 'never' str survives.
            {
                "id": "r_membership",
                "match": {
                    "fields": {
                        "transformers.engine_params.early_stopping": {
                            "not_in": [False, True, "never"]
                        }
                    }
                },
            },
            # message-template quoted mention -> candidate, attached to match leaf.
            {
                "id": "r_message",
                "match": {
                    "fields": {"transformers.engine_params.early_stopping": {"present": True}}
                },
                "message_template": "must be a boolean or 'never', but is X.",
            },
            # numeric comparands are skipped.
            {
                "id": "r_numeric",
                "match": {
                    "fields": {"transformers.engine_params.early_stopping": {"not_in": [1, 16]}}
                },
            },
            # @field_ref comparand skipped.
            {
                "id": "r_ref",
                "match": {
                    "fields": {
                        "transformers.engine_params.early_stopping": {
                            "==": "@transformers.sampling_params.x"
                        }
                    }
                },
            },
            # a str literal on a str-typed field is expressible -> not a candidate.
            {
                "id": "r_str_on_str",
                "match": {"fields": {"transformers.engine_params.cache_impl": {"==": "dynamic"}}},
            },
            # type_is_not member names are not comparands and must not be candidates.
            {
                "id": "r_type",
                "match": {
                    "fields": {
                        "transformers.engine_params.early_stopping": {
                            "type_is_not": ["bool", "int", "str"]
                        }
                    }
                },
            },
        ]
    }
    cands = rl.corpus_candidates(rules, fields)
    got = {(c.field, c.value) for c in cands}
    assert got == {("early_stopping", "never")}
    # neither the type_is_not member names nor the numeric / ref / str-on-str
    # values leaked in.
    assert not (got & {("cache_impl", "dynamic")})
    evidence = {e for c in cands for e in c.evidence}
    assert {"rule:r_membership", "rule:r_message"} <= evidence


# ---------------------------------------------------------------------------
# Source-text + docstring scans (pure, no engine import)
# ---------------------------------------------------------------------------


def test_scan_source_text() -> None:
    src = """
class GenerationConfig:
    def validate(self):
        if self.early_stopping not in {None, True, False, "never"}:
            raise ValueError("bad")
        if self.mode == "special":
            pass
        if self.num_beams > 0:
            pass
"""
    fields: rl.FieldSpecs = {
        "early_stopping": ("sampling_params", {"type": "bool"}),
        "mode": ("engine_params", {"type": "int"}),
        "num_beams": ("engine_params", {"type": "int"}),
    }
    label = "transformers/generation/configuration_utils.py"
    cands = rl.scan_source_text(src, label, fields)
    got = {(c.field, c.value) for c in cands}
    # Membership set members and the == right-hand side are both collected; the
    # ordering (>) comparison is ignored.
    assert got == {("early_stopping", "never"), ("mode", "special")}
    assert all(e.startswith(f"src:{label}:") for c in cands for e in c.evidence)


def test_scan_docstring_block_scoping() -> None:
    doc = """Summary line.

Args:
    early_stopping (`bool` or `str`, *optional*):
        Set to the value "never" to keep all beams alive.
    other_field (`str`, *optional*):
        Accepts "foo" as a special mode.
"""
    # other_field is deliberately NOT in the field map: its block must not leak.
    fields: rl.FieldSpecs = {"early_stopping": ("sampling_params", {"type": "bool"})}
    cands = rl.scan_docstring(doc, "GenerationConfig", fields)
    got = {(c.field, c.value) for c in cands}
    assert got == {("early_stopping", "never")}
    assert [e for c in cands for e in c.evidence] == ["doc:GenerationConfig.early_stopping"]


# ---------------------------------------------------------------------------
# LLM proposals reader
# ---------------------------------------------------------------------------


def test_llm_candidates_reader() -> None:
    fields: rl.FieldSpecs = {"early_stopping": ("sampling_params", {"type": "bool"})}
    proposals = {
        "candidates": [
            {
                "field": "early_stopping",
                "value": "never",
                "citation": {"file": "gen.py", "line": 42},
            },
            {"field": "early_stopping", "value": 123, "citation": {}},  # non-str value skipped
            {
                "field": "unknown",
                "value": "x",
                "citation": {"file": "a", "line": 1},
            },  # unknown field
        ]
    }
    cands = rl.llm_candidates(proposals, fields)
    assert [(c.field, c.value) for c in cands] == [("early_stopping", "never")]
    assert cands[0].evidence == ("llm:gen.py:42",)


# ---------------------------------------------------------------------------
# Two-leg probe matrix
# ---------------------------------------------------------------------------


def _candidate(value: str) -> rl.LiteralCandidate:
    return rl.LiteralCandidate(field="f", value=value, evidence=("rule:x",))


def test_probe_two_leg_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    fields: rl.FieldSpecs = {"f": ("sampling_params", {"type": "bool"})}

    class Dummy:
        pass

    monkeypatch.setattr(rl.ec, "candidate_classes", lambda engine, path, leaves: [Dummy])
    monkeypatch.setattr(rl.ec, "accepts", lambda cls, leaf: True)

    def construct_value_only(engine: str, cls: type, kwargs: dict) -> None:
        if rl._SENTINEL in kwargs.values():
            raise ValueError("sentinel rejected")

    verified, rejected, undiscriminating, errors = rl.probe_candidates_fn(
        "transformers", [_candidate("never")], fields, construct=construct_value_only
    )
    assert [c.value for c in verified] == ["never"]
    assert not rejected and not undiscriminating and not errors

    def construct_accept_all(engine: str, cls: type, kwargs: dict) -> None:
        return None

    verified, rejected, undiscriminating, errors = rl.probe_candidates_fn(
        "transformers", [_candidate("never")], fields, construct=construct_accept_all
    )
    assert [c.value for c in undiscriminating] == ["never"]
    assert not verified and not rejected and not errors

    def construct_reject(engine: str, cls: type, kwargs: dict) -> None:
        raise ValueError("leg1-detail-snippet")

    verified, rejected, undiscriminating, errors = rl.probe_candidates_fn(
        "transformers", [_candidate("never")], fields, construct=construct_reject
    )
    assert [c.value for c, _ in rejected] == ["never"]
    assert "leg1-detail-snippet" in rejected[0][1]
    assert not verified and not undiscriminating and not errors

    def boom(engine: str, path: str, leaves: list) -> list:
        raise rl.ec.ConstructorResolutionError("no class here")

    monkeypatch.setattr(rl.ec, "candidate_classes", boom)
    verified, rejected, undiscriminating, errors = rl.probe_candidates_fn(
        "transformers", [_candidate("never")], fields, construct=construct_accept_all
    )
    assert [c.value for c, _ in errors] == ["never"]
    assert not verified and not rejected and not undiscriminating


# ---------------------------------------------------------------------------
# Merge determinism + shape
# ---------------------------------------------------------------------------


def _base_envelope() -> dict:
    return {
        "engine_version": "5.7.0",
        "engine_params": {"num_beams": {"type": "int", "default": None}},
        "sampling_params": {"early_stopping": {"type": "bool", "default": None}},
    }


def test_merge_entry_shape_and_key_order() -> None:
    verified = [rl.LiteralCandidate("early_stopping", "never", ("src:z:1", "rule:a"))]
    env = _base_envelope()
    rl.merge_runtime_literals(env, verified, "5.7.0")
    entries = env["sampling_params"]["early_stopping"]["runtime_literals"]
    assert entries == [
        {
            "value": "never",
            "verified": "construction",
            "pin": "5.7.0",
            "evidence": ["rule:a", "src:z:1"],  # sorted
        }
    ]
    # key order value, verified, pin, evidence
    assert list(entries[0].keys()) == ["value", "verified", "pin", "evidence"]


def test_merge_shuffled_order_is_byte_identical() -> None:
    cands = [
        rl.LiteralCandidate("early_stopping", "never", ("rule:a",)),
        rl.LiteralCandidate("early_stopping", "always", ("rule:b",)),
    ]
    env_a = _base_envelope()
    rl.merge_runtime_literals(env_a, cands, "5.7.0")
    env_b = _base_envelope()
    rl.merge_runtime_literals(env_b, list(reversed(cands)), "5.7.0")
    assert json.dumps(env_a, sort_keys=False) == json.dumps(env_b, sort_keys=False)


def test_zero_verified_leaves_envelope_byte_identical() -> None:
    env = _base_envelope()
    before = json.dumps(env, sort_keys=False)
    rl.merge_runtime_literals(env, [], "5.7.0")
    assert json.dumps(env, sort_keys=False) == before


# ---------------------------------------------------------------------------
# Auto-narrow surfacing
# ---------------------------------------------------------------------------


def test_narrowing_line_for_dropped_literal() -> None:
    previous = _base_envelope()
    previous["sampling_params"]["early_stopping"]["runtime_literals"] = [
        {"value": "never", "verified": "construction", "pin": "5.6.0", "evidence": ["rule:a"]}
    ]
    new = _base_envelope()  # this run verified nothing
    lines = rl.narrowing_lines(previous, new)
    assert len(lines) == 1
    assert lines[0].startswith("NARROWED: early_stopping literal 'never'")
    assert "auto-narrow" in lines[0]


# ---------------------------------------------------------------------------
# run_stage end-to-end
# ---------------------------------------------------------------------------


def test_run_stage_records_literal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules_dir = tmp_path / "src/llenergymeasure/engines/transformers"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yaml").write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "id": "transformers_raises_early_stopping_not_in_set",
                        "engine": "transformers",
                        "match": {
                            "fields": {
                                "transformers.engine_params.early_stopping": {
                                    "not_in": [None, True, False, "never"]
                                }
                            }
                        },
                        "message_template": "must be a boolean or 'never'",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    envelope: dict[str, Any] = {
        "engine_version": "5.7.0",
        "engine_params": {},
        "sampling_params": {"early_stopping": {"type": "bool", "default": None}},
    }

    class Dummy:
        pass

    monkeypatch.setattr(rl, "source_scan_candidates", lambda engine, fields: [])
    monkeypatch.setattr(rl.ec, "candidate_classes", lambda engine, path, leaves: [Dummy])
    monkeypatch.setattr(rl.ec, "accepts", lambda cls, leaf: True)

    def construct(engine: str, cls: type, kwargs: dict) -> None:
        if rl._SENTINEL in kwargs.values():
            raise ValueError("sentinel rejected")

    monkeypatch.setattr(rl, "_construct", construct)

    report = rl.run_stage("transformers", envelope, tmp_path, None)

    assert envelope["sampling_params"]["early_stopping"]["runtime_literals"] == [
        {
            "value": "never",
            "verified": "construction",
            "pin": "5.7.0",
            "evidence": ["rule:transformers_raises_early_stopping_not_in_set"],
        }
    ]
    joined = "\n".join(report.lines)
    # The rule contributes a not_in comparand AND a message-template mention (two
    # raw corpus candidates) that pool to one unique (field, value).
    assert (
        "runtime-literals: candidates corpus=2 source=0 doc=0 llm=0 previous=0 "
        "-> 1 unique (field,value)" in joined
    )
    assert "runtime-literals: verified=1 rejected=0 undiscriminating=0 errors=0" in joined
    assert "runtime-literals: RECORDED sampling_params.early_stopping 'never'" in joined
    assert "runtime-literals: census after merge: 0 corpus literal(s) inexpressible" in joined


# ---------------------------------------------------------------------------
# Standing census check
# ---------------------------------------------------------------------------


# This test pins the corpus/type consistency census at zero for every engine. It
# WILL FAIL for transformers until the caller regenerates the shipped artifacts
# in the pinned container (early_stopping/'never' is the known finding, resolved
# once the schema records the verified runtime literal).
@pytest.mark.parametrize("engine", list(_outputs.ENGINES))
def test_census_zero_for_shipped_artifacts(engine: str) -> None:
    base = REPO_ROOT / "src/llenergymeasure/engines" / engine
    schema = json.loads((base / "schema.discovered.json").read_text(encoding="utf-8"))
    rules = yaml.safe_load((base / "rules.yaml").read_text(encoding="utf-8")) or {}
    assert rl.census(schema, rules) == []
