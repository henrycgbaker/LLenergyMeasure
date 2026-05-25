"""Smoke tests for agentic_tool_harness.

Covers:

1. parse_tool_call: well-formed JSON; fenced; with prose; malformed.
2. tool_read_file: sandbox enforcement; line range clamping; MAX_READ_LINES cap.
3. tool_list_validators: AST class discovery; method-name filtering.
4. tool_run_miner: returns count + namespaces + ids from canonical reference.
5. tool_score_against: scores emitted YAML against active reference; recall=1.0
   when emitted == reference.
6. AgenticLoop: end-to-end with StubLLM; finalised flow; max_calls budget;
   parse-error recovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root sits four parents above research/mining-substrate-trial/scripts/strategies/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from strategies.agentic_tool_harness import (  # noqa: E402
    MAX_READ_LINES,
    AgenticLoop,
    ReadSandbox,
    StubLLM,
    build_default_sandbox,
    parse_tool_call,
    tool_list_validators,
    tool_read_file,
    tool_run_miner,
    tool_score_against,
)

# ---------------------------------------------------------------------------
# 1. parse_tool_call
# ---------------------------------------------------------------------------


def test_parse_well_formed_object():
    raw = '{"tool": "read_file", "args": {"path": "config.py"}}'
    parsed, err = parse_tool_call(raw)
    assert parsed is not None
    assert err == ""
    assert parsed["tool"] == "read_file"
    assert parsed["args"]["path"] == "config.py"


def test_parse_with_code_fence():
    raw = (
        '```json\n{"tool": "run_miner", "args": {"engine": "vllm", "version_slug": "v0_7_3"}}\n```'
    )
    parsed, err = parse_tool_call(raw)
    assert parsed is not None
    assert err == ""
    assert parsed["tool"] == "run_miner"


def test_parse_with_leading_prose():
    raw = (
        "Sure! Let me start by exploring the engine source.\n"
        '{"tool": "list_validators", "args": {"engine": "vllm", "class_name": "ModelConfig"}}'
    )
    parsed, _err = parse_tool_call(raw)
    assert parsed is not None
    assert parsed["tool"] == "list_validators"


def test_parse_malformed():
    raw = "I don't want to call any tools."
    parsed, err = parse_tool_call(raw)
    assert parsed is None
    assert "no JSON object" in err


def test_parse_unbalanced_braces():
    raw = '{"tool": "read_file", "args": {"path": "x"'
    parsed, err = parse_tool_call(raw)
    assert parsed is None
    assert "unbalanced braces" in err.lower() or "json parse" in err.lower()


def test_parse_missing_tool_field():
    raw = '{"args": {"path": "x"}}'
    parsed, err = parse_tool_call(raw)
    assert parsed is None
    assert "tool" in err


def test_parse_empty_args_ok():
    raw = '{"tool": "finalise"}'
    parsed, _err = parse_tool_call(raw)
    assert parsed is not None
    assert parsed["args"] == {}


# ---------------------------------------------------------------------------
# 2. tool_read_file
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_sandbox(tmp_path):
    """Build a sandbox over a small tmp tree with a multi-line file."""
    root = tmp_path / "src"
    root.mkdir()
    f = root / "example.py"
    lines = [f"# line {i}" for i in range(1, 250)]
    f.write_text("\n".join(lines) + "\n")
    return ReadSandbox(roots=[root])


def test_read_file_within_sandbox(tmp_sandbox):
    out = tool_read_file(
        {"path": "example.py", "line_start": 0, "line_end": 5}, sandbox=tmp_sandbox
    )
    assert "line 1" in out
    assert "line 5" in out
    # Past line 5 (i.e. line 6) should NOT be in the slice
    assert "line 6" not in out


def test_read_file_clamps_to_max_lines(tmp_sandbox):
    out = tool_read_file(
        {"path": "example.py", "line_start": 0, "line_end": 500}, sandbox=tmp_sandbox
    )
    # File has 249 lines; line_end clamps to 249. That's under MAX_READ_LINES (200).
    # Actually MAX_READ_LINES=200, so span clamps to 200.
    rendered_lines = out.split("\n")
    # Count lines with "# line " (skip the TRUNCATED note)
    content_lines = [line for line in rendered_lines if "# line " in line]
    assert len(content_lines) <= MAX_READ_LINES


def test_read_file_blocks_traversal(tmp_sandbox):
    with pytest.raises(PermissionError):
        tool_read_file({"path": "../etc/passwd"}, sandbox=tmp_sandbox)


def test_read_file_blocks_absolute_outside_sandbox(tmp_sandbox):
    with pytest.raises(PermissionError):
        tool_read_file({"path": "/etc/passwd"}, sandbox=tmp_sandbox)


def test_read_file_missing_path_arg(tmp_sandbox):
    with pytest.raises(ValueError):
        tool_read_file({}, sandbox=tmp_sandbox)


def test_read_file_negative_line_start(tmp_sandbox):
    """Negative line_start counts from the end (e.g. -10 = last 10 lines)."""
    out = tool_read_file(
        {"path": "example.py", "line_start": -5, "line_end": -1}, sandbox=tmp_sandbox
    )
    # When line_end == -1 we treat it as "to end"
    assert "line 245" in out or "line 249" in out


# ---------------------------------------------------------------------------
# 3. tool_list_validators
# ---------------------------------------------------------------------------


def test_list_validators_ast_discovery(tmp_path):
    root = tmp_path / "engine_src"
    root.mkdir()
    (root / "config.py").write_text(
        "class ModelConfig:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def __post_init__(self):\n"
        "        pass\n"
        "    def _verify_dtype(self):\n"
        "        pass\n"
        "    def _verify_quantization(self):\n"
        "        pass\n"
        "    def other_method(self):\n"
        "        pass\n"
        "\n"
        "class OtherClass:\n"
        "    def _verify_unrelated(self):\n"
        "        pass\n"
    )
    sandbox = ReadSandbox(roots=[root])
    out = tool_list_validators({"engine": "vllm", "class_name": "ModelConfig"}, sandbox=sandbox)
    assert "__post_init__" in out
    assert "_verify_dtype" in out
    assert "_verify_quantization" in out
    assert "other_method" not in out
    assert "_verify_unrelated" not in out


def test_list_validators_missing_class(tmp_path):
    root = tmp_path / "engine_src"
    root.mkdir()
    (root / "config.py").write_text("class Foo:\n    pass\n")
    sandbox = ReadSandbox(roots=[root])
    out = tool_list_validators({"engine": "vllm", "class_name": "Nonexistent"}, sandbox=sandbox)
    assert out == []


# ---------------------------------------------------------------------------
# 4. tool_run_miner (uses real project tree)
# ---------------------------------------------------------------------------


def test_run_miner_returns_summary():
    out = tool_run_miner(
        {"engine": "transformers", "version_slug": "v4_57_3"},
        project_root=_PROJECT_ROOT,
    )
    assert "count" in out
    assert isinstance(out["count"], int)
    assert out["count"] > 0
    assert "namespaces" in out
    assert "invariant_ids" in out
    assert len(out["invariant_ids"]) == out["count"]


def test_run_miner_unknown_engine_raises():
    with pytest.raises(FileNotFoundError):
        tool_run_miner(
            {"engine": "fake", "version_slug": "v0"},
            project_root=_PROJECT_ROOT,
        )


# ---------------------------------------------------------------------------
# 5. tool_score_against
# ---------------------------------------------------------------------------


def test_score_against_self_yields_recall_one():
    """Score the canonical reference against itself; expect recall=1.0."""
    reference_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / "transformers"
        / "v4_57_3"
        / "outputs"
        / "invariants.proposed.yaml"
    )
    emitted_yaml = reference_path.read_text()
    out = tool_score_against(
        {
            "emitted_yaml": emitted_yaml,
            "engine": "transformers",
            "version_slug": "v4_57_3",
        },
        project_root=_PROJECT_ROOT,
    )
    assert out["recall"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(1.0)
    assert out["intersection_count"] > 0


def test_score_against_empty_emitted_yields_zero():
    out = tool_score_against(
        {
            "emitted_yaml": "schema_version: 1.0.0\ninvariants: []\n",
            "engine": "transformers",
            "version_slug": "v4_57_3",
        },
        project_root=_PROJECT_ROOT,
    )
    assert out["recall"] == 0.0
    assert out["ref_count"] > 0
    assert out["cell_count"] == 0


def test_score_against_invalid_yaml_raises():
    with pytest.raises(ValueError):
        tool_score_against(
            {
                "emitted_yaml": "not: valid: yaml: : :",
                "engine": "transformers",
                "version_slug": "v4_57_3",
            },
            project_root=_PROJECT_ROOT,
        )


# ---------------------------------------------------------------------------
# 6. AgenticLoop end-to-end
# ---------------------------------------------------------------------------


def test_loop_finalises_immediately(tmp_path):
    """Stub LLM immediately calls finalise; loop ends with finalised=True."""
    sandbox = ReadSandbox(roots=[tmp_path])
    yaml_payload = "schema_version: 1.0.0\ninvariants: []\n"
    llm = StubLLM(
        responses=[
            json.dumps({"tool": "finalise", "args": {"invariants_yaml": yaml_payload}}),
        ]
    )
    loop = AgenticLoop(
        llm=llm,
        sandbox=sandbox,
        project_root=_PROJECT_ROOT,
        max_tool_calls=10,
    )
    result = loop.run(engine="transformers", version_slug="v4_57_3")
    assert result.finalised is True
    assert result.stop_reason == "finalised"
    assert result.final_yaml == yaml_payload
    assert result.total_tool_calls == 1


def test_loop_max_calls_budget(tmp_path):
    """LLM keeps calling read_file; loop stops at max_tool_calls budget."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "x.py").write_text("x = 1\n")
    sandbox = ReadSandbox(roots=[src_root])
    llm = StubLLM(responses=[json.dumps({"tool": "read_file", "args": {"path": "x.py"}})] * 10)
    loop = AgenticLoop(
        llm=llm,
        sandbox=sandbox,
        project_root=_PROJECT_ROOT,
        max_tool_calls=3,
    )
    result = loop.run(engine="transformers", version_slug="v4_57_3")
    assert result.finalised is False
    assert result.stop_reason == "max_calls"
    assert result.total_tool_calls == 3


def test_loop_parse_error_counts_toward_budget(tmp_path):
    """LLM emits malformed responses; parse errors count as turns."""
    sandbox = ReadSandbox(roots=[tmp_path])
    llm = StubLLM(
        responses=[
            "I won't call a tool.",
            "Still not a JSON object.",
            json.dumps({"tool": "finalise", "args": {"invariants_yaml": "invariants: []"}}),
        ]
    )
    loop = AgenticLoop(
        llm=llm,
        sandbox=sandbox,
        project_root=_PROJECT_ROOT,
        max_tool_calls=10,
    )
    result = loop.run(engine="transformers", version_slug="v4_57_3")
    assert result.finalised is True
    # 2 parse errors + 1 finalise = 3 turns
    assert len(result.turns) == 3
    assert result.turns[0].parsed_call is None
    assert result.turns[1].parsed_call is None
    assert result.turns[2].parsed_call is not None


def test_loop_unknown_tool_returns_error_result(tmp_path):
    sandbox = ReadSandbox(roots=[tmp_path])
    llm = StubLLM(
        responses=[
            json.dumps({"tool": "nonexistent_tool", "args": {}}),
            json.dumps({"tool": "finalise", "args": {"invariants_yaml": "invariants: []"}}),
        ]
    )
    loop = AgenticLoop(
        llm=llm,
        sandbox=sandbox,
        project_root=_PROJECT_ROOT,
        max_tool_calls=10,
    )
    result = loop.run(engine="transformers", version_slug="v4_57_3")
    assert result.finalised is True
    # First turn's tool_result should reflect the unknown-tool error
    assert result.turns[0].tool_result is not None
    assert result.turns[0].tool_result.ok is False
    assert "unknown tool" in result.turns[0].tool_result.error


def test_build_default_sandbox_transformers():
    sb = build_default_sandbox(
        project_root=_PROJECT_ROOT,
        engine="transformers",
        version_slug="v4_57_3",
    )
    # Transformers sandbox should include engine_versions/transformers/v4_57_3 at minimum.
    paths = [str(r) for r in sb.roots]
    assert any("engine_versions/transformers/v4_57_3" in p for p in paths)
