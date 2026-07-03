"""Tests for scripts/diff_schema_surface.py.

Unit tests exercise the pure diff functions on small synthetic schemas; the
real-data tests assert known deltas across the committed vLLM and TensorRT
version-pair snapshots.
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

from engine_versions import _outputs  # noqa: E402
from scripts import diff_schema_surface as dss  # noqa: E402


def _schema(
    *,
    engine: str = "vllm",
    engine_version: str = "0.0.0",
    engine_params: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
    defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "engine": engine,
        "engine_version": engine_version,
        "engine_params": engine_params or {},
        "sampling_params": sampling_params or {},
    }
    if defs is not None:
        schema["$defs"] = defs
    return schema


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestTypeToken:
    def test_explicit_type(self) -> None:
        assert dss._type_token({"type": "int | None"}) == "int | None"

    def test_ref(self) -> None:
        assert dss._type_token({"$ref": "#/$defs/Foo"}) == "#/$defs/Foo"

    def test_any_of_union(self) -> None:
        token = dss._type_token({"anyOf": [{"type": "string"}, {"type": "null"}]})
        assert token == "string | null"

    def test_no_type(self) -> None:
        assert dss._type_token({"default": 1}) is None


class TestEnumMembers:
    def test_present(self) -> None:
        assert dss._enum_members({"enum": ["a", "b"]}) == ("a", "b")

    def test_absent(self) -> None:
        assert dss._enum_members({"type": "str"}) is None


# ---------------------------------------------------------------------------
# Field-shape diff
# ---------------------------------------------------------------------------


def _diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    return dss._diff_shapes(dss._normalise(old), dss._normalise(new))


class TestDiffShapes:
    def test_identical(self) -> None:
        assert _diff({"type": "int", "default": 0}, {"type": "int", "default": 0}) == {}

    def test_type_change(self) -> None:
        delta = _diff({"type": "int | None"}, {"type": "int"})
        assert delta["type"] == {"old": "int | None", "new": "int"}

    def test_default_change(self) -> None:
        delta = _diff({"type": "float", "default": 0.9}, {"type": "float", "default": 0.95})
        assert delta["default"] == {"old": 0.9, "new": 0.95}

    def test_default_absent_vs_none(self) -> None:
        # A field gaining an explicit ``default: null`` is a change distinct
        # from having had no default at all.
        delta = _diff({"type": "int"}, {"type": "int", "default": None})
        assert delta["default"] == {"old": dss._ABSENT, "new": None}

    def test_bounds_added(self) -> None:
        delta = _diff({"type": "int"}, {"type": "int", "minimum": 0})
        assert delta["bounds"] == {"old": {}, "new": {"minimum": 0}}

    def test_enum_members_added(self) -> None:
        delta = _diff({"enum": ["a"]}, {"enum": ["a", "b"]})
        assert delta["enum"] == {"added": ["b"], "removed": []}

    def test_enum_introduced(self) -> None:
        delta = _diff({"type": "str"}, {"type": "str", "enum": ["a", "b"]})
        assert delta["enum"] == {"added": ["a", "b"], "removed": []}

    def test_enum_reorder_is_not_a_change(self) -> None:
        # Membership, not order: a reordered enum with the same members is inert.
        assert "enum" not in _diff({"enum": ["a", "b"]}, {"enum": ["b", "a"]})

    def test_unhashable_enum_members(self) -> None:
        # vLLM CUDAGraphMode carries list-valued members; containment, not sets.
        delta = _diff({"enum": [0, 1]}, {"enum": [0, 1, [2, 0]]})
        assert delta["enum"] == {"added": [[2, 0]], "removed": []}


# ---------------------------------------------------------------------------
# Field-map and $defs diff
# ---------------------------------------------------------------------------


class TestFieldMap:
    def test_added_removed_changed_sorted(self) -> None:
        diff = dss._diff_field_map(
            {"keep": {"type": "int"}, "gone": {"type": "str"}, "morph": {"type": "int"}},
            {"keep": {"type": "int"}, "morph": {"type": "int | None"}, "fresh": {"type": "bool"}},
        )
        assert diff["added"] == {"fresh": "bool"}
        assert diff["removed"] == {"gone": "str"}
        assert list(diff["changed"]) == ["morph"]

    def test_non_dict_entries_ignored(self) -> None:
        # Malformed (non-dict) entries are skipped, never crash the diff.
        diff = dss._diff_field_map({"x": "not-a-dict"}, {"x": "still-not"})
        assert diff["added"] == {} and diff["removed"] == {} and diff["changed"] == {}


class TestDefs:
    def test_class_added_and_removed(self) -> None:
        diff = dss._diff_defs({"Old": {"enum": [1]}}, {"New": {"enum": [1]}})
        assert diff["added"] == ["New"]
        assert diff["removed"] == ["Old"]

    def test_enum_class_membership(self) -> None:
        diff = dss._diff_defs({"Mode": {"enum": [0, 1]}}, {"Mode": {"enum": [0, 1, 2]}})
        assert diff["changed"]["Mode"]["enum"] == {"added": [2], "removed": []}

    def test_object_class_property_diff(self) -> None:
        diff = dss._diff_defs(
            {"Cfg": {"type": "object", "properties": {"a": {"type": "int"}}}},
            {"Cfg": {"type": "object", "properties": {"a": {"type": "int"}, "b": {"type": "str"}}}},
        )
        assert diff["changed"]["Cfg"]["properties"]["added"] == {"b": "str"}


# ---------------------------------------------------------------------------
# Report + rendering
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_no_changes(self) -> None:
        schema = _schema(engine_params={"model": {"type": "str"}})
        report = dss.build_report(schema, schema)
        assert report["changed"] is False
        assert report["engine"] == "vllm"

    def test_change_flag_set(self) -> None:
        report = dss.build_report(
            _schema(engine_params={"model": {"type": "str"}}),
            _schema(engine_params={"model": {"type": "str"}, "extra": {"type": "int"}}),
        )
        assert report["changed"] is True
        assert "extra" in report["sections"]["engine_params"]["added"]

    def test_version_metadata(self) -> None:
        report = dss.build_report(_schema(engine_version="0.7.3"), _schema(engine_version="0.19.1"))
        assert report["old_version"] == "0.7.3"
        assert report["new_version"] == "0.19.1"


class TestRenderText:
    def test_no_change_message(self) -> None:
        schema = _schema(engine_params={"model": {"type": "str"}})
        assert "No surface changes." in dss.render_text(dss.build_report(schema, schema))

    def test_change_markers(self) -> None:
        report = dss.build_report(
            _schema(engine_params={"gone": {"type": "str"}, "morph": {"type": "int"}}),
            _schema(engine_params={"fresh": {"type": "bool"}, "morph": {"type": "int | None"}}),
        )
        text = dss.render_text(report)
        assert "+ fresh: bool" in text
        assert "- gone" in text
        assert "~ morph" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_no_change_exit_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = dss.main(["--engine", "vllm", "--old", "0.7.3", "--new", "0.7.3"])
        assert code == 0
        assert "No surface changes." in capsys.readouterr().out

    def test_changes_exit_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = dss.main(["--engine", "vllm", "--old", "0.7.3", "--new", "0.19.1"])
        assert code == 1

    def test_json_format_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = dss.main(
            ["--engine", "vllm", "--old", "0.7.3", "--new", "0.19.1", "--format", "json"]
        )
        assert code == 1
        report = json.loads(capsys.readouterr().out)
        assert report["engine"] == "vllm"
        assert report["changed"] is True

    def test_unknown_version_exit_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = dss.main(["--engine", "vllm", "--old", "9.9.9", "--new", "0.7.3"])
        assert code == 2
        assert "no discovered schema" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Real-data deltas (committed workspace snapshots)
# ---------------------------------------------------------------------------


def _load_real(engine: str, version: str) -> dict[str, Any]:
    path = _outputs.schema_path(engine, version)
    if not path.is_file():
        pytest.skip(f"snapshot missing: {path}")
    return json.loads(path.read_text())


class TestRealDataVllm:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, Any]:
        return dss.build_report(_load_real("vllm", "0.7.3"), _load_real("vllm", "0.19.1"))

    def test_engine_params_added(self, report: dict[str, Any]) -> None:
        added = report["sections"]["engine_params"]["added"]
        assert "async_scheduling" in added
        assert "attention_backend" in added

    def test_engine_params_removed(self, report: dict[str, Any]) -> None:
        removed = report["sections"]["engine_params"]["removed"]
        assert "device" in removed
        assert "guided_decoding_backend" in removed

    def test_dtype_gained_enum(self, report: dict[str, Any]) -> None:
        dtype = report["sections"]["engine_params"]["changed"]["dtype"]
        assert "float16" in dtype["enum"]["added"]
        assert dtype["enum"]["removed"] == []

    def test_cpu_offload_gb_gained_bound(self, report: dict[str, Any]) -> None:
        bounds = report["sections"]["engine_params"]["changed"]["cpu_offload_gb"]["bounds"]
        assert bounds["new"] == {"minimum": 0}

    def test_sampling_params_delta(self, report: dict[str, Any]) -> None:
        sampling = report["sections"]["sampling_params"]
        assert "structured_outputs" in sampling["added"]
        assert "best_of" in sampling["removed"]

    def test_defs_delta(self, report: dict[str, Any]) -> None:
        assert "AttentionConfig" in report["defs"]["added"]
        assert "GuidedDecodingParams" in report["defs"]["removed"]

    def test_changed_flag(self, report: dict[str, Any]) -> None:
        assert report["changed"] is True


class TestRealDataTensorrt:
    @pytest.fixture(scope="class")
    def report(self) -> dict[str, Any]:
        return dss.build_report(_load_real("tensorrt", "0.21.0"), _load_real("tensorrt", "1.0.0"))

    def test_engine_params_delta(self, report: dict[str, Any]) -> None:
        engine_params = report["sections"]["engine_params"]
        assert "fail_fast_on_attention_window_too_large" in engine_params["added"]
        assert "max_loras" in engine_params["removed"]

    def test_sampling_params_stable(self, report: dict[str, Any]) -> None:
        sampling = report["sections"]["sampling_params"]
        assert sampling["added"] == {}
        assert sampling["removed"] == {}
