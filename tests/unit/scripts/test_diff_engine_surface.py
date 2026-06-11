"""Tests for :mod:`scripts.diff_engine_surface` - the surface-trend census.

Pure artefact diff: no engine imports. Fixtures are small synthetic
``outputs/`` directories written to ``tmp_path``. The census must stay at
encoding-agnostic grain (field + coarse predicate bucket), so the tests assert
on enum/bound *presence* counts, never on which members or values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import diff_engine_surface as des  # noqa: E402


def _write_pin(
    base: Path,
    *,
    name: str,
    engine: str = "vllm",
    engine_version: str = "0.0.0",
    engine_params: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
    defs: dict[str, Any] | None = None,
    invariants: list[dict[str, Any]] | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a synthetic pin outputs/ directory and return its path."""
    pin = base / name
    pin.mkdir(parents=True, exist_ok=True)
    schema: dict[str, Any] = {
        "engine": engine,
        "engine_version": engine_version,
        "engine_params": engine_params or {},
        "sampling_params": sampling_params or {},
    }
    if defs is not None:
        schema["$defs"] = defs
    (pin / "schema.discovered.json").write_text(json.dumps(schema))
    if invariants is not None:
        (pin / "invariants.proposed.yaml").write_text(
            yaml.safe_dump({"engine": engine, "invariants": invariants})
        )
    if cases is not None:
        (pin / "invariants.validated.yaml").write_text(
            yaml.safe_dump({"engine": engine, "cases": cases})
        )
    return pin


class TestFieldPredicateDetection:
    def test_enum_via_explicit_key(self) -> None:
        assert des._field_carries_enum({"enum": ["a", "b"]}) is True

    def test_enum_via_literal_type_string(self) -> None:
        # Encoding-agnostic: a Literal-typed field counts as enum-carrying even
        # without an explicit enum key (the committed-schema encoding).
        assert des._field_carries_enum({"type": "Literal['x', 'y']"}) is True

    def test_no_enum(self) -> None:
        assert des._field_carries_enum({"type": "int | None"}) is False

    def test_empty_enum_list_not_counted(self) -> None:
        assert des._field_carries_enum({"enum": []}) is False

    def test_bound_via_ge(self) -> None:
        assert des._field_carries_bound({"ge": 0}) is True

    def test_bound_via_constraints(self) -> None:
        assert des._field_carries_bound({"constraints": {"minimum": 0}}) is True

    def test_no_bound(self) -> None:
        assert des._field_carries_bound({"type": "int"}) is False


class TestCensusSchema:
    def test_counts_fields_enums_bounds_per_section(self) -> None:
        schema = {
            "engine_params": {
                "a": {"type": "int", "ge": 0},
                "b": {"type": "Literal['x','y']"},
                "c": {"type": "str"},
            },
            "sampling_params": {"t": {"type": "float", "le": 1.0}},
            "$defs": {"CacheConfig": {}, "LoraConfig": {}},
        }
        sections, defs = des.census_schema(schema)
        assert sections["engine_params"].fields == 3
        assert sections["engine_params"].enum_fields == 1
        assert sections["engine_params"].bound_fields == 1
        assert sections["sampling_params"].bound_fields == 1
        assert defs == ["CacheConfig", "LoraConfig"]


class TestCensusProposed:
    def test_severity_and_provenance_buckets(self) -> None:
        corpus = {
            "invariants": [
                {"severity": "error", "miner_source": {"method": "validate"}},
                {"severity": "error", "miner_source": {"method": "_verify_args"}},
                {"severity": "dormant", "miner_source": {"method": "validate"}},
                {"severity": "dormant"},  # no miner_source -> unknown provenance
            ]
        }
        by_sev, by_prov = des.census_proposed(corpus)
        assert by_sev == {"error": 2, "dormant": 2}
        assert by_prov == {"validate": 2, "_verify_args": 1, "unknown": 1}


class TestDiffSurface:
    def test_growth_no_shrinkage_flags(self, tmp_path: Path) -> None:
        old = _write_pin(
            tmp_path,
            name="old",
            engine_version="0.19.1",
            engine_params={"a": {"type": "int"}},
            invariants=[{"severity": "error", "miner_source": {"method": "validate"}}],
        )
        new = _write_pin(
            tmp_path,
            name="new",
            engine_version="0.20.0",
            engine_params={"a": {"type": "int"}, "b": {"type": "Literal['x']"}},
            invariants=[
                {"severity": "error", "miner_source": {"method": "validate"}},
                {"severity": "error", "miner_source": {"method": "_verify_args"}},
            ],
        )
        report = des.build_report(old, new)
        assert report["engine"] == "vllm"
        assert report["old_version"] == "0.19.1"
        assert report["new_version"] == "0.20.0"
        assert report["schema"]["engine_params"]["fields"] == {"old": 1, "new": 2, "delta": 1}
        assert report["schema"]["engine_params"]["enum_fields"]["new"] == 1
        assert report["shrinkage_flags"] == []

    def test_shrinkage_flagged(self, tmp_path: Path) -> None:
        # The cliff class: the substrate emits FEWER fields/invariants after a
        # bump - a walker-regression candidate.
        old = _write_pin(
            tmp_path,
            name="old",
            engine_params={"a": {"type": "int"}, "b": {"type": "int"}},
            invariants=[
                {"severity": "error", "miner_source": {"method": "validate"}},
                {"severity": "error", "miner_source": {"method": "validate"}},
            ],
        )
        new = _write_pin(
            tmp_path,
            name="new",
            engine_params={"a": {"type": "int"}},
            invariants=[{"severity": "error", "miner_source": {"method": "validate"}}],
        )
        report = des.build_report(old, new)
        flags = report["shrinkage_flags"]
        assert any("engine_params.fields shrank 2 -> 1" in f for f in flags)
        assert any("invariants total shrank 2 -> 1" in f for f in flags)

    def test_defs_class_inventory_delta(self, tmp_path: Path) -> None:
        old = _write_pin(tmp_path, name="old", defs={"CacheConfig": {}, "OldConfig": {}})
        new = _write_pin(tmp_path, name="new", defs={"CacheConfig": {}, "NewConfig": {}})
        report = des.build_report(old, new)
        assert report["defs_classes"]["added"] == ["NewConfig"]
        assert report["defs_classes"]["removed"] == ["OldConfig"]

    def test_report_shape_stable(self, tmp_path: Path) -> None:
        old = _write_pin(tmp_path, name="old", invariants=[], cases=[])
        new = _write_pin(tmp_path, name="new", invariants=[], cases=[])
        report = des.build_report(old, new)
        assert set(report) == {
            "engine",
            "old_version",
            "new_version",
            "schema",
            "defs_classes",
            "invariants_by_severity",
            "invariants_by_provenance",
            "validated_cases",
            "validated_by_outcome",
            "shrinkage_flags",
        }

    def test_missing_validated_corpus_tolerated(self, tmp_path: Path) -> None:
        # vllm pre-#445 ships proposed-only - a missing validated.yaml must not
        # error (only a missing directory does).
        old = _write_pin(tmp_path, name="old", invariants=[])
        new = _write_pin(tmp_path, name="new", invariants=[])
        report = des.build_report(old, new)
        assert report["validated_cases"] == {"old": 0, "new": 0, "delta": 0}


class TestCli:
    def test_missing_directory_exits_2(self, tmp_path: Path) -> None:
        exit_code = des.main([str(tmp_path / "nope"), str(tmp_path / "also-nope")])
        assert exit_code == 2

    def test_writes_json_report(self, tmp_path: Path) -> None:
        old = _write_pin(tmp_path, name="old", invariants=[])
        new = _write_pin(tmp_path, name="new", invariants=[])
        out = tmp_path / "report.json"
        exit_code = des.main([str(old), str(new), "--out", str(out)])
        assert exit_code == 0
        assert out.exists()
        written = json.loads(out.read_text())
        assert "schema" in written and "shrinkage_flags" in written
