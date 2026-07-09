"""Tests for scripts/engine_producers/regen_engine_configs.py.

Concerns:

1. Pre-step (mined envelope -> JSON Schema): Python type-string mapping,
   enum/bounds passthrough.
2. Curated filter: only exposed fields reach the synthetic schema; a debt
   field becomes a permissive stub; a curated field discovered in the other
   section is union-resolved.
3. Nested-blob projection: a wholesale-forwarded vLLM blob becomes a typed
   one-level submodel.
4. End-to-end --write then --check (determinism, including the ruff post-step)
   against a synthetic snapshot in a tmp dir, plus the CLI exit codes.
5. Generated-file shape assertions: Literal for an enum, a MINED bound, the
   extra="allow" policy, and the DO NOT EDIT header.
6. Real workspace snapshots: byte-stable regeneration, generation off a
   non-active pin, and the vLLM nested-blob classes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.engine_producers import regen_engine_configs as rec  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-step: Python type-string -> JSON Schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_str", "expected"),
    [
        ("str", {"type": "string"}),
        ("bool", {"type": "boolean"}),
        ("int", {"type": "integer"}),
        ("float", {"type": "number"}),
        ("unknown", {}),
        (None, {}),
        # Trailing None drops out; a single scalar survives.
        ("bool | None", {"type": "boolean"}),
        # Multi-scalar union -> anyOf.
        ("str | bool | None", {"anyOf": [{"type": "string"}, {"type": "boolean"}]}),
        # Non-scalar member (engine class / PathLike) -> permissive.
        ("PretrainedConfig | str | PathLike | None", {}),
        # JSON-native spelling maps identically to the Python spelling - a
        # bounded native-typed field must not silently widen to Any | None.
        ("number", {"type": "number"}),
        ("integer", {"type": "integer"}),
        ("boolean", {"type": "boolean"}),
        ("string", {"type": "string"}),
        ("integer | None", {"type": "integer"}),
        ("string | number", {"anyOf": [{"type": "string"}, {"type": "number"}]}),
        # A JSON container is not a scalar -> stays permissive.
        ("array", {}),
        ("object | None", {}),
    ],
)
def test_python_type_to_json_schema(type_str: str | None, expected: dict) -> None:
    assert rec._python_type_to_json_schema(type_str) == expected


def test_field_shape_passes_through_enum_and_bounds() -> None:
    """JSON-Schema-native keys (enum, bounds, default, x-source) survive."""
    shape = {
        "type": "str",
        "default": "auto",
        "enum": ["half", "bfloat16", "auto"],
        "x-source": "module_validation_collection",
    }
    prop = rec._field_shape_to_property(shape)
    assert prop["type"] == "string"
    assert prop["enum"] == ["half", "bfloat16", "auto"]
    assert prop["default"] == "auto"
    assert prop["x-source"] == "module_validation_collection"


# ---------------------------------------------------------------------------
# compose_schema: curation, $defs, union-resolution
# ---------------------------------------------------------------------------

_DISCOVERED = {
    "engine_version": "9.9.9",
    "engine_params": {
        "dtype": {"type": "str", "default": "auto", "enum": ["half", "bf16", "auto"]},
        "secret_internal": {"type": "str", "default": "x"},
    },
    "sampling_params": {
        # Curated under engine_params below; discovered here. Tests union-resolution.
        "use_cache": {"type": "bool", "default": True},
        "temperature": {"type": "float", "default": 1.0},
        "uncurated": {"type": "int", "default": 0},
    },
}


def test_compose_filters_to_curated_only() -> None:
    curated = {"engine_params": ["dtype", "use_cache"], "sampling_params": ["temperature"]}
    schema = rec.compose_schema("transformers", _DISCOVERED, curated)

    ep = schema["$defs"]["EngineParams"]["properties"]
    sp = schema["$defs"]["SamplingParams"]["properties"]
    # Curated fields present; non-curated absent.
    assert set(ep) == {"dtype", "use_cache"}
    assert set(sp) == {"temperature"}
    assert "secret_internal" not in ep
    assert "uncurated" not in sp
    # enum survived to the curated property.
    assert ep["dtype"]["enum"] == ["half", "bf16", "auto"]
    # Union-resolution: use_cache (discovered under sampling) typed in engine section.
    assert ep["use_cache"] == {"type": "boolean", "default": True}


def test_compose_marks_sections_extra_allow() -> None:
    curated = {"engine_params": ["dtype"], "sampling_params": []}
    schema = rec.compose_schema("transformers", _DISCOVERED, curated)
    assert schema["$defs"]["EngineParams"]["additionalProperties"] is True
    assert schema["properties"]["engine_params"] == {"$ref": "#/$defs/EngineParams"}


def test_compose_debt_field_becomes_permissive_stub() -> None:
    """A curated field absent from discovery is a permissive Any | None stub."""
    curated = {"engine_params": ["device_map"], "sampling_params": []}
    schema = rec.compose_schema("transformers", _DISCOVERED, curated)
    assert schema["$defs"]["EngineParams"]["properties"]["device_map"] == {}


def test_compose_defs_are_sorted() -> None:
    """$defs are emitted in sorted key order (byte-stable codegen fixpoint)."""
    curated = {"engine_params": ["dtype"], "sampling_params": ["temperature"]}
    schema = rec.compose_schema("transformers", _DISCOVERED, curated)
    assert list(schema["$defs"]) == sorted(schema["$defs"])


# ---------------------------------------------------------------------------
# Nested-blob projection (wholesale-forwarded vLLM blobs)
# ---------------------------------------------------------------------------

_BLOB_DISCOVERED = {
    "engine_version": "0.19.1",
    "engine_params": {
        "compilation_config": {"$ref": "#/$defs/CompilationConfig"},
    },
    "sampling_params": {},
    "$defs": {
        "CompilationConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "level": {"type": "integer", "default": 0, "minimum": 0, "maximum": 3},
                "backend": {"type": "string", "default": "inductor"},
                # A $ref leaf stays permissive - no deeper recursion.
                "pass_config": {"$ref": "#/$defs/PassConfig"},
            },
        },
        "PassConfig": {"type": "object", "properties": {"x": {"type": "integer"}}},
    },
}


def test_nested_blob_projected_one_level() -> None:
    curated = {"engine_params": ["compilation_config"], "sampling_params": []}
    schema = rec.compose_schema("vllm", _BLOB_DISCOVERED, curated)
    # The blob field references a typed submodel, not Any | None.
    assert schema["$defs"]["EngineParams"]["properties"]["compilation_config"] == {
        "$ref": "#/$defs/CompilationConfig"
    }
    blob = schema["$defs"]["CompilationConfig"]
    assert blob["additionalProperties"] is True  # extra="allow" on the submodel
    props = blob["properties"]
    # Scalar/bounded leaf typed; its mined default is STRIPPED (nullable leaf).
    assert props["level"] == {"type": "integer", "minimum": 0, "maximum": 3}
    assert props["backend"] == {"type": "string"}
    # A $ref leaf stays permissive (no deeper projection).
    assert props["pass_config"] == {}
    # Properties emitted in sorted order for byte-stability.
    assert list(props) == sorted(props)


def test_nested_blob_not_projected_for_engine_without_blobs() -> None:
    """transformers lists no wholesale blobs: a $ref field collapses to Any | None."""
    curated = {"engine_params": ["compilation_config"], "sampling_params": []}
    schema = rec.compose_schema("transformers", _BLOB_DISCOVERED, curated)
    ep = schema["$defs"]["EngineParams"]["properties"]
    assert ep["compilation_config"] == {}
    assert "CompilationConfig" not in schema["$defs"]


# ---------------------------------------------------------------------------
# End-to-end --write / --check against a synthetic snapshot in a tmp dir
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a one-engine snapshot under tmp_path wired into the module.

    The discovered schema carries an enum (-> Literal) and a numeric field with
    MINED bounds (``minimum`` / ``maximum`` -> Field(ge= / le=), now the sole
    bound path). Returns ``(outputs_dir, output_config_path)``.
    """
    outputs = tmp_path / "engine_versions" / "transformers" / "v9_9_9" / "outputs"
    outputs.mkdir(parents=True)
    output = tmp_path / "config.py"

    (outputs / "schema.discovered.json").write_text(
        json.dumps(
            {
                "engine_version": "9.9.9",
                "engine_params": {
                    "dtype": {"type": "str", "default": "auto", "enum": ["half", "bf16", "auto"]},
                    "device_map": {"type": "unknown", "default": None},
                },
                "sampling_params": {
                    "temperature": {"type": "float", "default": 1.0},
                    # Mined bounds straight off the schema: proves the
                    # minimum/maximum -> Field(ge= / le=) projection.
                    "top_k": {"type": "int", "default": 50, "minimum": 0, "maximum": 100},
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "curated.yaml").write_text(
        yaml.safe_dump(
            {
                "engine": "transformers",
                "exposed_fields": {
                    "engine_params": ["dtype", "device_map"],
                    "sampling_params": ["temperature", "top_k"],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(rec._outputs, "outputs_dir", lambda engine, version: outputs)
    return outputs, output


def test_write_then_check_is_green(fake_snapshot: tuple[Path, Path]) -> None:
    """--write materialises config.py; an immediate --check is clean (determinism)."""
    _outputs, output = fake_snapshot
    assert (
        rec.main(
            ["--engine", "transformers", "--version", "9.9.9", "--write", "--output", str(output)]
        )
        == 0
    )
    assert output.exists()
    assert (
        rec.main(
            ["--engine", "transformers", "--version", "9.9.9", "--check", "--output", str(output)]
        )
        == 0
    )


def test_check_fails_on_drift(fake_snapshot: tuple[Path, Path]) -> None:
    _outputs, output = fake_snapshot
    rec.main(["--engine", "transformers", "--version", "9.9.9", "--write", "--output", str(output)])
    output.write_text(output.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    assert (
        rec.main(
            ["--engine", "transformers", "--version", "9.9.9", "--check", "--output", str(output)]
        )
        == 1
    )


def test_check_defaults_to_check_mode(fake_snapshot: tuple[Path, Path]) -> None:
    """No mode flag defaults to --check; against an absent target it reports drift."""
    _outputs, output = fake_snapshot
    assert (
        rec.main(["--engine", "transformers", "--version", "9.9.9", "--output", str(output)]) == 1
    )


def test_generated_shape(fake_snapshot: tuple[Path, Path]) -> None:
    """Literal for the enum, a MINED bound, extra=allow, header.

    Asserts the EXACT generated annotations so the enum -> Literal and
    minimum/maximum -> Field(ge= / le=) projections are pinned through the full
    pre-step + datamodel-codegen + ruff path.
    """
    _outputs, output = fake_snapshot
    rec.main(["--engine", "transformers", "--version", "9.9.9", "--write", "--output", str(output)])
    text = output.read_text(encoding="utf-8")

    assert text.startswith("# DO NOT EDIT")
    assert "--engine transformers --version 9.9.9 --write" in text
    # ruff post-step ran: double-quoted strings, not the generator's single quotes.
    assert 'extra="allow"' in text
    assert "extra='allow'" not in text
    # enum -> Literal, exact field annotation.
    assert 'dtype: Literal["half", "bf16", "auto"] | None = "auto"' in text
    # MINED minimum/maximum -> Field(ge= / le=), now the sole bound path.
    assert "ge=0" in text
    assert "le=100" in text
    # the three classes are present.
    for cls in ("class EngineParams", "class SamplingParams", "class Config"):
        assert cls in text


def test_missing_snapshot_dir_sync_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec._outputs, "outputs_dir", lambda engine, version: tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="snapshot outputs dir not found"):
        rec.sync("vllm", "9.9.9", tmp_path / "c.py", write=False)


def test_missing_snapshot_dir_main_exits_with_friendly_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A version bump lands current.yaml pointing at a pin whose snapshot has
    # not been mined yet; main() must exit non-zero with a remediation message
    # (naming the missing directory and the ritual), not a bare traceback.
    absent = tmp_path / "absent"
    monkeypatch.setattr(rec._outputs, "outputs_dir", lambda engine, version: absent)
    rc = rec.main(
        [
            "--engine",
            "vllm",
            "--version",
            "9.9.9",
            "--check",
            "--output",
            str(tmp_path / "c.py"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "No mined snapshot for vllm 9.9.9" in err
    assert str(absent) in err
    assert "make discover-schema ENGINE=vllm" in err


def test_default_output_is_the_src_shadow() -> None:
    assert rec._shadow_config_path("vllm") == (
        REPO_ROOT / "src" / "llenergymeasure" / "engines" / "vllm" / "config.py"
    )


# ---------------------------------------------------------------------------
# Real workspace snapshots
# ---------------------------------------------------------------------------


def test_real_snapshot_generation_is_byte_stable() -> None:
    """Two generations off the same real snapshot are byte-identical.

    This is the invariant the CI byte-gate rests on: the tool is a
    deterministic fixpoint over the committed workspace data.
    """
    outputs = rec._outputs.outputs_dir("transformers", "5.7.0")
    first = rec.generate_config("transformers", "5.7.0", outputs)
    second = rec.generate_config("transformers", "5.7.0", outputs)
    assert first == second
    assert first.startswith(b"# DO NOT EDIT")


def test_generates_for_non_active_pin(tmp_path: Path) -> None:
    """A prior (non-active) pin snapshot generates - the tool is version-scoped."""
    out = tmp_path / "vllm_073.py"
    assert (
        rec.main(["--engine", "vllm", "--version", "0.7.3", "--write", "--output", str(out)]) == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "v0_7_3" in text  # header stamps the snapshot version, not the active pin
    for cls in ("class EngineParams", "class SamplingParams", "class Config"):
        assert cls in text


def test_vllm_nested_blobs_generate_typed_submodels(tmp_path: Path) -> None:
    """vLLM wholesale-forwarded blobs become typed submodels against real data."""
    out = tmp_path / "vllm.py"
    rec.main(["--engine", "vllm", "--version", "0.19.1", "--write", "--output", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "class CompilationConfig" in text
    assert "class SpeculativeConfig" in text
    assert "compilation_config: CompilationConfig | None" in text
    assert "speculative_config: SpeculativeConfig | None" in text
