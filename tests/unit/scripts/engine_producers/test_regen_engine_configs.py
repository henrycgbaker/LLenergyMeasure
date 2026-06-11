"""Tests for scripts/engine_producers/regen_engine_configs.py.

Concerns:

1. Pre-step (envelope -> JSON Schema): legacy Python type-string mapping,
   enum/bounds passthrough, $defs/section materialisation.
2. Curated filter: only exposed fields reach the synthetic schema.
3. Overlay: narrow (tighten), complete (add), and contradiction-errors.
4. End-to-end --write then --check (determinism, including the ruff post-step),
   against synthetic envelopes in tmp dirs.
5. Generated-file shape assertions: Literal for an enum field, a bound, the
   extra="allow" policy, and the DO NOT EDIT header.
6. One run against the real committed transformers shadow.
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
# Pre-step: legacy Python type-string -> JSON Schema
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
# compose_synthetic_schema: curation, $defs, union-resolution
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


def _empty_overlay() -> dict:
    return {
        "narrowings": {s: {} for s in rec.SECTIONS},
        "completions": {s: {} for s in rec.SECTIONS},
    }


def test_compose_filters_to_curated_only() -> None:
    curated = {"engine_params": ["dtype", "use_cache"], "sampling_params": ["temperature"]}
    schema = rec.compose_synthetic_schema(_DISCOVERED, curated, _empty_overlay())

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
    schema = rec.compose_synthetic_schema(_DISCOVERED, curated, _empty_overlay())
    assert schema["$defs"]["EngineParams"]["additionalProperties"] is True
    assert schema["properties"]["engine_params"] == {"$ref": "#/$defs/EngineParams"}


def test_compose_debt_field_becomes_permissive_stub() -> None:
    """A curated field absent from discovery is a permissive Any | None stub."""
    curated = {"engine_params": ["device_map"], "sampling_params": []}
    schema = rec.compose_synthetic_schema(_DISCOVERED, curated, _empty_overlay())
    assert schema["$defs"]["EngineParams"]["properties"]["device_map"] == {}


# ---------------------------------------------------------------------------
# Overlay: narrow, complete, contradiction
# ---------------------------------------------------------------------------


def test_overlay_narrowing_tightens_mined_field() -> None:
    overlay = _empty_overlay()
    overlay["narrowings"]["sampling_params"]["temperature"] = {
        "minimum": 0.0,
        "x-narrowing-reason": "NaN softmax below 0",
    }
    curated = {"engine_params": [], "sampling_params": ["temperature"]}
    schema = rec.compose_synthetic_schema(_DISCOVERED, curated, overlay)
    prop = schema["$defs"]["SamplingParams"]["properties"]["temperature"]
    assert prop["minimum"] == 0.0
    assert prop["type"] == "number"  # mined type preserved
    assert prop["x-narrowing-applied"] == "NaN softmax below 0"


def test_overlay_narrowing_allows_subtype_tighten() -> None:
    """integer narrows number (a legal tighten)."""
    overlay = _empty_overlay()
    overlay["narrowings"]["sampling_params"]["temperature"] = {"type": "integer"}
    curated = {"engine_params": [], "sampling_params": ["temperature"]}
    schema = rec.compose_synthetic_schema(_DISCOVERED, curated, overlay)
    assert schema["$defs"]["SamplingParams"]["properties"]["temperature"]["type"] == "integer"


def test_overlay_narrowing_contradiction_errors() -> None:
    """string mined vs integer overlay is a contradiction, not a tighten."""
    overlay = _empty_overlay()
    overlay["narrowings"]["engine_params"]["dtype"] = {"type": "integer"}
    curated = {"engine_params": ["dtype"], "sampling_params": []}
    with pytest.raises(ValueError, match="contradicts mined type"):
        rec.compose_synthetic_schema(_DISCOVERED, curated, overlay)


def test_overlay_completion_adds_new_field() -> None:
    overlay = _empty_overlay()
    overlay["completions"]["sampling_params"]["compile_mode"] = {
        "type": "string",
        "x-completion-reason": "nested CompileConfig not walked yet",
    }
    curated = {"engine_params": [], "sampling_params": ["temperature"]}
    schema = rec.compose_synthetic_schema(_DISCOVERED, curated, overlay)
    sp = schema["$defs"]["SamplingParams"]["properties"]
    assert sp["compile_mode"]["type"] == "string"
    assert sp["compile_mode"]["x-source"] == "engine_overlay"
    assert sp["compile_mode"]["x-completion-applied"] == "nested CompileConfig not walked yet"


def test_overlay_completion_shadowing_curated_field_errors() -> None:
    overlay = _empty_overlay()
    overlay["completions"]["engine_params"]["dtype"] = {"type": "string"}
    curated = {"engine_params": ["dtype"], "sampling_params": []}
    with pytest.raises(ValueError, match="shadows a curated field"):
        rec.compose_synthetic_schema(_DISCOVERED, curated, overlay)


# ---------------------------------------------------------------------------
# End-to-end --write / --check against a synthetic SSOT in a tmp dir
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ssot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a one-engine SSOT + shadow under tmp_path wired into the module.

    The discovered schema carries an enum (-> Literal) and a numeric field an
    overlay narrows with a bound (-> Field(ge=)), so generated-shape assertions
    have something to bite on. Returns ``(outputs_dir, config_path)``.
    """
    outputs = tmp_path / "engine_versions" / "demo" / "v9_9_9" / "outputs"
    shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "demo"
    outputs.mkdir(parents=True)
    shadow.mkdir(parents=True)

    (outputs / "schema.discovered.json").write_text(
        json.dumps(
            {
                "engine_version": "9.9.9",
                "engine_params": {
                    "dtype": {"type": "str", "default": "auto", "enum": ["half", "bf16", "auto"]},
                    "device_map": {"type": "unknown", "default": None},
                },
                "sampling_params": {"temperature": {"type": "float", "default": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    (outputs / "curated.yaml").write_text(
        yaml.safe_dump(
            {
                "engine": "demo",
                "exposed_fields": {
                    "engine_params": ["dtype", "device_map"],
                    "sampling_params": ["temperature"],
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "overlay.yaml").write_text(
        yaml.safe_dump(
            {
                "narrowings": {
                    "sampling_params": {
                        "temperature": {"minimum": 0.0, "x-narrowing-reason": "no negatives"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(rec, "ENGINES", ("demo",))
    monkeypatch.setattr(rec, "current_outputs_dir", lambda engine: outputs)
    monkeypatch.setattr(rec, "_shadow_config_path", lambda engine: shadow / "config.py")
    return outputs, shadow / "config.py"


def test_write_then_check_is_green(fake_ssot: tuple[Path, Path]) -> None:
    """--write materialises config.py; an immediate --check is clean (determinism)."""
    _outputs, config_path = fake_ssot
    assert rec.main(["--write"]) == 0
    assert config_path.exists()
    assert rec.main(["--check"]) == 0


def test_check_fails_on_drift(fake_ssot: tuple[Path, Path]) -> None:
    _outputs, config_path = fake_ssot
    rec.main(["--write"])
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8"
    )
    assert rec.main(["--check"]) == 1


def test_generated_shape(fake_ssot: tuple[Path, Path]) -> None:
    """Literal for the enum field, ge= bound from the overlay, extra=allow, header."""
    _outputs, config_path = fake_ssot
    rec.main(["--write"])
    text = config_path.read_text(encoding="utf-8")

    assert text.startswith("# DO NOT EDIT")
    # ruff post-step ran: double-quoted strings, not the generator's single quotes.
    assert 'extra="allow"' in text
    assert "extra='allow'" not in text
    # enum -> Literal.
    assert 'Literal["half", "bf16", "auto"]' in text
    # overlay narrowing bound -> Field(ge=0.0).
    assert "ge=0.0" in text
    # the three classes are present.
    for cls in ("class EngineParams", "class SamplingParams", "class Config"):
        assert cls in text


def test_missing_ssot_dir_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec, "ENGINES", ("demo",))
    monkeypatch.setattr(rec, "current_outputs_dir", lambda engine: tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="SSOT outputs dir not found"):
        rec.main(["--check"])


# ---------------------------------------------------------------------------
# Real committed transformers shadow
# ---------------------------------------------------------------------------


def test_real_transformers_config_in_sync() -> None:
    """The committed transformers config.py matches what the script regenerates."""
    assert rec.main(["--check", "--engine", "transformers"]) == 0


def test_real_transformers_config_importable_and_shaped() -> None:
    """Module imports without transformers; carries the three classes + extra=allow."""
    from llenergymeasure.engines.transformers import config as tcfg

    assert tcfg.Config.model_config["extra"] == "allow"
    # A field mining typed survives as a real type; a debt field is permissive.
    inst = tcfg.Config(
        engine_params={"num_beams": 2, "novel_kwarg": True},
        sampling_params={"temperature": 0.5},
    )
    assert inst.engine_params.num_beams == 2
    assert inst.engine_params.novel_kwarg is True  # extra="allow" passthrough
    assert inst.sampling_params.temperature == 0.5
