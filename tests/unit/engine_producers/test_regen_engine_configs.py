"""Unit tests for ``scripts/engine_producers/regen_engine_configs.py``.

Phase 2-T pilot scope: transformers only. The wrapper composes a
synthetic JSON Schema from ``schema.discovered.json`` + ``curated.yaml``,
shells out to ``datamodel-codegen``, and writes
``src/llenergymeasure/engines/<engine>/config.py``.

Tests exercise:
- Pure-Python helpers (envelope -> JSON Schema, curated.yaml filter).
- ``sync_engine`` skip behaviour (Renovate-pre-vendor window).
- ``--check`` / ``--write`` CLI flag plumbing (the subprocess call
  itself is exercised end-to-end against the live transformers
  schema in ``test_codegen_round_trips_live_schema``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import regen_engine_configs  # noqa: E402


class TestFieldShapeToProperty:
    def test_typed_field_passes_through(self) -> None:
        out = regen_engine_configs._field_shape_to_property(
            {"type": "number", "default": 1.0}
        )
        assert out == {"type": "number", "default": 1.0}

    def test_untyped_field_skips_missing_type(self) -> None:
        out = regen_engine_configs._field_shape_to_property(
            {"default": None, "description": "discovery emitted 'unknown'"}
        )
        # No 'type' key, default + description preserved.
        assert "type" not in out
        assert out["default"] is None
        assert out["description"] == "discovery emitted 'unknown'"

    def test_extra_keys_dropped(self) -> None:
        # Non-JSON-Schema keys (e.g. 'miner_source') are not forwarded.
        out = regen_engine_configs._field_shape_to_property(
            {
                "type": "integer",
                "default": 50,
                "miner_source": {"path": "...", "method": "..."},
            }
        )
        assert out == {"type": "integer", "default": 50}

    def test_provenance_keys_forwarded(self) -> None:
        # x-source / x-source-ref are llem extensions preserved via
        # --field-extra-keys in the dmcg invocation; they MUST reach the
        # generated schema.
        out = regen_engine_configs._field_shape_to_property(
            {"type": "string", "x-source": "pydantic_field", "x-source-ref": "module:Class"}
        )
        assert out["x-source"] == "pydantic_field"
        assert out["x-source-ref"] == "module:Class"


class TestComposeSyntheticSchema:
    def test_synthetic_root_shape(self) -> None:
        discovered = {
            "engine": "transformers",
            "engine_params": {"a": {"type": "string", "default": "x"}},
            "sampling_params": {"b": {"type": "integer", "default": 1}},
        }
        curated = {"engine_params": ["a"], "sampling_params": ["b"]}
        out = regen_engine_configs._compose_synthetic_schema(discovered, curated, {"narrowings": {}, "completions": {}})
        assert out["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert out["type"] == "object"
        assert out["additionalProperties"] is True
        # Sections are $refs into $defs (uniform class generation - empty
        # sections still emit a named class).
        assert out["properties"]["engine_params"] == {
            "$ref": "#/$defs/EngineParams"
        }
        assert out["properties"]["sampling_params"] == {
            "$ref": "#/$defs/SamplingParams"
        }
        assert "EngineParams" in out["$defs"]
        assert "SamplingParams" in out["$defs"]

    def test_curated_filters_out_non_listed_fields(self) -> None:
        discovered = {
            "sampling_params": {
                "exposed": {"type": "number", "default": 1.0},
                "hidden": {"type": "number", "default": 2.0},
            },
            "engine_params": {},
        }
        curated = {"engine_params": [], "sampling_params": ["exposed"]}
        out = regen_engine_configs._compose_synthetic_schema(discovered, curated, {"narrowings": {}, "completions": {}})
        props = out["$defs"]["SamplingParams"]["properties"]
        assert "exposed" in props
        assert "hidden" not in props

    def test_curated_field_not_in_schema_is_silently_dropped(self) -> None:
        # If curated.yaml lists a field that doesn't exist in
        # schema.discovered.json (maintainer mistake, or upstream removed
        # the field on a Renovate bump), the field is dropped from the
        # generated class. This matches "you cannot expose what isn't
        # mined" - the schema is authoritative.
        discovered = {
            "sampling_params": {"only_real": {"type": "number"}},
            "engine_params": {},
        }
        curated = {
            "engine_params": [],
            "sampling_params": ["only_real", "ghost_field"],
        }
        out = regen_engine_configs._compose_synthetic_schema(discovered, curated, {"narrowings": {}, "completions": {}})
        props = out["$defs"]["SamplingParams"]["properties"]
        assert set(props.keys()) == {"only_real"}

    def test_empty_section_still_emits_named_class(self) -> None:
        # Forward-uniform API: EngineParams class must exist even if no
        # fields are curated for it. Achieved via $defs/$ref - dmcg can't
        # collapse the section into an inline dict.
        discovered = {"sampling_params": {}, "engine_params": {}}
        curated = {"engine_params": [], "sampling_params": []}
        out = regen_engine_configs._compose_synthetic_schema(discovered, curated, {"narrowings": {}, "completions": {}})
        assert out["$defs"]["EngineParams"]["properties"] == {}
        assert out["$defs"]["EngineParams"]["additionalProperties"] is True
        assert out["$defs"]["SamplingParams"]["properties"] == {}


class TestLoadCurated:
    def test_missing_curated_yaml_returns_empty_lists(self, tmp_path: Path) -> None:
        # Renovate-pre-vendor window or new engine: no curated.yaml yet.
        # _load_curated must NOT raise.
        out = regen_engine_configs._load_curated(tmp_path)
        assert out == {"engine_params": [], "sampling_params": []}

    def test_partial_curated_yaml_defaults_missing_sections(self, tmp_path: Path) -> None:
        (tmp_path / "curated.yaml").write_text(
            "schema_version: 1.0.0\n"
            "exposed_fields:\n"
            "  sampling_params:\n"
            "    - temperature\n",
            encoding="utf-8",
        )
        out = regen_engine_configs._load_curated(tmp_path)
        assert out["engine_params"] == []
        assert out["sampling_params"] == ["temperature"]

    def test_non_mapping_exposed_fields_raises(self, tmp_path: Path) -> None:
        (tmp_path / "curated.yaml").write_text(
            "exposed_fields:\n  - just_a_list\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            regen_engine_configs._load_curated(tmp_path)


class TestSyncEngineSkipBehaviour:
    def test_missing_outputs_dir_skips_not_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Point current_outputs_dir at a non-existent path.
        bogus = tmp_path / "engine_versions" / "transformers" / "v9_9_9" / "outputs"
        monkeypatch.setattr(
            regen_engine_configs, "current_outputs_dir", lambda engine: bogus
        )
        drift, skipped = regen_engine_configs.sync_engine("transformers", write=False)
        assert drift == []
        assert len(skipped) == 1
        assert "outputs directory not present" in skipped[0]

    def test_missing_schema_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # outputs/ dir exists but is empty.
        outputs = tmp_path / "engine_versions" / "transformers" / "v0_0_0" / "outputs"
        outputs.mkdir(parents=True)
        monkeypatch.setattr(
            regen_engine_configs, "current_outputs_dir", lambda engine: outputs
        )
        drift, skipped = regen_engine_configs.sync_engine("transformers", write=False)
        assert drift == []
        assert len(skipped) == 1
        assert "schema.discovered.json missing" in skipped[0]


class TestLiveCodegen:
    """End-to-end against the committed transformers outputs.

    Exercises the real subprocess invocation of datamodel-codegen. Slow
    (~1-2s per invocation) but the right place to catch flag-combo
    regressions.
    """

    def test_check_against_committed_config_passes(self) -> None:
        # The committed config.py was produced by this script's --write
        # mode. Re-running --check must therefore exit 0.
        rc = regen_engine_configs.main(["--check"])
        assert rc == 0

    def test_generated_class_is_importable_and_round_trips(self) -> None:
        # Sanity: the committed config.py imports cleanly + accepts a
        # field from the curated set.
        from llenergymeasure.engines.transformers.config import Config

        cfg = Config(sampling_params={"temperature": 0.7, "top_k": 40})
        assert cfg.sampling_params is not None
        assert cfg.sampling_params.temperature == 0.7
        assert cfg.sampling_params.top_k == 40
        # extra='allow' contract: unknown fields survive into model_dump.
        cfg_extra = Config(sampling_params={"temperature": 0.5, "novel": "x"})
        assert cfg_extra.sampling_params is not None
        dumped = cfg_extra.sampling_params.model_dump()
        assert dumped["novel"] == "x"


class TestCliFlags:
    def test_check_and_write_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            regen_engine_configs.main(["--check", "--write"])
        assert exc_info.value.code == 2

    def test_default_mode_is_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default-mode test must not mutate the working tree. We point
        # the target at tmp_path then assert no file appears there.
        # Use --engine transformers to scope to one engine - the
        # monkeypatched _shadow_config_path returns the same target for
        # all engines, which would compose drift incorrectly under the
        # default all-engines iteration.
        target = tmp_path / "config.py"
        monkeypatch.setattr(
            regen_engine_configs, "_shadow_config_path", lambda engine: target
        )
        rc = regen_engine_configs.main(["--engine", "transformers"])
        assert rc == 1
        assert not target.exists()


# ---------------------------------------------------------------------------
# Round-trip: --write produces output that --check accepts (idempotency).
# ---------------------------------------------------------------------------


class TestRoundtripIdempotency:
    def test_write_then_check_clean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Scope to one engine since the monkeypatched _shadow_config_path
        # collapses all engines onto the same target file.
        target = tmp_path / "config.py"
        monkeypatch.setattr(
            regen_engine_configs, "_shadow_config_path", lambda engine: target
        )
        rc_write = regen_engine_configs.main(["--write", "--engine", "transformers"])
        assert rc_write == 0
        assert target.is_file()
        rc_check = regen_engine_configs.main(["--check", "--engine", "transformers"])
        assert rc_check == 0

    def test_write_twice_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # --disable-timestamp guarantees byte-identical output across
        # back-to-back invocations. If this test ever fails, dmcg has
        # regressed or our flag combo lost --disable-timestamp.
        target = tmp_path / "config.py"
        monkeypatch.setattr(
            regen_engine_configs, "_shadow_config_path", lambda engine: target
        )
        regen_engine_configs.main(["--write", "--engine", "transformers"])
        first = target.read_bytes()
        regen_engine_configs.main(["--write", "--engine", "transformers"])
        second = target.read_bytes()
        assert first == second
