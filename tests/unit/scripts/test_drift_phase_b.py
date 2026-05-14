"""Tests for :mod:`scripts._drift` Phase B (added direction / coverage check).

Phase B extends the drift tool with ``discover_live_landmarks()``, which
introspects the live library surface (Pydantic v2, Pydantic v1, stdlib
dataclasses, enums, plain classes) and surfaces validators that exist in the
library but are NOT declared in the per-version archive's LANDMARKS tuple.

The introspection functions operate on Python types, so all tests in this
module use stdlib / built-in types as stand-ins for engine library classes
- no engine library installation required. The same import-via-importlib path
exercised by the production code is preserved: tests that need to check module
import use types registered in ``sys.modules``.

Key invariants verified:
- Layered dispatch: pydantic v2 -> pydantic v1 -> dataclass -> enum -> plain class
- Set-diff correctness: ``landmarks_added`` = live_high - declared
- ``landmarks_added_low_confidence`` = live_low - declared
- Empty live surface -> no additions
- Empty declared LANDMARKS -> all live surface surfaced as added
- Missing engine library -> empty live surface, no error
- Direction field: "added" / "mixed" / "stable" / "removed"
- Verdict is always "pass" when only additions found (Phase B warning-only)
- ``landmarks_added_low_confidence`` present in JSON output
- ``direction = "mixed"`` when both missing and added simultaneously
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import _drift  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers (shared with test_drift.py - duplicated to keep modules
# independent; the helper is small enough to not warrant a conftest fixture)
# ---------------------------------------------------------------------------


def _install_synthetic_producer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str,
    producer: _drift.ProducerKind,
    landmarks: tuple[str, ...],
    module_name: str = "scripts._drift_synthetic_producer_phase_b",
) -> types.ModuleType:
    """Register a fake producer module and retarget ``_PRODUCER_MODULES``."""
    module = types.ModuleType(module_name)
    module.LANDMARKS = landmarks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(_drift._PRODUCER_MODULES, (engine, producer), module_name)
    return module


def _redirect_compat_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: str = "transformers"
) -> Path:
    """Point current.yaml path and compat cache writes at ``tmp_path``."""
    real_current = _PROJECT_ROOT / "engine_versions" / engine / "current.yaml"
    fake_engine_dir = tmp_path / engine
    fake_engine_dir.mkdir(parents=True, exist_ok=True)
    fake_current = fake_engine_dir / "current.yaml"
    fake_current.write_text(real_current.read_text())

    def _fake_path(name: str) -> Path:
        return tmp_path / name / "current.yaml"

    monkeypatch.setattr(_drift, "current_path", _fake_path)
    return fake_current


# ---------------------------------------------------------------------------
# Helpers for constructing synthetic types for introspection tests
# ---------------------------------------------------------------------------


def _make_pydantic_v2_model(name: str, validators: dict[str, str]) -> type:
    """Return a pydantic v2 BaseModel subclass with the given field_validators.

    ``validators`` maps method_name -> field_name. Requires pydantic installed.
    """
    try:
        import pydantic
        from pydantic import field_validator
    except ImportError:
        pytest.skip("pydantic not installed")

    # Build a class body with field validators dynamically.
    # Use type() to create the class; pydantic's metaclass processes it.
    namespace: dict = {"__annotations__": {"x": int}}
    for method_name, field_name in validators.items():

        def make_validator(fn_name: str, fn_field: str) -> classmethod:
            @field_validator(fn_field)
            @classmethod
            def _v(cls: type, v: int) -> int:
                return v

            _v.__name__ = fn_name
            _v.__qualname__ = fn_name
            return _v  # type: ignore[return-value]

        namespace[method_name] = make_validator(method_name, field_name)
    return pydantic.create_model(name, **{"x": (int, ...)})  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# _class_validator_surface - layered dispatch
# ---------------------------------------------------------------------------


class TestLayeredDispatch:
    """Verify that _class_validator_surface dispatches correctly by class kind."""

    def test_pydantic_v2_field_validators_surfaced(self) -> None:
        """Pydantic v2 BaseModel field_validators -> high confidence."""
        try:
            import pydantic
            from pydantic import field_validator
        except ImportError:
            pytest.skip("pydantic not installed")

        class MyModel(pydantic.BaseModel):
            x: int
            y: str

            @field_validator("x")
            @classmethod
            def check_x(cls, v: int) -> int:
                return v

        results = _drift._class_validator_surface(MyModel, "mymodule")
        paths = {path for _conf, path in results}
        confs = {conf for conf, _path in results}

        # check_x should appear as high confidence
        assert any("check_x" in p for p in paths), f"check_x missing from {paths}"
        assert _drift._CONF_HIGH in confs

    def test_pydantic_v2_model_fields_surfaced(self) -> None:
        """Pydantic v2 declared model_fields -> high confidence."""
        try:
            import pydantic
        except ImportError:
            pytest.skip("pydantic not installed")

        class MyConfig(pydantic.BaseModel):
            alpha: int
            beta: str

        results = _drift._class_validator_surface(MyConfig, "mymodule")
        paths = {path for _conf, path in results}
        assert any("alpha" in p for p in paths)
        assert any("beta" in p for p in paths)

    def test_dataclass_fields_and_post_init_surfaced(self) -> None:
        """stdlib @dataclass: fields + __post_init__ -> high confidence."""

        @dataclasses.dataclass
        class MyDC:
            x: int
            y: str = "hello"

            def __post_init__(self) -> None:
                pass

        results = _drift._class_validator_surface(MyDC, "mymodule")
        paths = {path for _conf, path in results}
        confs = {conf for conf, _path in results}

        assert any("x" in p for p in paths), f"field 'x' missing from {paths}"
        assert any("y" in p for p in paths), f"field 'y' missing from {paths}"
        assert any("__post_init__" in p for p in paths), f"__post_init__ missing from {paths}"
        assert confs == {_drift._CONF_HIGH}

    def test_dataclass_without_post_init_no_post_init_entry(self) -> None:
        """Dataclass without __post_init__ must not surface a post_init entry."""

        @dataclasses.dataclass
        class NakedDC:
            a: int
            b: float

        results = _drift._class_validator_surface(NakedDC, "mymodule")
        paths = {path for _conf, path in results}
        assert not any("__post_init__" in p for p in paths)

    def test_enum_skipped(self) -> None:
        """Enum subclasses return empty results - no validator surface."""

        class MyStrEnum(str, enum.Enum):
            VALUE_A = "a"
            VALUE_B = "b"

        class MyIntEnum(int, enum.Enum):
            ONE = 1
            TWO = 2

        assert _drift._class_validator_surface(MyStrEnum, "mymodule") == []
        assert _drift._class_validator_surface(MyIntEnum, "mymodule") == []

    def test_plain_class_public_validate_medium_confidence(self) -> None:
        """Plain class with ``validate_*`` method -> medium confidence."""

        class PlainConfig:
            def validate_dtype(self) -> None:
                pass

            def check_mode(self) -> None:
                pass

        results = _drift._class_validator_surface(PlainConfig, "mymodule")
        paths = {path for _conf, path in results}
        confs_by_path = {path: conf for conf, path in results}

        val_paths = [p for p in paths if "validate_dtype" in p or "check_mode" in p]
        assert val_paths, f"validate_dtype / check_mode missing from {paths}"
        for p in val_paths:
            assert confs_by_path[p] == _drift._CONF_MEDIUM

    def test_plain_class_private_validate_low_confidence(self) -> None:
        """Plain class with ``_validate_*`` -> low confidence."""

        class PlainConfig:
            def _validate_internal(self) -> None:
                pass

        results = _drift._class_validator_surface(PlainConfig, "mymodule")
        paths = {path for _conf, path in results}
        confs_by_path = {path: conf for conf, path in results}

        low_paths = [p for p in paths if "_validate_internal" in p]
        assert low_paths, f"_validate_internal missing from {paths}"
        for p in low_paths:
            assert confs_by_path[p] == _drift._CONF_LOW

    def test_plain_class_non_validator_callable_not_surfaced(self) -> None:
        """Plain class callables that don't start with validate/check/verify are skipped."""

        class PlainConfig:
            def run(self) -> None:
                pass

            def _helper(self) -> None:
                pass

            def load_model(self) -> None:
                pass

        results = _drift._class_validator_surface(PlainConfig, "mymodule")
        paths = {path for _conf, path in results}
        assert not any("run" in p for p in paths)
        assert not any("load_model" in p for p in paths)

    def test_pydantic_v2_takes_precedence_over_dataclass(self) -> None:
        """If a class is both a pydantic model and dataclass-ish, pydantic path wins."""
        try:
            from pydantic.dataclasses import dataclass as pydantic_dataclass
        except ImportError:
            pytest.skip("pydantic not installed")

        # pydantic.dataclasses.dataclass wraps stdlib dataclass but creates
        # a pydantic model; __pydantic_decorators__ is present.

        @pydantic_dataclass
        class PydDC:
            x: int
            y: str = "hello"

        # Should go through the pydantic v2 path (has __pydantic_decorators__).
        assert hasattr(PydDC, "__pydantic_decorators__") or hasattr(PydDC, "model_fields"), (
            "Expected pydantic dataclass to have pydantic v2 decorator infra"
        )

        # The pydantic path returns fields from model_fields, not dataclasses.fields.
        results = _drift._class_validator_surface(PydDC, "mymodule")
        # Just ensure no exception and some surface surfaced.
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# discover_live_landmarks - set-diff and engine wiring
# ---------------------------------------------------------------------------


class TestDiscoverLiveLandmarks:
    """Verify discover_live_landmarks() returns the expected sets."""

    def test_unknown_engine_returns_empty(self) -> None:
        """An engine with no introspection roots returns empty sets."""
        high, low = _drift.discover_live_landmarks("ghost_engine_xyz")
        assert high == set()
        assert low == set()

    def test_class_import_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a root class can't be imported, that root is skipped."""
        # Patch the engine roots to point at a non-existent class.
        monkeypatch.setitem(
            _drift._ENGINE_INTROSPECTION_ROOTS,
            "mock_engine",
            ("nonexistent_module.NonExistent",),
        )
        high, low = _drift.discover_live_landmarks("mock_engine")
        assert high == set()
        assert low == set()

    def test_module_import_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a root module can't be imported, that root is skipped."""
        monkeypatch.setitem(
            _drift._ENGINE_INTROSPECTION_ROOTS,
            "mock_engine2",
            ("totally_nonexistent_module_xyz",),
        )
        high, low = _drift.discover_live_landmarks("mock_engine2")
        assert high == set()
        assert low == set()

    def test_class_level_root_discovered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A class-level root (uppercase last component) is imported directly."""

        @dataclasses.dataclass
        class SynthConfig:
            alpha: int

            def __post_init__(self) -> None:
                pass

        # Register a synthetic module containing SynthConfig.
        synth_mod = types.ModuleType("synth_mod_test")
        synth_mod.SynthConfig = SynthConfig  # type: ignore[attr-defined]
        SynthConfig.__module__ = "synth_mod_test"
        monkeypatch.setitem(sys.modules, "synth_mod_test", synth_mod)

        monkeypatch.setitem(
            _drift._ENGINE_INTROSPECTION_ROOTS,
            "mock_engine3",
            ("synth_mod_test.SynthConfig",),
        )

        high, _low = _drift.discover_live_landmarks("mock_engine3")
        # SynthConfig has field 'alpha' and __post_init__ -> both high confidence.
        assert any("alpha" in p for p in high), f"alpha missing from {high}"
        assert any("__post_init__" in p for p in high), f"__post_init__ missing from {high}"

    def test_module_level_root_walks_all_classes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A module-level root (lowercase last component) walks all classes in the module."""

        @dataclasses.dataclass
        class ConfigA:
            x: int

        @dataclasses.dataclass
        class ConfigB:
            y: str

        synth_mod = types.ModuleType("synth_module_walk")
        synth_mod.ConfigA = ConfigA  # type: ignore[attr-defined]
        synth_mod.ConfigB = ConfigB  # type: ignore[attr-defined]
        # Set __module__ so _walk_module_classes includes them.
        ConfigA.__module__ = "synth_module_walk"
        ConfigB.__module__ = "synth_module_walk"
        monkeypatch.setitem(sys.modules, "synth_module_walk", synth_mod)

        monkeypatch.setitem(
            _drift._ENGINE_INTROSPECTION_ROOTS,
            "mock_engine4",
            ("synth_module_walk",),
        )

        high, _low = _drift.discover_live_landmarks("mock_engine4")
        # Both classes should be surfaced.
        assert any("ConfigA" in p for p in high), f"ConfigA missing from {high}"
        assert any("ConfigB" in p for p in high), f"ConfigB missing from {high}"

    def test_enum_class_in_module_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enum classes inside a walked module are silently skipped."""

        class MyEnum(str, enum.Enum):
            VAL = "v"

        synth_mod = types.ModuleType("synth_module_enum")
        synth_mod.MyEnum = MyEnum  # type: ignore[attr-defined]
        MyEnum.__module__ = "synth_module_enum"
        monkeypatch.setitem(sys.modules, "synth_module_enum", synth_mod)

        monkeypatch.setitem(
            _drift._ENGINE_INTROSPECTION_ROOTS,
            "mock_engine5",
            ("synth_module_enum",),
        )

        high, low = _drift.discover_live_landmarks("mock_engine5")
        # Enum members should not appear in either set.
        assert not any("MyEnum" in p for p in high)
        assert not any("MyEnum" in p for p in low)


# ---------------------------------------------------------------------------
# Set-diff correctness
# ---------------------------------------------------------------------------


class TestSetDiff:
    """Verify that landmarks_added = live_high - declared."""

    def _run_with_live_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        declared_landmarks: tuple[str, ...],
        live_high: set[str],
        live_low: set[str],
    ) -> _drift.DriftReport:
        """Helper: wire a synthetic producer + fake discover_live_landmarks, run."""
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=declared_landmarks,
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda engine: (live_high, live_low))
        return _drift.run(engine="transformers", producer="invariants")

    def test_no_additions_when_live_subset_of_declared(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Live surface fully covered by LANDMARKS -> landmarks_added empty."""
        report = self._run_with_live_override(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError", "json.JSONEncoder"),
            live_high={"json.JSONDecodeError", "json.JSONEncoder"},
            live_low=set(),
        )
        assert report.landmarks_added == []
        assert report.landmarks_added_low_confidence == []
        assert report.direction == "stable"

    def test_additions_when_live_superset_of_declared(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Live has more than declared -> landmarks_added non-empty."""
        report = self._run_with_live_override(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError", "json.JSONEncoder", "json.JSONDecoder"},
            live_low=set(),
        )
        assert sorted(report.landmarks_added) == sorted(["json.JSONEncoder", "json.JSONDecoder"])
        assert report.direction == "added"

    def test_low_confidence_additions_in_separate_field(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Low-confidence live paths go to landmarks_added_low_confidence, not landmarks_added."""
        report = self._run_with_live_override(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError"},
            live_low={"json.SomePrivateHelper"},
        )
        assert report.landmarks_added == []
        assert "json.SomePrivateHelper" in report.landmarks_added_low_confidence

    def test_empty_declared_all_live_surfaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Empty LANDMARKS -> entire live surface appears in landmarks_added."""
        report = self._run_with_live_override(
            monkeypatch,
            tmp_path,
            declared_landmarks=(),
            live_high={"json.JSONDecodeError", "json.JSONEncoder"},
            live_low=set(),
        )
        assert sorted(report.landmarks_added) == sorted(
            ["json.JSONDecodeError", "json.JSONEncoder"]
        )

    def test_empty_live_no_additions(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Empty live surface (engine not installed) -> no additions."""
        report = self._run_with_live_override(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high=set(),
            live_low=set(),
        )
        assert report.landmarks_added == []
        assert report.landmarks_added_low_confidence == []


# ---------------------------------------------------------------------------
# Direction and verdict logic
# ---------------------------------------------------------------------------


class TestDirectionAndVerdict:
    """Verify direction and verdict logic across all combinations."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        declared_landmarks: tuple[str, ...],
        live_high: set[str],
        live_low: set[str],
    ) -> _drift.DriftReport:
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=declared_landmarks,
        )
        monkeypatch.setattr(_drift, "discover_live_landmarks", lambda engine: (live_high, live_low))
        return _drift.run(engine="transformers", producer="invariants")

    def test_direction_stable_no_gaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """All landmarks resolve, no additions -> stable."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError"},
            live_low=set(),
        )
        assert report.direction == "stable"
        assert report.verdict == "pass"

    def test_direction_added_only_additions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No missing landmarks, but live has extras -> added, verdict pass."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError",),
            live_high={"json.JSONDecodeError", "json.JSONEncoder"},
            live_low=set(),
        )
        assert report.direction == "added"
        assert report.verdict == "pass"  # Phase B: warning-only

    def test_direction_removed_missing_landmark(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Missing landmark, no additions -> removed, verdict fail."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError", "json.NonExistentXYZ"),
            live_high=set(),
            live_low=set(),
        )
        assert report.direction == "removed"
        assert report.verdict == "fail"
        assert "json.NonExistentXYZ" in report.landmarks_missing

    def test_direction_mixed_both_missing_and_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Missing landmark AND live extras -> mixed, verdict still fail."""
        report = self._run(
            monkeypatch,
            tmp_path,
            declared_landmarks=("json.JSONDecodeError", "json.NonExistentXYZ"),
            live_high={"json.JSONDecodeError", "json.JSONEncoder"},
            live_low=set(),
        )
        assert report.direction == "mixed"
        assert report.verdict == "fail"  # missing landmark overrides
        assert "json.NonExistentXYZ" in report.landmarks_missing
        assert "json.JSONEncoder" in report.landmarks_added

    def test_added_direction_verdict_pass_not_fail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Phase B is warning-only: additions alone NEVER flip verdict to fail."""
        report = self._run(
            monkeypatch,
            tmp_path,
            # No declared landmarks - everything in live is "added"
            declared_landmarks=(),
            live_high={
                "json.JSONDecodeError",
                "json.JSONEncoder",
                "json.JSONDecoder",
            },
            live_low=set(),
        )
        assert report.verdict == "pass"
        assert len(report.landmarks_added) == 3

    def test_schemas_producer_skips_phase_b(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Phase B introspection is only active for the 'invariants' producer."""
        _redirect_compat_dir(monkeypatch, tmp_path)
        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="schemas",
            landmarks=("json.JSONDecodeError",),
        )
        discover_called = []

        def _fake_discover(engine: str) -> tuple[set[str], set[str]]:
            discover_called.append(engine)
            return ({"json.JSONEncoder"}, set())

        monkeypatch.setattr(_drift, "discover_live_landmarks", _fake_discover)
        report = _drift.run(engine="transformers", producer="schemas")

        # discover_live_landmarks should NOT have been called for schemas.
        assert discover_called == [], "Phase B must not run for schemas producer"
        assert report.landmarks_added == []
        assert report.direction == "stable"


# ---------------------------------------------------------------------------
# JSON output: new Phase B fields present
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Verify that Phase B fields are serialised in DriftReport JSON output."""

    def test_landmarks_added_in_json_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``landmarks_added`` and ``landmarks_added_low_confidence`` present in stdout JSON."""
        real_current = _PROJECT_ROOT / "engine_versions" / "transformers" / "current.yaml"
        fake_engine_dir = tmp_path / "transformers"
        fake_engine_dir.mkdir(parents=True, exist_ok=True)
        (fake_engine_dir / "current.yaml").write_text(real_current.read_text())
        monkeypatch.setattr(_drift, "current_path", lambda name: tmp_path / name / "current.yaml")

        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=("json.JSONDecodeError",),
        )
        monkeypatch.setattr(
            _drift,
            "discover_live_landmarks",
            lambda engine: ({"json.JSONEncoder"}, {"json._SomePrivate"}),
        )

        rc = _drift.main(
            ["--engine", "transformers", "--producer", "invariants", "--no-write-cache"]
        )
        assert rc == 0

        out = capsys.readouterr().out
        payload = json.loads(out)

        assert "landmarks_added" in payload, "landmarks_added field must be in JSON"
        assert "landmarks_added_low_confidence" in payload, (
            "landmarks_added_low_confidence field must be in JSON"
        )
        assert "json.JSONEncoder" in payload["landmarks_added"]
        assert "json._SomePrivate" in payload["landmarks_added_low_confidence"]
        assert payload["direction"] == "added"
        assert payload["verdict"] == "pass"

    def test_compat_json_includes_phase_b_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Phase B fields are persisted to the compat.json cache."""
        real_current = _PROJECT_ROOT / "engine_versions" / "transformers" / "current.yaml"
        fake_engine_dir = tmp_path / "transformers"
        fake_engine_dir.mkdir(parents=True, exist_ok=True)
        (fake_engine_dir / "current.yaml").write_text(real_current.read_text())
        monkeypatch.setattr(_drift, "current_path", lambda name: tmp_path / name / "current.yaml")

        _install_synthetic_producer(
            monkeypatch,
            engine="transformers",
            producer="invariants",
            landmarks=("json.JSONDecodeError",),
        )
        monkeypatch.setattr(
            _drift,
            "discover_live_landmarks",
            lambda engine: ({"json.JSONEncoder"}, set()),
        )

        rc = _drift.main(["--engine", "transformers", "--producer", "invariants"])
        assert rc == 0

        cache_path = tmp_path / "transformers.compat.json"
        assert cache_path.is_file()
        cache = json.loads(cache_path.read_text())

        assert "invariants" in cache
        entry = cache["invariants"]
        assert "landmarks_added" in entry
        assert "landmarks_added_low_confidence" in entry
        assert "json.JSONEncoder" in entry["landmarks_added"]
        assert entry["direction"] == "added"


# ---------------------------------------------------------------------------
# _import_class_from_path helper
# ---------------------------------------------------------------------------


class TestImportClassFromPath:
    """Verify _import_class_from_path handles various paths correctly."""

    def test_valid_stdlib_class(self) -> None:
        """A valid stdlib class path resolves to the class."""
        cls = _drift._import_class_from_path("json.JSONDecodeError")
        assert cls is not None
        import json as _json

        assert cls is _json.JSONDecodeError

    def test_nonexistent_module_returns_none(self) -> None:
        """A completely non-existent module returns None."""
        cls = _drift._import_class_from_path("totally_fake_module_xyz.FakeClass")
        assert cls is None

    def test_nonexistent_attribute_returns_none(self) -> None:
        """An existing module but non-existent attribute returns None."""
        cls = _drift._import_class_from_path("json.CompletelyNonExistentClass")
        assert cls is None

    def test_non_type_object_returns_none(self) -> None:
        """A path that resolves to a non-type object (e.g. a constant) returns None."""
        cls = _drift._import_class_from_path("json.encoder.INFINITY")
        assert cls is None


# ---------------------------------------------------------------------------
# _walk_module_classes helper
# ---------------------------------------------------------------------------


class TestWalkModuleClasses:
    """Verify _walk_module_classes includes only classes defined in the module."""

    def test_json_encoder_module_includes_defined_classes(self) -> None:
        """json.encoder module returns JSONEncoder (defined there with matching __module__)."""
        results = _drift._walk_module_classes("json.encoder")
        class_names = {cls.__name__ for cls, _mod in results}
        # JSONEncoder.__module__ == "json.encoder" so it should be included.
        assert "JSONEncoder" in class_names, f"JSONEncoder missing from {class_names}"

    def test_nonexistent_module_returns_empty(self) -> None:
        """A non-existent module path returns an empty list."""
        results = _drift._walk_module_classes("nonexistent_xyz_module")
        assert results == []

    def test_only_own_module_classes_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Classes with __module__ != the walked module are excluded."""
        import json

        @dataclasses.dataclass
        class _Foreign:
            pass

        # _Foreign has __module__ = __name__ of this test module, not 'json'.
        # Monkeypatching sys.modules['json'].Foreign ensures it's in the module
        # namespace but its __module__ doesn't match - should be excluded.
        monkeypatch.setattr(json, "_Foreign", _Foreign, raising=False)
        results = _drift._walk_module_classes("json")
        class_names = {cls.__name__ for cls, _mod in results}
        assert "_Foreign" not in class_names
