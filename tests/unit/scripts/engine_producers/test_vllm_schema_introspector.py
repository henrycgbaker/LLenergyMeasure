"""Host test for the vLLM schema introspector's msgspec type recovery (W1.2).

The introspector recovers ``SamplingParams`` field types via
``msgspec.json.schema``, then folds ``_msgspec_lift.recover_field_types`` onto
any field that came out ``"unknown"`` (the union / enum / nested-struct fields
``json.schema`` renders as an untyped anyOf). This test drives the real per-pin
``discover`` against a synthetic ``vllm`` module so it runs on a CPU host with
no vLLM installed, and asserts:

- previously-``unknown`` union / container fields gain their concrete type;
- a field ``json.schema`` already typed is NOT overwritten;
- a genuinely-opaque ``Any | None`` field stays ``unknown``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

msgspec = pytest.importorskip("msgspec")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _FakeSamplingParams(msgspec.Struct):  # type: ignore[name-defined,misc]
    """Synthetic stand-in mirroring the vLLM ``SamplingParams`` type shape."""

    n: int = 1  # json.schema renders this typed - must NOT be overwritten
    seed: int | None = None  # optional union -> json.schema "unknown"
    logit_bias: dict[int, float] | None = None  # dict union -> "unknown"
    opaque: Any | None = None  # genuinely opaque -> stays "unknown"


@dataclass
class _FakeEngineArgs:
    model: str = "x"


@pytest.fixture
def discover(monkeypatch: pytest.MonkeyPatch):
    """Load the real per-pin ``discover`` with a synthetic ``vllm`` injected."""
    from scripts.engine_producers._current import current_version

    pin = current_version("vllm")
    vllm = types.ModuleType("vllm")
    vllm.__version__ = f"{pin}-test"  # type: ignore[attr-defined]
    vllm.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = _FakeEngineArgs  # type: ignore[attr-defined]
    modules = {
        "vllm": vllm,
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": arg_utils,
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.sampling_params": types.ModuleType("vllm.sampling_params"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    from engine_versions._dispatcher import load_producer

    producer = load_producer(engine="vllm", version=pin, producer="schema_introspector")
    return producer.discover


def test_introspector_recovers_unknown_sampling_types(discover) -> None:
    envelope = discover(Path("."), image_ref=None)
    sampling = envelope["sampling_params"]
    assert sampling["seed"]["type"] == "int | None"
    assert sampling["logit_bias"]["type"] == "dict[int, float] | None"


def test_introspector_does_not_overwrite_json_schema_type(discover) -> None:
    envelope = discover(Path("."), image_ref=None)
    # json.schema renders a plain ``int`` field as "integer"; the msgspec fold
    # only touches fields still marked "unknown", so this stays untouched.
    assert envelope["sampling_params"]["n"]["type"] == "integer"


def test_introspector_leaves_opaque_any_field_unknown(discover) -> None:
    envelope = discover(Path("."), image_ref=None)
    assert envelope["sampling_params"]["opaque"]["type"] == "unknown"
