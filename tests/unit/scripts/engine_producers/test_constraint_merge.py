"""D3 plumbing tests: source-text constraints fold onto discovered schema fields.

The schema introspectors capture type + default from runtime introspection but
not the ``Field(ge/le/...)`` numeric bounds or ``Literal[...]`` membership sets
that live in class source. :func:`merge_source_constraints` walks the source and
overlays those onto the matching discovered fields. These tests drive the merge
with a synthetic source file (no live engine needed), pinning that a
``Field(ge=..., le=...)`` becomes ``minimum`` / ``maximum`` and a ``Literal[...]``
becomes ``enum`` on the envelope, and that discovery's type/default win on
conflict.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._common import merge_source_constraints  # noqa: E402

_SYNTHETIC_SOURCE = """
from typing import Literal, Optional

from pydantic import Field


class WidgetConfig:
    block_size: int = Field(8, ge=1, le=256, multiple_of=8)
    ratio: float = Field(1.0, gt=0.0, le=1.0)
    mode: Literal["auto", "fast", "slow"]
    cache_dtype: Optional[Literal["fp8", "fp16"]] = None
    # No declarative constraint -> contributes nothing.
    name: str = "widget"
"""


def _write_source(tmp_path: Path) -> Path:
    src = tmp_path / "widget_config.py"
    src.write_text(_SYNTHETIC_SOURCE)
    return src


def test_field_bounds_and_literal_fold_onto_schema(tmp_path: Path) -> None:
    # Discovered fields carry type/default; the merge adds bounds + enum.
    schema_fields: dict[str, dict[str, Any]] = {
        "block_size": {"type": "int", "default": 8},
        "ratio": {"type": "float", "default": 1.0},
        "mode": {"type": "str", "default": "auto"},
        "cache_dtype": {"type": "unknown", "default": None},
        "name": {"type": "str", "default": "widget"},
    }
    touched = merge_source_constraints(schema_fields, [_write_source(tmp_path)])

    assert schema_fields["block_size"] == {
        "type": "int",
        "default": 8,
        "minimum": 1,
        "maximum": 256,
        "multipleOf": 8,
    }
    assert schema_fields["ratio"] == {
        "type": "float",
        "default": 1.0,
        "exclusiveMinimum": 0.0,
        "maximum": 1.0,
    }
    assert schema_fields["mode"]["enum"] == ["auto", "fast", "slow"]
    # Optional[Literal[...]] unwraps to the membership set.
    assert schema_fields["cache_dtype"]["enum"] == ["fp8", "fp16"]
    # No constraint field is left untouched.
    assert schema_fields["name"] == {"type": "str", "default": "widget"}
    # Four fields gained a constraint key.
    assert touched == 4


def test_discovery_type_and_default_are_not_overwritten(tmp_path: Path) -> None:
    # A pre-existing key (here a deliberately-wrong default) is preserved;
    # discovery is the source of truth for type/default.
    schema_fields = {"block_size": {"type": "int", "default": 99}}
    merge_source_constraints(schema_fields, [_write_source(tmp_path)])
    assert schema_fields["block_size"]["default"] == 99
    assert schema_fields["block_size"]["minimum"] == 1


def test_fields_absent_from_discovery_are_not_added(tmp_path: Path) -> None:
    # The source declares block_size/ratio/mode/cache_dtype, but discovery only
    # found ``ratio`` - the others must not be invented (discovery is canonical).
    schema_fields = {"ratio": {"type": "float", "default": 1.0}}
    touched = merge_source_constraints(schema_fields, [_write_source(tmp_path)])
    assert set(schema_fields) == {"ratio"}
    assert touched == 1


def test_missing_source_path_is_a_noop(tmp_path: Path) -> None:
    schema_fields = {"block_size": {"type": "int", "default": 8}}
    touched = merge_source_constraints(schema_fields, [tmp_path / "does_not_exist.py"])
    assert touched == 0
    assert schema_fields == {"block_size": {"type": "int", "default": 8}}
