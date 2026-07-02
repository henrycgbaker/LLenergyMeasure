"""Acceptance tests for :mod:`scripts.engine_producers._source_walker`.

Exercises the source-level walkers against trimmed engine source-excerpt
fixtures WITHOUT any engine installed: the walkers read source text via ``ast``,
so the fixtures only need to parse, never import or run.

Two fixture cells:

- ``fixtures/vllm_019/`` - a ``config/*.py`` subpackage with pydantic ``Field``
  bounds, ``Literal`` membership, and a nested pydantic sub-config.
- ``fixtures/tensorrt_12/`` - a ``plugin/plugin.py`` PluginConfig-like class with
  inline ``Literal`` membership fields.

The assertions pin the class / field / constraint inventories the walkers must
surface, so a regression in any walker fails here rather than silently shrinking
the discovered constraints on the next bump.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import _source_walker as sw  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_VLLM_ROOT = _FIXTURES / "vllm_019"
_TRT_ROOT = _FIXTURES / "tensorrt_12"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


# ---------------------------------------------------------------------------
# Subpackage glob
# ---------------------------------------------------------------------------


def test_expand_files_finds_config_subpackage() -> None:
    found = sw.expand_files(_VLLM_ROOT, ["config.py", "config/*.py", "sampling_params.py"])
    names = sorted(p.name for p in found)
    # config.py and sampling_params.py do not exist here (flat layout absent);
    # the glob finds the two real subpackage files and skips the rest.
    assert names == ["cache.py", "scheduler.py"]


def test_expand_files_graceful_when_surface_absent() -> None:
    """vllm 0.7.3 has no config/ subpackage: the glob simply finds nothing."""
    found = sw.expand_files(_VLLM_ROOT, ["nonexistent/*.py", "also_missing.py"])
    assert found == []


def test_expand_files_finds_plugin_module() -> None:
    found = sw.expand_files(_TRT_ROOT, ["plugin/*.py", "llmapi/llm_args.py"])
    assert [p.name for p in found] == ["plugin.py"]


# ---------------------------------------------------------------------------
# Class-surface enumeration
# ---------------------------------------------------------------------------


def test_iter_config_classes_finds_sibling_and_nested_classes() -> None:
    module = _parse(_VLLM_ROOT / "config" / "cache.py")
    names = {c.name for c in sw.iter_config_classes(module)}
    # Both the entry-ish CacheConfig and the sibling PrefixCacheConfig surface;
    # underscore-prefixed names never do.
    assert names == {"CacheConfig", "PrefixCacheConfig"}


def test_iter_config_classes_matches_params_and_args_suffixes() -> None:
    src = "class FooParams: ...\nclass BarArgs: ...\nclass Helper: ...\nclass _Hidden: ...\n"
    module = ast.parse(src)
    names = {c.name for c in sw.iter_config_classes(module)}
    assert names == {"FooParams", "BarArgs"}


# ---------------------------------------------------------------------------
# Declarative-constraint walk - vllm cell
# ---------------------------------------------------------------------------


def test_vllm_field_bounds_to_jsonschema() -> None:
    module = _parse(_VLLM_ROOT / "config" / "cache.py")
    constraints = sw.walk_declarative_constraints(module)
    cache = constraints["CacheConfig"]
    # block_size: ge=1, le=256, multiple_of=8 -> canonical JSON Schema keys.
    assert cache["block_size"] == {"minimum": 1, "maximum": 256, "multipleOf": 8}
    # gpu_memory_utilization: gt=0.0, le=1.0 -> exclusiveMinimum + maximum.
    assert cache["gpu_memory_utilization"] == {"exclusiveMinimum": 0.0, "maximum": 1.0}
    assert cache["swap_space"] == {"minimum": 0}


def test_vllm_literal_membership_and_optional_unwrap() -> None:
    module = _parse(_VLLM_ROOT / "config" / "cache.py")
    cache = sw.walk_declarative_constraints(module)["CacheConfig"]
    assert cache["cache_dtype"]["enum"] == ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
    # Optional[Literal[...]] still surfaces its membership set.
    assert cache["mamba_ssm_cache_dtype"]["enum"] == ["auto", "float32"]
    # Private fields never surface.
    assert "_private_internal" not in cache


def test_vllm_annotated_field_bounds_lifted_from_annotation() -> None:
    module = _parse(_VLLM_ROOT / "config" / "scheduler.py")
    sched = sw.walk_declarative_constraints(module)["SchedulerConfig"]
    # Annotated[int, Field(ge=1)] - bound lives in the annotation, not the RHS.
    assert sched["max_num_batched_tokens"] == {"minimum": 1}
    assert sched["max_num_seqs"] == {"minimum": 1}
    # A plain ``str`` field with no constraints does not appear.
    assert "policy" not in sched


# ---------------------------------------------------------------------------
# Declarative-constraint walk - tensorrt PluginConfig cell
# ---------------------------------------------------------------------------


def test_tensorrt_plugin_literal_membership_inventory() -> None:
    module = _parse(_TRT_ROOT / "plugin" / "plugin.py")
    plugin = sw.walk_declarative_constraints(module)["PluginConfig"]

    # Inline Literal fields surface their own value sets (incl. an int Literal).
    assert plugin["context_fmha"]["enum"] == ["enabled", "disabled"]
    assert plugin["tokens_per_block"]["enum"] == [32, 64, 128]

    # The 10 inline Literal fields each carry a membership set. Fields typed
    # against a module-level Literal alias (``DefaultPluginDtype``) are NOT
    # closed without alias resolution, so they do not surface an enum.
    membership_fields = {f for f, frag in plugin.items() if "enum" in frag}
    assert len(membership_fields) == 10
    assert "enum" not in plugin.get("gpt_attention_plugin", {})
    assert "enum" not in plugin.get("moe_plugin", {})
    assert "max_lora_rank" not in plugin  # plain int, no constraint
    assert "_internal" not in plugin
