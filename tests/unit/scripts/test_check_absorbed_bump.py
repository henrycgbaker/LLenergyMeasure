"""Tests for :mod:`scripts.ci.check_absorbed_bump` - the half-absorbed-bump guard.

The guard is a pure path check: given the PR's changed-file list, it flags any
engine whose ``current.yaml`` pin moved without the accompanying regenerated
snapshot outputs AND packaged src copies. These tests pin the pass/fail
boundary (bare bump fails, full absorb passes, per-leg omissions fail) and the
multi-engine / non-bump cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts" / "ci"))

import check_absorbed_bump as cab


def _full_absorb(engine: str) -> list[str]:
    """Changed-file list for a complete absorb of ``engine`` (mirrors PR #792)."""
    return [
        f"engine_versions/{engine}/current.yaml",
        f"engine_versions/{engine}/v9_9_9/outputs/schema.discovered.json",
        f"engine_versions/{engine}/v9_9_9/outputs/curated.yaml",
        f"src/llenergymeasure/engines/{engine}/config.py",
        f"src/llenergymeasure/engines/{engine}/rules.yaml",
        f"src/llenergymeasure/engines/{engine}/schema.discovered.json",
    ]


def test_full_absorb_passes() -> None:
    assert cab.check_absorbed_bump(_full_absorb("tensorrt")) == []


def test_no_pin_change_passes() -> None:
    """An unrelated PR that never touches a pin is not the guard's concern."""
    changed = [
        "src/llenergymeasure/cli/run.py",
        "docs/reference/cli.md",
        "src/llenergymeasure/engines/vllm/plugin.py",
    ]
    assert cab.check_absorbed_bump(changed) == []


def test_bare_pin_bump_fails() -> None:
    """The Renovate shape: only current.yaml moves. Both legs are missing."""
    errors = cab.check_absorbed_bump(["engine_versions/vllm/current.yaml"])
    assert len(errors) == 1
    msg = errors[0]
    assert msg.startswith("vllm:")
    assert "engine_versions/vllm/<version>/outputs/" in msg
    assert "src/llenergymeasure/engines/vllm/" in msg
    assert "make absorb ENGINE=vllm" in msg


def test_missing_src_copies_fails() -> None:
    """Snapshot regenerated but the packaged src copies were not promoted."""
    changed = [
        "engine_versions/transformers/current.yaml",
        "engine_versions/transformers/v9_9_9/outputs/schema.discovered.json",
    ]
    errors = cab.check_absorbed_bump(changed)
    assert len(errors) == 1
    assert "src/llenergymeasure/engines/transformers/" in errors[0]
    assert "engine_versions/transformers/<version>/outputs/" not in errors[0]


def test_missing_snapshot_outputs_fails() -> None:
    """src copies changed but no versioned snapshot output was committed."""
    changed = [
        "engine_versions/tensorrt/current.yaml",
        "src/llenergymeasure/engines/tensorrt/config.py",
    ]
    errors = cab.check_absorbed_bump(changed)
    assert len(errors) == 1
    assert "engine_versions/tensorrt/<version>/outputs/" in errors[0]
    assert "src/llenergymeasure/engines/tensorrt/" not in errors[0]


def test_plugin_only_does_not_count_as_src_evidence() -> None:
    """plugin.py is hand-written glue, not an absorb output - it is not evidence."""
    changed = [
        "engine_versions/vllm/current.yaml",
        "engine_versions/vllm/v9_9_9/outputs/curated.yaml",
        "src/llenergymeasure/engines/vllm/plugin.py",
    ]
    errors = cab.check_absorbed_bump(changed)
    assert len(errors) == 1
    assert "src/llenergymeasure/engines/vllm/" in errors[0]


def test_multiple_engines_reported_independently() -> None:
    """A full absorb of one engine plus a bare bump of another: only the bare one fails."""
    changed = [*_full_absorb("vllm"), "engine_versions/tensorrt/current.yaml"]
    errors = cab.check_absorbed_bump(changed)
    assert len(errors) == 1
    assert errors[0].startswith("tensorrt:")


def test_blank_lines_ignored() -> None:
    """Trailing blank lines from a piped `git diff` are stripped, not misparsed."""
    changed = ["", "engine_versions/vllm/current.yaml", "  ", ""]
    errors = cab.check_absorbed_bump(changed)
    assert len(errors) == 1
    assert errors[0].startswith("vllm:")


def test_main_file_input_bare_bump(tmp_path: Path) -> None:
    diff = tmp_path / "changed.txt"
    diff.write_text("engine_versions/vllm/current.yaml\n", encoding="utf-8")
    assert cab.main(["--changed-files", str(diff)]) == 1


def test_main_file_input_full_absorb(tmp_path: Path) -> None:
    diff = tmp_path / "changed.txt"
    diff.write_text("\n".join(_full_absorb("tensorrt")) + "\n", encoding="utf-8")
    assert cab.main(["--changed-files", str(diff)]) == 0
