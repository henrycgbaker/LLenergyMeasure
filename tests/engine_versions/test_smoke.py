"""Smoke test for the engine_versions package.

Asserts that ``engine_versions.__file__`` resolves to a path under
``<repo>/engine_versions/``. Catches the case where a stale pip-installed
wheel of llenergymeasure might somehow shadow the in-tree source. Because
``engine_versions/`` is at the repo root (outside ``src/``), it should never
appear in a wheel, making shadowing effectively impossible - but the test
is cheap and provides a clear failure message if the import path is wrong.
"""

from __future__ import annotations

from pathlib import Path

import engine_versions as archive

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_engine_versions_loaded_from_source_tree() -> None:
    """``engine_versions.__file__`` must resolve under the repo's engine_versions/ tree.

    If a stale pip-installed wheel is somehow masking the source tree,
    this test fails loud rather than letting the dispatcher silently load
    empty/missing machinery.
    """
    archive_file = Path(archive.__file__).resolve()
    expected = _REPO_ROOT / "engine_versions" / "__init__.py"
    assert archive_file == expected.resolve(), (
        f"engine_versions.__file__ resolved to {archive_file}, expected {expected}. "
        "A stale installation may be masking the in-tree archive. "
        "Re-install in editable mode (`uv sync --dev`) or remove the cached wheel."
    )
