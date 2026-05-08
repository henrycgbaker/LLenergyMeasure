"""Smoke test for the _engine_archive package.

Asserts that ``llenergymeasure._engine_archive.__file__`` resolves to a
path under ``<repo>/src/llenergymeasure/_engine_archive/``. Catches the
case where a stale pip-installed wheel of llenergymeasure (without the
archive package) masks the in-tree source. This is the FM 1 mitigation
recommended by the second adversarial pass on PR-0.
"""

from __future__ import annotations

from pathlib import Path

import llenergymeasure._engine_archive as archive

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_engine_archive_loaded_from_source_tree() -> None:
    """``_engine_archive.__file__`` must resolve under the repo's src/ tree.

    If a stale pip-installed wheel is masking the source tree (e.g. user
    has ``pip install llenergymeasure==0.9.0`` from before the archive
    existed), this test fails loud rather than letting the dispatcher
    silently load empty/missing machinery.
    """
    archive_file = Path(archive.__file__).resolve()
    expected = _REPO_ROOT / "src" / "llenergymeasure" / "_engine_archive" / "__init__.py"
    assert archive_file == expected.resolve(), (
        f"_engine_archive.__file__ resolved to {archive_file}, expected {expected}. "
        "A stale pip-installed wheel may be masking the in-tree archive. "
        "Re-install in editable mode (`uv sync --dev`) or remove the cached wheel."
    )
