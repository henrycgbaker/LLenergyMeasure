"""Tests for scripts/check_discovered_schema_versions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the script's main function directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from check_discovered_schema_versions import main


def _current_toml(version: str) -> str:
    """Render a minimal engine current.toml carrying ``library.current_version``."""
    return f'[library]\ncurrent_version = "{version}"\n'


def _setup_repo(
    tmp_path: Path,
    *,
    vllm_current: str = "0.7.3",
    vllm_schema_version: str = "0.7.3",
    trt_current: str = "0.21.0",
    trt_schema_version: str = "0.21.0",
    transformers_current: str = "5.5.4",
    transformers_schema_version: str = "5.5.4",
    skip_vllm_schema: bool = False,
) -> Path:
    """Create a minimal repo structure for the version check script."""
    repo = tmp_path / "repo"
    engine_versions_dir = repo / "engine_versions"

    for engine, version in [
        ("vllm", vllm_current),
        ("tensorrt", trt_current),
        ("transformers", transformers_current),
    ]:
        engine_dir = engine_versions_dir / engine
        engine_dir.mkdir(parents=True)
        (engine_dir / "current.toml").write_text(_current_toml(version))

    engines_dir = repo / "src" / "llenergymeasure" / "engines"

    def _write_schema(engine: str, version: str) -> None:
        engine_dir = engines_dir / engine
        engine_dir.mkdir(parents=True, exist_ok=True)
        (engine_dir / "schema.discovered.json").write_text(json.dumps({"engine_version": version}))

    if not skip_vllm_schema:
        _write_schema("vllm", vllm_schema_version)
    _write_schema("tensorrt", trt_schema_version)
    _write_schema("transformers", transformers_schema_version)

    return repo


class TestVersionsMatch:
    def test_all_match(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert main(repo_root=repo) == 0

    def test_v_prefix_normalised(self, tmp_path: Path):
        """v0.7.3 in current.toml should match 0.7.3 in schema."""
        repo = _setup_repo(tmp_path, vllm_current="v0.7.3", vllm_schema_version="0.7.3")
        assert main(repo_root=repo) == 0


class TestMismatch:
    def test_version_mismatch(self, tmp_path: Path, capsys):
        repo = _setup_repo(tmp_path, vllm_current="0.8.0", vllm_schema_version="0.7.3")
        code = main(repo_root=repo)
        assert code == 1
        captured = capsys.readouterr()
        assert "MISMATCH" in captured.err
        assert "refresh_discovered_schemas.sh" in captured.err

    def test_transformers_mismatch(self, tmp_path: Path, capsys):
        repo = _setup_repo(
            tmp_path,
            transformers_current="5.6.0",
            transformers_schema_version="5.5.4",
        )
        code = main(repo_root=repo)
        assert code == 1
        captured = capsys.readouterr()
        assert "MISMATCH" in captured.err
        assert "transformers" in captured.err


class TestErrors:
    def test_missing_schema_file(self, tmp_path: Path, capsys):
        repo = _setup_repo(tmp_path, skip_vllm_schema=True)
        code = main(repo_root=repo)
        assert code == 2
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
