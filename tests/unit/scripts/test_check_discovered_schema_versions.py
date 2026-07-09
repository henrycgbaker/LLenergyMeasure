"""Tests for scripts/check_discovered_schema_versions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the script's main function directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from check_discovered_schema_versions import main

from engine_versions._outputs import safe_version as _safe_version


def _current_yaml(version: str) -> str:
    """Render a minimal engine current.yaml carrying ``library.current_version``."""
    return f"library:\n  current_version: {version}\n"


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
    vllm_src_surface: dict | None = None,
    vllm_outputs_surface: dict | None = None,
) -> Path:
    """Create a minimal repo structure for the version check script.

    Each engine gets a current.yaml pin, a src/ shadow schema, and a versioned
    outputs/ snapshot schema. The outputs dir is named by the pin (``*_current``)
    while the schemas carry ``*_schema_version``, so the two can diverge to drive
    a version-mismatch test. ``vllm_src_surface`` / ``vllm_outputs_surface``
    override the two vllm parameter surfaces so a surface divergence can be
    simulated (they default to the same empty surface).
    """
    repo = tmp_path / "repo"
    engine_versions_dir = repo / "engine_versions"
    engines_dir = repo / "src" / "llenergymeasure" / "engines"
    empty_surface: dict[str, dict] = {"engine_params": {}, "sampling_params": {}}

    specs = [
        ("vllm", vllm_current, vllm_schema_version, vllm_src_surface, vllm_outputs_surface),
        ("tensorrt", trt_current, trt_schema_version, None, None),
        ("transformers", transformers_current, transformers_schema_version, None, None),
    ]
    for engine, current, schema_version, src_surface, outputs_surface in specs:
        engine_dir = engine_versions_dir / engine
        engine_dir.mkdir(parents=True)
        (engine_dir / "current.yaml").write_text(_current_yaml(current))
        if engine == "vllm" and skip_vllm_schema:
            continue
        # src/ shadow (runtime loader + absorb read this copy)
        src_dir = engines_dir / engine
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "schema.discovered.json").write_text(
            json.dumps({"engine_version": schema_version, **(src_surface or empty_surface)})
        )
        # versioned outputs/ snapshot codegen reads (dir named by the pin)
        out_dir = engine_versions_dir / engine / _safe_version(current) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "schema.discovered.json").write_text(
            json.dumps({"engine_version": schema_version, **(outputs_surface or empty_surface)})
        )

    return repo


class TestVersionsMatch:
    def test_all_match(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert main(repo_root=repo) == 0

    def test_v_prefix_normalised(self, tmp_path: Path):
        """v0.7.3 in current.yaml should match 0.7.3 in schema."""
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

    def test_surface_divergence_between_copies(self, tmp_path: Path, capsys):
        """Versions agree, but the src/ and outputs/ parameter surfaces differ."""
        repo = _setup_repo(
            tmp_path,
            vllm_src_surface={"engine_params": {"max_num_seqs": {}}, "sampling_params": {}},
            vllm_outputs_surface={"engine_params": {}, "sampling_params": {}},
        )
        code = main(repo_root=repo)
        assert code == 1
        captured = capsys.readouterr()
        assert "SURFACE MISMATCH" in captured.err
        assert "vllm" in captured.err


class TestErrors:
    def test_missing_schema_file(self, tmp_path: Path, capsys):
        repo = _setup_repo(tmp_path, skip_vllm_schema=True)
        code = main(repo_root=repo)
        assert code == 2
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
