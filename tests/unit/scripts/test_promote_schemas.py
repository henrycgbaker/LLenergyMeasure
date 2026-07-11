"""Tests for scripts/promote_schemas.py.

Promotion must be a verbatim byte-copy of the versioned outputs/ snapshot into
the packaged src copy, and re-running it must be a no-op. The tests drive the
copy on a throwaway fixture repo so the real committed schema bytes are never
touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from promote_schemas import main, promote_engine

from engine_versions._outputs import safe_version as _safe_version

ENGINES = ("vllm", "tensorrt", "transformers")
_DEFAULT_VERSIONS = {"vllm": "0.7.3", "tensorrt": "0.21.0", "transformers": "5.5.4"}
_STALE_SRC = b'{"engine_version": "OLD"}\n'


def _current_yaml(version: str) -> str:
    return f"library:\n  current_version: {version}\n"


def _snapshot_bytes(engine: str, version: str) -> bytes:
    """A realistic multi-line snapshot: envelope keys plus a parameter surface.

    The bytes vary by engine and version so a cross-wired copy would be caught,
    and the envelope (image_ref/discovered_at) proves the copy is the full file,
    not just the surface the CI guard compares.
    """
    lines = [
        "{",
        f'  "engine": "{engine}",',
        f'  "engine_version": "{version}",',
        f'  "image_ref": "pristine/{engine}:{version}",',
        '  "discovered_at": "2026-01-01T00:00:00+00:00",',
        '  "engine_params": {"max_num_seqs": {"type": "integer"}},',
        '  "sampling_params": {}',
        "}",
        "",
    ]
    return "\n".join(lines).encode()


def _src(repo: Path, engine: str) -> Path:
    return repo / "src" / "llenergymeasure" / "engines" / engine / "schema.discovered.json"


def _out(repo: Path, engine: str, version: str) -> Path:
    return (
        repo / "engine_versions" / engine / _safe_version(version) / "outputs"
    ) / "schema.discovered.json"


def _setup_repo(
    tmp_path: Path,
    *,
    src_stale: bool = True,
    skip_vllm_outputs: bool = False,
) -> Path:
    """Create a throwaway repo: a pin, an outputs/ snapshot, and a src/ copy per engine.

    The src copy starts with different (``_STALE_SRC``) bytes so a successful
    promotion is observable. ``skip_vllm_outputs`` drops vllm's snapshot to
    exercise the missing-snapshot error path.
    """
    repo = tmp_path / "repo"
    engine_versions_dir = repo / "engine_versions"

    for engine in ENGINES:
        version = _DEFAULT_VERSIONS[engine]
        eng_dir = engine_versions_dir / engine
        eng_dir.mkdir(parents=True)
        (eng_dir / "current.yaml").write_text(_current_yaml(version))

        if engine == "vllm" and skip_vllm_outputs:
            continue

        out_path = _out(repo, engine, version)
        out_path.parent.mkdir(parents=True)
        out_path.write_bytes(_snapshot_bytes(engine, version))

        src_path = _src(repo, engine)
        src_path.parent.mkdir(parents=True)
        if src_stale:
            src_path.write_bytes(_STALE_SRC)

    return repo


class TestByteCopy:
    def test_promote_is_verbatim_byte_copy(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        changed = promote_engine("vllm", repo)
        assert changed is True
        # Full-file identity, envelope included - not just the compared surface.
        assert _src(repo, "vllm").read_bytes() == _out(repo, "vllm", "0.7.3").read_bytes()

    def test_promote_idempotent(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert promote_engine("vllm", repo) is True  # first run rewrites the stale src
        first = _src(repo, "vllm").read_bytes()
        assert promote_engine("vllm", repo) is False  # second run is a no-op
        second = _src(repo, "vllm").read_bytes()
        assert first == second == _out(repo, "vllm", "0.7.3").read_bytes()

    def test_promote_creates_missing_src(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        _src(repo, "vllm").unlink()
        assert promote_engine("vllm", repo) is True
        assert _src(repo, "vllm").read_bytes() == _out(repo, "vllm", "0.7.3").read_bytes()


class TestMain:
    def test_main_promotes_all_engines(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert main(repo_root=repo) == 0
        for engine, version in _DEFAULT_VERSIONS.items():
            assert _src(repo, engine).read_bytes() == _out(repo, engine, version).read_bytes()

    def test_main_single_engine_leaves_others_untouched(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert main(repo_root=repo, engines=("vllm",)) == 0
        assert _src(repo, "vllm").read_bytes() == _out(repo, "vllm", "0.7.3").read_bytes()
        assert _src(repo, "tensorrt").read_bytes() == _STALE_SRC
        assert _src(repo, "transformers").read_bytes() == _STALE_SRC

    def test_main_double_run_is_byte_stable(self, tmp_path: Path):
        repo = _setup_repo(tmp_path)
        assert main(repo_root=repo) == 0
        after_first = {engine: _src(repo, engine).read_bytes() for engine in ENGINES}
        assert main(repo_root=repo) == 0
        after_second = {engine: _src(repo, engine).read_bytes() for engine in ENGINES}
        assert after_first == after_second

    def test_main_missing_snapshot_errors(self, tmp_path: Path, capsys):
        repo = _setup_repo(tmp_path, skip_vllm_outputs=True)
        code = main(repo_root=repo)
        assert code == 2
        assert "ERROR" in capsys.readouterr().err
