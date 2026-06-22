"""Tests for scripts/engine_producers/regen_engine_corpus.py.

Two concerns:

1. The sync mechanism (tmp-dir fixtures): in-sync passes --check, a drifted
   file fails --check with a per-file report, and --write resyncs.
2. Curated.yaml integrity against the real repo: every exposed field is in
   the matching pin's discovered schema OR carries a discovery-debt marker;
   all 3 curated.yaml files load; the 7 harness knobs are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.engine_producers import regen_engine_corpus as rec  # noqa: E402

# ---------------------------------------------------------------------------
# Sync mechanism (hermetic tmp-dir fixtures)
# ---------------------------------------------------------------------------

_CORPUS = {
    "schema.discovered.json": '{"engine_version": "1.2.3"}\n',
    "rules.proposed.yaml": "invariants: []\n",
    "rules.validated.yaml": "invariants: []\n",
    "curated.yaml": "engine: demo\nexposed_fields:\n  engine_params: []\n",
}


@pytest.fixture
def fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a one-engine SSOT + shadow under tmp_path, wired into the module.

    Returns ``(outputs_dir, shadow_dir)`` both populated with identical
    corpus content (in-sync starting state).
    """
    outputs = tmp_path / "engine_versions" / "demo" / "v1_2_3" / "outputs"
    shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "demo"
    outputs.mkdir(parents=True)
    shadow.mkdir(parents=True)
    for name, body in _CORPUS.items():
        (outputs / name).write_text(body, encoding="utf-8")
        (shadow / name).write_text(body, encoding="utf-8")

    monkeypatch.setattr(rec, "ENGINES", ("demo",))
    monkeypatch.setattr(rec, "current_outputs_dir", lambda engine: outputs)
    monkeypatch.setattr(rec, "_shadow_dir", lambda engine: shadow)
    return outputs, shadow


def test_in_sync_passes_check(fake_corpus: tuple[Path, Path]) -> None:
    assert rec.main(["--check"]) == 0


def test_default_mode_is_check(fake_corpus: tuple[Path, Path]) -> None:
    """Bare invocation is --check (the safe, non-mutating default)."""
    _outputs, shadow = fake_corpus
    (shadow / "curated.yaml").write_text("engine: tampered\n", encoding="utf-8")
    assert rec.main([]) == 1
    # The shadow was not mutated by the bare (check) run.
    assert (shadow / "curated.yaml").read_text(encoding="utf-8") == "engine: tampered\n"


def test_drifted_file_fails_check_with_report(
    fake_corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _outputs, shadow = fake_corpus
    (shadow / "curated.yaml").write_text("engine: drifted\n", encoding="utf-8")

    assert rec.main(["--check"]) == 1
    err = capsys.readouterr().err
    assert "demo/curated.yaml drift:" in err
    # The unified diff names the offending content from both sides.
    assert "drifted" in err
    # Unaffected files are not reported.
    assert "demo/schema.discovered.json" not in err


def test_write_resyncs(fake_corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    _outputs, shadow = fake_corpus
    (shadow / "rules.proposed.yaml").write_text("invariants: [stale]\n", encoding="utf-8")

    assert rec.main(["--write"]) == 0
    out = capsys.readouterr().out
    assert "demo/rules.proposed.yaml" in out
    # Shadow now matches the SSOT, so a follow-up check is clean.
    assert (shadow / "rules.proposed.yaml").read_text(encoding="utf-8") == _CORPUS[
        "rules.proposed.yaml"
    ]
    assert rec.main(["--check"]) == 0


def test_write_is_noop_when_in_sync(
    fake_corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert rec.main(["--write"]) == 0
    assert "already in sync" in capsys.readouterr().out


def test_optional_overlay_synced_only_when_present(fake_corpus: tuple[Path, Path]) -> None:
    outputs, shadow = fake_corpus
    # No overlay yet: absence is not drift.
    assert rec.main(["--check"]) == 0

    (outputs / "overlay.yaml").write_text("narrowings: {}\n", encoding="utf-8")
    # Now the SSOT carries an overlay the shadow lacks -> drift, then synced.
    assert rec.main(["--check"]) == 1
    assert rec.main(["--write"]) == 0
    assert (shadow / "overlay.yaml").read_text(encoding="utf-8") == "narrowings: {}\n"


def test_missing_ssot_dir_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec, "ENGINES", ("demo",))
    monkeypatch.setattr(rec, "current_outputs_dir", lambda engine: tmp_path / "absent")
    monkeypatch.setattr(rec, "_shadow_dir", lambda engine: tmp_path / "shadow")
    with pytest.raises(FileNotFoundError, match="SSOT outputs dir not found"):
        rec.main(["--check"])


# ---------------------------------------------------------------------------
# --only scoping (parallel-cell bootstrap ordering)
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_schema_only_ssot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fresh-version SSOT carrying ONLY schema.discovered.json + curated.yaml.

    Mirrors the schemas cell's state on a fresh-version bump: the rules cell
    runs in parallel and has not written rules.*.yaml yet. Returns
    ``(outputs_dir, shadow_dir)`` with an empty shadow.
    """
    outputs = tmp_path / "engine_versions" / "demo" / "v9_0_0" / "outputs"
    shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "demo"
    outputs.mkdir(parents=True)
    shadow.mkdir(parents=True)
    (outputs / "schema.discovered.json").write_text(
        _CORPUS["schema.discovered.json"], encoding="utf-8"
    )
    (outputs / "curated.yaml").write_text(_CORPUS["curated.yaml"], encoding="utf-8")
    monkeypatch.setattr(rec, "ENGINES", ("demo",))
    monkeypatch.setattr(rec, "current_outputs_dir", lambda engine: outputs)
    monkeypatch.setattr(rec, "_shadow_dir", lambda engine: shadow)
    return outputs, shadow


def test_only_schema_derives_schema_without_rules_ssot(
    fresh_schema_only_ssot: tuple[Path, Path],
) -> None:
    """--only schema succeeds on a dir lacking the rules SSOT, syncs just schema."""
    outputs, shadow = fresh_schema_only_ssot
    assert not (outputs / "rules.proposed.yaml").exists()
    assert rec.main(["--write", "--only", "schema"]) == 0
    assert (shadow / "schema.discovered.json").read_text(encoding="utf-8") == _CORPUS[
        "schema.discovered.json"
    ]
    # Out-of-scope files are not pulled into the shadow.
    assert not (shadow / "rules.proposed.yaml").exists()
    assert not (shadow / "curated.yaml").exists()


def test_only_rules_derives_rules_without_schema_ssot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--only rules succeeds on a dir lacking the schema SSOT, syncs just rules."""
    outputs = tmp_path / "engine_versions" / "demo" / "v9_0_0" / "outputs"
    shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "demo"
    outputs.mkdir(parents=True)
    shadow.mkdir(parents=True)
    (outputs / "rules.proposed.yaml").write_text(_CORPUS["rules.proposed.yaml"], encoding="utf-8")
    (outputs / "rules.validated.yaml").write_text(_CORPUS["rules.validated.yaml"], encoding="utf-8")
    monkeypatch.setattr(rec, "ENGINES", ("demo",))
    monkeypatch.setattr(rec, "current_outputs_dir", lambda engine: outputs)
    monkeypatch.setattr(rec, "_shadow_dir", lambda engine: shadow)

    assert not (outputs / "schema.discovered.json").exists()
    assert rec.main(["--write", "--only", "rules"]) == 0
    assert (shadow / "rules.proposed.yaml").exists()
    assert (shadow / "rules.validated.yaml").exists()
    # Out-of-scope files are not pulled in (would FileNotFoundError on a full sync).
    assert not (shadow / "schema.discovered.json").exists()


def test_full_check_unchanged_by_scope_addition(fake_corpus: tuple[Path, Path]) -> None:
    """The unscoped full --check/--write still covers the whole corpus."""
    assert rec.main(["--check"]) == 0
    assert rec.main(["--check", "--only", "schema"]) == 0
    assert rec.main(["--check", "--only", "rules"]) == 0


# ---------------------------------------------------------------------------
# Curated.yaml integrity (real repo artefacts)
# ---------------------------------------------------------------------------

# Pin -> SSOT outputs dir for the 3 current engines.
_PINS = {
    "transformers": "v5_7_0",
    "vllm": "v0_7_3",
    "tensorrt": "v1_0_0",
}

# Genuine llem-orchestration knobs that belong to a future HarnessConfig,
# not engine curation. None of these may appear in any curated.yaml.
_HARNESS_KNOBS = {
    "batch_size",
    "torch_compile",
    "torch_compile_mode",
    "torch_compile_backend",
    "allow_tf32",
    "autocast_enabled",
    "autocast_dtype",
}

_DEBT_MARKER = "discovery debt"


def _outputs_dir(engine: str) -> Path:
    return REPO_ROOT / "engine_versions" / engine / _PINS[engine] / "outputs"


def _load_curated(engine: str) -> dict:
    return yaml.safe_load((_outputs_dir(engine) / "curated.yaml").read_text(encoding="utf-8"))


def _debt_marked_fields(engine: str) -> set[str]:
    """Field names whose curated.yaml line carries the discovery-debt marker."""
    marked: set[str] = set()
    for line in (_outputs_dir(engine) / "curated.yaml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and _DEBT_MARKER in line:
            # "- field_name  # discovery debt: ..."  ->  "field_name"
            marked.add(stripped[2:].split("#", 1)[0].strip())
    return marked


@pytest.mark.parametrize("engine", list(_PINS))
def test_curated_loads_and_shapes(engine: str) -> None:
    data = _load_curated(engine)
    assert data["engine"] == engine
    assert isinstance(data["schema_version"], str)
    exposed = data["exposed_fields"]
    assert set(exposed) <= {"engine_params", "sampling_params"}
    for section in exposed.values():
        assert isinstance(section, list)
        assert all(isinstance(f, str) for f in section)


@pytest.mark.parametrize("engine", list(_PINS))
def test_no_harness_knobs_in_curation(engine: str) -> None:
    data = _load_curated(engine)
    exposed: set[str] = set()
    for section in data["exposed_fields"].values():
        exposed.update(section)
    leaked = exposed & _HARNESS_KNOBS
    assert not leaked, f"{engine} curated.yaml leaks harness knobs: {sorted(leaked)}"


@pytest.mark.parametrize("engine", list(_PINS))
def test_every_field_discovered_or_debt_marked(engine: str) -> None:
    """Each exposed field is in the pin's discovered schema, or is debt-marked.

    The discovered engine_params/sampling_params split does not line up with
    the curated split (e.g. transformers generation knobs are discovered under
    sampling_params but curated under engine_params), so presence is checked
    against the union of both discovered sections - matching the
    LLEM_NATIVE_FIELDS allowlist semantics.
    """
    schema = json.loads((_outputs_dir(engine) / "schema.discovered.json").read_text("utf-8"))
    discovered = set(schema.get("engine_params", {})) | set(schema.get("sampling_params", {}))
    debt = _debt_marked_fields(engine)

    data = _load_curated(engine)
    offenders: list[str] = []
    for section in data["exposed_fields"].values():
        for field in section:
            if field not in discovered and field not in debt:
                offenders.append(field)
    assert not offenders, (
        f"{engine}: fields neither discovered nor debt-marked: {sorted(offenders)}"
    )


@pytest.mark.parametrize("engine", list(_PINS))
def test_debt_markers_are_genuine(engine: str) -> None:
    """A debt marker must only sit on a field actually absent from discovery.

    Guards against marking a field as debt when discovery in fact covers it
    (which would hide real coverage and mislead miner-deepening work).
    """
    schema = json.loads((_outputs_dir(engine) / "schema.discovered.json").read_text("utf-8"))
    discovered = set(schema.get("engine_params", {})) | set(schema.get("sampling_params", {}))
    spurious = sorted(_debt_marked_fields(engine) & discovered)
    assert not spurious, f"{engine}: fields marked debt but present in discovery: {spurious}"
