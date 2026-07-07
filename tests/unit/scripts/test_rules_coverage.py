"""Host-side tests for scripts/rules_coverage.py (the validator coverage check).

No engine import, no container: each test builds a tiny source tree and a tiny
rules corpus on disk and exercises the check's own logic - validator-site
detection (pydantic decorators by any alias, plus __post_init__), the public
field-read extraction (method calls and private attributes excluded), the
materiality gate (raise AND a field), the field-name coverage intersection,
IGNORED_SITES suppression, deterministic path-relative reporting, and CLI codes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

import rules_coverage as rc


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _pi(cls: str, field: str) -> str:
    """A minimal class whose __post_init__ raises on one field read."""
    return f"class {cls}:\n    def __post_init__(self):\n        if self.{field}:\n            raise ValueError('x')\n"


def _corpus(root: Path, engine: str, fields: list[str]) -> Path:
    rules = "".join(
        f"- id: r{i}\n  engine: {engine}\n  severity: error\n"
        f"  match:\n    fields:\n      {engine}.engine_params.{leaf}:\n        <: 1\n"
        f"  provenance:\n    source: manual\n    verified: human\n"
        for i, leaf in enumerate(fields)
    )
    corpus_root = root / "corpus"
    (corpus_root / engine).mkdir(parents=True, exist_ok=True)
    (corpus_root / engine / "rules.yaml").write_text(
        f"schema_version: 1.0.0\nengine: {engine}\nengine_version: 9.9.9\n"
        f"rules:{chr(10) + rules if rules else ' []'}\n"
    )
    return corpus_root


def _cli(tmp: Path, engine: str, fields: list[str], *extra: str) -> list[str]:
    return [
        "--engine",
        engine,
        "--source-root",
        str(tmp),
        "--corpus-root",
        str(_corpus(tmp, engine, fields)),
        *extra,
    ]


def test_detects_pydantic_and_post_init(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cfg.py",
        "from pydantic import field_validator, model_validator\n"
        "class Cfg:\n"
        "    @field_validator('alpha')\n"
        "    def _v(cls, v):\n"
        "        if v < 0:\n            raise ValueError('bad')\n        return v\n"
        "    @model_validator(mode='after')\n"
        "    def _m(self):\n"
        "        if self.beta:\n            raise ValueError('bad')\n        return self\n"
        "    def __post_init__(self):\n"
        "        if self.gamma < 1:\n            raise ValueError('bad')\n",
    )
    quals = {s.qualname: s.fields for s in rc.scan_source(tmp_path)}
    assert quals == {"Cfg._v": ("alpha",), "Cfg._m": ("beta",), "Cfg.__post_init__": ("gamma",)}


def test_decorator_alias_matched_by_last_name(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cfg.py",
        "import pydantic as pyd\n"
        "class Cfg:\n"
        "    @pyd.field_validator('alpha')\n"
        "    def _v(cls, v):\n        if v:\n            raise ValueError('x')\n",
    )
    (site,) = rc.scan_source(tmp_path)
    assert site.qualname == "Cfg._v" and site.fields == ("alpha",)


def test_field_extraction_excludes_calls_and_private(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cfg.py",
        "class Cfg:\n"
        "    def __post_init__(self):\n"
        "        self.helper()\n"  # method call, not a field
        "        _ = self._private\n"  # private attr, excluded
        "        _ = self.__dict__\n"  # dunder attr, excluded
        "        if self.public and self.other:\n            raise ValueError('x')\n",
    )
    (site,) = rc.scan_source(tmp_path)
    assert site.fields == ("other", "public")


def test_data_arg_attribute_read(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cfg.py",
        "from pydantic import model_validator\n"
        "class Cfg:\n"
        "    @model_validator(mode='before')\n"
        "    def _m(cls, data):\n        if data.knob:\n            raise ValueError('x')\n        return data\n",
    )
    (site,) = rc.scan_source(tmp_path)
    assert site.fields == ("knob",)


def test_materiality_requires_raise_and_field(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cfg.py",
        # normaliser: reads a field, never raises -> skipped
        "class Norm:\n    def __post_init__(self):\n        self.x = self.y or 1\n"
        # guard: raises but reads no field -> skipped
        "class Guard:\n    def __post_init__(self):\n        raise ValueError('always')\n"
        # real: raises and reads a field -> material
        + _pi("Real", "z"),
    )
    assert {s.qualname for s in rc.scan_source(tmp_path)} == {"Real.__post_init__"}


def test_unparseable_file_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "broken.py", "def (:\n")
    _write(tmp_path, "ok.py", _pi("Cfg", "z"))
    assert [s.qualname for s in rc.scan_source(tmp_path)] == ["Cfg.__post_init__"]


def test_coverage_intersection() -> None:
    sites = [
        rc.ValidatorSite("a.py", 1, "A.f", ("covered_field",)),
        rc.ValidatorSite("b.py", 2, "B.g", ("unknown_field",)),
    ]
    covered, uncovered, ignored = rc.classify("vllm", sites, {"covered_field"})
    assert [s.qualname for s in covered] == ["A.f"]
    assert [s.qualname for s in uncovered] == ["B.g"]
    assert ignored == 0


def test_ignored_sites_suppressed(monkeypatch) -> None:
    monkeypatch.setattr(rc, "IGNORED_SITES", frozenset({("vllm", "b.py", "B.g")}))
    sites = [
        rc.ValidatorSite("a.py", 1, "A.f", ("x",)),
        rc.ValidatorSite("b.py", 2, "B.g", ("x",)),
    ]
    covered, uncovered, ignored = rc.classify("vllm", sites, set())
    assert not covered
    assert [s.qualname for s in uncovered] == ["A.f"]
    assert ignored == 1
    # the same site under a different engine is NOT ignored
    _, unc2, ig2 = rc.classify("tensorrt", sites, set())
    assert ig2 == 0 and len(unc2) == 2


def test_corpus_leaf_names(tmp_path: Path) -> None:
    version, leaves = rc.corpus_leaf_names(
        "vllm", _corpus(tmp_path, "vllm", ["max_model_len", "seed"])
    )
    assert version == "9.9.9"
    assert leaves == {"max_model_len", "seed"}


def test_report_sorted_relative_and_deterministic(tmp_path: Path) -> None:
    _write(tmp_path, "z_last.py", _pi("Z", "q"))
    _write(tmp_path, "a_first.py", _pi("A", "p"))
    corpus_root = _corpus(tmp_path, "vllm", [])
    report_a, n_a = rc.run("vllm", tmp_path, corpus_root)
    report_b, n_b = rc.run("vllm", tmp_path, corpus_root)
    assert report_a == report_b  # byte-identical
    assert n_a == n_b == 2
    assert str(tmp_path) not in report_a  # no absolute paths leak
    assert report_a.index("a_first.py") < report_a.index("z_last.py")  # sorted


def test_main_advisory_default_and_fail_flag(tmp_path: Path, capsys) -> None:
    _write(tmp_path, "cfg.py", _pi("Cfg", "x"))
    args = _cli(tmp_path, "vllm", [])
    assert rc.main(args) == 0  # advisory by default even with an uncovered site
    assert "uncovered: 1" in capsys.readouterr().out
    assert rc.main([*args, "--fail-on-uncovered"]) == 1


def test_main_clean_source_exit_zero(tmp_path: Path) -> None:
    _write(tmp_path, "cfg.py", _pi("Cfg", "covered"))
    assert rc.main(_cli(tmp_path, "vllm", ["covered"], "--fail-on-uncovered")) == 0


def test_main_missing_source_root(tmp_path: Path) -> None:
    corpus_root = _corpus(tmp_path, "vllm", [])
    args = [
        "--engine",
        "vllm",
        "--source-root",
        str(tmp_path / "nope"),
        "--corpus-root",
        str(corpus_root),
    ]
    assert rc.main(args) == 2
