"""Host-side tests for scripts/check_plugin_kwargs.py.

No engine import, no container. The extractor tests feed synthetic plugin-like
source to :func:`extract_literal_kwargs` and assert exactly which string-literal
kwarg names are captured (dict-literal seed, subscript assign, literal update)
and which are ignored (dynamic update, nested sub-config kwargs on a different
variable, computed keys). The integration tests build a tiny synthetic repo
(current.yaml pin + versioned schema snapshot + plugin.py) and drive ``main`` /
``check_engine`` through it, including the historical regression: the removed
``quantization`` name checked against a tensorrt 1.2.1 schema must fail. A final
test asserts the real shipped plugins all pass against their mined schemas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import check_plugin_kwargs as cpk

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Extractor: which string-literal kwarg names are captured
# ---------------------------------------------------------------------------

_SPEC = cpk.PluginSpec(builders=("_build",), kwargs_vars=frozenset({"kwargs"}))


def _names(source: str, spec: cpk.PluginSpec = _SPEC) -> set[str]:
    names, missing = cpk.extract_literal_kwargs(source, spec)
    assert not missing
    return set(names)


def test_captures_dict_literal_seed():
    src = 'def _build(self):\n    kwargs = {"model": m, "seed": s}\n    return kwargs\n'
    assert _names(src) == {"model", "seed"}


def test_captures_annotated_dict_seed():
    src = 'def _build(self):\n    kwargs: dict = {"model": m}\n    return kwargs\n'
    assert _names(src) == {"model"}


def test_captures_subscript_assignment():
    src = 'def _build(self):\n    kwargs = {}\n    kwargs["quantization"] = q\n    return kwargs\n'
    assert _names(src) == {"quantization"}


def test_captures_literal_update():
    src = 'def _build(self):\n    kwargs = {}\n    kwargs.update({"model": m, "dtype": d})\n'
    assert _names(src) == {"model", "dtype"}


def test_ignores_dynamic_update():
    """A model_dump()-style ``.update(<var>)`` is schema-sourced, not hand-typed."""
    src = 'def _build(self):\n    kwargs = {"model": m}\n    kwargs.update(dumped)\n'
    assert _names(src) == {"model"}


def test_ignores_nested_subconfig_kwargs():
    """Keys on a different variable (a nested sub-config's kwargs) are out of scope."""
    src = (
        "def _build(self):\n"
        '    kwargs = {"model": m}\n'
        "    qc_kwargs = {}\n"
        '    qc_kwargs["quant_algo"] = a\n'
        '    kwargs["quant_config"] = QuantConfig(**qc_kwargs)\n'
    )
    assert _names(src) == {"model", "quant_config"}


def test_ignores_computed_and_variable_keys():
    src = (
        "def _build(self):\n"
        "    kwargs = {}\n"
        "    kwargs[name] = v\n"  # variable key
        '    kwargs[f"x_{i}"] = v\n'  # f-string key
        "    return kwargs\n"
    )
    assert _names(src) == set()


def test_scopes_to_multiple_kwargs_vars():
    """tensorrt threads an early-return ``early_kwargs`` dict too."""
    spec = cpk.PluginSpec(builders=("_build",), kwargs_vars=frozenset({"kwargs", "early_kwargs"}))
    src = (
        "def _build(self):\n"
        '    early_kwargs = {"model": m}\n'
        '    early_kwargs["backend"] = b\n'
        '    kwargs = {"model": m}\n'
    )
    assert _names(src, spec) == {"model", "backend"}


def test_scans_all_configured_builders():
    """A helper the kwargs dict is threaded through is scanned too."""
    spec = cpk.PluginSpec(builders=("_build", "_helper"), kwargs_vars=frozenset({"kwargs"}))
    src = (
        "def _build(self):\n"
        '    kwargs = {"model": m}\n'
        "def _helper(kwargs):\n"
        '    kwargs["enable_build_cache"] = True\n'
    )
    assert _names(src, spec) == {"model", "enable_build_cache"}


def test_reports_missing_builder():
    """A renamed builder is reported so the lint fails loudly instead of dormant."""
    _names_out, missing = cpk.extract_literal_kwargs("def other():\n    pass\n", _SPEC)
    assert missing == frozenset({"_build"})


# ---------------------------------------------------------------------------
# Integration: synthetic repo (pin + versioned schema snapshot + plugin.py)
# ---------------------------------------------------------------------------


def _setup_engine(
    repo: Path,
    engine: str,
    *,
    version: str,
    engine_params: list[str],
    plugin_src: str,
) -> None:
    """Write a current.yaml pin, a versioned schema snapshot, and a plugin.py."""
    from engine_versions._outputs import SCHEMA_FILENAME, safe_version

    current = repo / "engine_versions" / engine
    current.mkdir(parents=True, exist_ok=True)
    (current / "current.yaml").write_text(f"library:\n  current_version: {version}\n")

    out_dir = repo / "engine_versions" / engine / safe_version(version) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SCHEMA_FILENAME).write_text(
        json.dumps(
            {
                "engine_version": version,
                "engine_params": {name: {} for name in engine_params},
                "sampling_params": {},
            }
        )
    )

    plugin_dir = repo / "src" / "llenergymeasure" / "engines" / engine
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(plugin_src)


_TRT_QUANTIZATION_PLUGIN = (
    "class TensorRTEngine:\n"
    "    def _build_llm_kwargs(self, config):\n"
    '        kwargs = {"model": config.task.model}\n'
    "        qc_kwargs = {}\n"
    '        qc_kwargs["quant_algo"] = 1\n'
    '        kwargs["quantization"] = QuantConfig(**qc_kwargs)\n'
    "        return kwargs\n"
    "\n\n"
    "def _apply_default_build_cache(kwargs):\n"
    '    kwargs["enable_build_cache"] = True\n'
)


def test_regression_quantization_against_trt_1_2_1_fails(tmp_path, capsys):
    """The historical case: ``quantization`` is not in the tensorrt 1.2.1 schema.

    At 1.2.1 the mined ``TrtLlmArgs`` surface carries ``quant_config`` (the native
    name), never the removed ``quantization``. A plugin that hand-types
    ``quantization`` must be caught.
    """
    _setup_engine(
        tmp_path,
        "tensorrt",
        version="1.2.1",
        engine_params=["model", "quant_config", "enable_build_cache"],
        plugin_src=_TRT_QUANTIZATION_PLUGIN,
    )
    code = cpk.main(repo_root=tmp_path, engines=("tensorrt",))
    assert code == 1
    out = capsys.readouterr().out
    assert "quantization" in out
    assert "tensorrt" in out
    # The nested QuantConfig kwarg must not be reported as a top-level violation.
    assert "quant_algo" not in out


def test_schema_name_passes(tmp_path):
    _setup_engine(
        tmp_path,
        "tensorrt",
        version="1.2.1",
        engine_params=["model", "quant_config"],
        plugin_src=(
            "class E:\n"
            "    def _build_llm_kwargs(self, config):\n"
            '        kwargs = {"model": config.task.model}\n'
            '        kwargs["quant_config"] = q\n'
            "        return kwargs\n"
            "\n\n"
            "def _apply_default_build_cache(kwargs):\n"
            "    return None\n"
        ),
    )
    assert cpk.main(repo_root=tmp_path, engines=("tensorrt",)) == 0


def test_bad_name_not_allowlisted_fails(tmp_path, capsys):
    _setup_engine(
        tmp_path,
        "vllm",
        version="0.19.1",
        engine_params=["model", "seed"],
        plugin_src=(
            "class E:\n"
            "    def _build_llm_kwargs(self, config):\n"
            '        kwargs = {"model": m, "made_up_name": x}\n'
            "        return kwargs\n"
        ),
    )
    code = cpk.main(repo_root=tmp_path, engines=("vllm",))
    assert code == 1
    assert "made_up_name" in capsys.readouterr().out


def test_allowlisted_offsurface_name_passes(tmp_path, capsys):
    """A transformers from_pretrained **kwarg is off-surface but allowlisted."""
    assert ("transformers", "torch_dtype") in cpk.ALLOWLIST
    _setup_engine(
        tmp_path,
        "transformers",
        version="5.7.0",
        engine_params=["cache_dir", "revision"],  # signature-only surface
        plugin_src=(
            "class E:\n"
            "    def _model_load_kwargs(self, config):\n"
            '        kwargs = {"torch_dtype": d}\n'
            "        return kwargs\n"
        ),
    )
    code = cpk.main(repo_root=tmp_path, engines=("transformers",))
    assert code == 0
    assert "torch_dtype" in capsys.readouterr().out  # reported as allowlisted


def test_missing_builder_is_exit_2(tmp_path, capsys):
    _setup_engine(
        tmp_path,
        "vllm",
        version="0.19.1",
        engine_params=["model"],
        plugin_src="class E:\n    def some_other_method(self):\n        return {}\n",
    )
    code = cpk.main(repo_root=tmp_path, engines=("vllm",))
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_missing_schema_is_exit_2(tmp_path, capsys):
    (tmp_path / "engine_versions" / "vllm").mkdir(parents=True)
    (tmp_path / "engine_versions" / "vllm" / "current.yaml").write_text(
        "library:\n  current_version: 0.19.1\n"
    )
    plugin_dir = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "class E:\n    def _build_llm_kwargs(self, config):\n        return {}\n"
    )
    code = cpk.main(repo_root=tmp_path, engines=("vllm",))
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real shipped plugins must pass against their mined schemas
# ---------------------------------------------------------------------------


def test_real_plugins_pass():
    """Every shipped plugin's hand-typed kwargs match the mined schema at the pin."""
    assert cpk.main(repo_root=_REPO_ROOT) == 0


def test_engine_registry_matches_config_enum():
    """The workspace registry the script enumerates equals the config-side enum.

    The script reads engine_versions._outputs.ENGINES (stdlib-light, so CI runs
    it on bare python3); this cross-check pins that registry to the Engine enum
    so the two SSOT-adjacent lists cannot silently diverge.
    """
    from llenergymeasure.config.ssot import Engine

    assert set(cpk.ENGINES) == {e.value for e in Engine}
    assert all((engine in cpk._PLUGIN_SPECS) for engine in cpk.ENGINES)
