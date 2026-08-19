"""Tests for corpus validation of ``<engine>.engine_params`` keys.

The defect this guards: ``engine_params`` is ``extra="allow"``, so a misspelt key
used to parse. A study sweeping ``vllm.engine_params.max_num_seq`` (for
``max_num_seqs``) expanded into three distinctly-hashed experiments that all
carried the same effective engine configuration, and the whole sweep ran.

Covered here:

- the reported typo case errors, naming the key, the engine, and a suggestion,
  both on a single config and through study sweep expansion;
- legitimate keys are admitted unchanged, including corpus-visible fields well
  outside the commonly-used set, and nested blocks (both a curated nested block
  and one visible only through the corpus ``$defs``);
- a nested typo inside a corpus-visible block errors, naming the dotted path;
- a block the corpus leaves opaque admits any key below it;
- every engine is covered, each according to whether its corpus enumerates a
  closed surface (vllm, tensorrt) or an open var-kwargs one (transformers);
- a missing corpus warns and admits rather than crashing or rejecting.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from llenergymeasure.config import engine_params_keys
from llenergymeasure.config.grid import expand_grid
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.schema_loader import SchemaLoader
from llenergymeasure.config.ssot import ALL_ENGINES, Engine
from llenergymeasure.config.warnings import ConfigValidationWarning
from llenergymeasure.utils.exceptions import ConfigError

# Engines whose discovered surface is closed (every argument enumerated), so an
# unrecognised key is a typo rather than a possibly-legitimate kwarg.
CLOSED_SURFACE_ENGINES = (Engine.VLLM, Engine.TENSORRT)

# Engines whose corpus records a var-kwargs limitation on engine_params, so an
# unrecognised key may still be legitimate.
OPEN_SURFACE_ENGINES = (Engine.TRANSFORMERS,)


def build(engine: str, engine_params: dict) -> ExperimentConfig:
    """An otherwise-valid offline config carrying ``engine_params`` for ``engine``."""
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine=engine,
        serving_mode="offline",
        **{engine: {"engine_params": engine_params}},
    )


# ---------------------------------------------------------------------------
# (a) The reported typo case
# ---------------------------------------------------------------------------


def test_vllm_engine_params_typo_is_rejected() -> None:
    """``max_num_seq`` names the key, the engine, and the intended field."""
    with pytest.raises(ValidationError) as excinfo:
        build("vllm", {"max_num_seq": 64})
    message = str(excinfo.value)
    assert "max_num_seq" in message
    assert "vllm.engine_params" in message
    assert "did you mean max_num_seqs?" in message


def test_swept_typo_axis_fails_the_whole_study() -> None:
    """A sweep over the typo'd key errors instead of expanding into wasted runs."""
    raw_study = {
        "task": {"model": "gpt2"},
        "engine": "vllm",
        "serving_mode": "offline",
        "sweep": {"vllm.engine_params.max_num_seq": [64, 256, 1024]},
    }
    with pytest.raises(ConfigError) as excinfo:
        expand_grid(raw_study)
    message = str(excinfo.value)
    assert "all 3 generated config(s) are invalid" in message
    assert "max_num_seq" in message


def test_unknown_key_with_no_close_match_is_still_rejected() -> None:
    """Rejection is not gated on a suggestion being available."""
    with pytest.raises(
        ValidationError, match=r"unknown field 'zzz_nonsense' in vllm\.engine_params"
    ):
        build("vllm", {"zzz_nonsense": 1})


# ---------------------------------------------------------------------------
# (b) Legitimate keys are admitted unchanged
# ---------------------------------------------------------------------------


def test_uncurated_corpus_field_is_admitted() -> None:
    """A corpus-visible field outside the curated set parses and passes through.

    ``long_prefill_token_threshold`` is a real vLLM ``EngineArgs`` field that the
    generated config does not curate: the check validates against the full
    discovered surface, not the curated subset.
    """
    cfg = build("vllm", {"long_prefill_token_threshold": 2048})
    assert cfg.vllm is not None
    assert cfg.vllm.engine_params.model_extra == {"long_prefill_token_threshold": 2048}


def test_curated_field_is_admitted() -> None:
    """The correctly-spelt curated field the typo case meant still works."""
    cfg = build("vllm", {"max_num_seqs": 64})
    assert cfg.vllm is not None
    assert cfg.vllm.engine_params.max_num_seqs == 64


def test_curated_nested_block_is_admitted() -> None:
    """A nested block the generated config models is admitted with its own keys."""
    cfg = build("vllm", {"speculative_config": {"num_speculative_tokens": 3, "method": "eagle"}})
    assert cfg.vllm is not None
    spec = cfg.vllm.engine_params.speculative_config
    assert spec is not None
    assert spec.num_speculative_tokens == 3
    assert spec.method == "eagle"


def test_uncurated_nested_block_is_admitted_via_corpus_definitions() -> None:
    """A nested block visible only through the corpus ``$defs`` is admitted.

    ``kv_transfer_config`` is not curated onto the generated model, so it arrives
    as an untyped dict; its inner keys are still checked, against the
    ``KVTransferConfig`` definition the discovered envelope carries.
    """
    cfg = build("vllm", {"kv_transfer_config": {"kv_connector": "SharedStorageConnector"}})
    assert cfg.vllm is not None
    assert cfg.vllm.engine_params.model_extra == {
        "kv_transfer_config": {"kv_connector": "SharedStorageConnector"}
    }


def test_opaque_nested_block_admits_any_key() -> None:
    """A block whose structure the corpus does not model admits every key below it.

    ``attention`` is a curated llem-side grouping with no corpus definition, so
    there is no visible surface to validate its contents against.
    """
    cfg = build("vllm", {"attention": {"backend": "flash_attn", "future_attn_opt": 42}})
    assert cfg.vllm is not None
    assert cfg.vllm.engine_params.attention == {
        "backend": "flash_attn",
        "future_attn_opt": 42,
    }


# ---------------------------------------------------------------------------
# (c) Nested typos
# ---------------------------------------------------------------------------


def test_nested_typo_in_curated_block_is_rejected() -> None:
    """A typo inside a curated nested block errors, naming the dotted path."""
    with pytest.raises(ValidationError) as excinfo:
        build("vllm", {"speculative_config": {"num_speculative_tokenz": 3}})
    message = str(excinfo.value)
    assert "vllm.engine_params.speculative_config" in message
    assert "did you mean num_speculative_tokens?" in message


def test_nested_typo_in_uncurated_block_is_rejected() -> None:
    """A typo inside an un-curated nested block is caught from the corpus defs."""
    with pytest.raises(ValidationError) as excinfo:
        build("vllm", {"kv_transfer_config": {"kv_connectorz": "x"}})
    message = str(excinfo.value)
    assert "vllm.engine_params.kv_transfer_config" in message
    assert "did you mean kv_connector?" in message


# ---------------------------------------------------------------------------
# (d) tensorrt
# ---------------------------------------------------------------------------


def test_tensorrt_typo_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build("tensorrt", {"max_batch_sizes": 4})
    message = str(excinfo.value)
    assert "tensorrt.engine_params" in message
    assert "did you mean max_batch_size?" in message


def test_tensorrt_uncurated_corpus_field_is_admitted() -> None:
    cfg = build("tensorrt", {"enable_attention_dp": True})
    assert cfg.tensorrt is not None
    assert cfg.tensorrt.engine_params.model_extra == {"enable_attention_dp": True}


def test_tensorrt_llem_owned_passthrough_key_is_admitted() -> None:
    """``engine_path`` is llem-owned, absent from the corpus, and still legal.

    It is declared on the engine descriptor rather than mined, so the check must
    admit it (the plugin reads it to load a prebuilt engine directory).
    """
    cfg = build("tensorrt", {"engine_path": "/models/prebuilt", "backend": "trt"})
    assert cfg.tensorrt is not None
    assert cfg.tensorrt.engine_params.model_extra == {"engine_path": "/models/prebuilt"}


# ---------------------------------------------------------------------------
# (e) transformers: an open var-kwargs surface
# ---------------------------------------------------------------------------


def test_transformers_corpus_records_an_open_surface() -> None:
    """The openness that suppresses rejection is read from the corpus itself.

    ``from_pretrained`` is discovered by signature and that signature ends in
    ``**kwargs``, whose documented members live only in a class docstring. If a
    regenerated corpus ever stops recording that, this assertion fails rather than
    the check silently starting to reject legitimate transformers kwargs.
    """
    schema = SchemaLoader().load_schema(Engine.TRANSFORMERS.value)
    var_kwargs = [
        name
        for limitation in schema.discovery_limitations
        if limitation.section == "engine_params"
        for name in limitation.fields
        if "**" in name
    ]
    assert var_kwargs


def test_transformers_unknown_key_is_admitted() -> None:
    """An undiscoverable-but-plausible kwarg parses (no closed surface to check)."""
    cfg = build("transformers", {"brand_new_hf_kwarg_xyz": 1})
    assert cfg.transformers is not None
    assert cfg.transformers.engine_params.model_extra == {"brand_new_hf_kwarg_xyz": 1}


def test_transformers_close_typo_warns_but_parses() -> None:
    """On an open surface a close typo is a warning, never a rejection."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        build("transformers", {"dtypee": "float16"})
    messages = [str(w.message) for w in caught]
    assert any("dtypee" in m and "did you mean dtype" in m for m in messages)


# ---------------------------------------------------------------------------
# (f) Every engine is covered
# ---------------------------------------------------------------------------


def test_engine_surface_expectations_cover_every_engine() -> None:
    """The closed/open split above accounts for every engine in the SSOT."""
    assert set(CLOSED_SURFACE_ENGINES) | set(OPEN_SURFACE_ENGINES) == set(ALL_ENGINES)


@pytest.mark.parametrize("engine", sorted(ALL_ENGINES))
def test_every_engine_ships_an_engine_params_corpus(engine: Engine) -> None:
    schema = SchemaLoader().load_schema(engine.value)
    assert schema.engine_params


@pytest.mark.parametrize("engine", sorted(CLOSED_SURFACE_ENGINES))
def test_closed_surface_engines_reject_a_nonsense_key(engine: Engine) -> None:
    with pytest.raises(ValidationError, match=r"unknown field 'not_a_real_engine_field_zz'"):
        build(engine.value, {"not_a_real_engine_field_zz": 1})


@pytest.mark.parametrize("engine", sorted(OPEN_SURFACE_ENGINES))
def test_open_surface_engines_admit_a_nonsense_key(engine: Engine) -> None:
    cfg = build(engine.value, {"not_a_real_engine_field_zz": 1})
    section = getattr(cfg, engine.value)
    assert section.engine_params.model_extra == {"not_a_real_engine_field_zz": 1}


# ---------------------------------------------------------------------------
# (g) No corpus installed
# ---------------------------------------------------------------------------


def test_missing_corpus_warns_and_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no corpus on the install the check warns and admits, never crashes."""

    def _missing(self: SchemaLoader, engine: str):
        raise FileNotFoundError(f"no schema for {engine}")

    monkeypatch.setattr(SchemaLoader, "load_schema", _missing)
    engine_params_keys.reset_corpus_cache()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConfigValidationWarning)
            cfg = build("vllm", {"max_num_seq": 64})
        messages = [str(w.message) for w in caught]
        assert any("no engine-knowledge corpus is installed" in m for m in messages)
        assert cfg.vllm is not None
        assert cfg.vllm.engine_params.model_extra == {"max_num_seq": 64}
    finally:
        monkeypatch.undo()
        engine_params_keys.reset_corpus_cache()
