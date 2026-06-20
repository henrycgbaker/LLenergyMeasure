"""Unit tests for domain/model_info.py - architecture field normalisation."""

from __future__ import annotations

from llenergymeasure.domain.model_info import ModelArchInfo, extract_model_arch


def test_extract_standard_fields() -> None:
    """Canonical HuggingFace field names are read directly."""
    cfg = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
    }
    arch = extract_model_arch(cfg)
    assert arch == ModelArchInfo(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,  # 4096 // 32
        intermediate_size=11008,
        vocab_size=32000,
        tie_word_embeddings=False,
    )


def test_extract_gpt2_field_variants() -> None:
    """GPT-2-style aliases (n_layer/n_embd/n_head/n_inner) are resolved."""
    cfg = {
        "n_layer": 12,
        "n_embd": 768,
        "n_head": 12,
        "n_inner": 3072,
        "vocab_size": 50257,
    }
    arch = extract_model_arch(cfg)
    assert arch is not None
    assert arch.num_layers == 12
    assert arch.hidden_size == 768
    assert arch.num_attention_heads == 12
    assert arch.intermediate_size == 3072
    assert arch.head_dim == 64  # 768 // 12


def test_mha_fallback_when_kv_heads_absent() -> None:
    """num_key_value_heads defaults to num_attention_heads (MHA)."""
    arch = extract_model_arch(
        {"num_hidden_layers": 4, "hidden_size": 512, "num_attention_heads": 8}
    )
    assert arch is not None
    assert arch.num_key_value_heads == arch.num_attention_heads == 8


def test_explicit_head_dim_overrides_ratio() -> None:
    """An explicit head_dim is honoured even when it != hidden // heads."""
    arch = extract_model_arch(
        {
            "num_hidden_layers": 4,
            "hidden_size": 512,
            "num_attention_heads": 8,
            "head_dim": 128,
        }
    )
    assert arch is not None
    assert arch.head_dim == 128  # not 512 // 8 == 64


def test_intermediate_size_default() -> None:
    """Missing intermediate_size defaults to 4 * hidden_size."""
    arch = extract_model_arch(
        {"num_hidden_layers": 4, "hidden_size": 512, "num_attention_heads": 8}
    )
    assert arch is not None
    assert arch.intermediate_size == 2048


def test_tie_word_embeddings_truthy() -> None:
    """tie_word_embeddings is normalised to a bool."""
    tied = extract_model_arch(
        {
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "tie_word_embeddings": True,
        }
    )
    untied = extract_model_arch(
        {"num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 4}
    )
    assert tied is not None and tied.tie_word_embeddings is True
    assert untied is not None and untied.tie_word_embeddings is False


def test_missing_mandatory_returns_none() -> None:
    """None when any of layers / hidden / heads cannot be resolved."""
    assert extract_model_arch({"hidden_size": 512}) is None
    assert extract_model_arch({"num_hidden_layers": 4, "hidden_size": 512}) is None
    assert extract_model_arch({}) is None


def test_accepts_attribute_object() -> None:
    """Works with attribute-bearing config objects (e.g. PretrainedConfig)."""

    class Cfg:
        num_hidden_layers = 4
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 2

    arch = extract_model_arch(Cfg())
    assert arch is not None
    assert arch.num_key_value_heads == 2
    assert arch.num_attention_heads == 8
