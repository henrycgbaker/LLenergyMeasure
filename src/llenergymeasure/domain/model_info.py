"""Model architecture introspection - normalise HuggingFace config field variants.

Transformer configs use inconsistent field names across model families
(``num_hidden_layers`` vs ``n_layer``, ``hidden_size`` vs ``d_model`` vs
``n_embd``, etc.). This module pulls the architecture dimensions needed for
parameter counting into a single normalised shape, handling the common
field-name variants and GQA/MQA grouped-query attention.

Low-layer (domain) so both the harness FLOPs estimator and later VRAM
KV-cache sizing can reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _first_attr(config: Any, names: tuple[str, ...]) -> Any:
    """Return the first present (non-None) value among *names* on *config*.

    Accepts either a mapping (dict-like) or an attribute-bearing object
    (e.g. a HuggingFace ``PretrainedConfig``). Returns None when none match.
    """
    for name in names:
        value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class ModelArchInfo:
    """Normalised transformer architecture dimensions for parameter counting.

    GQA/MQA aware: ``num_key_value_heads`` may be smaller than
    ``num_attention_heads`` (grouped-query / multi-query attention), which
    shrinks the K and V projection matrices.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int | None
    tie_word_embeddings: bool


def extract_model_arch(config: Any) -> ModelArchInfo | None:
    """Extract normalised architecture dimensions from a model config.

    Handles the common field-name variants across model families and falls
    back to multi-head attention (kv heads = attention heads) when
    ``num_key_value_heads`` is absent.

    Args:
        config: A HuggingFace config object or a plain dict of config values.

    Returns:
        ModelArchInfo, or None when the mandatory dimensions
        (layers, hidden size, attention heads) cannot be resolved.
    """
    num_layers = _first_attr(config, ("num_hidden_layers", "n_layer", "num_layers"))
    hidden_size = _first_attr(config, ("hidden_size", "d_model", "n_embd"))
    num_attention_heads = _first_attr(config, ("num_attention_heads", "n_head", "num_heads"))

    if num_layers is None or hidden_size is None or num_attention_heads is None:
        return None

    num_layers = int(num_layers)
    hidden_size = int(hidden_size)
    num_attention_heads = int(num_attention_heads)

    # GQA/MQA: kv heads < attention heads. Fall back to MHA when absent.
    kv_heads = _first_attr(config, ("num_key_value_heads", "num_kv_heads"))
    num_key_value_heads = int(kv_heads) if kv_heads is not None else num_attention_heads

    # head_dim is usually hidden_size // num_attention_heads but some configs
    # set it explicitly (and it need not equal that ratio).
    explicit_head_dim = _first_attr(config, ("head_dim",))
    head_dim = (
        int(explicit_head_dim)
        if explicit_head_dim is not None
        else hidden_size // num_attention_heads
    )

    intermediate = _first_attr(config, ("intermediate_size", "ffn_dim", "n_inner"))
    intermediate_size = int(intermediate) if intermediate is not None else hidden_size * 4

    vocab = _first_attr(config, ("vocab_size",))
    vocab_size = int(vocab) if vocab is not None else None

    tie_word_embeddings = bool(_first_attr(config, ("tie_word_embeddings",)) or False)

    return ModelArchInfo(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
    )


__all__ = ["ModelArchInfo", "extract_model_arch"]
