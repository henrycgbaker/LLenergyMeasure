"""FLOPs estimation for LLM inference.

PaLM/Chinchilla formula: FLOPs = 2 * N_non_embedding_params * total_tokens
Split into prefill (input) and decode (output) phases.

Two entry points:
- estimate_flops_palm() - requires a loaded model object (higher confidence).
- estimate_flops_palm_from_config() - uses AutoConfig only (no weights needed).
"""

from __future__ import annotations

import logging
from typing import Any

from llenergymeasure.domain.metrics import FlopsResult
from llenergymeasure.domain.model_info import ModelArchInfo, extract_model_arch

logger = logging.getLogger(__name__)


def _count_non_embedding_params(model: Any) -> int:
    """Count non-embedding parameters. Embeddings are memory-bound lookups, not MAC ops."""
    total = 0
    for name, param in model.named_parameters():
        if "embed" not in name.lower():
            total += param.numel()
    return total


def estimate_flops_palm(
    model: Any,
    n_input_tokens: int,
    n_output_tokens: int,
    batch_size: int = 1,
) -> FlopsResult:
    """PaLM/Chinchilla inference FLOPs estimate.

    Formula: FLOPs = 2 * N_non_embedding_params * total_tokens
    Split: prefill (input) + decode (output).
    Caller must exclude warmup tokens from n_input_tokens/n_output_tokens.

    Args:
        model: The model (must support named_parameters()).
        n_input_tokens: Number of input/prefill tokens (measurement phase only).
        n_output_tokens: Number of output/decode tokens (measurement phase only).
        batch_size: Batch size multiplier.

    Returns:
        FlopsResult with PaLM formula estimate and high confidence.
    """
    n_params = _count_non_embedding_params(model)
    flops_prefill = 2 * n_params * batch_size * n_input_tokens
    flops_decode = 2 * n_params * batch_size * n_output_tokens
    flops_total = flops_prefill + flops_decode

    return FlopsResult(
        value=float(flops_total),
        method="palm_formula",
        confidence="high",
        precision="n/a",  # precision doesn't affect FLOPs (forward pass)
        notes=(
            f"PaLM formula: 2x{n_params:,}x({n_input_tokens}+{n_output_tokens}) tokens. "
            f"Attention FLOPs omitted (v2.0 limitation, significant only for seq_len>=2048)."
        ),
    )


def _count_params_from_arch(arch: ModelArchInfo) -> int:
    """Non-embedding parameter count from normalised architecture dimensions.

    GQA/MQA aware: Q and O projections scale with the full attention-head
    width, but K and V projections scale with the (possibly smaller) number
    of key/value heads. Embeddings are excluded (PaLM formula counts
    non-embedding params only; they are memory-bound lookups, not MAC ops),
    so tied vs untied embeddings do not affect this count.
    """
    h = arch.hidden_size
    q_dim = arch.num_attention_heads * arch.head_dim
    kv_dim = arch.num_key_value_heads * arch.head_dim
    # Attention per layer: Q (h x q_dim) + K,V (h x kv_dim each) + O (q_dim x h)
    attn = (2 * h * q_dim + 2 * h * kv_dim) * arch.num_layers
    # FFN per layer: up (h x intermediate) + down (intermediate x h)
    ffn = 2 * h * arch.intermediate_size * arch.num_layers
    return int(attn + ffn)


def _count_params_from_config(model_name: str) -> int | None:
    """Extract non-embedding parameter count from HuggingFace model config.

    Uses AutoConfig.from_pretrained() to read architecture dimensions
    without loading model weights. Returns None on any failure.

    This enables FLOPs estimation for engines that do not expose an hf_model
    object (e.g. vLLM, TensorRT-LLM).
    """
    try:
        from transformers import AutoConfig

        from llenergymeasure.utils.security import trust_remote_code_enabled

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code_enabled())
        arch = extract_model_arch(cfg)
        if arch is None:
            return None
        return _count_params_from_arch(arch)
    except Exception:
        return None


def estimate_flops_palm_from_config(
    model_name: str,
    n_input_tokens: int,
    n_output_tokens: int,
    batch_size: int = 1,
) -> FlopsResult | None:
    """PaLM/Chinchilla inference FLOPs estimate using AutoConfig (no model weights needed).

    Uses _count_params_from_config() to extract architecture dimensions from
    the HuggingFace model config file. Returns None if config cannot be loaded.

    Formula: FLOPs = 2 * N_non_embedding_params * total_tokens
    Split: prefill (input) + decode (output).

    Args:
        model_name: HuggingFace model name or path (passed to AutoConfig.from_pretrained).
        n_input_tokens: Number of input/prefill tokens (measurement phase only).
        n_output_tokens: Number of output/decode tokens (measurement phase only).
        batch_size: Batch size multiplier.

    Returns:
        FlopsResult with config-based estimate and medium confidence, or None on failure.
    """
    n_params = _count_params_from_config(model_name)
    if n_params is None:
        return None

    flops_prefill = 2 * n_params * batch_size * n_input_tokens
    flops_decode = 2 * n_params * batch_size * n_output_tokens
    flops_total = flops_prefill + flops_decode

    return FlopsResult(
        value=float(flops_total),
        method="palm_formula",
        confidence="medium",
        precision="n/a",
        notes=(
            f"PaLM formula via AutoConfig (no weights loaded): "
            f"2x{n_params:,}x({n_input_tokens}+{n_output_tokens}) tokens."
        ),
    )


__all__ = [
    "_count_non_embedding_params",
    "_count_params_from_arch",
    "_count_params_from_config",
    "estimate_flops_palm",
    "estimate_flops_palm_from_config",
]
