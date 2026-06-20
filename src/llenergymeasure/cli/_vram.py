"""VRAM estimation for --dry-run.

Non-blocking: all external calls (HuggingFace Hub, pynvml) are wrapped in
try/except and return None on any failure. Network errors never block the CLI.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.config.models import ExperimentConfig

# Bytes per parameter for each dtype.
DTYPE_BYTES: dict[str, float] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}


def estimate_vram(config: ExperimentConfig) -> dict[str, float] | None:
    """Estimate VRAM usage for a given experiment config.

    Returns a dict with keys: weights_gb, kv_cache_gb, overhead_gb, total_gb.
    Returns None if the model parameter count cannot be determined (e.g., network
    failure, model not found, or HuggingFace Hub unavailable).

    This function is non-blocking: all network operations have a 5-second timeout
    and are wrapped in try/except.
    """
    import socket

    # Try to fetch model metadata from HuggingFace Hub
    param_count: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    head_dim: int | None = None

    try:
        # Apply a short connection timeout to avoid blocking the CLI
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(5)
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            model_info = api.model_info(config.task.model)

            # Extract parameter count from safetensors metadata
            if (
                hasattr(model_info, "safetensors")
                and model_info.safetensors is not None
                and hasattr(model_info.safetensors, "total")
                and model_info.safetensors.total is not None
            ):
                param_count = int(model_info.safetensors.total)

            # Try to extract architecture details for KV cache estimation.
            # HF Hub's ModelInfo.config is a plain dict (the model's config.json
            # contents) in production, so read via dict access; tolerate an
            # attribute-style object too at this external-API edge.
            if hasattr(model_info, "config") and model_info.config is not None:
                cfg = model_info.config

                def _cfg_get(*names: str) -> Any:
                    for name in names:
                        value = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
                        if value is not None:
                            return value
                    return None

                # Common field names across model families
                _n_layers = _cfg_get("num_hidden_layers", "n_layer", "num_layers")
                _n_heads = _cfg_get("num_attention_heads", "n_head", "num_heads")
                _hidden = _cfg_get("hidden_size", "d_model", "n_embd")
                n_layers = int(_n_layers) if _n_layers is not None else None
                n_heads = int(_n_heads) if _n_heads is not None else None
                if n_heads and _hidden is not None:
                    head_dim = int(_hidden) // n_heads
        finally:
            socket.setdefaulttimeout(original_timeout)

    except Exception:
        # Non-blocking: network errors, model not found, HF Hub unavailable
        pass

    if param_count is None:
        return None

    # Weights memory
    engine_section = getattr(config, config.engine, None)
    dtype = getattr(engine_section, "dtype", None)
    bytes_per_param = DTYPE_BYTES.get(dtype or "bfloat16", 2)
    weights_gb = (param_count * bytes_per_param) / 1e9

    # KV cache estimation (sequence length 1 = single inference pass)
    kv_gb = 0.0
    if n_layers is not None and n_heads is not None and head_dim is not None:
        # 2 = key + value, 1 = batch size 1
        seq_len = config.task.max_input_tokens or 512  # fallback for VRAM estimate
        kv_bytes = 2 * int(n_layers) * 1 * seq_len * int(n_heads) * int(head_dim) * bytes_per_param
        kv_gb = kv_bytes / 1e9

    # Empirical 15% overhead for activations and framework buffers
    overhead_gb = weights_gb * 0.15

    total_gb = weights_gb + kv_gb + overhead_gb

    return {
        "weights_gb": weights_gb,
        "kv_cache_gb": kv_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_gb,
    }


def get_gpu_vram_gb() -> float | None:
    """Return total VRAM of the first GPU in GB, or None if unavailable.

    Uses pynvml. Returns None on any failure (pynvml not installed, no GPU,
    driver error, etc.).
    """
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        vram_gb: float | None = None
        with nvml_context():
            # GPU:0 is intentional for dry-run - we don't yet know which GPU will be
            # assigned at runtime, so the first GPU serves as a conservative estimate.
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem_info.total / 1e9
        return vram_gb
    except Exception:
        return None
