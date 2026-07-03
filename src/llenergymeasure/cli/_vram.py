"""VRAM estimation for --dry-run.

Non-blocking: all external calls (HuggingFace Hub, pynvml) are wrapped in
try/except and return None on any failure. Network errors never block the CLI.
"""

from __future__ import annotations

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.model_info import ModelArchInfo, extract_model_arch

# Bytes per parameter for each dtype.
DTYPE_BYTES: dict[str, float] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}

# Per-engine engine_params field that carries the effective inference batch
# size for vLLM/tensorrt (their max concurrency knob, which bounds the KV
# cache the same way for a worst-case estimate). transformers prompt-batching
# is an llem-orchestration knob and lives on harness.transformers.batch_size.
_BATCH_FIELDS: dict[str, str] = {
    "vllm": "max_num_seqs",
    "tensorrt": "max_batch_size",
}


def _effective_batch_size(config: ExperimentConfig) -> int:
    """Return the configured batch size for the active engine (defaults to 1)."""
    if str(config.engine) == "transformers":
        harness = config.active_harness()
        if harness is not None and harness.batch_size is not None:
            return int(harness.batch_size)
        return 1
    engine_params = config.active_engine_params()
    field = _BATCH_FIELDS.get(str(config.engine))
    if engine_params is not None and field is not None:
        value = getattr(engine_params, field, None)
        if value is not None:
            return int(value)
    return 1


def _fetch_model_info(model: str) -> tuple[int | None, ModelArchInfo | None]:
    """Fetch (param_count, arch) from HuggingFace Hub. Non-blocking.

    Returns (None, None) on any failure (network error, model not found, or
    HuggingFace Hub unavailable). Applies a short socket timeout so the CLI
    never blocks on a slow network.
    """
    import socket

    param_count: int | None = None
    arch: ModelArchInfo | None = None
    try:
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(5)
        try:
            from huggingface_hub import HfApi

            model_info = HfApi().model_info(model)

            # Parameter count from safetensors metadata.
            safetensors = getattr(model_info, "safetensors", None)
            total = getattr(safetensors, "total", None) if safetensors is not None else None
            if total is not None:
                param_count = int(total)

            # Architecture for KV-cache sizing. HF Hub's ModelInfo.config is the
            # model's config.json contents (a dict); extract_model_arch handles
            # both dict and attribute-style configs.
            cfg = getattr(model_info, "config", None)
            if cfg is not None:
                arch = extract_model_arch(cfg)
        finally:
            socket.setdefaulttimeout(original_timeout)
    except Exception:
        # Non-blocking: network errors, model not found, HF Hub unavailable.
        return None, None

    return param_count, arch


def estimate_vram(config: ExperimentConfig) -> dict[str, float] | None:
    """Estimate VRAM usage for a given experiment config.

    Returns a dict with keys: weights_gb, kv_cache_gb, overhead_gb, total_gb.
    Returns None if the model parameter count cannot be determined (e.g., network
    failure, model not found, or HuggingFace Hub unavailable).

    This function is non-blocking: all network operations have a 5-second timeout
    and are wrapped in try/except.
    """
    param_count, arch = _fetch_model_info(config.task.model)
    if param_count is None:
        return None

    # Weights memory
    dtype = getattr(config.active_engine_params(), "dtype", None)
    bytes_per_param = DTYPE_BYTES.get(dtype or "bfloat16", 2)
    weights_gb = (param_count * bytes_per_param) / 1e9

    # KV cache: 2 (key + value) * layers * batch * seq_len * kv_heads * head_dim.
    # GQA/MQA models size K/V by num_key_value_heads (< num_attention_heads).
    kv_gb = 0.0
    if arch is not None:
        seq_len = config.task.max_input_tokens or 512  # fallback for VRAM estimate
        batch_size = _effective_batch_size(config)
        kv_bytes = (
            2
            * arch.num_layers
            * batch_size
            * seq_len
            * arch.num_key_value_heads
            * arch.head_dim
            * bytes_per_param
        )
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
