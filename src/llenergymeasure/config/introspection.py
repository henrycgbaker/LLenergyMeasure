"""Configuration introspection for SSOT architecture.

This module provides the Single Source of Truth (SSOT) for parameter metadata
by introspecting Pydantic models. All downstream consumers (tests, CLI, docs)
should use these functions to derive parameter information rather than
maintaining separate parameter lists.

Usage:
    from llenergymeasure.config.introspection import (
        get_backend_params,
        get_shared_params,
        get_all_params,
        get_param_test_values,
    )

    # Get all params for a backend
    pytorch_params = get_backend_params("pytorch")

    # Get test values for a param
    values = get_param_test_values("pytorch.batching_strategy")
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def _extract_param_metadata(
    field_name: str,
    field_info: FieldInfo,
    prefix: str = "",
) -> dict[str, Any]:
    """Extract metadata from a Pydantic field.

    Returns:
        Dict with keys: type, default, description, optional, constraints,
        options (for Literal), test_values.
    """
    param_path = f"{prefix}.{field_name}" if prefix else field_name
    annotation = field_info.annotation

    # Handle Optional types (X | None)
    origin = get_origin(annotation)
    args = get_args(annotation)

    is_optional = False
    if origin is type(None) or (args and type(None) in args):
        is_optional = True
        actual_types = [a for a in args if a is not type(None)]
        if actual_types:
            annotation = actual_types[0]
            origin = get_origin(annotation)
            args = get_args(annotation)

    # Build metadata
    meta: dict[str, Any] = {
        "path": param_path,
        "name": field_name,
        "default": field_info.default if field_info.default is not ... else None,
        "description": field_info.description or "",
        "optional": is_optional,
        "constraints": {},
        "options": None,
        "test_values": [],
        "type_str": "unknown",
    }

    # Extract constraints from field metadata
    if hasattr(field_info, "metadata"):
        for constraint in field_info.metadata:
            if hasattr(constraint, "ge"):
                meta["constraints"]["ge"] = constraint.ge
            if hasattr(constraint, "le"):
                meta["constraints"]["le"] = constraint.le
            if hasattr(constraint, "gt"):
                meta["constraints"]["gt"] = constraint.gt
            if hasattr(constraint, "lt"):
                meta["constraints"]["lt"] = constraint.lt

    # Determine type and generate test values
    if origin is Literal:
        meta["type_str"] = "literal"
        meta["options"] = list(args)
        meta["test_values"] = list(args)  # Test ALL Literal values

    elif annotation is bool:
        meta["type_str"] = "bool"
        meta["test_values"] = [False, True]

    elif annotation is int:
        meta["type_str"] = "int"
        ge = meta["constraints"].get("ge")
        le = meta["constraints"].get("le")
        default = meta["default"]

        if ge is not None and le is not None:
            # Test min, mid, max
            meta["test_values"] = sorted(set([ge, (ge + le) // 2, le]))
        elif ge is not None:
            meta["test_values"] = [ge, ge * 2, ge * 4]
        elif default is not None and isinstance(default, int):
            meta["test_values"] = [max(1, default // 2), default, default * 2]
        else:
            meta["test_values"] = [1, 4, 8]

    elif annotation is float:
        meta["type_str"] = "float"
        default = meta["default"]
        if default is not None and isinstance(default, int | float):
            meta["test_values"] = [
                round(default * 0.5, 2),
                default,
                round(default * 1.5, 2),
            ]
        else:
            meta["test_values"] = [0.5, 0.7, 0.9]

    elif annotation is str:
        meta["type_str"] = "str"
        meta["test_values"] = []  # Strings need context-specific values

    else:
        meta["type_str"] = str(annotation)

    return meta


def get_params_from_model(
    model_class: type[BaseModel],
    prefix: str = "",
    include_nested: bool = True,
) -> dict[str, dict[str, Any]]:
    """Extract all parameters from a Pydantic model.

    Args:
        model_class: Pydantic model to introspect.
        prefix: Prefix for param paths (e.g., "pytorch").
        include_nested: Whether to recurse into nested models.

    Returns:
        Dict mapping param paths to metadata dicts.
    """
    params: dict[str, dict[str, Any]] = {}

    for field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation

        # Handle Optional wrapper
        args = get_args(annotation)
        if args and type(None) in args:
            actual_types = [a for a in args if a is not type(None)]
            if actual_types:
                annotation = actual_types[0]

        # Check if nested Pydantic model
        if include_nested and hasattr(annotation, "model_fields"):
            nested_prefix = f"{prefix}.{field_name}" if prefix else field_name
            # Cast annotation to BaseModel subclass (we know it is from model_fields check)
            nested_params = get_params_from_model(
                annotation,  # type: ignore[arg-type]
                prefix=nested_prefix,
                include_nested=True,
            )
            params.update(nested_params)
        else:
            meta = _extract_param_metadata(field_name, field_info, prefix)
            params[meta["path"]] = meta

    return params


def get_backend_params(backend: str) -> dict[str, dict[str, Any]]:
    """Get all parameters for a backend from its Pydantic model.

    Args:
        backend: One of "pytorch", "vllm", "tensorrt".

    Returns:
        Dict mapping param paths to metadata.
    """
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

    backend_models = {
        "pytorch": PyTorchConfig,
        "vllm": VLLMConfig,
        "tensorrt": TensorRTConfig,
    }

    if backend not in backend_models:
        raise ValueError(
            f"Unknown backend: {backend}. Must be one of {list(backend_models.keys())}"
        )

    model_class = backend_models[backend]
    # All values are Pydantic BaseModel subclasses, mypy can't infer this from dict
    return get_params_from_model(model_class, prefix=backend)  # type: ignore[arg-type]


def get_shared_params() -> dict[str, dict[str, Any]]:
    """Get shared/universal parameters from ExperimentConfig and DecoderConfig.

    Returns params that are universal across all backends:
    - Top-level: fp_precision, streaming, max_input_tokens, max_output_tokens
    - Decoder: temperature, do_sample, top_p, top_k, repetition_penalty, preset
    """
    from llenergymeasure.config.models import DecoderConfig

    shared: dict[str, dict[str, Any]] = {}

    # Decoder params
    decoder_params = get_params_from_model(DecoderConfig, prefix="decoder")
    shared.update(decoder_params)

    # Top-level universal params (manually defined since they're simple)
    shared["fp_precision"] = {
        "path": "fp_precision",
        "name": "fp_precision",
        "type_str": "literal",
        "default": "float16",
        "description": "Floating point precision",
        "options": ["float32", "float16", "bfloat16"],
        "test_values": ["float32", "float16", "bfloat16"],
        "constraints": {},
        "optional": False,
    }
    shared["streaming"] = {
        "path": "streaming",
        "name": "streaming",
        "type_str": "bool",
        "default": False,
        "description": "Enable streaming mode for TTFT/ITL measurement",
        "options": None,
        "test_values": [False, True],
        "constraints": {},
        "optional": False,
    }
    shared["max_input_tokens"] = {
        "path": "max_input_tokens",
        "name": "max_input_tokens",
        "type_str": "int",
        "default": 512,
        "description": "Maximum input token length",
        "options": None,
        "test_values": [128, 512, 1024],
        "constraints": {"ge": 1},
        "optional": False,
    }
    shared["max_output_tokens"] = {
        "path": "max_output_tokens",
        "name": "max_output_tokens",
        "type_str": "int",
        "default": 128,
        "description": "Maximum output token length",
        "options": None,
        "test_values": [32, 128, 256],
        "constraints": {"ge": 1},
        "optional": False,
    }

    return shared


def get_all_params() -> dict[str, dict[str, dict[str, Any]]]:
    """Get all parameters organised by backend + shared.

    Returns:
        {
            "shared": {...},
            "pytorch": {...},
            "vllm": {...},
            "tensorrt": {...},
        }
    """
    return {
        "shared": get_shared_params(),
        "pytorch": get_backend_params("pytorch"),
        "vllm": get_backend_params("vllm"),
        "tensorrt": get_backend_params("tensorrt"),
    }


def get_param_test_values(param_path: str) -> list[Any]:
    """Get test values for a specific parameter.

    Args:
        param_path: Full param path, e.g., "pytorch.batch_size" or "decoder.temperature".

    Returns:
        List of test values.
    """
    all_params = get_all_params()

    for section in all_params.values():
        if param_path in section:
            test_values: list[Any] = section[param_path].get("test_values", [])
            return test_values

    return []


def get_param_options(param_path: str) -> list[Any] | None:
    """Get valid options for a Literal-typed parameter.

    Args:
        param_path: Full param path.

    Returns:
        List of options for Literal types, None otherwise.
    """
    all_params = get_all_params()

    for section in all_params.values():
        if param_path in section:
            return section[param_path].get("options")

    return None


def list_all_param_paths(backend: str | None = None) -> list[str]:
    """List all parameter paths, optionally filtered by backend.

    Args:
        backend: Optional backend filter ("pytorch", "vllm", "tensorrt", "shared").

    Returns:
        Sorted list of param paths.
    """
    all_params = get_all_params()

    if backend:
        if backend not in all_params:
            raise ValueError(f"Unknown backend: {backend}")
        return sorted(all_params[backend].keys())

    paths: list[str] = []
    for section in all_params.values():
        paths.extend(section.keys())
    return sorted(set(paths))


# =============================================================================
# Constraint Metadata for SSOT Architecture Hardening
# =============================================================================


def get_mutual_exclusions() -> dict[str, list[str]]:
    """Get parameters that are mutually exclusive.

    Returns:
        Dict mapping param path to list of params it cannot be used with.
        These combinations should be skipped during runtime testing.
    """
    return {
        # PyTorch: can't use both 4-bit and 8-bit quantization
        "pytorch.load_in_4bit": ["pytorch.load_in_8bit"],
        "pytorch.load_in_8bit": ["pytorch.load_in_4bit"],
        # vLLM: chunked prefill can conflict with speculative decoding
        "vllm.enable_chunked_prefill": ["vllm.speculative.model"],
        # vLLM: fp8 KV cache not compatible with all GPUs
        "vllm.kv_cache_dtype": [],  # Handled by test infrastructure
        # TensorRT: calibration only needed for int8_sq quantization
        "tensorrt.quantization": [],  # Contextual - calibration depends on method
    }


def get_backend_specific_params() -> dict[str, list[str]]:
    """Get params that are only valid for specific backends.

    Returns:
        Dict mapping backend name to list of exclusive param paths.
    """
    return {
        "pytorch": [
            "pytorch.batch_size",
            "pytorch.batching_strategy",
            "pytorch.load_in_4bit",
            "pytorch.load_in_8bit",
            "pytorch.bnb_4bit_quant_type",
            "pytorch.bnb_4bit_compute_dtype",
            "pytorch.bnb_4bit_use_double_quant",
            "pytorch.attn_implementation",
            "pytorch.torch_compile",
            "pytorch.torch_compile_backend",
            "pytorch.use_bettertransformer",
            "pytorch.cache_implementation",
            "pytorch.assisted_generation",
            "pytorch.beam_search",
        ],
        "vllm": [
            "vllm.max_num_seqs",
            "vllm.max_num_batched_tokens",
            "vllm.enable_prefix_caching",
            "vllm.enable_chunked_prefill",
            "vllm.kv_cache_dtype",
            "vllm.block_size",
            "vllm.enforce_eager",
            "vllm.attention",
            "vllm.speculative",
            "vllm.lora",
            "vllm.quantization",
            "vllm.logprobs",
        ],
        "tensorrt": [
            "tensorrt.engine_path",
            "tensorrt.max_batch_size",
            "tensorrt.builder_opt_level",
            "tensorrt.strongly_typed",
            "tensorrt.multiple_profiles",
            "tensorrt.kv_cache_type",
            "tensorrt.enable_chunked_context",
            "tensorrt.enable_kv_cache_reuse",
            "tensorrt.quantization",
            "tensorrt.calibration",
        ],
    }


def get_special_test_models() -> dict[str, str]:
    """Get parameters that require special pre-quantized test models.

    Some parameters (like AWQ/GPTQ quantization) require models that have
    been pre-quantized with that method. Using a non-quantized model will fail.

    Returns:
        Dict mapping param value patterns to appropriate test model names.
    """
    return {
        # vLLM quantization methods requiring pre-quantized models
        "vllm.quantization=awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "vllm.quantization=gptq": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        "vllm.quantization=marlin": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        # TensorRT quantization methods requiring pre-quantized models
        "tensorrt.quantization=int4_awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "tensorrt.quantization=int4_gptq": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    }


def get_params_requiring_gpu_capability(min_compute_capability: float = 8.0) -> list[str]:
    """Get params that require specific GPU compute capabilities.

    Args:
        min_compute_capability: Minimum compute capability (default 8.0 = Ampere).

    Returns:
        List of param paths that require the specified compute capability.
    """
    # These features require Ampere (8.0) or newer GPUs
    ampere_required = [
        "vllm.kv_cache_dtype=fp8",
        "tensorrt.quantization=fp8",
        "pytorch.attn_implementation=flash_attention_2",
    ]

    # Hopper (9.0) required features
    hopper_required = [
        "vllm.attention.flash_version=3",
    ]

    if min_compute_capability >= 9.0:
        return ampere_required + hopper_required
    return ampere_required


def get_param_skip_conditions() -> dict[str, str]:
    """Get conditions under which params should be skipped during testing.

    Returns:
        Dict mapping param paths to skip reasons for documentation/logging.
    """
    return {
        # Multi-GPU params - skip if single GPU
        "pytorch.parallelism_strategy=tensor_parallel": "Requires 2+ GPUs",
        "pytorch.parallelism_strategy=data_parallel": "Requires 2+ GPUs",
        "vllm.tensor_parallel_size>1": "Requires 2+ GPUs",
        "vllm.pipeline_parallel_size>1": "Requires 2+ GPUs",
        "tensorrt.tp_size>1": "Requires 2+ GPUs",
        "tensorrt.pp_size>1": "Requires 2+ GPUs",
        # Ray backend - requires ray installation
        "vllm.distributed_backend=ray": "Requires ray installation",
        # Flash Attention 3 - Hopper only
        "vllm.attention.flash_version=3": "Requires Hopper GPU (H100)",
        # FP8 - Ampere or newer
        "vllm.kv_cache_dtype=fp8": "Requires Ampere+ GPU",
        "tensorrt.quantization=fp8": "Requires Ampere+ GPU",
    }


def get_streaming_constraints() -> dict[str, str]:
    """Get parameters that are affected by streaming=True.

    When streaming=True, certain parameters are ignored or behave differently
    because streaming requires sequential per-request processing for accurate
    TTFT (Time To First Token) and ITL (Inter-Token Latency) measurement.

    Returns:
        Dict mapping param paths to explanations of streaming impact.
    """
    return {
        # Parameters ignored when streaming=True
        "pytorch.batch_size": "Ignored - streaming processes 1 request at a time",
        "pytorch.batching_strategy": "Ignored - always sequential in streaming mode",
        "vllm.max_num_seqs": "Effectively 1 in streaming mode for accurate TTFT",
        # Parameters that may conflict with streaming
        "pytorch.torch_compile": "May cause graph-tracing errors, falls back to uncompiled",
        "vllm.enable_chunked_prefill": "May interfere with TTFT measurement accuracy",
    }


def get_streaming_incompatible_tests() -> list[tuple[str, str]]:
    """Get parameter combinations that should not be tested with streaming=True.

    These combinations either:
    - Are known to fail or be unreliable
    - Would give misleading results
    - Have no meaningful interaction with streaming

    Returns:
        List of (param_path, reason) tuples.
    """
    return [
        # torch.compile + streaming is unreliable
        ("pytorch.torch_compile", "Graph compilation incompatible with streaming callbacks"),
        # Large batch sizes are pointless with streaming (they're ignored)
        ("pytorch.batch_size>1", "Batch size ignored in streaming mode"),
    ]


# =============================================================================
# SSOT Backend Capability Matrix
# =============================================================================


def get_backend_capabilities() -> dict[str, dict[str, bool | str]]:
    """Derive backend capability matrix from Pydantic model structure.

    This is the SSOT for the capability matrix shown in documentation.
    Capabilities are inferred by checking which fields exist in each
    backend config and their allowed values.

    Returns:
        Dict mapping capability names to per-backend support status.
        Values are True/False for simple support, or str for notes.
    """
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

    # Get field names for each backend
    pytorch_fields = set(PyTorchConfig.model_fields.keys())
    vllm_fields = set(VLLMConfig.model_fields.keys())
    tensorrt_fields = set(TensorRTConfig.model_fields.keys())

    # Get quantization Literal values for vLLM and TensorRT
    vllm_quant_field = VLLMConfig.model_fields.get("quantization")
    vllm_quant_options: list[str] = []
    if vllm_quant_field and vllm_quant_field.annotation:
        args = get_args(vllm_quant_field.annotation)
        # Filter out None from Optional[Literal[...]]
        for arg in args:
            if arg is not type(None):
                inner_args = get_args(arg)
                if inner_args:
                    vllm_quant_options = [a for a in inner_args if a is not None]

    trt_quant_field = TensorRTConfig.model_fields.get("quantization")
    trt_quant_options: list[str] = []
    if trt_quant_field and trt_quant_field.annotation:
        args = get_args(trt_quant_field.annotation)
        if args:
            trt_quant_options = [a for a in args if a != "none"]

    return {
        "tensor_parallel": {
            "pytorch": "parallelism_strategy" in pytorch_fields,  # has tensor_parallel option
            "vllm": "tensor_parallel_size" in vllm_fields,
            "tensorrt": "tp_size" in tensorrt_fields,
        },
        "pipeline_parallel": {
            # PyTorch's generate() doesn't support PP (requires full model for autoregressive)
            "pytorch": False,
            "vllm": "pipeline_parallel_size" in vllm_fields,
            "tensorrt": "pp_size" in tensorrt_fields,
        },
        "data_parallel": {
            "pytorch": "parallelism_strategy" in pytorch_fields,  # has data_parallel option
            # vLLM manages multi-GPU internally via tensor parallel
            "vllm": False,
            "tensorrt": True,  # Can use multiple TRT instances in DP mode
        },
        "bitsandbytes_4bit": {
            "pytorch": "load_in_4bit" in pytorch_fields,
            "vllm": False,  # vLLM uses native quantization, not bitsandbytes
            "tensorrt": False,  # TensorRT uses native quantization
        },
        "bitsandbytes_8bit": {
            "pytorch": "load_in_8bit" in pytorch_fields,
            "vllm": False,
            "tensorrt": False,
        },
        "native_quantization": {
            "pytorch": False,  # PyTorch relies on bitsandbytes, not native
            "vllm": "AWQ/GPTQ/FP8" if vllm_quant_options else False,
            "tensorrt": "FP8/INT8" if trt_quant_options else False,
        },
        "float32_precision": {
            "pytorch": True,
            "vllm": True,
            # TensorRT-LLM is optimised for lower precision
            "tensorrt": False,
        },
        "float16_precision": {
            "pytorch": True,
            "vllm": True,
            "tensorrt": True,
        },
        "bfloat16_precision": {
            "pytorch": True,
            "vllm": True,
            "tensorrt": True,
        },
        "streaming": {
            # All backends support streaming for TTFT/ITL measurement
            "pytorch": True,
            "vllm": True,
            "tensorrt": True,
        },
        "lora_adapters": {
            "pytorch": True,  # Via peft library
            "vllm": "lora" in vllm_fields,
            "tensorrt": "draft_model" in tensorrt_fields,  # Limited LoRA support
        },
        "speculative_decoding": {
            "pytorch": "assisted_generation" in pytorch_fields,
            "vllm": "speculative" in vllm_fields,
            "tensorrt": "draft_model" in tensorrt_fields,
        },
    }


def get_capability_matrix_markdown() -> str:
    """Generate the capability matrix as a markdown table.

    This is used by doc generation scripts to create the capability
    matrix section in documentation files.

    Returns:
        Markdown table string.
    """
    capabilities = get_backend_capabilities()

    # Define display names and footnotes
    display_names = {
        "tensor_parallel": "Tensor Parallel",
        "pipeline_parallel": "Pipeline Parallel",
        "data_parallel": "Data Parallel",
        "bitsandbytes_4bit": "BitsAndBytes (4-bit)",
        "bitsandbytes_8bit": "BitsAndBytes (8-bit)",
        "native_quantization": "Native Quantization",
        "float32_precision": "float32 precision",
        "float16_precision": "float16 precision",
        "bfloat16_precision": "bfloat16 precision",
        "streaming": "Streaming (TTFT/ITL)",
        "lora_adapters": "LoRA Adapters",
        "speculative_decoding": "Speculative Decoding",
    }

    lines = [
        "| Feature | PyTorch | vLLM | TensorRT |",
        "|---------|---------|------|----------|",
    ]

    for cap_key, cap_values in capabilities.items():
        display_name = display_names.get(cap_key, cap_key)
        cells = []

        for backend in ["pytorch", "vllm", "tensorrt"]:
            value = cap_values.get(backend, False)
            if value is True:
                cells.append("✅")
            elif value is False:
                cells.append("❌")
            elif isinstance(value, str):
                # Has a note - add footnote
                cells.append(f"✅ ({value})")
            else:
                cells.append("❌")

        lines.append(f"| {display_name} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # Add standard footnotes
    lines.append("")
    lines.append("**Notes:**")
    lines.append("- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes")
    lines.append("- TensorRT-LLM is optimised for FP16/BF16/INT8 precision, not FP32")

    return "\n".join(lines)


def get_validation_rules() -> list[dict[str, str]]:
    """Get validation rules from config validators for documentation.

    Extracts cross-backend validation rules that are enforced at config
    load time. These rules are the SSOT for the "Config Validation Errors"
    section in invalid-combos.md.

    Returns:
        List of dicts with keys: backend, combination, reason, resolution.
    """
    # These are extracted from ExperimentConfig and backend validators
    # TODO: In future, could use AST parsing to extract these automatically
    return [
        {
            "backend": "pytorch",
            "combination": "parallelism.strategy=pipeline_parallel",
            "reason": "PyTorch's generate() requires full model access for autoregressive generation",
            "resolution": "Use backend='vllm' or backend='tensorrt' for pipeline parallel",
        },
        {
            "backend": "vllm",
            "combination": "parallelism.strategy=data_parallel",
            "reason": "vLLM manages multi-GPU internally via Ray/tensor parallel",
            "resolution": "Use tensor_parallel_size or pipeline_parallel_size",
        },
        {
            "backend": "vllm",
            "combination": "quantization.load_in_8bit=True",
            "reason": "vLLM does not support bitsandbytes 8-bit quantization",
            "resolution": "Use vllm.quantization (awq, gptq, fp8) for quantized inference",
        },
        {
            "backend": "tensorrt",
            "combination": "fp_precision=float32",
            "reason": "TensorRT-LLM is optimised for lower precision inference",
            "resolution": "Use fp_precision='float16' or 'bfloat16'",
        },
        {
            "backend": "tensorrt",
            "combination": "quantization.load_in_4bit=True",
            "reason": "TensorRT does not support bitsandbytes quantization",
            "resolution": "Use tensorrt.quantization (fp8, int8_sq, int4_awq)",
        },
        {
            "backend": "tensorrt",
            "combination": "quantization.load_in_8bit=True",
            "reason": "TensorRT does not support bitsandbytes quantization",
            "resolution": "Use tensorrt.quantization (fp8, int8_sq, int8_weight_only)",
        },
        {
            "backend": "all",
            "combination": "quantization.load_in_4bit + load_in_8bit",
            "reason": "Cannot use both 4-bit and 8-bit quantization simultaneously",
            "resolution": "Choose one: load_in_4bit=True OR load_in_8bit=True",
        },
    ]
