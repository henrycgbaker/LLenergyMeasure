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
        get_experiment_config_schema,
    )

    # Get all params for a backend
    pytorch_params = get_backend_params("pytorch")

    # Get test values for a param
    values = get_param_test_values("pytorch.batch_size")

    # Get full JSON schema
    schema = get_experiment_config_schema()
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


def _get_custom_test_values() -> dict[str, list[Any]]:
    """Get custom test value overrides for params that need special handling.

    Returns:
        Empty dict — v2.0 minimal backend configs have no special overrides needed.
        (v1.x entries removed: vllm.max_model_len, vllm.max_num_batched_tokens,
        tensorrt.max_input_len — these fields no longer exist in v2.0 minimal configs.)
    """
    return {}


def get_backend_params(backend: str) -> dict[str, dict[str, Any]]:
    """Get all parameters for a backend from its Pydantic model.

    Args:
        backend: One of "pytorch", "vllm", "tensorrt".

    Returns:
        Dict mapping param paths to metadata. Each param includes
        ``backend_support: list[str]`` indicating which backends expose it.
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
    params = get_params_from_model(model_class, prefix=backend)  # type: ignore[arg-type]

    # Add backend_support to every param
    for param in params.values():
        param["backend_support"] = [backend]

    # Apply custom test value overrides
    custom_values = _get_custom_test_values()
    for param_path, values in custom_values.items():
        if param_path in params:
            params[param_path]["test_values"] = values

    return params


def get_shared_params() -> dict[str, dict[str, Any]]:
    """Get shared/universal parameters from ExperimentConfig and DecoderConfig.

    Returns params that are universal across all backends:
    - Top-level: precision, n, max_input_tokens, max_output_tokens, random_seed
    - Decoder: temperature, do_sample, top_p, top_k, repetition_penalty, preset

    Each param includes ``backend_support: list[str]`` indicating which backends
    expose each parameter.
    """
    from llenergymeasure.config.models import DecoderConfig

    shared: dict[str, dict[str, Any]] = {}

    # Decoder params (introspected from model)
    decoder_params = get_params_from_model(DecoderConfig, prefix="decoder")
    # Add backend_support to decoder params
    for param in decoder_params.values():
        param["backend_support"] = ["pytorch", "vllm", "tensorrt"]
    shared.update(decoder_params)

    # Top-level universal params — defined manually for explicit backend_support
    shared["precision"] = {
        "path": "precision",
        "name": "precision",
        "type_str": "literal",
        "default": "bf16",
        "description": "Floating point precision",
        "options": ["fp32", "fp16", "bf16"],
        "test_values": ["fp32", "fp16", "bf16"],
        "constraints": {},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["n"] = {
        "path": "n",
        "name": "n",
        "type_str": "int",
        "default": 100,
        "description": "Number of prompts from dataset",
        "options": None,
        "test_values": [10, 100, 500],
        "constraints": {"ge": 1},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["max_input_tokens"] = {
        "path": "max_input_tokens",
        "name": "max_input_tokens",
        "type_str": "int",
        "default": 512,
        "description": "Maximum input token length",
        "options": None,
        "test_values": [64, 128, 256],
        "constraints": {"ge": 1},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
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
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }

    return shared


def get_experiment_config_schema() -> dict[str, Any]:
    """Return the full ExperimentConfig JSON schema (Pydantic v2 schema).

    Returns:
        JSON-serialisable dict with the complete schema including all
        properties, types, constraints, and nested model schemas.
        Uses Pydantic's built-in model_json_schema() — always in sync
        with the actual model definition.
    """
    from llenergymeasure.config.models import ExperimentConfig

    return ExperimentConfig.model_json_schema()


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
        # vLLM: quantization method is exclusive (can only use one)
        "vllm.quantization": [],  # Handled by Literal type constraint
        # TensorRT: quantization method is exclusive
        "tensorrt.quantization": [],  # Handled by Literal type constraint
    }


def get_backend_specific_params() -> dict[str, list[str]]:
    """Get params that are only valid for specific backends.

    Returns:
        Dict mapping backend name to list of exclusive param paths.
        Derived from v2.0 minimal backend config fields.
    """
    return {
        "pytorch": [
            "pytorch.batch_size",
            "pytorch.attn_implementation",
            "pytorch.torch_compile",
            "pytorch.load_in_4bit",
            "pytorch.load_in_8bit",
            "pytorch.num_processes",
        ],
        "vllm": [
            "vllm.max_num_seqs",
            "vllm.tensor_parallel_size",
            "vllm.gpu_memory_utilization",
            "vllm.enable_prefix_caching",
            "vllm.quantization",
        ],
        "tensorrt": [
            "tensorrt.max_batch_size",
            "tensorrt.tp_size",
            "tensorrt.quantization",
            "tensorrt.engine_path",
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
        "vllm.quantization=fp8",
        "tensorrt.quantization=fp8",
        "pytorch.attn_implementation=flash_attention_2",
    ]

    if min_compute_capability >= 9.0:
        # Hopper (9.0) required features — none in v2.0 minimal config
        return ampere_required
    return ampere_required


def get_param_skip_conditions() -> dict[str, str]:
    """Get conditions under which params should be skipped during testing.

    Returns:
        Dict mapping param paths to skip reasons for documentation/logging.
    """
    return {
        # Multi-GPU params - skip if single GPU
        "pytorch.num_processes>1": "Requires 2+ GPUs (data parallelism)",
        "vllm.tensor_parallel_size>1": "Requires 2+ GPUs",
        "tensorrt.tp_size>1": "Requires 2+ GPUs",
        # Flash Attention 2 - requires flash-attn package
        "pytorch.attn_implementation=flash_attention_2": "Requires flash-attn package",
        # FP8 - Ampere or newer
        "vllm.quantization=fp8": "Requires Ampere+ GPU",
        "tensorrt.quantization=fp8": "Requires Ampere+ GPU",
        # Quantization - requires pre-quantized models (see get_special_test_models)
        "vllm.quantization=awq": "Requires AWQ-quantized model",
        "vllm.quantization=gptq": "Requires GPTQ-quantized model",
        # PyTorch optional dependencies
        "pytorch.load_in_4bit": "Requires compatible bitsandbytes version",
        "pytorch.load_in_8bit": "Requires compatible bitsandbytes version",
    }


def get_streaming_constraints() -> dict[str, str]:
    """Streaming is a Phase 5 concern.

    Returns:
        Empty dict — streaming parameters are not part of v2.0 M1 scope.
    """
    return {}


def get_streaming_incompatible_tests() -> list[tuple[str, str]]:
    """Streaming is a Phase 5 concern.

    Returns:
        Empty list — streaming parameters are not part of v2.0 M1 scope.
    """
    return []


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
            # PyTorch does NOT support tensor parallelism for HuggingFace models
            "pytorch": False,
            "vllm": "tensor_parallel_size" in vllm_fields,
            "tensorrt": "tp_size" in tensorrt_fields,
        },
        "data_parallel": {
            # PyTorch uses data parallelism via Accelerate (num_processes)
            "pytorch": "num_processes" in pytorch_fields,
            # vLLM/TensorRT manage parallelism internally
            "vllm": False,
            "tensorrt": False,
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
            "tensorrt": "INT8/INT4/FP8" if trt_quant_options else False,
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
        "prefix_caching": {
            "pytorch": False,
            "vllm": "enable_prefix_caching" in vllm_fields,
            "tensorrt": False,
        },
        "lora_adapters": {
            "pytorch": True,  # Via peft library
            "vllm": False,  # Not in v2.0 minimal VLLMConfig
            "tensorrt": False,  # Not in v2.0 minimal TensorRTConfig
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

    # Define display names
    display_names = {
        "tensor_parallel": "Tensor Parallel",
        "data_parallel": "Data Parallel",
        "bitsandbytes_4bit": "BitsAndBytes (4-bit)",
        "bitsandbytes_8bit": "BitsAndBytes (8-bit)",
        "native_quantization": "Native Quantization",
        "float32_precision": "float32 precision",
        "float16_precision": "float16 precision",
        "bfloat16_precision": "bfloat16 precision",
        "prefix_caching": "Prefix Caching",
        "lora_adapters": "LoRA Adapters",
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
                cells.append("Yes")
            elif value is False:
                cells.append("No")
            elif isinstance(value, str):
                cells.append(value)
            else:
                cells.append("No")

        lines.append(f"| {display_name} | {cells[0]} | {cells[1]} | {cells[2]} |")

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
    return [
        {
            "backend": "pytorch",
            "combination": "load_in_4bit=True + load_in_8bit=True",
            "reason": "Cannot use both 4-bit and 8-bit quantization simultaneously",
            "resolution": "Choose one: pytorch.load_in_4bit=true OR pytorch.load_in_8bit=true",
        },
        {
            "backend": "all",
            "combination": "backend section mismatch",
            "reason": "Backend section must match the backend field",
            "resolution": "Ensure pytorch:/vllm:/tensorrt: section matches backend: field",
        },
        {
            "backend": "all",
            "combination": "passthrough_kwargs key collision",
            "reason": "passthrough_kwargs keys must not collide with ExperimentConfig fields",
            "resolution": "Use named fields directly instead of passthrough_kwargs",
        },
        {
            "backend": "tensorrt",
            "combination": "precision=fp32",
            "reason": "TensorRT-LLM is optimised for lower precision inference",
            "resolution": "Use precision='fp16' or 'bf16'",
        },
        {
            "backend": "vllm",
            "combination": "load_in_4bit or load_in_8bit",
            "reason": "vLLM does not support bitsandbytes quantization",
            "resolution": "Use vllm.quantization (awq, gptq, fp8) for quantized inference",
        },
    ]
