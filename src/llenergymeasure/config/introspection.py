"""Configuration introspection for SSOT architecture.

This module provides the Single Source of Truth (SSOT) for parameter metadata
by introspecting Pydantic models. All downstream consumers (tests, CLI, docs)
should use these functions to derive parameter information rather than
maintaining separate parameter lists.

Usage:
    from llenergymeasure.config.introspection import (
        get_engine_params,
        get_experiment_config_schema,
    )

    # Get all params for an engine
    transformers_params = get_engine_params("transformers")

    # Get full JSON schema
    schema = get_experiment_config_schema()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from llenergymeasure.config.ssot import Engine

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


def _literal_options(field_info: FieldInfo | None) -> list[str]:
    """Extract the string options from an ``Optional[Literal[...]]`` field.

    Returns an empty list when the field is missing, unannotated, or not a
    Literal union.
    """
    if not field_info or not field_info.annotation:
        return []
    for arg in get_args(field_info.annotation):
        if arg is type(None):
            continue
        inner_args = get_args(arg)
        if inner_args:
            return [a for a in inner_args if a is not None]
    return []


# =============================================================================
# Field Metadata Helpers (SSOT display labels and roles)
# =============================================================================


def get_display_label(field_info: FieldInfo, field_name: str) -> str:
    """Return display label from json_schema_extra, falling back to title-cased name."""
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        label = extra.get("display_label")
        return str(label) if label is not None else field_name.replace("_", " ").title()
    return field_name.replace("_", " ").title()


def get_field_role(field_info: FieldInfo) -> str | None:
    """Return 'workload' or 'experimental', or None if not annotated."""
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        role = extra.get("role")
        return str(role) if role is not None else None
    return None


def get_swept_field_paths(experiments: list[ExperimentConfig]) -> set[str]:
    """Return dotted field paths that vary across experiments in a study.

    Inspects the experiment list and identifies fields where values differ,
    indicating they are sweep dimensions (independent variables). Recurses
    into nested BaseModel sub-configs (e.g. task.dataset.n_prompts).

    Args:
        experiments: List of resolved ExperimentConfig objects from a study.

    Returns:
        Set of dotted field paths (e.g. {"engine", "task.dataset.n_prompts"}).
    """
    if len(experiments) <= 1:
        return set()

    swept: set[str] = set()

    def _compare_fields(objects: Sequence[BaseModel], prefix: str, *, max_depth: int = 3) -> None:
        if max_depth <= 0:
            return
        first_obj = objects[0]
        for field_name in type(first_obj).model_fields:
            path = f"{prefix}.{field_name}" if prefix else field_name
            values = [getattr(obj, field_name) for obj in objects]

            first_non_none = next((v for v in values if v is not None), None)

            if first_non_none is not None and isinstance(first_non_none, BaseModel):
                non_none_objs = [v for v in values if v is not None]
                if len(non_none_objs) < len(values):
                    swept.add(path)
                if len(non_none_objs) >= 2:
                    _compare_fields(non_none_objs, path, max_depth=max_depth - 1)
            else:
                if len({str(v) for v in values}) > 1:
                    swept.add(path)

    _compare_fields(experiments, "")

    return swept


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
        prefix: Prefix for param paths (e.g., "transformers").
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

    Returns known-invalid values for constrained fields - used by runtime
    parameter tests to verify validation rejects out-of-range inputs.
    One invalid value per constrained field (the simplest violation).
    """
    return {
        # VLLMEngineConfig: one known-invalid value per constrained field
        "vllm.engine.gpu_memory_utilization": [1.5],  # ge=0.0, lt=1.0: 1.5 violates lt
        "vllm.engine.swap_space": [-1.0],  # ge=0.0: negative violates ge
        "vllm.engine.cpu_offload_gb": [-0.5],  # ge=0.0: negative violates ge
        "vllm.engine.max_num_seqs": [0],  # ge=1: 0 violates ge
        "vllm.engine.max_num_batched_tokens": [0],  # ge=1: 0 violates ge
        "vllm.engine.max_model_len": [0],  # ge=1: 0 violates ge
        "vllm.engine.tensor_parallel_size": [0],  # ge=1: 0 violates ge
        "vllm.engine.pipeline_parallel_size": [0],  # ge=1: 0 violates ge
        "vllm.engine.num_speculative_tokens": [0],  # ge=1: 0 violates ge
        # VLLMSamplingConfig: one known-invalid value per constrained field
        "vllm.sampling.presence_penalty": [3.0],  # ge=-2.0, le=2.0: 3.0 violates le
        "vllm.sampling.frequency_penalty": [-3.0],  # ge=-2.0, le=2.0: -3.0 violates ge
        # VLLMEngineConfig: constrained fields
        "vllm.engine.offload_num_in_group": [0],  # ge=1: 0 violates ge
        "vllm.engine.kv_cache_memory_bytes": [0],  # ge=1: 0 violates ge
        # VLLMSamplingConfig: constrained field
        "vllm.sampling.n": [0],  # ge=1: 0 violates ge
        # VLLMBeamSearchConfig: constrained fields
        "vllm.beam_search.beam_width": [0],  # ge=1: 0 violates ge
        # TensorRTConfig: compile-time params
        "tensorrt.max_batch_size": [0],  # ge=1: 0 violates ge
        "tensorrt.tensor_parallel_size": [0],  # ge=1: 0 violates ge
        "tensorrt.max_input_len": [0],  # ge=1: 0 violates ge
        "tensorrt.max_seq_len": [0],  # ge=1: 0 violates ge
        # TensorRTKvCacheConfig: cache params
        "tensorrt.kv_cache_config.max_tokens": [0],  # ge=1: 0 violates ge
        # TensorRTSamplingConfig: sampling params
        "tensorrt.sampling.n": [0],  # ge=1: 0 violates ge
    }


def get_engine_params(engine: str) -> dict[str, dict[str, Any]]:
    """Get all parameters for an engine from its Pydantic model.

    Args:
        engine: One of "transformers", "vllm", "tensorrt".

    Returns:
        Dict mapping param paths to metadata. Each param includes
        ``engine_support: list[str]`` indicating which engines expose it.
    """
    from llenergymeasure.config.engine_configs import (
        TensorRTConfig,
        TransformersConfig,
        VLLMConfig,
    )

    engine_models = {
        "transformers": TransformersConfig,
        "vllm": VLLMConfig,
        "tensorrt": TensorRTConfig,
    }

    if engine not in engine_models:
        raise ValueError(f"Unknown engine: {engine}. Must be one of {list(engine_models.keys())}")

    model_class = engine_models[engine]
    # All values are Pydantic BaseModel subclasses, mypy can't infer this from dict
    params = get_params_from_model(model_class, prefix=engine)  # type: ignore[arg-type]

    # Add engine_support to every param
    for param in params.values():
        param["engine_support"] = [engine]

    # Apply custom test value overrides
    custom_values = _get_custom_test_values()
    for param_path, values in custom_values.items():
        if param_path in params:
            params[param_path]["test_values"] = values

    return params


def get_experiment_config_schema() -> dict[str, Any]:
    """Return the full ExperimentConfig JSON schema (Pydantic v2 schema).

    Returns:
        JSON-serialisable dict with the complete schema including all
        properties, types, constraints, and nested model schemas.
        Uses Pydantic's built-in model_json_schema() - always in sync
        with the actual model definition.
    """
    from llenergymeasure.config.models import ExperimentConfig

    return ExperimentConfig.model_json_schema()


# =============================================================================
# SSOT Engine Capability Matrix
# =============================================================================


def get_engine_capabilities() -> dict[str, dict[str, bool | str]]:
    """Derive engine capability matrix from Pydantic model structure.

    This is the SSOT for the capability matrix shown in documentation.
    Capabilities are inferred by checking which fields exist in each
    engine config and their allowed values.

    Returns:
        Dict mapping capability names to per-engine support status.
        Values are True/False for simple support, or str for notes.
    """
    from llenergymeasure.config.engine_configs import (
        TensorRTConfig,
        TensorRTQuantConfig,
        TransformersConfig,
        VLLMEngineConfig,
    )

    # Get field names for each engine
    # VLLMConfig is nested: engine fields are in VLLMEngineConfig
    transformers_fields = set(TransformersConfig.model_fields.keys())
    vllm_fields = set(VLLMEngineConfig.model_fields.keys())
    tensorrt_fields = set(TensorRTConfig.model_fields.keys())

    # Get quantization Literal values for vLLM and TensorRT
    vllm_quant_options = _literal_options(VLLMEngineConfig.model_fields.get("quantization"))
    trt_quant_options = _literal_options(TensorRTQuantConfig.model_fields.get("quant_algo"))

    return {
        "tensor_parallel": {
            # Transformers exposes HF-native tensor parallelism via tp_plan/tp_size
            "transformers": "tp_plan" in transformers_fields,
            "vllm": "tensor_parallel_size" in vllm_fields,
            "tensorrt": "tensor_parallel_size" in tensorrt_fields,
        },
        "data_parallel": {
            # Transformers data parallelism via Accelerate is not supported in this version
            "transformers": False,
            # vLLM/TensorRT manage parallelism internally
            "vllm": False,
            "tensorrt": False,
        },
        "bitsandbytes_4bit": {
            "transformers": "load_in_4bit" in transformers_fields,
            "vllm": False,  # vLLM uses native quantization, not bitsandbytes
            "tensorrt": False,  # TensorRT uses native quantization
        },
        "bitsandbytes_8bit": {
            "transformers": "load_in_8bit" in transformers_fields,
            "vllm": False,
            "tensorrt": False,
        },
        "native_quantization": {
            "transformers": False,  # Transformers relies on bitsandbytes, not native
            "vllm": "AWQ/GPTQ/FP8" if vllm_quant_options else False,
            "tensorrt": "INT8/W4A16_AWQ/W4A16_GPTQ/FP8" if trt_quant_options else False,
        },
        "float32_precision": {
            "transformers": True,
            # vLLM rejects float32: dtype is Literal["float16", "bfloat16", "auto"]
            "vllm": False,
            # TensorRT-LLM is optimised for lower precision
            "tensorrt": False,
        },
        "float16_precision": {
            "transformers": True,
            "vllm": True,
            "tensorrt": True,
        },
        "bfloat16_precision": {
            "transformers": True,
            "vllm": True,
            "tensorrt": True,
        },
        "prefix_caching": {
            "transformers": False,
            "vllm": "enable_prefix_caching" in vllm_fields,
            "tensorrt": False,
        },
        "torch_compile": {
            "transformers": "torch_compile" in transformers_fields,
            "vllm": False,
            "tensorrt": False,
        },
        "beam_search": {
            "transformers": "num_beams" in transformers_fields,
            "vllm": True,
            "tensorrt": False,
        },
        "speculative_decoding": {
            "transformers": "prompt_lookup_num_tokens" in transformers_fields,
            "vllm": "speculative_config" in vllm_fields,
            "tensorrt": False,
        },
        "static_kv_cache": {
            "transformers": "cache_implementation" in transformers_fields,
            "vllm": False,
            "tensorrt": False,
        },
    }


def get_capability_matrix_markdown() -> str:
    """Generate the capability matrix as a markdown table.

    This is used by doc generation scripts to create the capability
    matrix section in documentation files.

    Returns:
        Markdown table string.
    """
    capabilities = get_engine_capabilities()

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
        "torch_compile": "torch.compile",
        "beam_search": "Beam Search",
        "speculative_decoding": "Speculative Decoding",
        "static_kv_cache": "Static KV Cache",
    }

    lines = [
        "| Feature | Transformers | vLLM | TensorRT |",
        "|---------|---------|------|----------|",
    ]

    for cap_key, cap_values in capabilities.items():
        display_name = display_names.get(cap_key, cap_key)
        cells = []

        # Engine definition order matches the header above; ALL_ENGINES (a frozenset)
        # has no stable order, which reordered/mislabelled the columns.
        for engine in Engine:
            value = cap_values.get(engine, False)
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
    lines.append("- TensorRT-LLM is optimised for FP16/BF16/INT8, not FP32")

    return "\n".join(lines)


def get_validation_rules() -> list[dict[str, str]]:
    """Get validation rules from config validators for documentation.

    Extracts cross-engine validation rules that are enforced at config
    load time. These rules are the SSOT for the "Config Validation Errors"
    section in invalid-combos.md.

    Returns:
        List of dicts with keys: engine, combination, reason, resolution.
    """
    return [
        {
            "engine": "transformers",
            "combination": "load_in_4bit=True + load_in_8bit=True",
            "reason": "Cannot use both 4-bit and 8-bit quantization simultaneously",
            "resolution": "Choose one: transformers.load_in_4bit=true OR transformers.load_in_8bit=true",
        },
        {
            "engine": "transformers",
            "combination": "torch_compile_mode without torch_compile=True",
            "reason": "torch_compile_mode/torch_compile_backend only take effect when torch_compile=True",
            "resolution": "Set transformers.torch_compile=true when using torch_compile_mode or torch_compile_backend",
        },
        {
            "engine": "transformers",
            "combination": "bnb_4bit_* without load_in_4bit=True",
            "reason": "BitsAndBytes 4-bit options require 4-bit quantization to be enabled",
            "resolution": "Set transformers.load_in_4bit=true when using bnb_4bit_compute_dtype, bnb_4bit_quant_type, or bnb_4bit_use_double_quant",
        },
        {
            "engine": "transformers",
            "combination": "cache_implementation with use_cache=False",
            "reason": "Cannot specify a cache strategy when caching is explicitly disabled",
            "resolution": "Remove use_cache=false or remove cache_implementation",
        },
        {
            "engine": "all",
            "combination": "engine section mismatch",
            "reason": "Engine section must match the engine field",
            "resolution": "Ensure transformers:/vllm:/tensorrt: section matches engine: field",
        },
        {
            "engine": "all",
            "combination": "passthrough_kwargs key collision",
            "reason": "passthrough_kwargs keys must not collide with ExperimentConfig fields",
            "resolution": "Use named fields directly instead of passthrough_kwargs",
        },
        {
            "engine": "tensorrt",
            "combination": "dtype=float32",
            "reason": "TensorRT-LLM is optimised for lower-precision inference",
            "resolution": "Use dtype='float16' or 'bfloat16'",
        },
        {
            "engine": "vllm",
            "combination": "load_in_4bit or load_in_8bit",
            "reason": "vLLM does not support bitsandbytes quantization",
            "resolution": "Use vllm.quantization (awq, gptq, fp8) for quantized inference",
        },
    ]


def get_runtime_limitations() -> list[dict[str, str]]:
    """Get known runtime limitations for documentation.

    These combinations pass config validation but may fail at runtime
    due to hardware, model, or package requirements.

    Returns:
        List of dicts with keys: engine, parameter, limitation, resolution.
    """
    return [
        {
            "engine": "transformers",
            "parameter": "transformers.attn_implementation=flash_attention_2",
            "limitation": "flash-attn requires Ampere+ GPU (SM80+); fails on older architectures",
            "resolution": "Use attn_implementation='sdpa' on pre-Ampere GPUs",
        },
        {
            "engine": "transformers",
            "parameter": "transformers.attn_implementation=flash_attention_3",
            "limitation": "FA3 requires the flash_attn_3 package (built from flash-attn hopper/ directory) and Ampere+ GPU (SM80+). The Docker PyTorch image includes it pre-built",
            "resolution": "Install flash_attn_3 from source, or use the Docker runner",
        },
        {
            "engine": "vllm",
            "parameter": "vllm.engine.kv_cache_dtype=fp8",
            "limitation": "FP8 KV cache requires Hopper (H100) or newer GPU",
            "resolution": "Use kv_cache_dtype='auto' for automatic selection",
        },
        {
            "engine": "vllm",
            "parameter": "vllm.engine.attention.backend=flashinfer",
            "limitation": "FlashInfer requires JIT compilation on first use",
            "resolution": "Leave attention.backend unset (auto) or use 'flash_attn'",
        },
        {
            "engine": "vllm",
            "parameter": "vllm.engine.quantization=awq/gptq",
            "limitation": "Requires a pre-quantized model checkpoint",
            "resolution": "Use a quantized model (e.g., TheBloke/*-AWQ) or omit",
        },
        {
            "engine": "tensorrt",
            "parameter": "tensorrt.quant_config.quant_algo=FP8",
            "limitation": "FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM80) raises ConfigurationError - no silent emulation or fallback",
            "resolution": "Use INT8, W4A16_AWQ, W4A16_GPTQ, or W8A16 on A100",
        },
        {
            "engine": "tensorrt",
            "parameter": "tensorrt.quant_config.quant_algo=INT8",
            "limitation": "INT8 quantisation requires a calibrated checkpoint; uncalibrated weights degrade accuracy",
            "resolution": "Use a pre-quantised checkpoint or a weight-only algo (W4A16_AWQ, W4A16_GPTQ, W8A16)",
        },
    ]
