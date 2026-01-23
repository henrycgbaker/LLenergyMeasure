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
