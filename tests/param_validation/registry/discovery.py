"""Pydantic model field discovery for automatic parameter extraction.

Provides utilities to introspect Pydantic models and extract field
information for generating ParamSpec definitions.
"""

from __future__ import annotations

import contextlib
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass
class DiscoveredField:
    """Information extracted from a Pydantic field."""

    name: str
    type_annotation: Any
    field_info: FieldInfo
    default_value: Any
    description: str
    is_optional: bool
    is_nested_model: bool
    nested_model_class: type | None

    # Constraints extracted from field
    ge: float | int | None = None  # >= constraint
    le: float | int | None = None  # <= constraint
    gt: float | int | None = None  # > constraint
    lt: float | int | None = None  # < constraint

    # Type-specific info
    literal_values: list[Any] = field(default_factory=list)
    is_literal: bool = False
    is_bool: bool = False
    is_numeric: bool = False
    is_string: bool = False
    is_dict: bool = False

    @property
    def has_constraints(self) -> bool:
        return any(x is not None for x in [self.ge, self.le, self.gt, self.lt])


def discover_model_fields(model_class: type[BaseModel]) -> dict[str, DiscoveredField]:
    """Extract all fields from a Pydantic model.

    Args:
        model_class: Pydantic model class to introspect.

    Returns:
        Dict mapping field name to DiscoveredField.
    """
    fields = {}

    for name, field_info in model_class.model_fields.items():
        annotation = model_class.model_fields[name].annotation
        discovered = _extract_field_info(name, annotation, field_info)
        fields[name] = discovered

    return fields


def _extract_field_info(name: str, annotation: Any, field_info: FieldInfo) -> DiscoveredField:
    """Extract detailed information from a single field."""
    # Handle Optional types
    is_optional = False
    inner_type = annotation

    origin = get_origin(annotation)
    if origin is typing.Union:
        args = get_args(annotation)
        # Check if it's Optional[X] (Union[X, None])
        if type(None) in args:
            is_optional = True
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                inner_type = non_none_args[0]

    # Check for nested Pydantic models
    is_nested_model = False
    nested_model_class = None
    if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
        is_nested_model = True
        nested_model_class = inner_type

    # Extract Literal values
    literal_values = []
    is_literal = False
    inner_origin = get_origin(inner_type)
    if inner_origin is Literal:
        is_literal = True
        literal_values = list(get_args(inner_type))

    # Determine base type
    is_bool = inner_type is bool or (
        is_literal and all(isinstance(v, bool) for v in literal_values)
    )
    is_numeric = inner_type in (int, float) or (
        is_literal and all(isinstance(v, int | float) for v in literal_values)
    )
    is_string = inner_type is str or (
        is_literal and all(isinstance(v, str) for v in literal_values)
    )
    is_dict = inner_origin is dict or inner_type is dict

    # Extract constraints from field_info metadata
    ge = le = gt = lt = None
    for metadata in field_info.metadata:
        if hasattr(metadata, "ge"):
            ge = metadata.ge
        if hasattr(metadata, "le"):
            le = metadata.le
        if hasattr(metadata, "gt"):
            gt = metadata.gt
        if hasattr(metadata, "lt"):
            lt = metadata.lt

    # Get default value
    default_value = field_info.default
    if default_value is None and field_info.default_factory is not None:
        with contextlib.suppress(Exception):
            default_value = field_info.default_factory()

    return DiscoveredField(
        name=name,
        type_annotation=annotation,
        field_info=field_info,
        default_value=default_value,
        description=field_info.description or "",
        is_optional=is_optional,
        is_nested_model=is_nested_model,
        nested_model_class=nested_model_class,
        ge=ge,
        le=le,
        gt=gt,
        lt=lt,
        literal_values=literal_values,
        is_literal=is_literal,
        is_bool=is_bool,
        is_numeric=is_numeric,
        is_string=is_string,
        is_dict=is_dict,
    )


def infer_test_values(discovered: DiscoveredField, max_values: int = 4) -> list[Any]:
    """Infer reasonable test values from a discovered field.

    Args:
        discovered: The discovered field information.
        max_values: Maximum number of test values to generate.

    Returns:
        List of values suitable for testing this field.
    """
    values = []

    # Literal types - use all values (up to max)
    if discovered.is_literal and discovered.literal_values:
        return discovered.literal_values[:max_values]

    # Boolean - test both
    if discovered.is_bool:
        return [True, False]

    # Numeric with constraints - sample from valid range
    if discovered.is_numeric:
        return _infer_numeric_values(discovered, max_values)

    # String without Literal - provide placeholder
    if discovered.is_string:
        return ["test_value"]

    # Dict - empty and single-entry
    if discovered.is_dict:
        return [{}, {"test_key": "test_value"}]

    # Default - use the default value if available
    if discovered.default_value is not None:
        values.append(discovered.default_value)

    return values


def _infer_numeric_values(discovered: DiscoveredField, max_values: int) -> list[Any]:
    """Infer test values for numeric fields based on constraints."""
    # Determine bounds
    min_val = discovered.ge if discovered.ge is not None else discovered.gt
    max_val = discovered.le if discovered.le is not None else discovered.lt

    # Adjust for exclusive bounds
    if discovered.gt is not None and min_val == discovered.gt:
        min_val = discovered.gt + 1 if isinstance(discovered.gt, int) else discovered.gt + 0.1
    if discovered.lt is not None and max_val == discovered.lt:
        max_val = discovered.lt - 1 if isinstance(discovered.lt, int) else discovered.lt - 0.1

    # Default bounds if not specified
    if min_val is None:
        min_val = 0
    if max_val is None:
        max_val = min_val + 100  # Arbitrary default range

    # Check if integer type
    is_int = discovered.type_annotation in (int, int | None)

    values = []

    # For small ranges, use all values
    if is_int and (max_val - min_val) <= max_values:
        return list(range(int(min_val), int(max_val) + 1))

    # Sample: min, middle points, max
    if max_values >= 1:
        values.append(int(min_val) if is_int else min_val)

    if max_values >= 2:
        values.append(int(max_val) if is_int else max_val)

    if max_values >= 3:
        mid = (min_val + max_val) / 2
        values.insert(1, int(mid) if is_int else mid)

    if max_values >= 4 and (max_val - min_val) > 2:
        # Add quartiles
        q1 = min_val + (max_val - min_val) * 0.25
        q3 = min_val + (max_val - min_val) * 0.75
        if is_int:
            q1, q3 = int(q1), int(q3)
        if q1 not in values:
            values.insert(1, q1)
        if q3 not in values and len(values) < max_values:
            values.insert(-1, q3)

    return values[:max_values]


def discover_all_backend_params(
    include_nested: bool = True,
) -> dict[str, dict[str, DiscoveredField]]:
    """Discover all parameters from backend config models.

    Args:
        include_nested: Whether to recursively discover nested model fields.

    Returns:
        Dict mapping backend name to dict of field name to DiscoveredField.
    """
    from llm_energy_measure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

    backends = {
        "vllm": VLLMConfig,
        "pytorch": PyTorchConfig,
        "tensorrt": TensorRTConfig,
    }

    all_params: dict[str, dict[str, DiscoveredField]] = {}

    for backend_name, model_class in backends.items():
        fields = discover_model_fields(model_class)

        if include_nested:
            # Recursively discover nested model fields
            nested_fields = {}
            for field_name, discovered in list(fields.items()):
                if discovered.is_nested_model and discovered.nested_model_class:
                    nested = discover_model_fields(discovered.nested_model_class)
                    for nested_name, nested_discovered in nested.items():
                        nested_fields[f"{field_name}.{nested_name}"] = nested_discovered
            fields.update(nested_fields)

        all_params[backend_name] = fields

    return all_params


def get_coverage_report(registered_params: list[str]) -> dict[str, Any]:
    """Generate a coverage report comparing registered params to discovered params.

    Args:
        registered_params: List of param names that have ParamSpecs registered.

    Returns:
        Dict with coverage statistics.
    """
    discovered = discover_all_backend_params()

    total_discovered = 0
    total_covered = 0
    uncovered = []
    coverage_by_backend = {}

    for backend_name, fields in discovered.items():
        backend_total = len(fields)
        backend_covered = 0

        for field_name in fields:
            full_name = f"{backend_name}.{field_name}"
            total_discovered += 1
            if full_name in registered_params:
                total_covered += 1
                backend_covered += 1
            else:
                uncovered.append(full_name)

        coverage_by_backend[backend_name] = (
            (backend_covered / backend_total * 100) if backend_total > 0 else 100.0
        )

    return {
        "total_discovered": total_discovered,
        "total_covered": total_covered,
        "coverage_percent": (total_covered / total_discovered * 100)
        if total_discovered > 0
        else 100.0,
        "uncovered_params": uncovered,
        "coverage_by_backend": coverage_by_backend,
    }
