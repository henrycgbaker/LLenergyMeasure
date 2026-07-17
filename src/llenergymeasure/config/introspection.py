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


def get_engine_config_model(engine: str) -> type[BaseModel]:
    """Return the generated ``Config`` model class for an engine.

    Raises ``ValueError`` for an unknown engine name.
    """
    from llenergymeasure.engines.tensorrt.config import Config as TensorRTConfig
    from llenergymeasure.engines.transformers.config import Config as TransformersConfig
    from llenergymeasure.engines.vllm.config import Config as VLLMConfig

    engine_models: dict[str, type[BaseModel]] = {
        "transformers": TransformersConfig,
        "vllm": VLLMConfig,
        "tensorrt": TensorRTConfig,
    }
    if engine not in engine_models:
        raise ValueError(f"Unknown engine: {engine}. Must be one of {list(engine_models.keys())}")
    return engine_models[engine]


def _engine_params_field_names(engine: str) -> set[str]:
    """Field names on an engine's generated ``engine_params`` sub-model."""
    ep_field = get_engine_config_model(engine).model_fields.get("engine_params")
    if ep_field is None:
        return set()
    for arg in get_args(ep_field.annotation) or (ep_field.annotation,):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return set(arg.model_fields.keys())
    return set()


def get_engine_params(engine: str) -> dict[str, dict[str, Any]]:
    """Get all parameters for an engine from its generated ``Config`` model.

    Paths use the generated nested shape: ``<engine>.engine_params.<field>`` and
    ``<engine>.sampling_params.<field>``. For transformers, the llem-orchestration
    residual (batch_size, torch_compile, ...) is also included under the engine
    prefix from ``TransformersHarness``.

    Args:
        engine: One of "transformers", "vllm", "tensorrt".

    Returns:
        Dict mapping param paths to metadata. Each param includes
        ``engine_support: list[str]`` indicating which engines expose it.
    """
    model_class = get_engine_config_model(engine)
    params = get_params_from_model(model_class, prefix=engine)

    if engine == "transformers":
        from llenergymeasure.config.harness import TransformersHarness

        params.update(get_params_from_model(TransformersHarness, prefix=engine))

    for param in params.values():
        param["engine_support"] = [engine]

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
    from llenergymeasure.config.harness import TransformersHarness

    # Engine fields live on each generated Config's ``engine_params`` sub-model.
    transformers_fields = _engine_params_field_names("transformers")
    vllm_fields = _engine_params_field_names("vllm")
    tensorrt_fields = _engine_params_field_names("tensorrt")

    # torch.compile is an llem-orchestration knob on TransformersHarness, not an
    # engine field.
    transformers_harness_fields = set(TransformersHarness.model_fields.keys())

    # vLLM/TRT quantization are Any-typed in the generated projection (discovery
    # debt: the mined schema ships no enum), so support is derived from field
    # presence rather than readable Literal options.
    vllm_has_quant = "quantization" in vllm_fields
    trt_has_quant = "quant_config" in tensorrt_fields

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
            "vllm": "AWQ/GPTQ/FP8" if vllm_has_quant else False,
            "tensorrt": "INT8/W4A16_AWQ/W4A16_GPTQ/FP8" if trt_has_quant else False,
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
            "transformers": "torch_compile" in transformers_harness_fields,
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


# Config-load-time errors enforced by ExperimentConfig @model_validator rules
# (not corpus-derived). These live in ``llenergymeasure.config.models`` as
# ``@model_validator`` methods; there is no introspectable data table for them,
# so they are named here explicitly. The "validator" key names the method that
# raises; get_validation_rules cross-checks this set against the live validators
# so the list cannot silently drift out of step (see the tripwire there).
_MODEL_VALIDATOR_RULES: list[dict[str, str]] = [
    {
        "engine": "all",
        "validator": "validate_engine_section_match",
        "combination": "engine section mismatch",
        "reason": "The engine section must match the engine field (validate_engine_section_match).",
        "resolution": "Ensure the transformers:/vllm:/tensorrt: section matches the engine: field.",
    },
    {
        "engine": "all",
        "validator": "validate_passthrough_kwargs_no_collision",
        "combination": "passthrough_kwargs key collision",
        "reason": "passthrough_kwargs keys must not collide with ExperimentConfig "
        "fields (validate_passthrough_kwargs_no_collision).",
        "resolution": "Set the named field directly instead of via passthrough_kwargs.",
    },
    {
        "engine": "all",
        "validator": "validate_engine_section_extras",
        "combination": "unknown field on the engine section wrapper",
        "reason": "A key placed directly on the engine section (not under "
        "engine_params/sampling_params) is never forwarded to the engine "
        "(validate_engine_section_extras).",
        "resolution": "Move the key under <engine>.engine_params or <engine>.sampling_params.",
    },
    {
        "engine": "transformers",
        "validator": "validate_transformers_flash_attn_dtype",
        "combination": "attn_implementation in [flash_attention_2, flash_attention_3] "
        "and dtype=float32",
        "reason": "attn_implementation='flash_attention_2'/'flash_attention_3' requires "
        "dtype='float16' or dtype='bfloat16'; FlashAttention does not support float32 "
        "computation (validate_transformers_flash_attn_dtype).",
        "resolution": "Set transformers.engine_params.dtype to float16 or bfloat16.",
    },
    {
        "engine": "tensorrt",
        "validator": "validate_tensorrt_engine_path_backend",
        "combination": "engine_path is set and backend != trt",
        "reason": "engine_path loads a prebuilt compiled-TensorRT engine directory, which "
        "only the trt constructor can read; the pytorch backend (the default) would "
        "misinterpret it as a checkpoint (validate_tensorrt_engine_path_backend).",
        "resolution": "Set tensorrt.engine_params.backend to trt, or drop engine_path to "
        "build from the model checkpoint.",
    },
]


def _render_predicate(field_path: str, spec: Any) -> str:
    """Render one match predicate as a compact human-readable string.

    ``field_path`` is a dotted path (``vllm.sampling_params.top_p``); it is
    shortened to its leaf name for legibility. ``spec`` is the corpus predicate:
    a bare value (equality) or an operator dict (``{"<": 1}``,
    ``{"in": [...]}``). ``@field`` references are shown verbatim.
    """
    leaf = field_path.rsplit(".", 1)[-1]
    if not isinstance(spec, dict):
        return f"{leaf}={spec}"
    parts: list[str] = []
    for op, value in spec.items():
        if op == "present":
            parts.append(f"{leaf} is set")
        elif op == "absent":
            parts.append(f"{leaf} is unset")
        elif op in ("in", "not_in"):
            joined = ", ".join(str(v) for v in value)
            word = "in" if op == "in" else "not in"
            parts.append(f"{leaf} {word} [{joined}]")
        elif op in ("type_is", "type_is_not"):
            names = value if isinstance(value, str) else ", ".join(str(v) for v in value)
            word = "is" if op == "type_is" else "is not"
            parts.append(f"type({leaf}) {word} {names}")
        elif op in ("divisible_by", "not_divisible_by"):
            word = "divisible by" if op == "divisible_by" else "not divisible by"
            parts.append(f"{leaf} {word} {value}")
        else:
            parts.append(f"{leaf} {op} {value}")
    return " and ".join(parts)


def _render_combination(rule: Any) -> str:
    """Render a rule's full match as ``field-a op-a and field-b op-b``."""
    return " and ".join(_render_predicate(path, spec) for path, spec in rule.match_fields.items())


def _rule_reason(rule: Any) -> str:
    """Human-readable reason for a rule.

    Prefer the rule's message template (rendered with the substitution
    placeholders left literal, since there is no concrete config here); fall
    back to the rule id when a rule ships no template.
    """
    if rule.message_template:
        return " ".join(rule.message_template.split())
    return f"Enforced by rule {rule.id}."


def _corpus_rules_by_severity(severity: str) -> list[tuple[str, Any]]:
    """Return ``(engine_name, rule)`` pairs of one severity, sorted by (engine, id).

    Every engine's shipped rules.yaml is the SSOT; this reads them through the
    same loader the runtime uses, so the doc can never drift from what actually
    fires. Ordering is stable so the generated doc is byte-stable across runs.
    Each caller projects the rule into its own row shape.
    """
    from llenergymeasure.config.engine_rules.loader import EngineRulesLoader

    loader = EngineRulesLoader()
    pairs: list[tuple[str, Any]] = []
    for engine in Engine:
        engine_name = engine.value
        for rule in loader.load_rules(engine_name).rules:
            if rule.severity != severity:
                continue
            pairs.append((engine_name, rule))
    pairs.sort(key=lambda p: (p[0], p[1].id))
    return pairs


def get_validation_rules() -> list[dict[str, str]]:
    """Config-load-time error rules, derived from the live rule corpus + validators.

    These are the SSOT for the "Config Validation Errors" section in
    invalid-combos.md. Rows come from two places, unified into one shape:

    - The ExperimentConfig ``@model_validator`` rules, which raise before any
      engine rule runs (engine-section mismatch, passthrough collision,
      wrapper-level extras, and the transformers flash-attention dtype check).
    - The per-engine rule corpus (``src/llenergymeasure/engines/<e>/rules.yaml``),
      every ``error``-severity rule, read through :class:`EngineRulesLoader`.

    A completeness tripwire cross-checks ``_MODEL_VALIDATOR_RULES`` against the
    live ExperimentConfig validators so adding or renaming a validator without
    updating the list fails ``make docs-check`` instead of silently drifting.

    Returns:
        List of dicts with keys: engine, combination, reason, resolution.
        Deterministically ordered (model validators first, then corpus rules by
        engine then rule id) so the generated doc is byte-stable.
    """
    from llenergymeasure.config.models import ExperimentConfig

    listed = {e["validator"] for e in _MODEL_VALIDATOR_RULES}
    live = {
        name
        for name in ExperimentConfig.__pydantic_decorators__.model_validators
        if name.startswith("validate_")
    }
    if listed != live:
        missing = live - listed
        extra = listed - live
        raise RuntimeError(
            "_MODEL_VALIDATOR_RULES is out of step with ExperimentConfig validators. "
            f"Missing (add a row): {sorted(missing)}. "
            f"Extra (remove a row): {sorted(extra)}."
        )

    rows: list[dict[str, str]] = [
        {
            "engine": e["engine"],
            "combination": e["combination"],
            "reason": e["reason"],
            "resolution": e["resolution"],
        }
        for e in _MODEL_VALIDATOR_RULES
    ]
    for engine_name, rule in _corpus_rules_by_severity("error"):
        rows.append(
            {
                "engine": engine_name,
                "combination": _render_combination(rule),
                "reason": _rule_reason(rule),
                "resolution": "Adjust the field(s) so the condition no longer holds; "
                f"see rule {rule.id}.",
            }
        )
    return rows


def get_dormant_rules() -> list[dict[str, str]]:
    """Dormant (silently-normalised) rules, derived from the live rule corpus.

    A ``dormant`` rule describes a field the engine accepts but silently
    normalises or ignores: the declared value is not the effective value. The
    study planner uses these to deduplicate configs that resolve to the same
    effective configuration, so they never reject a config - they are surfaced
    here so users know which declared values do not take effect.

    Returns:
        List of dicts with keys: engine, combination, effect, normalised_fields.
        Deterministically ordered by engine then rule id.
    """
    rows: list[dict[str, str]] = []
    for engine_name, rule in _corpus_rules_by_severity("dormant"):
        rows.append(
            {
                "engine": engine_name,
                "combination": _render_combination(rule),
                "effect": _rule_reason(rule),
                "normalised_fields": ", ".join(rule.normalised_fields) or "-",
            }
        )
    return rows


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
            "parameter": "transformers.engine_params.attn_implementation=flash_attention_2",
            "limitation": "flash-attn requires Ampere+ GPU (SM80+); fails on older architectures",
            "resolution": "Use attn_implementation='sdpa' on pre-Ampere GPUs",
        },
        {
            "engine": "transformers",
            "parameter": "transformers.engine_params.attn_implementation=flash_attention_3",
            "limitation": "FA3 requires the flash_attn_3 package (built from flash-attn hopper/ directory) and Ampere+ GPU (SM80+). The Docker PyTorch image includes it pre-built",
            "resolution": "Install flash_attn_3 from source, or use the Docker runner",
        },
        {
            "engine": "vllm",
            "parameter": "vllm.engine_params.kv_cache_dtype=fp8",
            "limitation": "FP8 KV cache requires Hopper (H100) or newer GPU",
            "resolution": "Use kv_cache_dtype='auto' for automatic selection",
        },
        {
            "engine": "vllm",
            "parameter": "vllm.engine_params.attention.backend=flashinfer",
            "limitation": "FlashInfer requires JIT compilation on first use",
            "resolution": "Leave attention.backend unset (auto) or use 'flash_attn'",
        },
        {
            "engine": "vllm",
            "parameter": "vllm.engine_params.quantization=awq/gptq",
            "limitation": "Requires a pre-quantized model checkpoint",
            "resolution": "Use a quantized model (e.g., TheBloke/*-AWQ) or omit",
        },
        {
            "engine": "tensorrt",
            "parameter": "tensorrt.engine_params.quant_config.quant_algo=FP8",
            "limitation": "FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM80) raises ConfigurationError - no silent emulation or fallback",
            "resolution": "Use INT8, W4A16_AWQ, W4A16_GPTQ, or W8A16 on A100",
        },
        {
            "engine": "tensorrt",
            "parameter": "tensorrt.engine_params.quant_config.quant_algo=INT8",
            "limitation": "INT8 quantisation requires a calibrated checkpoint; uncalibrated weights degrade accuracy",
            "resolution": "Use a pre-quantised checkpoint or a weight-only algo (W4A16_AWQ, W4A16_GPTQ, W8A16)",
        },
    ]
