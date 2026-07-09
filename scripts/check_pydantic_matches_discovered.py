#!/usr/bin/env python3
"""Check Pydantic engine configs align with discovered schemas.

Detects type drift between llem's hand-authored Pydantic models and the
machine-discovered engine parameter schemas. Catches:
- Pydantic Literal values going stale relative to engine enums
- Type narrowing/widening between Pydantic and discovered
- Pydantic fields with no discovered counterpart (unless whitelisted)

Exit 0: clean alignment. Exit 1: unexplained drift detected.
Structured JSON on stdout; human-readable details on stderr.

Run: python scripts/check_pydantic_matches_discovered.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llenergymeasure.config.introspection import get_engine_params
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.schema_loader import SchemaLoader
from llenergymeasure.config.ssot import Engine

ENGINES = tuple(e.value for e in Engine)

# Pydantic fields intentionally added by llem without an engine counterpart.
# Each entry: (engine, leaf_name) with explanation.
LLEM_NATIVE_FIELDS: set[tuple[str, str]] = {
    # -- transformers --
    # Quantization params surfaced at engine level for consistent interface
    ("transformers", "batch_size"),
    # dtype is HF-native (torch_dtype is a deprecated BC alias). Passed via
    # from_pretrained **kwargs, so signature-based discovery misses it.
    ("transformers", "dtype"),
    ("transformers", "load_in_4bit"),
    ("transformers", "load_in_8bit"),
    ("transformers", "bnb_4bit_compute_dtype"),
    ("transformers", "bnb_4bit_quant_type"),
    ("transformers", "bnb_4bit_use_double_quant"),
    # Runtime/compile params not in from_pretrained or GenerationConfig
    ("transformers", "attn_implementation"),
    ("transformers", "torch_compile"),
    ("transformers", "torch_compile_mode"),
    ("transformers", "torch_compile_backend"),
    # Device/memory params
    ("transformers", "device_map"),
    ("transformers", "max_memory"),
    ("transformers", "allow_tf32"),
    ("transformers", "autocast_enabled"),
    ("transformers", "autocast_dtype"),
    ("transformers", "low_cpu_mem_usage"),
    # Parallelism
    ("transformers", "tp_plan"),
    ("transformers", "tp_size"),
    # -- vLLM --
    # Speculative decoding sub-config
    ("vllm", "method"),
    ("vllm", "offload_group_size"),
    ("vllm", "offload_num_in_group"),
    ("vllm", "offload_prefetch_step"),
    ("vllm", "offload_params"),
    ("vllm", "kv_cache_memory_bytes"),
    # Attention sub-config (engine-internal knobs)
    ("vllm", "backend"),
    ("vllm", "flash_attn_version"),
    ("vllm", "flash_attn_max_num_splits_for_cuda_graph"),
    ("vllm", "use_prefill_decode_attention"),
    ("vllm", "use_prefill_query_quantization"),
    ("vllm", "use_cudnn_prefill"),
    ("vllm", "disable_flashinfer_prefill"),
    ("vllm", "disable_flashinfer_q_quantization"),
    ("vllm", "use_trtllm_attention"),
    ("vllm", "use_trtllm_ragged_deepseek_prefill"),
    # Beam search params (llem surfaces from vLLM internals)
    ("vllm", "beam_width"),
    ("vllm", "length_penalty"),
    ("vllm", "early_stopping"),
    # Nested passthrough-config containers. The generated vLLM config exposes
    # these full sub-config models; signature-based discovery introspects the
    # top-level EngineArgs and represents the nested configs opaquely, so it does
    # not recurse into their fields.
    ("vllm", "attention"),
    ("vllm", "beam_search"),
    # SpeculativeConfig knobs (full vLLM speculative-decoding sub-config).
    ("vllm", "disable_padded_drafter_batch"),
    ("vllm", "draft_load_config"),
    ("vllm", "draft_model_config"),
    ("vllm", "draft_parallel_config"),
    ("vllm", "draft_tensor_parallel_size"),
    ("vllm", "num_speculative_tokens"),
    ("vllm", "parallel_drafting"),
    ("vllm", "prompt_lookup_max"),
    ("vllm", "prompt_lookup_min"),
    ("vllm", "rejection_sample_method"),
    ("vllm", "speculative_token_tree"),
    ("vllm", "suffix_decoding_max_cached_requests"),
    ("vllm", "suffix_decoding_max_spec_factor"),
    ("vllm", "suffix_decoding_max_tree_depth"),
    ("vllm", "suffix_decoding_min_token_prob"),
    ("vllm", "synthetic_acceptance_rate"),
    ("vllm", "target_model_config"),
    ("vllm", "target_parallel_config"),
    ("vllm", "use_local_argmax_reduction"),
    # CompilationConfig knobs (full vLLM torch.compile / cudagraph sub-config).
    ("vllm", "cache_dir"),
    ("vllm", "compilation_time"),
    ("vllm", "compile_cache_save_format"),
    ("vllm", "compile_mm_encoder"),
    ("vllm", "compile_ranges_endpoints"),
    ("vllm", "compile_sizes"),
    ("vllm", "cudagraph_copy_inputs"),
    ("vllm", "cudagraph_mm_encoder"),
    ("vllm", "cudagraph_mode"),
    ("vllm", "cudagraph_num_of_warmups"),
    ("vllm", "cudagraph_specialize_lora"),
    ("vllm", "custom_ops"),
    ("vllm", "debug_dump_path"),
    ("vllm", "disabled_custom_ops"),
    ("vllm", "dynamic_shapes_config"),
    ("vllm", "enabled_custom_ops"),
    ("vllm", "encoder_cudagraph_max_images_per_batch"),
    ("vllm", "encoder_cudagraph_token_budgets"),
    ("vllm", "fast_moe_cold_start"),
    ("vllm", "inductor_compile_config"),
    ("vllm", "inductor_passes"),
    ("vllm", "local_cache_dir"),
    ("vllm", "mode"),
    ("vllm", "pass_config"),
    ("vllm", "splitting_ops"),
    ("vllm", "static_all_moe_layers"),
    ("vllm", "static_forward_context"),
    ("vllm", "traced_files"),
    ("vllm", "use_inductor_graph_partition"),
    # -- TensorRT --
    # Sub-config structure differs from engine API
    ("tensorrt", "max_batch_size"),
    ("tensorrt", "max_input_len"),
    ("tensorrt", "max_seq_len"),
    ("tensorrt", "max_num_tokens"),
    ("tensorrt", "free_gpu_memory_fraction"),
    ("tensorrt", "quant_algo"),
    ("tensorrt", "kv_cache_quant_algo"),
    ("tensorrt", "enable_block_reuse"),
    ("tensorrt", "host_cache_size"),
    ("tensorrt", "capacity_scheduling_policy"),
    # Sampling params (TRT-LLM SamplingConfig; sub-config differs from flat engine API)
    ("tensorrt", "top_k"),
    ("tensorrt", "top_p"),
    ("tensorrt", "temperature"),
    ("tensorrt", "repetition_penalty"),
}


# ---------------------------------------------------------------------------
# Type canonicalisation
# ---------------------------------------------------------------------------

_JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "string": "str",
    "array": "list",
    "object": "dict",
}


def _canonicalise_discovered_type(type_str: str) -> str:
    """Canonicalise a discovered schema type string."""
    type_str = type_str.strip()

    # Remove | None suffix (llem always wraps in Optional)
    type_str = re.sub(r"\s*\|\s*None\s*$", "", type_str)

    # Handle Literal types - extract and sort values
    literal_match = re.match(r"Literal\[(.+)\]", type_str)
    if literal_match:
        inner = literal_match.group(1)
        values = sorted(v.strip().strip("'\"") for v in inner.split(","))
        return f"Literal[{', '.join(repr(v) for v in values)}]"

    # Normalise compound types (int | str → sorted)
    if "|" in type_str:
        parts = sorted(_JSON_TO_PYTHON_TYPE.get(p.strip(), p.strip()) for p in type_str.split("|"))
        return " | ".join(parts)

    # Normalise single JSON Schema type names to Python
    return _JSON_TO_PYTHON_TYPE.get(type_str, type_str)


def _canonicalise_pydantic_type(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    """Canonicalise a Pydantic JSON schema property type."""
    # Handle anyOf (Optional[X] → anyOf: [X, null])
    any_of = prop.get("anyOf") or prop.get("allOf")
    if any_of:
        non_null = [p for p in any_of if p.get("type") != "null"]
        if len(non_null) == 1:
            return _canonicalise_pydantic_type(non_null[0], defs)
        # Multiple non-null types
        parts = sorted(_canonicalise_pydantic_type(p, defs) for p in non_null)
        return " | ".join(parts)

    # Handle $ref
    if "$ref" in prop:
        ref_name = prop["$ref"].split("/")[-1]
        ref_def = defs.get(ref_name, {})
        if "enum" in ref_def:
            values = sorted(str(v) for v in ref_def["enum"])
            return f"Literal[{', '.join(repr(v) for v in values)}]"
        return ref_name

    # Handle enum (Literal)
    if "enum" in prop:
        values = sorted(str(v) for v in prop["enum"])
        return f"Literal[{', '.join(repr(v) for v in values)}]"

    # Handle array
    if prop.get("type") == "array":
        items = prop.get("items", {})
        inner = _canonicalise_pydantic_type(items, defs)
        return f"list[{inner}]"

    # Base type
    base = prop.get("type", "any")
    return _JSON_TO_PYTHON_TYPE.get(base, base)


def _is_intentional_narrowing(discovered: str, pydantic: str) -> bool:
    """Check if Pydantic intentionally narrows or opaquely passes a broad engine type.

    Allowed patterns:
    - str -> Literal[...] (curating valid string values)
    - int -> Literal[...] (curating valid int values)
    - Complex class type -> simpler Pydantic type (e.g. CompilationConfig -> dict)
    - anything -> any (opaque passthrough: llem declines to type a complex field
      and lets it through the extra="allow" surface with soft validation against
      the discovered schema - a nested sub-config, a set/list container, or a
      broad str/type union. Widening to Any is the maximal non-narrowing, so it
      is not drift.)
    """
    if pydantic == "any":
        return True
    if pydantic.startswith("Literal["):
        # Simple base type -> Literal (str -> Literal['a', 'b'])
        if discovered in ("str", "int", "float"):
            return True
        # Compound type containing str -> Literal (str | SomeClass -> Literal['a', 'b'])
        if "|" in discovered and any(p.strip() == "str" for p in discovered.split("|")):
            return True
    # Complex discovered type (class name) mapped to simple Pydantic type
    return (
        discovered[0].isupper()
        and not discovered.startswith("Literal[")
        and pydantic in ("dict", "str", "list")
    )


# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------


def _refs_in(prop: dict[str, Any]) -> list[str]:
    """$defs names a property references directly (bare, or inside anyOf/allOf)."""
    refs: list[str] = []
    ref = prop.get("$ref")
    if isinstance(ref, str):
        refs.append(ref.split("/")[-1])
    for key in ("anyOf", "allOf"):
        for member in prop.get(key) or []:
            member_ref = member.get("$ref") if isinstance(member, dict) else None
            if isinstance(member_ref, str):
                refs.append(member_ref.split("/")[-1])
    return refs


def _engine_config_def(engine: str, schema: dict[str, Any]) -> str:
    """Root $defs name of an engine's generated Config model, resolved structurally.

    The top-level ``ExperimentConfig`` schema carries one property per engine
    (``schema["properties"][engine]``) whose $ref (inside an anyOf with null)
    points at that engine's generated Config def. Following the ref here binds
    the walk root to the actual schema structure rather than to the codegen
    module path (``llenergymeasure__engines__<engine>__config__Config``), so a
    codegen rename cannot silently leave the type check dormant - the exact
    failure that hit when the pre-codegen names (``VLLMEngineConfig`` ...)
    stopped matching. A missing property or unresolvable ref is loud, not silent.
    """
    prop = (schema.get("properties") or {}).get(engine)
    if not isinstance(prop, dict):
        raise SystemExit(
            f"{engine}: no top-level property on ExperimentConfig; expected a $ref "
            "to the engine's generated Config def. The type check would otherwise "
            "go dormant - fix the structural walk root."
        )
    refs = [r for r in _refs_in(prop) if r in (schema.get("$defs") or {})]
    if not refs:
        raised = _refs_in(prop) or ["(no $ref)"]
        raise SystemExit(
            f"{engine}: ExperimentConfig property {engine!r} $ref {raised[0]!r} is "
            "not in schema $defs; expected a $ref to the engine's generated Config "
            "def. The type check would otherwise go dormant - fix the structural "
            "walk root."
        )
    return refs[0]


def _collect_props(engine: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """All leaf property schemas reachable from an engine's Config def.

    Resolves the walk root structurally from the top-level ExperimentConfig
    property for the engine, then walks $refs from that generated Config def
    (which fans out to EngineParams / SamplingParams and any nested passthrough
    sub-configs the vllm surface exposes, e.g. CompilationConfig /
    SpeculativeConfig). Resolving structurally instead of via a hardcoded
    class-name list keeps this immune to codegen renames - the exact failure
    that left the type check silently dormant when the pre-codegen names
    (``VLLMEngineConfig`` ...) stopped matching.
    """
    defs = schema.get("$defs") or {}
    root = _engine_config_def(engine, schema)
    props: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    queue = [root]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        for leaf, prop in (defs.get(name, {}).get("properties", {}) or {}).items():
            props.setdefault(leaf, prop)
            queue += _refs_in(prop)
    return props


def _get_pydantic_leaves(engine: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Get flattened Pydantic leaves for an engine with their JSON schema props.

    Returns dict mapping leaf_name -> JSON schema property dict.
    """
    params = get_engine_params(engine)

    all_props = _collect_props(engine, schema)
    if not all_props:
        raise SystemExit(
            f"{engine}: no properties reachable from the generated Config def; "
            "the type-equality check would be silently dormant. Fix the schema walk."
        )

    # Match introspection output to JSON schema props
    return {meta["name"]: all_props.get(meta["name"], {}) for meta in params.values()}


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def check_engine(engine: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    """Check one engine for drift. Returns list of drift records."""
    drifts: list[dict[str, str]] = []
    defs = schema.get("$defs", {})

    loader = SchemaLoader()
    discovered = loader.load_schema(engine)

    # Combine engine_params and sampling_params from discovered
    all_discovered: dict[str, dict[str, Any]] = {}
    all_discovered.update(discovered.engine_params)
    all_discovered.update(discovered.sampling_params)

    # Get Pydantic leaves
    pydantic_leaves = _get_pydantic_leaves(engine, schema)

    # Check Pydantic fields against discovered
    for leaf_name, prop in pydantic_leaves.items():
        if leaf_name in all_discovered:
            # Both sides have it - compare types
            discovered_type = all_discovered[leaf_name].get("type", "")
            if not discovered_type or not prop or discovered_type == "unknown":
                continue

            canon_discovered = _canonicalise_discovered_type(discovered_type)
            canon_pydantic = _canonicalise_pydantic_type(prop, defs)

            if canon_discovered != canon_pydantic:
                # Allow intentional narrowing: engine exposes broad type,
                # llem curates to specific Literal values
                if _is_intentional_narrowing(canon_discovered, canon_pydantic):
                    continue
                drifts.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "kind": "type_mismatch",
                        "discovered": canon_discovered,
                        "pydantic": canon_pydantic,
                    }
                )
        else:
            # Pydantic has it, discovered doesn't
            if (engine, leaf_name) not in LLEM_NATIVE_FIELDS:
                drifts.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "kind": "pydantic_only",
                        "discovered": "(not present)",
                        "pydantic": _canonicalise_pydantic_type(prop, defs) if prop else "unknown",
                    }
                )

    return drifts


def main() -> None:
    schema = ExperimentConfig.model_json_schema()
    all_drifts: list[dict[str, str]] = []

    for engine in ENGINES:
        drifts = check_engine(engine, schema)
        all_drifts.extend(drifts)

        if drifts:
            print(f"\n[{engine}] {len(drifts)} drift(s) detected:", file=sys.stderr)
            for d in drifts:
                print(
                    f"  {d['field']}: {d['kind']} "
                    f"(discovered={d['discovered']}, pydantic={d['pydantic']})",
                    file=sys.stderr,
                )
        else:
            print(f"[{engine}] OK - no drift", file=sys.stderr)

    # Structured output on stdout
    json.dump({"drifts": all_drifts, "total": len(all_drifts)}, sys.stdout, indent=2)
    print(file=sys.stdout)

    sys.exit(1 if all_drifts else 0)


if __name__ == "__main__":
    main()
