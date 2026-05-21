#!/usr/bin/env python3
"""Check Pydantic engine configs are a valid subset of discovered schemas.

Detects type CONTRADICTIONS between llem's hand-authored Pydantic models
and the machine-discovered engine parameter schemas. llem curates which
fields to expose; the value set within an exposed field is the engine's
contract. Pydantic may provide more precise type information than the
engine signature exposes (e.g. ``Literal[...]`` over a discovered
``str``), but may not silently narrow an already-enumerated engine
type.

Three-state classification per (engine, field):
- SUBSET-COMPATIBLE: pydantic type is a valid subset/restriction of discovered
- LLEM-EXTENSION:    pydantic field absent from discovered, listed in
                     LLEM_NATIVE_FIELDS (kwargs-passthrough, depth-below-
                     introspection, llem-orchestration, or pre-transform;
                     see issue #655 for the planned 4-category encoding)
- CONTRADICTION:     pydantic widens beyond discovered, OR is absent from
                     discovered without a whitelist entry. Only this state
                     fails the gate.

Exit 0: no contradictions. Exit 1: at least one CONTRADICTION.
Structured JSON on stdout (full 3-state classification per field).
Human-readable contradictions on stderr.

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


def _is_subset(discovered: str, pydantic: str) -> bool:
    """Return True iff ``pydantic`` is a valid restriction of ``discovered``.

    llem curates which fields to expose, but the value set within an
    exposed field is the engine's contract. The subset relation accepts
    only TYPE-ENRICHMENT cases - where llem provides more precise type
    information than the engine's signature alone exposes - never
    VALUE-NARROWING of an already-enumerated engine type.

    Patterns recognised:

    1. **Equal types** - trivial pass.
    2. **T <= any** - any concrete pydantic type is a valid view of an
       unconstrained discovered ``any`` (signature carried no type info).
    3. **Literal[A] <= scalar** (str/int/float) - llem types the
       concrete value set the engine accepts at runtime, where the
       signature only said ``str`` / ``int`` / ``float``.
    4. **Literal[A] <= union containing the scalar** - same enrichment
       pattern for ``str | SomeClass``-style signatures.
    5. **dict|str|list <= ClassName** - llem abstracts a complex upstream
       class to a JSON-schema-friendly primitive for serialisation.

    Literal-to-Literal narrowing is NOT accepted: when the engine
    enumerates a value set, llem must match it. A maintainer who wants
    to drop values should remove the field or surface the divergence
    explicitly, not silently shrink the accepted set.
    """
    if discovered == pydantic:
        return True
    if discovered == "any":
        return True

    pydantic_is_literal = pydantic.startswith("Literal[")

    if pydantic_is_literal:
        if discovered in ("str", "int", "float"):
            return True
        if "|" in discovered and any(
            p.strip() in ("str", "int", "float") for p in discovered.split("|")
        ):
            return True

    return bool(
        discovered
        and discovered[0].isupper()
        and not discovered.startswith("Literal[")
        and pydantic in ("dict", "str", "list")
    )


# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------


def _get_pydantic_leaves(engine: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Get flattened Pydantic leaves for an engine with their JSON schema props.

    Returns dict mapping leaf_name -> JSON schema property dict.
    """
    defs = schema.get("$defs", {})
    params = get_engine_params(engine)
    result: dict[str, dict[str, Any]] = {}

    # Build a lookup from the JSON schema for detailed type info
    engine_config_names = {
        "transformers": ["TransformersConfig"],
        "vllm": [
            "VLLMEngineConfig",
            "VLLMSamplingConfig",
            "VLLMBeamSearchConfig",
            "VLLMAttentionConfig",
            "VLLMSpeculativeConfig",
        ],
        "tensorrt": [
            "TensorRTConfig",
            "TensorRTQuantConfig",
            "TensorRTKvCacheConfig",
            "TensorRTSamplingConfig",
            "TensorRTSchedulerConfig",
        ],
    }

    # Collect all properties from relevant $defs
    all_props: dict[str, dict[str, Any]] = {}
    for config_name in engine_config_names.get(engine, []):
        if config_name in defs:
            props = defs[config_name].get("properties", {})
            all_props.update(props)

    # Match introspection output to JSON schema props
    for _path, meta in params.items():
        leaf_name = meta["name"]
        result[leaf_name] = all_props.get(leaf_name, {})

    return result


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


# Classification labels emitted in the per-field record. Only CONTRADICTION
# fails the gate; the other two are silent passes (kept in the JSON
# diagnostic for downstream consumers like the audit-bot in #655).
CLASSIFICATION_SUBSET = "SUBSET-COMPATIBLE"
CLASSIFICATION_EXTENSION = "LLEM-EXTENSION"
CLASSIFICATION_CONTRADICTION = "CONTRADICTION"


def check_engine(engine: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    """Classify each Pydantic field for ``engine`` against the discovered schema.

    Returns one record per (engine, field) pair where the field appears in
    Pydantic, regardless of classification - downstream consumers (the
    audit-bot in #655) want the full 3-state picture, not just failures.
    Discovered-only fields (in upstream but not curated by llem) are out
    of scope here; they're surveyed separately by the curation doc.
    """
    classifications: list[dict[str, str]] = []
    defs = schema.get("$defs", {})

    loader = SchemaLoader()
    discovered = loader.load_schema(engine)

    all_discovered: dict[str, dict[str, Any]] = {}
    all_discovered.update(discovered.engine_params)
    all_discovered.update(discovered.sampling_params)

    pydantic_leaves = _get_pydantic_leaves(engine, schema)

    for leaf_name, prop in pydantic_leaves.items():
        if leaf_name in all_discovered:
            discovered_type = all_discovered[leaf_name].get("type", "")
            if not discovered_type or not prop or discovered_type == "unknown":
                # Discovered type unknown -> can't classify subset relation;
                # treat as silent pass (no information either way).
                continue

            canon_discovered = _canonicalise_discovered_type(discovered_type)
            canon_pydantic = _canonicalise_pydantic_type(prop, defs)

            if _is_subset(canon_discovered, canon_pydantic):
                classifications.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "classification": CLASSIFICATION_SUBSET,
                        "discovered": canon_discovered,
                        "pydantic": canon_pydantic,
                    }
                )
            else:
                classifications.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "classification": CLASSIFICATION_CONTRADICTION,
                        "kind": "type_widens_or_disagrees",
                        "discovered": canon_discovered,
                        "pydantic": canon_pydantic,
                    }
                )
        else:
            canon_pydantic = _canonicalise_pydantic_type(prop, defs) if prop else "unknown"
            if (engine, leaf_name) in LLEM_NATIVE_FIELDS:
                classifications.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "classification": CLASSIFICATION_EXTENSION,
                        "discovered": "(not present)",
                        "pydantic": canon_pydantic,
                    }
                )
            else:
                classifications.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "classification": CLASSIFICATION_CONTRADICTION,
                        "kind": "pydantic_only_unwhitelisted",
                        "discovered": "(not present)",
                        "pydantic": canon_pydantic,
                    }
                )

    return classifications


def main() -> None:
    schema = ExperimentConfig.model_json_schema()
    all_classifications: list[dict[str, str]] = []
    contradictions_total = 0

    for engine in ENGINES:
        records = check_engine(engine, schema)
        all_classifications.extend(records)

        contradictions = [r for r in records if r["classification"] == CLASSIFICATION_CONTRADICTION]
        contradictions_total += len(contradictions)

        if contradictions:
            print(
                f"\n[{engine}] {len(contradictions)} contradiction(s) detected:",
                file=sys.stderr,
            )
            for d in contradictions:
                print(
                    f"  {d['field']}: {d['kind']} "
                    f"(discovered={d['discovered']}, pydantic={d['pydantic']})",
                    file=sys.stderr,
                )
        else:
            subset_n = sum(1 for r in records if r["classification"] == CLASSIFICATION_SUBSET)
            ext_n = sum(1 for r in records if r["classification"] == CLASSIFICATION_EXTENSION)
            print(
                f"[{engine}] OK - no contradictions "
                f"(subset-compatible: {subset_n}, llem-extension: {ext_n})",
                file=sys.stderr,
            )

    json.dump(
        {
            "classifications": all_classifications,
            "total_contradictions": contradictions_total,
            "total_records": len(all_classifications),
        },
        sys.stdout,
        indent=2,
    )
    print(file=sys.stdout)

    sys.exit(1 if contradictions_total else 0)


if __name__ == "__main__":
    main()
