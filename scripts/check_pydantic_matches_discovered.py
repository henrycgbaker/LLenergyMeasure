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

# Pydantic fields present in the generated class but with no engine-discovered
# counterpart. Post engine-knowledge-as-data option-A migration the list is
# empty: the codegen pipeline (regen_engine_configs.py) projects the curated
# subset of schema.discovered.json into Pydantic, so any Pydantic-only field
# necessarily comes from a curated mining surface. The previous 5
# CompileConfig entries dissolved when #671 (transformers nested-dataclass
# walker) landed: CompileConfig now mines as a proper $def and the codegen
# emits a nested Pydantic sub-class driven by mining rather than overlay.
#
# Audit: _spike/findings/phase3_audit_llem_fields.md (52 -> 5 -> 0).
LLEM_NATIVE_FIELDS: set[tuple[str, str]] = set()


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


def _canonicalise_discovered_type(type_repr: Any) -> str:
    """Canonicalise a discovered schema type representation into a Python-ish string.

    Accepts the legacy compact-string form (``"int | None"``), the canonical
    JSON-Schema-2020-12 array form (``["integer", "null"]``), or the
    canonical primitive name (``"string"``). Returns the equivalent Python
    type string with ``| None`` stripped (llem always wraps optionals) and
    JSON Schema primitives normalised to their Python counterparts.
    """
    if isinstance(type_repr, list):
        # ["string", "null"] -> "string" (after stripping null)
        parts = sorted(
            _JSON_TO_PYTHON_TYPE.get(str(t), str(t)) for t in type_repr if str(t) != "null"
        )
        return " | ".join(parts) if parts else "None"
    type_str = str(type_repr).strip() if type_repr is not None else ""

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


def _discovered_to_python_str(spec: dict[str, Any]) -> str:
    """Render a discovered field spec (v2 canonical or v1 compact-string) as a Python-ish string.

    v2 ``anyOf`` branches are unwound and joined with ``|``; v2 ``enum`` lifts
    to ``Literal[...]``. v1 legacy ``type`` strings pass through
    :func:`_canonicalise_discovered_type` unchanged.
    """
    if not isinstance(spec, dict):
        return ""
    # Enum lifts to Literal regardless of branch shape.
    if "enum" in spec:
        values = sorted(str(v) for v in spec["enum"])
        return f"Literal[{', '.join(repr(v) for v in values)}]"
    if "anyOf" in spec:
        # Mirror legacy ``_canonicalise_discovered_type``: drop the ``null``
        # branch (llem always wraps optionals in Optional) and union the rest.
        parts: list[str] = []
        for branch in spec["anyOf"]:
            if not isinstance(branch, dict):
                continue
            if branch.get("type") == "null":
                continue
            parts.append(_discovered_to_python_str(branch))
        deduped = sorted({p for p in parts if p})
        return " | ".join(deduped) if deduped else "None"
    return _canonicalise_discovered_type(spec.get("type"))


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
    """Check if Pydantic intentionally narrows a broad engine type.

    Allowed patterns:
    - str → Literal[...] (curating valid string values)
    - int → Literal[...] (curating valid int values)
    - Complex class type → simpler Pydantic type (e.g. CompilationConfig → dict)
    """
    if pydantic.startswith("Literal["):
        # Simple base type → Literal (str → Literal['a', 'b'])
        if discovered in ("str", "int", "float"):
            return True
        # Compound type containing str → Literal (str | SomeClass → Literal['a', 'b'])
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


def _get_pydantic_leaves(engine: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Get flattened Pydantic leaves for an engine with their JSON schema props.

    Returns dict mapping leaf_name -> JSON schema property dict.
    """
    defs = schema.get("$defs", {})
    params = get_engine_params(engine)
    result: dict[str, dict[str, Any]] = {}

    # Per-engine def names in the ExperimentConfig JSON schema. Generated
    # nested classes share names (EngineParams, SamplingParams) so Pydantic
    # disambiguates via the module-path-qualified key. CompileConfig has no
    # collision and lands as a bare def.
    engine_config_names = {
        "transformers": [
            "llenergymeasure__engines__transformers__config__EngineParams",
            "llenergymeasure__engines__transformers__config__SamplingParams",
            "CompileConfig",
        ],
        "vllm": [
            "llenergymeasure__engines__vllm__config__EngineParams",
            "llenergymeasure__engines__vllm__config__SamplingParams",
        ],
        "tensorrt": [
            "llenergymeasure__engines__tensorrt__config__EngineParams",
            "llenergymeasure__engines__tensorrt__config__SamplingParams",
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


def check_engine(engine: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    """Check one engine for drift. Returns list of drift records."""
    drifts: list[dict[str, str]] = []
    defs = schema.get("$defs", {})

    loader = SchemaLoader()
    discovered = loader.load_schema(engine)

    # Combine engine_params + sampling_params + flattened $defs properties.
    # The Pydantic-side leaf walker descends into nested sub-classes
    # (CompileConfig, KvCacheConfig etc.) and surfaces inner field names as
    # leaves. To detect drift correctly, the discovered side must include
    # the corresponding inner fields too - they live under
    # ``$defs.<ClassName>.properties.<field>`` in the canonical JSON Schema
    # envelope. Without this flattening, every nested sub-class field looks
    # like a "pydantic_only" drift even when it's a faithful mining of an
    # engine-side dataclass.
    all_discovered: dict[str, dict[str, Any]] = {}
    all_discovered.update(discovered.engine_params)
    all_discovered.update(discovered.sampling_params)
    for def_spec in discovered.defs.values():
        if not isinstance(def_spec, dict):
            continue
        props = def_spec.get("properties")
        if isinstance(props, dict):
            for field_name, field_spec in props.items():
                if isinstance(field_spec, dict):
                    all_discovered.setdefault(field_name, field_spec)

    # Get Pydantic leaves
    pydantic_leaves = _get_pydantic_leaves(engine, schema)

    # Check Pydantic fields against discovered
    for leaf_name, prop in pydantic_leaves.items():
        if leaf_name in all_discovered:
            # Both sides have it - compare types
            discovered_spec = all_discovered[leaf_name]
            if not prop or not isinstance(discovered_spec, dict):
                continue
            # ``type=='unknown'`` (v1) and untyped specs (v2 ``{"description": ...}``
            # without ``type``/``anyOf``) carry no useful drift signal.
            if discovered_spec.get("type") == "unknown" or (
                not discovered_spec.get("type") and not discovered_spec.get("anyOf")
            ):
                continue

            # Docstring-mined types (Move 1 kwargs walker) are loose by
            # construction - HF/upstream Sphinx docs often type kwargs as
            # `str` when the actual semantic type is `int` (e.g.
            # ``tp_size``). Skip the type-mismatch check for these; the
            # Pydantic type is authoritative when the source is a
            # docstring. The schema gate still catches signature-mined
            # mismatches (the high-signal case).
            if discovered_spec.get("x-source") == "kwargs_docstring":
                continue

            canon_discovered = _discovered_to_python_str(discovered_spec)
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
