#!/usr/bin/env python3
"""Validate a discovered engine schema against the live engine (link 1 -> 2).

The engine-owned-SSOT gate for parameter SCHEMA, sibling to
``validate_invariants.py`` (which gates invariants). It answers "is the
discovered schema VALID?" - i.e. does ``schema.discovered.json`` still tell the
truth about the live engine library in its container.

It does this the "observe, don't re-encode" way: it RE-RUNS the engine's own
schema introspector (the SSOT producer at
``engine_versions/<engine>/<v>/producers/schema_introspector.py``) inside the
container and diffs the stored schema against the freshly reflected one,
field by field, for ``{exists, type, default}``. Then, for caller-touchable
enum fields, it CONSTRUCT-PROBES the live config: a valid value must be
ACCEPTED and an out-of-domain value must be REJECTED. Divergence (field
absent / type mismatch / default mismatch / wrong accept-reject) fails.

This is link 1 -> 2 (live engine -> discovered schema). The separate
``check_pydantic_matches_discovered.py`` covers link 2 -> 3 (is the packaged
pydantic the same as discovered).

Usage (inside the engine's container)::

    python scripts/validate_schema.py \\
        --engine tensorrt \\
        --discovered src/llenergymeasure/engines/tensorrt/schema.discovered.json \\
        --out src/llenergymeasure/engines/tensorrt/schema.validated.json \\
        --image-ref nvcr.io/nvidia/tensorrt-llm/release:1.2.1

Exit codes: 0 clean, 1 divergence (envelope still written), 2 hard error.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the invariant gate's TRT-LLM construction (bare-name resolution across
# tensorrt_llm export modules + model-placeholder injection for *LlmArgs).
from scripts.validate_invariants import _construct_trtllm  # noqa: E402

SCHEMA_VERSION = "1.0.0"
_SECTIONS = ("engine_params", "sampling_params")
_PROBE_STRING_SENTINEL = "__llem_invalid_schema_probe__"

# Per-engine, per-section live class the discovered schema was reflected FROM -
# mirrors each engine's schema_introspector so the construct-probe exercises
# the exact type the discovery observed. Only enum-bearing scalar fields are
# probed; the reflection diff covers the rest.
_PROBE_TARGETS: dict[str, dict[str, str]] = {
    "tensorrt": {
        "engine_params": "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
        "sampling_params": "tensorrt_llm.SamplingParams",
    },
}


class SchemaValidationError(Exception):
    """Hard error (engine not importable, discovered schema malformed)."""


def _json_safe(obj: Any) -> Any:
    """Fallback serialiser: some engine introspectors emit sets (e.g. enum
    value collections) which json can't encode. Render sets as sorted lists,
    everything else by repr, so the validated envelope always writes."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return repr(obj)


# ---------------------------------------------------------------------------
# Live reflection: re-run the engine's own introspector
# ---------------------------------------------------------------------------


def _reflect_live_schema(engine: str, image_ref: str | None) -> dict[str, Any]:
    """Re-run the engine's SSOT schema introspector and return its envelope."""
    try:
        module = importlib.import_module(f"scripts.engine_producers.{engine}_schema_introspector")
    except ImportError as exc:
        raise SchemaValidationError(
            f"no schema introspector importable for engine {engine!r}: {exc}"
        ) from exc
    discover = getattr(module, "discover", None)
    if discover is None:
        raise SchemaValidationError(f"introspector for {engine!r} has no discover()")
    return discover(_PROJECT_ROOT, image_ref)


# ---------------------------------------------------------------------------
# Reflection diff
# ---------------------------------------------------------------------------


def _type_signature(spec: Any) -> str | None:
    """Raw signature of a field spec's TYPE shape (type/enum/anyOf/items), kept
    for the divergence record so a human can see exactly what differs."""
    if not isinstance(spec, dict):
        return None
    relevant = {k: spec[k] for k in ("type", "enum", "anyOf", "items") if k in spec}
    return json.dumps(relevant, sort_keys=True, default=_json_safe)


# Semantic type canonicalisation (mirrors check_pydantic_matches_discovered.py):
# the SAME engine field can be rendered differently as the introspector evolves
# (e.g. ``"string"`` -> ``anyOf[string, string+format:path]``; ``{enum, "string"}``
# -> ``"Literal[...]"``). For a link-1->2 engine-TRUTH check we compare on the
# canonical semantic type, not the literal representation, so introspector-version
# skew doesn't masquerade as engine drift. (The raw signatures are still recorded.)
_JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "string": "str",
    "array": "list",
    "object": "dict",
}


def _canonicalise_type(type_repr: Any) -> str:
    if isinstance(type_repr, list):
        parts = sorted(
            _JSON_TO_PYTHON_TYPE.get(str(t), str(t)) for t in type_repr if str(t) != "null"
        )
        return " | ".join(parts) if parts else "None"
    type_str = str(type_repr).strip() if type_repr is not None else ""
    type_str = re.sub(r"\s*\|\s*None\s*$", "", type_str)
    literal_match = re.match(r"Literal\[(.+)\]", type_str)
    if literal_match:
        values = sorted(v.strip().strip("'\"") for v in literal_match.group(1).split(","))
        return f"Literal[{', '.join(repr(v) for v in values)}]"
    if "|" in type_str:
        parts = sorted(_JSON_TO_PYTHON_TYPE.get(p.strip(), p.strip()) for p in type_str.split("|"))
        return " | ".join(parts)
    return _JSON_TO_PYTHON_TYPE.get(type_str, type_str)


def _ref_target(ref: Any) -> str:
    """Last path segment of a JSON-Schema ``$ref`` ("#/$defs/CompileConfig" ->
    "CompileConfig"), so a nested-config field canonicalises to the type it
    points AT rather than vanishing to "". We compare ref IDENTITY (does the
    field still point at the same nested type?), not the dereferenced internals
    - those nested fields are reflected as their own sections."""
    return str(ref).rstrip("/").rsplit("/", 1)[-1] or str(ref)


def _semantic_type(spec: Any) -> str:
    """Canonical semantic type of a field spec (enum -> Literal, anyOf unwound,
    $ref -> target type name, JSON-Schema primitives -> Python, null/optional
    dropped)."""
    if not isinstance(spec, dict):
        return ""
    if "$ref" in spec:
        return f"ref:{_ref_target(spec['$ref'])}"
    if "enum" in spec:
        return f"Literal[{', '.join(repr(str(v)) for v in sorted(str(v) for v in spec['enum']))}]"
    if "anyOf" in spec:
        parts: list[str] = []
        for branch in spec["anyOf"]:
            if not isinstance(branch, dict) or branch.get("type") == "null":
                continue
            parts.append(_semantic_type(branch))
        deduped = sorted({p for p in parts if p})
        return " | ".join(deduped) if deduped else "None"
    return _canonicalise_type(spec.get("type"))


def _diff_section(
    section: str, stored: dict[str, Any], live: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Diff one section's stored vs live field specs.

    Returns ``(field_results, divergences)``. A field result records exists /
    type_match / default_match per stored field; divergences capture stored
    fields missing or mismatched against live, plus live fields the stored
    schema lacks (drift / GT-growth).
    """
    results: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []

    for name, stored_spec in stored.items():
        live_spec = live.get(name)
        exists = live_spec is not None
        type_match = exists and _semantic_type(stored_spec) == _semantic_type(live_spec)
        default_match = exists and (
            (stored_spec or {}).get("default") == (live_spec or {}).get("default")
        )
        results.append(
            {
                "section": section,
                "field": name,
                "exists": exists,
                "type_match": bool(type_match),
                "default_match": bool(default_match),
            }
        )
        if not exists:
            divergences.append(
                {"section": section, "field": name, "check": "exists", "observed": "absent_in_live"}
            )
            continue
        if not type_match:
            divergences.append(
                {
                    "section": section,
                    "field": name,
                    "check": "type",
                    "declared": _semantic_type(stored_spec),
                    "observed": _semantic_type(live_spec),
                    "declared_raw": _type_signature(stored_spec),
                    "observed_raw": _type_signature(live_spec),
                }
            )
        if not default_match:
            divergences.append(
                {
                    "section": section,
                    "field": name,
                    "check": "default",
                    "declared": (stored_spec or {}).get("default"),
                    "observed": (live_spec or {}).get("default"),
                }
            )

    # Live fields the stored schema does not carry: drift the gate should flag
    # (a bump may have ADDED a field; for GT this is growth, not an error, but
    # we record it so the caller can decide).
    for name in live:
        if name not in stored:
            divergences.append(
                {"section": section, "field": name, "check": "new_in_live", "observed": "present"}
            )

    return results, divergences


# ---------------------------------------------------------------------------
# Construct-probe: enum fields must accept a valid value and reject an invalid one
# ---------------------------------------------------------------------------


def _enum_probe_values(spec: dict[str, Any]) -> tuple[Any, Any] | None:
    """Return ``(valid, invalid)`` probe values for an enum field, or None."""
    enum = spec.get("enum")
    if not isinstance(enum, list) or not enum:
        return None
    valid = enum[0]
    if all(isinstance(v, str) for v in enum):
        invalid: Any = _PROBE_STRING_SENTINEL
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in enum):
        invalid = max(enum) + 1
    else:
        return None
    return valid, invalid


def _probe_field(
    engine: str, native_type: str, field: str, spec: dict[str, Any]
) -> dict[str, Any] | None:
    """Construct-probe one enum field. Returns a probe result or None if the
    field isn't a probeable enum / the engine has no probe target."""
    values = _enum_probe_values(spec)
    if values is None:
        return None
    valid, invalid = values

    def _construct(value: Any) -> str:
        if engine != "tensorrt":
            raise SchemaValidationError(f"no construct-probe wired for engine {engine!r}")
        try:
            _construct_trtllm(native_type, {field: value})
            return "accepted"
        except Exception as exc:
            return f"rejected:{type(exc).__name__}"

    valid_outcome = _construct(valid)
    invalid_outcome = _construct(invalid)
    accept_ok = valid_outcome == "accepted"
    reject_ok = invalid_outcome.startswith("rejected")
    return {
        "section": None,  # filled by caller
        "field": field,
        "valid_value": valid,
        "invalid_value": invalid,
        "valid_outcome": valid_outcome,
        "invalid_outcome": invalid_outcome,
        "accept_ok": accept_ok,
        "reject_ok": reject_ok,
        "probe_confirmed": accept_ok and reject_ok,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def validate_schema(
    *, engine: str, discovered_path: Path, out_path: Path, image_ref: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate ``discovered_path`` against the live engine; write the envelope.

    Returns ``(envelope, divergences)``.
    """
    try:
        stored = json.loads(discovered_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(
            f"cannot read discovered schema {discovered_path}: {exc}"
        ) from exc

    live = _reflect_live_schema(engine, image_ref)

    all_results: list[dict[str, Any]] = []
    all_divergences: list[dict[str, Any]] = []
    all_probes: list[dict[str, Any]] = []
    probe_targets = _PROBE_TARGETS.get(engine, {})

    for section in _SECTIONS:
        stored_section = stored.get(section) or {}
        live_section = live.get(section) or {}
        results, divergences = _diff_section(section, stored_section, live_section)
        all_results.extend(results)
        all_divergences.extend(divergences)

        native_type = probe_targets.get(section)
        if native_type:
            for name, spec in stored_section.items():
                if not isinstance(spec, dict):
                    continue
                probe = _probe_field(engine, native_type, name, spec)
                if probe is None:
                    continue
                probe["section"] = section
                all_probes.append(probe)
                if not probe["probe_confirmed"]:
                    all_divergences.append(
                        {
                            "section": section,
                            "field": name,
                            "check": "construct_probe",
                            "declared": "accept_valid_reject_invalid",
                            "observed": {
                                "valid": probe["valid_outcome"],
                                "invalid": probe["invalid_outcome"],
                            },
                        }
                    )

    confirmed = [r for r in all_results if r["exists"] and r["type_match"] and r["default_match"]]
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_version": live.get("engine_version"),
        "image_ref": image_ref,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "n_fields": len(all_results),
        "n_confirmed": len(confirmed),
        "n_divergences": len(all_divergences),
        "n_probes": len(all_probes),
        "n_probes_confirmed": sum(1 for p in all_probes if p["probe_confirmed"]),
        "fields": all_results,
        "probes": all_probes,
        "divergences": all_divergences,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2, sort_keys=False, default=_json_safe))
    return envelope, all_divergences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", required=True, choices=[*sorted(_PROBE_TARGETS), "transformers", "vllm"]
    )
    parser.add_argument(
        "--discovered", required=True, type=Path, help="schema.discovered.json to validate"
    )
    parser.add_argument("--out", required=True, type=Path, help="schema.validated.json to write")
    parser.add_argument("--image-ref", default=None, help="container image reference to record")
    args = parser.parse_args(argv)

    try:
        envelope, divergences = validate_schema(
            engine=args.engine,
            discovered_path=args.discovered,
            out_path=args.out,
            image_ref=args.image_ref,
        )
    except SchemaValidationError as exc:
        print(f"[validate_schema] HARD ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"[validate_schema] {args.engine}: fields={envelope['n_fields']} "
        f"confirmed={envelope['n_confirmed']} divergences={envelope['n_divergences']} "
        f"probes={envelope['n_probes']} probes_confirmed={envelope['n_probes_confirmed']}",
        flush=True,
    )
    return 1 if divergences else 0


if __name__ == "__main__":
    sys.exit(main())
