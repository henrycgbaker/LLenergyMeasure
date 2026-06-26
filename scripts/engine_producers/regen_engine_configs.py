"""Generate per-engine typed Pydantic config.py from mined schema data.

Projects three files from an engine's active-pin SSOT
(``engine_versions/<engine>/v<safe>/outputs/``) into a vendored, committed,
typed ``src/llenergymeasure/engines/<engine>/config.py``:

- ``schema.discovered.json`` - the mined schema envelope (types, defaults,
  enums, bounds);
- ``curated.yaml`` - the exposure allowlist (only ``exposed_fields`` entries
  become first-class typed fields);
- ``overlay.yaml`` (optional, absent on the live pin) - hand-authored
  narrowings (tighten a mined field) and completions (add a missed field).

Generation is delegated to ``datamodel-code-generator`` (dev-only); this
wrapper owns the llem-specific parts: reshaping the custom envelope into a
JSON Schema 2020-12 document, applying curation + overlay, and ruff-normalising
the output so byte-comparison is stable. The emitted ``Config`` /
``EngineParams`` / ``SamplingParams`` classes mirror the schema's native
section split and carry ``extra="allow"`` (the live engine config policy).

Two modes mirror ``regen_engine_corpus.py``: ``--check`` (default) regenerates
in memory and byte-compares against the committed file (exit 1 with a diff on
drift); ``--write`` regenerates and writes. ``--engine`` restricts the run; the
default is every engine in ``ENGINES`` (transformers pilot only today).

The committed schema carries structured ``enum``/``minimum``/``maximum``/
``exclusiveMinimum``/``exclusiveMaximum`` keys (mined in-container), which this
wrapper projects into real ``Literal``/bounded fields. Both the mined Python
type spelling and the JSON-native spelling a model-schema lift emits are mapped,
so a bounded native-typed field is never silently widened to ``Any | None``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._current import current_outputs_dir  # noqa: E402

# Engines with config generation enabled. All three engines now generate their
# config.py from the discovered schema + curated.yaml.
ENGINES: tuple[str, ...] = ("transformers", "vllm", "tensorrt")

# Native schema sections, projected into same-named sub-models.
SECTIONS: tuple[str, ...] = ("engine_params", "sampling_params")

# Verified flag combo from
# .product/research/datamodel-codegen-spike-2026-05-23.md, re-checked against
# the pinned datamodel-code-generator 0.62.0 (every flag still present).
# --disable-timestamp is mandatory for --check (the banner timestamp would
# otherwise change every run).
_DMCG_FLAGS: tuple[str, ...] = (
    "--input-file-type",
    "jsonschema",
    "--output-model-type",
    "pydantic_v2.BaseModel",
    "--target-python-version",
    "3.10",
    "--use-annotated",
    "--enum-field-as-literal",
    "all",
    "--use-union-operator",
    "--use-attribute-docstrings",
    "--use-field-description",
    # Preserve x-source / x-source-ref provenance as json_schema_extra on
    # Field(). The bare --field-extra-keys form (literal key names including
    # the "x-" prefix) is what works empirically; --field-extra-keys-without-
    # x-prefix does not (noted during the 2026-05-24 spike audit).
    "--field-extra-keys",
    "x-source",
    "x-source-ref",
    "x-narrowing-applied",
    "x-completion-applied",
    "--disable-timestamp",
)


def _shadow_config_path(engine: str) -> Path:
    """Return ``src/llenergymeasure/engines/<engine>/config.py``."""
    return _PROJECT_ROOT / "src" / "llenergymeasure" / "engines" / engine / "config.py"


# ---------------------------------------------------------------------------
# Envelope -> JSON Schema 2020-12 pre-step
# ---------------------------------------------------------------------------

# Legacy Python-string types in the envelope mapped to a JSON Schema scalar.
# (PR-0.7 envelope canonicalisation, which would make this a no-op, has not
# landed on this branch; the envelope still records types as Python strings.)
_SCALAR_TYPES: dict[str, str] = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
}

# The same scalars in JSON-Schema-native spelling. Sections lifted via a
# pydantic/msgspec ``model_json_schema`` (vLLM ``sampling_params``) record types
# already JSON-native (``"number"``/``"integer"``/``"boolean"``/``"string"``),
# unlike the dataclass-introspected sections that record Python strings. Both
# vocabularies must map, or a bounded native-typed field (``temperature`` ->
# ``number`` with ``ge``/``le``) silently collapses to ``Any | None``.
_JSON_SCALAR_TYPES: frozenset[str] = frozenset(_SCALAR_TYPES.values())


def _scalar_for(member: str) -> str | None:
    """JSON Schema scalar for one union member, or ``None`` if it is not scalar.

    Accepts both vocabularies: the legacy Python spelling (``int`` ->
    ``integer``) and a JSON-native scalar already emitted by a model-schema lift
    (``integer`` -> ``integer``).
    """
    if member in _SCALAR_TYPES:
        return _SCALAR_TYPES[member]
    if member in _JSON_SCALAR_TYPES:
        return member
    return None


def _python_type_to_json_schema(type_str: str | None) -> dict[str, Any]:
    """Translate one envelope ``type`` string into JSON Schema keys.

    Handles the live envelope's vocabulary:

    - scalars in either Python (``"str"``) or JSON-native (``"string"``)
      spelling -> ``{"type": ...}``;
    - the ``"unknown"`` / ``None`` sentinel (fields with no annotation) -> ``{}``
      (datamodel-codegen renders ``Any | None``);
    - union strings (``"str | bool | None"``). Scalar members map to a JSON
      Schema ``anyOf``; any non-scalar member (engine class, PathLike, JSON
      container like ``array``/``object``) collapses the whole field to
      permissive ``Any | None`` (``{}``), since the generated class validates
      SHAPE only and the engine owns the rest.
    """
    if not type_str or type_str == "unknown":
        return {}

    members = [m.strip() for m in type_str.split("|")]
    scalars = [s for m in members if (s := _scalar_for(m)) is not None]
    has_unmappable = any(_scalar_for(m) is None and m != "None" for m in members)

    if has_unmappable or not scalars:
        return {}
    if len(scalars) == 1:
        return {"type": scalars[0]}
    return {"anyOf": [{"type": s} for s in scalars]}


# JSON-Schema-native keys passed through from a mined field shape unchanged.
_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "default",
    "description",
    "enum",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "x-source",
    "x-source-ref",
)


def _field_shape_to_property(shape: dict[str, Any]) -> dict[str, Any]:
    """Translate one ``schema.discovered.json`` field shape to a JSON Schema property."""
    prop = _python_type_to_json_schema(shape.get("type"))
    for key in _PASSTHROUGH_KEYS:
        if key in shape:
            prop[key] = shape[key]
    return prop


# ---------------------------------------------------------------------------
# Curation + overlay
# ---------------------------------------------------------------------------


def _load_curated(outputs: Path) -> dict[str, list[str]]:
    """Read curated.yaml; return per-section exposure allowlists.

    curated.yaml is part of the synced SSOT (always present); a missing
    section just exposes nothing.
    """
    raw: dict[str, Any] = yaml.safe_load((outputs / "curated.yaml").read_text("utf-8")) or {}
    exposed = raw.get("exposed_fields", {})
    return {section: list(exposed.get(section) or []) for section in SECTIONS}


# Non-bound keys an overlay narrowing may TIGHTEN or complete on a mined field.
# Numeric bounds are handled per-edge by _apply_narrowing (keep-tighter), not
# layered blindly, so a mined and a hand-enforced bound on the same edge never
# coexist (which would emit contradictory ge+gt / le+lt on one field).
_NON_BOUND_NARROWING_KEYS: tuple[str, ...] = ("type", "enum", "description")

# Lower/upper numeric edges as (inclusive_key, exclusive_key) JSON Schema names.
_BOUND_EDGES: dict[str, tuple[str, str]] = {
    "lower": ("minimum", "exclusiveMinimum"),
    "upper": ("maximum", "exclusiveMaximum"),
}


def _load_overlay(outputs: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the optional overlay.yaml; return narrowings + completions per section.

    Missing overlay.yaml -> empty (the live path on this branch). Shape::

        {"narrowings": {<section>: {<field>: {...}}},
         "completions": {<section>: {<field>: {...}}}}
    """
    path = outputs / "overlay.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text("utf-8")) or {} if path.is_file() else {}
    return {
        top: {section: (raw.get(top) or {}).get(section) or {} for section in SECTIONS}
        for top in ("narrowings", "completions")
    }


# JSON Schema "type" subtype relations the overlay may tighten TO. A narrowing
# may replace a broader mined type with a strictly narrower one; anything else
# is a contradiction and errors loudly.
_TYPE_NARROWINGS: dict[str, frozenset[str]] = {
    "number": frozenset({"number", "integer"}),
    "integer": frozenset({"integer"}),
    "string": frozenset({"string"}),
    "boolean": frozenset({"boolean"}),
}


def _bound_on_edge(prop: dict[str, Any], edge: str) -> tuple[float, bool] | None:
    """Return ``(value, is_exclusive)`` for the bound on *edge* in *prop*, else None."""
    inclusive_key, exclusive_key = _BOUND_EDGES[edge]
    if exclusive_key in prop:
        return prop[exclusive_key], True
    if inclusive_key in prop:
        return prop[inclusive_key], False
    return None


def _tighter_bound(edge: str, a: tuple[float, bool], b: tuple[float, bool]) -> tuple[float, bool]:
    """Return the tighter (more restrictive) of two bounds on *edge*.

    On the lower edge the larger value is tighter; on the upper edge the smaller
    value is tighter; on a tie the exclusive variant (gt/lt) is tighter than the
    inclusive one (ge/le).
    """
    (a_val, a_excl), (b_val, _) = a, b
    if a_val == b_val:
        return a if a_excl else b
    if edge == "lower":
        return a if a_val > b_val else b
    return a if a_val < b_val else b


def _apply_narrowing(
    field: str, mined: dict[str, Any], narrowing: dict[str, Any]
) -> dict[str, Any]:
    """Tighten a mined property with an overlay narrowing.

    Tighten-only: a ``type`` narrowing must be a subtype of the mined type
    (integer narrows number; like narrows like). A contradiction (``string``
    mined, ``integer`` overlay) raises with both shapes.

    Numeric bounds are resolved per edge: the mined and the hand-enforced bound
    on the same edge never coexist (that would emit contradictory ge+gt / le+lt
    on one field); the tighter survives, so a stale hand-enforced bound retires
    once mining surfaces a stricter one. Enum, type and description complete or
    narrow additively.
    """
    out = dict(mined)
    new_type = narrowing.get("type")
    mined_type = mined.get("type")
    if new_type is not None and mined_type is not None:
        allowed = _TYPE_NARROWINGS.get(mined_type, frozenset({mined_type}))
        if new_type not in allowed:
            raise ValueError(
                f"overlay narrowing on {field!r} contradicts mined type: "
                f"mined {mined_type!r}, overlay {new_type!r} (narrowings may only tighten)."
            )
    for edge, (inclusive_key, exclusive_key) in _BOUND_EDGES.items():
        mined_bound = _bound_on_edge(mined, edge)
        overlay_bound = _bound_on_edge(narrowing, edge)
        if overlay_bound is None:
            continue  # a mined bound (if any) already rides in out unchanged
        chosen = (
            _tighter_bound(edge, mined_bound, overlay_bound)
            if mined_bound is not None
            else overlay_bound
        )
        out.pop(inclusive_key, None)
        out.pop(exclusive_key, None)
        value, is_exclusive = chosen
        out[exclusive_key if is_exclusive else inclusive_key] = value
    for key in _NON_BOUND_NARROWING_KEYS:
        if key in narrowing:
            out[key] = narrowing[key]
    reason = narrowing.get("x-narrowing-reason")
    if reason is not None:
        out["x-narrowing-applied"] = str(reason).strip()
    return out


def compose_synthetic_schema(
    discovered: dict[str, Any],
    curated: dict[str, list[str]],
    overlay: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Compose a JSON Schema 2020-12 doc from envelope + curation + overlay.

    Per section, in order:

    1. Filter the discovered schema through the curated allowlist. A curated
       field is resolved against the UNION of discovered sections (the curated
       split does not line up with the discovered split - e.g. ``use_cache`` is
       curated under engine_params but discovered under sampling_params), then
       placed in the curated section. A field absent from discovery entirely is
       a discovery-debt stub: a permissive ``Any | None`` field.
    2. Apply overlay narrowings (tighten mined fields).
    3. Apply overlay completions (add fields mining missed).

    Each section becomes a named ``$defs`` entry with ``additionalProperties:
    true`` (the ``extra="allow"`` policy) so datamodel-codegen emits a named
    sub-class. The root ``Config`` references them by ``$ref``.
    """
    narrowings = overlay["narrowings"]
    completions = overlay["completions"]
    # Resolve curated fields against the union of discovered sections.
    discovered_fields: dict[str, Any] = {}
    for section in SECTIONS:
        discovered_fields.update(discovered.get(section, {}) or {})

    defs: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    for section in SECTIONS:
        section_props: dict[str, Any] = {}
        for name in curated.get(section, []):
            mined_shape = discovered_fields.get(name)
            # Absent from discovery -> permissive debt stub ({} -> Any | None).
            prop = _field_shape_to_property(mined_shape) if mined_shape is not None else {}
            if name in narrowings[section]:
                prop = _apply_narrowing(name, prop, narrowings[section][name])
            section_props[name] = prop
        for name, completion in completions[section].items():
            if name in section_props:
                raise ValueError(
                    f"overlay completion {section}.{name!r} shadows a curated field; "
                    "use a narrowing to modify a field mining already surfaced."
                )
            prop = {k: v for k, v in completion.items() if k != "x-completion-reason"}
            prop.setdefault("x-source", "engine_overlay")
            if (reason := completion.get("x-completion-reason")) is not None:
                prop["x-completion-applied"] = str(reason).strip()
            section_props[name] = prop

        title = "".join(part.capitalize() for part in section.split("_"))
        defs[title] = {
            "type": "object",
            "additionalProperties": True,
            "properties": section_props,
        }
        properties[section] = {"$ref": f"#/$defs/{title}"}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Config",
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
        "$defs": defs,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _ruff_normalise(path: Path) -> None:
    """Apply repo ruff style to a generated file in place.

    Load-bearing: datamodel-codegen emits single-quoted strings and its own
    import order, both of which differ from repo ruff style. Without this the
    --check byte-compare goes forever-red on cosmetic quote/import drift.
    """
    for cmd in (
        ["ruff", "check", "--fix", "--quiet", str(path)],
        ["ruff", "format", "--quiet", str(path)],
    ):
        subprocess.run(["uv", "run", *cmd], check=True, cwd=_PROJECT_ROOT, capture_output=True)


def generate_config(engine: str, outputs: Path) -> bytes:
    """Generate one engine's config.py bytes (ruff-normalised). No disk writes to the shadow."""
    discovered = json.loads((outputs / "schema.discovered.json").read_text(encoding="utf-8"))
    curated = _load_curated(outputs)
    overlay = _load_overlay(outputs)
    synthetic = compose_synthetic_schema(discovered, curated, overlay)
    safe_version = "v" + str(discovered.get("engine_version", "unknown")).replace(".", "_")
    header = (
        f"# DO NOT EDIT - regenerated from engine_versions/{engine}/{safe_version}/outputs/"
        "{curated.yaml,schema.discovered.json}\n"
        "# Edit those upstream and run "
        "`uv run python scripts/engine_producers/regen_engine_configs.py --write`."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "schema.json"
        out_path = Path(tmpdir) / "config.py"
        in_path.write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
        cmd = [
            "uv",
            "run",
            "datamodel-codegen",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--custom-file-header",
            header,
            *_DMCG_FLAGS,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=_PROJECT_ROOT)
        if proc.returncode != 0:
            raise RuntimeError(
                f"datamodel-codegen failed for {engine}:\n{proc.stdout}\n{proc.stderr}"
            )
        _ruff_normalise(out_path)
        return out_path.read_bytes()


def _file_diff(generated: bytes, on_disk: Path) -> str:
    """Unified diff (committed vs generated); empty when byte-identical."""
    old = on_disk.read_text(encoding="utf-8").splitlines(keepends=True) if on_disk.exists() else []
    new = generated.decode("utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(old, new, fromfile=str(on_disk), tofile=f"{on_disk} (regenerated)")
    )


def sync_engine(engine: str, *, write: bool) -> list[str]:
    """Generate (or check) one engine's config.py. Returns drift reports (--check)."""
    outputs = current_outputs_dir(engine)
    if not outputs.is_dir():
        raise FileNotFoundError(
            f"{engine}: SSOT outputs dir not found ({outputs}). "
            f"Check engine_versions/{engine}/current.yaml and the vendored pin."
        )

    generated = generate_config(engine, outputs)
    target = _shadow_config_path(engine)
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(generated)
        return []
    if generated == (target.read_bytes() if target.exists() else b""):
        return []
    return [f"{engine}/config.py drift:\n{_file_diff(generated, target)}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-engine Pydantic config.py from schema.discovered.json "
            "+ curated.yaml (+ optional overlay.yaml) via datamodel-code-generator."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify generated/committed config.py parity; exit 1 with a diff. Default mode.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate config.py (mutates the working tree) and report what changed.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=ENGINES,
        help="Restrict to one or more engines (repeatable). Default: all enabled.",
    )
    args = parser.parse_args(argv)

    engines = tuple(args.engine) if args.engine else ENGINES
    all_drift: list[str] = []
    for engine in engines:
        drift = sync_engine(engine, write=args.write)
        all_drift.extend(drift)
        if args.write:
            print(f"[regen-configs] wrote: {engine}/config.py")

    if args.write:
        return 0
    if all_drift:
        for entry in all_drift:
            print(entry, file=sys.stderr)
        print(
            "\nDrift between generated config.py and the committed shadow.\n"
            "Regenerate:\n"
            "  uv run python scripts/engine_producers/regen_engine_configs.py --write",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
