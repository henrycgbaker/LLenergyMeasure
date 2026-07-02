"""Generate a per-engine typed Pydantic config.py from mined schema data.

Projects one engine-version snapshot from the workspace SSOT
(``engine_versions/<engine>/v<safe>/outputs/``) into a vendored, committed,
typed ``src/llenergymeasure/engines/<engine>/config.py``:

- ``schema.discovered.json`` - the mined configuration surface (types,
  defaults, enums, bounds);
- ``curated.yaml`` - the exposure allowlist (only ``exposed_fields`` entries
  become first-class typed fields).

Generation is delegated to ``datamodel-code-generator`` (a dev-only tool); this
wrapper owns the llem-specific parts: reshaping the mined envelope into a JSON
Schema 2020-12 document, filtering it through the curated allowlist, and
ruff-normalising the output so byte-comparison is stable. The emitted
``Config`` / ``EngineParams`` / ``SamplingParams`` classes mirror the schema's
native section split and carry ``extra="allow"`` (the live engine config
policy).

The snapshot is selected explicitly by ``--engine`` and ``--version``, so the
tool works against any vendored pin - the active one, or a higher pin being
prepared for a bump - and never assumes the active pin. ``--output`` picks the
target file (default: the ``src/`` shadow for that engine).

Two modes: ``--check`` (default) regenerates in memory and byte-compares
against the target file (exit 1 with a diff on drift); ``--write`` regenerates
and writes it.

The mined schema carries structured ``enum`` / ``minimum`` / ``maximum`` /
``exclusiveMinimum`` / ``exclusiveMaximum`` keys, which this wrapper projects
into real ``Literal`` / bounded fields. Both the mined Python type spelling and
the JSON-native spelling a model-schema lift emits are mapped, so a bounded
native-typed field is never silently widened to ``Any | None``.
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

from engine_versions import _outputs  # noqa: E402

# Native schema sections, projected into same-named sub-models.
SECTIONS: tuple[str, ...] = ("engine_params", "sampling_params")

# Verified datamodel-code-generator flag combo, pinned against the generator
# version in the dev dependency group. ``--disable-timestamp`` is mandatory for
# ``--check`` (the banner timestamp would otherwise change every run).
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
    # Field(). The bare --field-extra-keys form (literal key names including the
    # "x-" prefix) is the spelling that works empirically.
    "--field-extra-keys",
    "x-source",
    "x-source-ref",
    "--disable-timestamp",
)


def _shadow_config_path(engine: str) -> Path:
    """Return ``src/llenergymeasure/engines/<engine>/config.py``."""
    return _PROJECT_ROOT / "src" / "llenergymeasure" / "engines" / engine / "config.py"


# ---------------------------------------------------------------------------
# Envelope -> JSON Schema 2020-12 pre-step
# ---------------------------------------------------------------------------

# Legacy Python-string types in the envelope mapped to a JSON Schema scalar.
_SCALAR_TYPES: dict[str, str] = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
}

# The same scalars in JSON-Schema-native spelling. Sections lifted via a
# pydantic/msgspec ``model_json_schema`` record types already JSON-native
# (``"number"`` / ``"integer"`` / ...), unlike the dataclass-introspected
# sections that record Python strings. Both vocabularies must map, or a bounded
# native-typed field (``temperature`` -> ``number`` with ``ge`` / ``le``)
# silently collapses to ``Any | None``.
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

    - scalars in either Python (``"str"``) or JSON-native (``"string"``)
      spelling -> ``{"type": ...}``;
    - the ``"unknown"`` / ``None`` sentinel (fields with no annotation) -> ``{}``
      (datamodel-codegen renders ``Any | None``);
    - union strings (``"str | bool | None"``): scalar members map to a JSON
      Schema ``anyOf``; any non-scalar member (engine class, PathLike, JSON
      container like ``array`` / ``object``) collapses the whole field to
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
# Per-engine wholesale-forwarded nested blob projection
# ---------------------------------------------------------------------------

# Per-engine set of nested config blobs the engine forwards WHOLESALE and that
# should be projected into a typed submodel rather than collapsed to
# ``Any | None``. Each listed blob is projected ONE LEVEL: scalar/enum/bounded
# interior leaves are typed; any interior leaf that is itself a nested-config
# ``$ref`` stays permissive ``Any | None`` (no deeper recursion - that would
# balloon the transitive closure). Engines absent from this map project
# nothing, so their blobs stay ``Any | None``.
#
# vLLM forwards compilation_config / speculative_config WHOLESALE through
# ``engine_params.model_dump(exclude_none=True)``, so it lists exactly those
# two; tensorrt / transformers list nothing today.
_PROJECTED_NESTED_BLOBS: dict[str, frozenset[str]] = {
    "vllm": frozenset({"compilation_config", "speculative_config"}),
}


def _ref_target(shape: dict[str, Any]) -> str | None:
    """Return the bare ``$defs`` class name a field shape ``$ref``s, else None."""
    ref = shape.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return ref.rsplit("/", 1)[-1]
    return None


def _is_scalar_enum_member(value: Any) -> bool:
    """True if an enum member is a hashable scalar a Literal can carry.

    Guards against non-scalar enum members (e.g. vLLM ``CUDAGraphMode`` carries
    nested-list members like ``[2, 0]``) which cannot be spelled as a Literal;
    such an enum collapses the whole leaf to permissive ``Any | None``.
    """
    return isinstance(value, (str, int, float, bool)) and not isinstance(value, list)


def _project_nested_leaf(shape: dict[str, Any], discovered_defs: dict[str, Any]) -> dict[str, Any]:
    """Project one interior leaf of a wholesale blob to a NULLABLE JSON property.

    One-level projection. Returns a property carrying ``enum`` and numeric
    bounds where the leaf is a clean scalar / enum / bounded value, and a bare
    ``{}`` (-> ``Any | None``) for anything that is not: a ``$ref`` to a nested
    config, an enum with non-scalar members, an array / object / path shape.

    The mined ``default`` is deliberately STRIPPED: every projected leaf is
    nullable with default ``None`` so ``model_dump(exclude_none=True)`` forwards
    only user-set keys (no injected defaults that would change runtime
    behavior), while a set value is still validated against the typed leaf.
    """
    target_name = _ref_target(shape)
    if target_name is not None:
        target = discovered_defs.get(target_name) or {}
        # $ref to a nested config (has properties) -> no deeper projection.
        if "properties" in target or "enum" not in target:
            return {}
        members = target["enum"]
        # An enum with a non-scalar member cannot be spelled as a Literal.
        if not all(_is_scalar_enum_member(m) for m in members):
            return {}
        prop: dict[str, Any] = {"enum": list(members)}
        if "type" in target:
            prop["type"] = target["type"]
        return prop

    prop = _python_type_to_json_schema(shape.get("type"))
    for key in _PASSTHROUGH_KEYS:
        if key == "default":
            continue  # strip mined default -> nullable leaf
        if key in shape:
            prop[key] = shape[key]
    return prop


def _project_nested_blob(class_name: str, discovered_defs: dict[str, Any]) -> dict[str, Any]:
    """Build a typed ``$defs`` entry for one wholesale-forwarded blob class.

    ``additionalProperties: true`` forces ``extra="allow"`` (the upstream
    ``$def`` carries ``additionalProperties: false``, which would reject
    engine-accepted extras). Interior leaves are projected one level via
    ``_project_nested_leaf``; properties are emitted in SORTED key order so the
    codegen output is a deterministic byte-stable fixpoint.
    """
    source = discovered_defs.get(class_name) or {}
    source_props: dict[str, Any] = source.get("properties", {}) or {}
    projected: dict[str, Any] = {
        name: _project_nested_leaf(source_props[name], discovered_defs)
        for name in sorted(source_props)
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": projected,
    }


# ---------------------------------------------------------------------------
# Curation + schema composition
# ---------------------------------------------------------------------------


def _load_curated(outputs: Path) -> dict[str, list[str]]:
    """Read curated.yaml; return per-section exposure allowlists.

    curated.yaml is part of the synced SSOT (always present in a complete
    snapshot); a missing section just exposes nothing.
    """
    raw: dict[str, Any] = (
        yaml.safe_load((outputs / _outputs.CURATED_FILENAME).read_text("utf-8")) or {}
    )
    exposed = raw.get("exposed_fields", {})
    return {section: list(exposed.get(section) or []) for section in SECTIONS}


def compose_schema(
    engine: str,
    discovered: dict[str, Any],
    curated: dict[str, list[str]],
) -> dict[str, Any]:
    """Compose a JSON Schema 2020-12 doc from the mined envelope + curation.

    Per section: filter the discovered schema through the curated allowlist. A
    curated field is resolved against the UNION of discovered sections (the
    curated split does not line up with the discovered split - e.g. ``use_cache``
    is curated under engine_params but discovered under sampling_params), then
    placed in the curated section. A field absent from discovery entirely is a
    discovery-debt stub: a permissive ``Any | None`` field. A curated field
    naming a wholesale-forwarded nested blob is projected one level into a typed
    submodel.

    Each section becomes a named ``$defs`` entry with ``additionalProperties:
    true`` (the ``extra="allow"`` policy) so datamodel-codegen emits a named
    sub-class. The root ``Config`` references them by ``$ref``.
    """
    discovered_defs: dict[str, Any] = discovered.get("$defs", {}) or {}
    blob_fields = _PROJECTED_NESTED_BLOBS.get(engine, frozenset())
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
            blob_class = _ref_target(mined_shape) if mined_shape is not None else None
            if name in blob_fields and blob_class is not None and blob_class in discovered_defs:
                # Wholesale-forwarded nested blob: project ONE LEVEL into a typed
                # submodel instead of collapsing to Any | None.
                defs[blob_class] = _project_nested_blob(blob_class, discovered_defs)
                section_props[name] = {"$ref": f"#/$defs/{blob_class}"}
                continue
            # Absent from discovery -> permissive debt stub ({} -> Any | None).
            section_props[name] = (
                _field_shape_to_property(mined_shape) if mined_shape is not None else {}
            )

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
        # Sorted for a deterministic byte-stable codegen fixpoint.
        "$defs": {name: defs[name] for name in sorted(defs)},
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _ruff_normalise(path: Path) -> None:
    """Apply repo ruff style to a generated file in place.

    Load-bearing: datamodel-codegen emits single-quoted strings and its own
    import order, both of which differ from repo ruff style. Without this the
    ``--check`` byte-compare goes forever-red on cosmetic quote/import drift.
    """
    for cmd in (
        ["ruff", "check", "--fix", "--quiet", str(path)],
        ["ruff", "format", "--quiet", str(path)],
    ):
        subprocess.run(["uv", "run", *cmd], check=True, cwd=_PROJECT_ROOT, capture_output=True)


def generate_config(engine: str, version: str, outputs: Path) -> bytes:
    """Generate one engine snapshot's config.py bytes (ruff-normalised)."""
    discovered = json.loads((outputs / _outputs.SCHEMA_FILENAME).read_text(encoding="utf-8"))
    curated = _load_curated(outputs)
    synthetic = compose_schema(engine, discovered, curated)
    safe_version = _outputs.safe_version(str(discovered.get("engine_version", version)))
    header = (
        f"# DO NOT EDIT - regenerated from engine_versions/{engine}/{safe_version}/outputs/"
        "{curated.yaml,schema.discovered.json}\n"
        "# Edit those upstream and run `uv run python "
        f"scripts/engine_producers/regen_engine_configs.py --engine {engine} "
        f"--version {version} --write`."
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
                f"datamodel-codegen failed for {engine} {version}:\n{proc.stdout}\n{proc.stderr}"
            )
        _ruff_normalise(out_path)
        return out_path.read_bytes()


def _file_diff(generated: bytes, on_disk: Path) -> str:
    """Unified diff (target vs generated); empty when byte-identical."""
    old = on_disk.read_text(encoding="utf-8").splitlines(keepends=True) if on_disk.exists() else []
    new = generated.decode("utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(old, new, fromfile=str(on_disk), tofile=f"{on_disk} (regenerated)")
    )


def sync(engine: str, version: str, output: Path, *, write: bool) -> str | None:
    """Generate (or check) one engine snapshot's config.py.

    Returns a drift report string under ``--check`` when the target does not
    match the freshly generated bytes, else ``None``. Under ``--write`` writes
    the target and always returns ``None``.
    """
    outputs = _outputs.outputs_dir(engine, version)
    if not outputs.is_dir():
        raise FileNotFoundError(
            f"{engine} {version}: snapshot outputs dir not found ({outputs}). "
            f"Expected a mined {_outputs.SCHEMA_FILENAME} + {_outputs.CURATED_FILENAME} there."
        )

    generated = generate_config(engine, version, outputs)
    if write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(generated)
        return None
    if generated == (output.read_bytes() if output.exists() else b""):
        return None
    return f"{output} drift:\n{_file_diff(generated, output)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        required=True,
        choices=_outputs.ENGINES,
        help="Engine whose snapshot to generate from.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Dotted engine version (e.g. 0.19.1) - selects the workspace snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target config.py (default: the src/ shadow for the engine).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify the target matches regeneration; exit 1 with a diff on drift. Default mode.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate and write the target (mutates the working tree).",
    )
    args = parser.parse_args(argv)

    output = args.output if args.output is not None else _shadow_config_path(args.engine)
    drift = sync(args.engine, args.version, output, write=args.write)

    if args.write:
        print(f"[regen-configs] wrote: {output}")
        return 0
    if drift is not None:
        print(drift, file=sys.stderr)
        print(
            "\nDrift between the generated config.py and the target.\nRegenerate:\n"
            f"  uv run python scripts/engine_producers/regen_engine_configs.py "
            f"--engine {args.engine} --version {args.version} --write",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
