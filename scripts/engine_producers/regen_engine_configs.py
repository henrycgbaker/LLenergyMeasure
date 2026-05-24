"""Per-engine Pydantic config codegen wrapper (Move 3).

For each engine, reads ``engine_versions/<engine>/v<safe>/outputs/{
schema.discovered.json, curated.yaml, overlay.yaml (optional)}`` and emits
``src/llenergymeasure/engines/<engine>/config.py`` as a generated Pydantic
v2 class via ``datamodel-code-generator``.

Pipeline:

1. **Envelope -> JSON Schema** pre-step. ``schema.discovered.json`` is a
   custom envelope today (top-level keys: ``engine``, ``engine_version``,
   ``engine_params``, ``sampling_params``, ...). datamodel-codegen wants
   a JSON Schema 2020-12 document with ``type: object`` + ``properties``
   at the top level. The pre-step composes a synthetic root schema that
   nests engine_params + sampling_params, then converts each field-shape
   entry to a JSON Schema property.
2. **Curated.yaml field filter.** Restrict the synthetic schema to only
   the fields listed under ``exposed_fields.{engine_params,sampling_params}``
   so the generated class matches llem's curation - not the full mined
   surface.
3. **Overlay.yaml merge** (engine-knowledge only; llem-orchestration
   fields live in ``HarnessConfig`` at ``src/llenergymeasure/config/
   harness.py``, NOT in overlay). Two operations:
       a. **Narrowings**: tighten mined fields (e.g. minimum, narrower
          enum, type-correction for docstring inaccuracies).
       b. **Completions**: add engine-exposed fields mining missed
          entirely (e.g. nested CompileConfig dataclass not walked yet).
4. **Subprocess invoke ``datamodel-codegen``** with the verified flag combo
   from .product/research/datamodel-codegen-spike-2026-05-23.md.
5. **Write** the result to ``src/llenergymeasure/engines/<engine>/config.py``
   (``--write`` mode) or compare bytes for drift (``--check`` mode).

Sister script: ``regen_engine_corpus.py`` (data shadow). Same modes,
same pre-commit + CI wiring, same writeback bot integration point.
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

# Project root on sys.path for direct + module invocation parity with
# regen_engine_corpus.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._current import (  # noqa: E402
    _find_repo_root,
    current_outputs_dir,
)

# Phase 2-T pilot expanded to vllm + tensorrt in their own phases.
ENGINES: tuple[str, ...] = ("transformers", "vllm", "tensorrt")

# Subset of schema.discovered.json fields used by codegen. The pre-step
# walks both keys and emits one JSON Schema sub-object per key, nested
# under properties.{key} of the synthetic root.
SECTIONS: tuple[str, ...] = ("engine_params", "sampling_params")

# Verified flag combo from
# .product/research/datamodel-codegen-spike-2026-05-23.md.
# --disable-timestamp is mandatory for --check (deterministic output).
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
    # Preserve x-source / x-source-ref provenance keys as
    # json_schema_extra on Field(). The bare --field-extra-keys form
    # (passing the literal schema key names, including the "x-" prefix)
    # is what actually works in datamodel-codegen 0.57.0; the
    # --field-extra-keys-without-x-prefix variant fails empirically
    # despite being recommended in
    # `.product/research/datamodel-codegen-spike-2026-05-23.md`. Note
    # filed there during 2026-05-24 audit.
    "--field-extra-keys",
    "x-source",
    "x-source-ref",
    "x-narrowing-applied",
    "x-completion-applied",
    "--disable-timestamp",
)


def _shadow_config_path(engine: str) -> Path:
    """Return ``src/llenergymeasure/engines/<engine>/config.py``."""
    return (
        _find_repo_root(Path(__file__).resolve())
        / "src"
        / "llenergymeasure"
        / "engines"
        / engine
        / "config.py"
    )


def _load_curated(engine_outputs: Path) -> dict[str, list[str]]:
    """Read curated.yaml; return per-section field allowlists.

    Missing sections default to empty lists. Missing curated.yaml file is
    tolerated and treated as "no curation" -> empty class.
    """
    path = engine_outputs / "curated.yaml"
    if not path.is_file():
        return {section: [] for section in SECTIONS}
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    exposed = raw.get("exposed_fields", {})
    if not isinstance(exposed, dict):
        raise ValueError(
            f"{path}: 'exposed_fields' must be a mapping, got {type(exposed).__name__}"
        )
    return {
        section: list(exposed.get(section, []) or [])
        for section in SECTIONS
    }


def _load_overlay(engine_outputs: Path) -> dict[str, Any]:
    """Read overlay.yaml; return narrowings + completions sub-dicts.

    Missing overlay.yaml is tolerated and treated as "no overlay" (pure
    mined-schema class). Returns a dict with shape::

        {
          "narrowings": {"engine_params": {<name>: <constraint-keys>}, ...},
          "completions": {"engine_params": {<name>: <full-schema-shape>}, ...}
        }

    Missing sections become empty dicts. See
    ``.product/designs/engine-knowledge-as-data.md`` section on overlay.
    """
    empty: dict[str, Any] = {
        "narrowings": {section: {} for section in SECTIONS},
        "completions": {section: {} for section in SECTIONS},
    }
    path = engine_outputs / "overlay.yaml"
    if not path.is_file():
        return empty
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for top_key in ("narrowings", "completions"):
        block = raw.get(top_key)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(
                f"{path}: '{top_key}' must be a mapping, got {type(block).__name__}"
            )
        for section in SECTIONS:
            section_block = block.get(section)
            if section_block is None:
                continue
            if not isinstance(section_block, dict):
                raise ValueError(
                    f"{path}: '{top_key}.{section}' must be a mapping, "
                    f"got {type(section_block).__name__}"
                )
            empty[top_key][section] = section_block
    return empty


_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "type",
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

# Keys an overlay narrowing is allowed to TIGHTEN on a mined field.
# x-narrowing-reason is metadata; it lands as x-source-ref-narrowing.
_NARROWING_KEYS: tuple[str, ...] = (
    "type",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "enum",
    "description",
)

# Keys allowed in an overlay completion (the full JSON Schema shape for
# a brand-new field). 'properties' supports nested objects (e.g.
# compile_config); pass-through unmodified.
_COMPLETION_KEYS: tuple[str, ...] = (
    *_PASSTHROUGH_KEYS,
    "properties",
    "nullable",
)


def _field_shape_to_property(shape: dict[str, Any]) -> dict[str, Any]:
    """Translate one schema.discovered.json field shape -> JSON Schema property.

    The envelope shape is already close to JSON Schema after Move 1's
    canonicalisation (PR #667): types are canonical strings
    (``"string"``, ``"integer"``, ``"number"``, ``"boolean"``), and
    ``enum`` / ``minimum`` / ``maximum`` keys (when present) follow the
    JSON Schema vocabulary.

    Special cases:
    - Missing ``type`` (untyped upstream like GenerationConfig's None
      defaults): emit no ``type`` key; datamodel-codegen renders as
      ``Any | None``.
    - ``default: null``: emitted as ``"default": null`` which becomes
      the Pydantic Field default.
    - ``x-source`` / ``x-source-ref``: passed through via
      ``--field-extra-keys`` -> ``json_schema_extra``.
    """
    out: dict[str, Any] = {}
    for k, v in shape.items():
        if k in _PASSTHROUGH_KEYS:
            out[k] = v
    return out


def _apply_narrowing(
    mined_property: dict[str, Any], narrowing: dict[str, Any]
) -> dict[str, Any]:
    """Merge an overlay narrowing onto a mined field's property dict.

    Narrowing keys override mined keys; everything else (default,
    x-source/x-source-ref provenance) is preserved. The narrowing's
    `x-narrowing-reason` lands as a separate `x-narrowing-applied`
    string on the field so reviewers can trace why the type changed.

    Narrowing is an EXPLICIT correction channel - it can replace
    ``type`` (the tp_size docstring-vs-runtime case). Codegen does not
    enforce subset-relations between mined and narrowing types;
    reviewer guards that via the x-narrowing-reason.
    """
    out = dict(mined_property)
    for key in _NARROWING_KEYS:
        if key in narrowing:
            out[key] = narrowing[key]
    reason = narrowing.get("x-narrowing-reason")
    if reason is not None:
        out["x-narrowing-applied"] = str(reason).strip()
    return out


def _completion_to_property(completion: dict[str, Any]) -> dict[str, Any]:
    """Translate an overlay completion entry into a JSON Schema property.

    Pass-through for the standard JSON Schema keys plus ``properties``
    (nested objects like ``compile_config``) and ``nullable``. The
    ``x-completion-reason`` metadata lands as
    ``x-completion-applied`` on the emitted field; ``x-source``
    defaults to ``engine_overlay`` if absent.
    """
    out: dict[str, Any] = {}
    for k, v in completion.items():
        if k in _COMPLETION_KEYS:
            out[k] = v
    out.setdefault("x-source", "engine_overlay")
    reason = completion.get("x-completion-reason")
    if reason is not None:
        out["x-completion-applied"] = str(reason).strip()
    return out


def _compose_synthetic_schema(
    discovered: dict[str, Any],
    curated: dict[str, list[str]],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Compose a JSON Schema 2020-12 doc from envelope + curation + overlay.

    Order of operations per section:
    1. Filter mined fields through curated allowlist (mined_property dict).
    2. Apply overlay narrowings: tighten/correct mined fields in-place.
    3. Apply overlay completions: ADD fields not present in mining.

    Curated.exposed_fields is the user-visible field-name allowlist; an
    overlay completion is treated as IMPLICITLY curated (the maintainer
    explicitly added it). Overlay narrowings only apply to fields the
    curated allowlist already exposed (no point narrowing an unexposed
    field).

    Uses ``$defs`` + ``$ref`` so datamodel-codegen ALWAYS generates a named
    class per section.
    """
    narrowings = overlay.get("narrowings", {})
    completions = overlay.get("completions", {})

    defs: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    for section in SECTIONS:
        section_fields = discovered.get(section, {}) or {}
        allowlist = curated.get(section, [])
        section_narrowings = narrowings.get(section, {}) or {}
        section_completions = completions.get(section, {}) or {}

        section_props: dict[str, Any] = {}
        # 1 + 2: mined + narrowings, curated-allowlist-ordered.
        for name in allowlist:
            if name not in section_fields:
                continue
            mined = _field_shape_to_property(section_fields[name])
            if name in section_narrowings:
                mined = _apply_narrowing(mined, section_narrowings[name])
            section_props[name] = mined
        # 3: completions, appended in overlay-declaration order.
        for name, completion in section_completions.items():
            if name in section_props:
                # Completion would shadow a mined field; that's an
                # overlay author error. Mining surfaces the field;
                # use a narrowing instead.
                raise ValueError(
                    f"overlay.completions.{section}.{name}: field is "
                    "already in the mined schema (via curated.yaml); "
                    "use overlay.narrowings to modify it instead."
                )
            section_props[name] = _completion_to_property(completion)

        section_title = "".join(p.capitalize() for p in section.split("_"))
        defs[section_title] = {
            "type": "object",
            "additionalProperties": True,
            "properties": section_props,
        }
        properties[section] = {"$ref": f"#/$defs/{section_title}"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Config",
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
        "$defs": defs,
    }


def _custom_header(engine: str, version: str) -> str:
    """Per-engine 'DO NOT EDIT' header passed via --custom-file-header."""
    return (
        "# DO NOT EDIT - regenerated from "
        f"engine_versions/{engine}/v{version.replace('.', '_')}/outputs/"
        "{curated.yaml,schema.discovered.json}\n"
        "# Edit those upstream and run "
        "`uv run python scripts/engine_producers/regen_engine_configs.py --write`."
    )


def _generate_config(
    engine: str, *, outputs_dir: Path, target: Path
) -> bytes:
    """Run datamodel-codegen; return the generated bytes.

    Doesn't touch ``target`` itself - callers compare/write as appropriate.
    """
    discovered = json.loads(
        (outputs_dir / "schema.discovered.json").read_text(encoding="utf-8")
    )
    curated = _load_curated(outputs_dir)
    overlay = _load_overlay(outputs_dir)
    synthetic = _compose_synthetic_schema(discovered, curated, overlay)

    engine_version = str(discovered.get("engine_version", "unknown"))

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
            _custom_header(engine, engine_version),
            *_DMCG_FLAGS,
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"datamodel-codegen failed for {engine}:\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return out_path.read_bytes()


def _file_diff(generated: bytes, on_disk_path: Path) -> str:
    """Return unified diff string (on_disk vs generated); empty if match."""
    on_disk = on_disk_path.read_text(encoding="utf-8") if on_disk_path.exists() else ""
    new = generated.decode("utf-8")
    return "".join(
        difflib.unified_diff(
            on_disk.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(on_disk_path),
            tofile=f"{on_disk_path} (regenerated)",
        )
    )


def sync_engine(engine: str, *, write: bool) -> tuple[list[str], list[str]]:
    """Sync (or check) one engine's generated config.py.

    Returns ``(drift_messages, skip_messages)`` parallel to
    ``regen_engine_corpus.sync_engine`` for symmetry. Renovate-pre-vendor
    window (missing outputs dir) is tolerated.
    """
    drift: list[str] = []
    skipped: list[str] = []
    try:
        outputs = current_outputs_dir(engine)
    except FileNotFoundError as exc:
        skipped.append(f"{engine}: {exc}")
        return drift, skipped
    if not outputs.is_dir():
        skipped.append(
            f"{engine}: outputs directory not present ({outputs}); "
            f"cells haven't produced this version yet. Skipping."
        )
        return drift, skipped
    if not (outputs / "schema.discovered.json").is_file():
        skipped.append(
            f"{engine}: schema.discovered.json missing in {outputs}; "
            f"cells incomplete. Skipping."
        )
        return drift, skipped

    target = _shadow_config_path(engine)
    generated = _generate_config(engine, outputs_dir=outputs, target=target)

    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(generated)
        return drift, skipped

    # --check: compare bytes.
    on_disk = target.read_bytes() if target.exists() else b""
    if generated == on_disk:
        return drift, skipped
    drift.append(f"{engine}/config.py drift:\n{_file_diff(generated, target)}")
    return drift, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate per-engine Pydantic config.py from "
            "schema.discovered.json + curated.yaml."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify parity between generated and committed config.py. Exit 1 on drift. CI gate. Default mode.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate config.py and write to src/<engine>/config.py.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=ENGINES,
        default=None,
        help="Restrict to one or more engines (repeatable). Default: all.",
    )
    args = parser.parse_args(argv)

    write = args.write
    engines = tuple(args.engine) if args.engine else ENGINES

    all_drift: list[str] = []
    all_skipped: list[str] = []
    for engine in engines:
        drift, skipped = sync_engine(engine, write=write)
        all_drift.extend(drift)
        all_skipped.extend(skipped)

    for entry in all_skipped:
        print(f"[regen-configs] skipped: {entry}", file=sys.stderr)

    if not write and all_drift:
        for entry in all_drift:
            print(entry, file=sys.stderr)
        print(
            "\nDrift detected between generated config.py and committed shadow.\n"
            "Resolution:\n"
            "    uv run python scripts/engine_producers/regen_engine_configs.py --write",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
