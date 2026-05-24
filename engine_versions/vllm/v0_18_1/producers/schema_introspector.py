"""Schema-introspector machinery for vLLM 0.18.1.

Vendored machinery: the full ``discover`` body lives here, version-pinned.
The global ``scripts/engine_producers/vllm_introspector.py`` is a
thin dispatcher that resolves ``library.current_version`` from the SSOT
and delegates to this module.

vLLM 0.18.1 retains the runtime-introspection surface contract from
0.16.0: ``vllm.engine.arg_utils.EngineArgs`` is a stdlib dataclass and
``vllm.SamplingParams`` is msgspec-typed. The introspector emits the
common envelope with engine + sampling parameter specs.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    canonicalize_defs,
    dataclass_fields_to_specs,
    discover_validation_collections,
    jsonschema_property_to_canonical,
    make_envelope,
    merge_validation_collections,
)

LANDMARKS: tuple[str, ...] = (
    "vllm.SamplingParams",
    "vllm.engine.arg_utils.EngineArgs",
)

# Validation-collection source modules. v0.16+ splits the config surface
# into subpackages (vs v0.7.3's monolithic ``vllm.config``). Per-version
# filter: modules absent in this vllm version are skipped silently.
_VALIDATION_COLLECTION_MODULES: tuple[tuple[str, str], ...] = (
    ("vllm.config.model", "engine_params"),
    ("vllm.config.compilation", "engine_params"),
    ("vllm.transformers_utils.config", "engine_params"),
    ("vllm.transformers_utils.configs.speculators.base", "engine_params"),
    ("vllm.model_executor.model_loader", "engine_params"),
    ("vllm.distributed.eplb.policy", "engine_params"),
    ("vllm.v1.structured_output.backend_xgrammar", "engine_params"),
)


def _collect_validation_collections() -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``{section: {field_name: enum_spec}}`` lifted from validator gates.

    See :data:`_VALIDATION_COLLECTION_MODULES` for sources. Per-version
    filter: modules absent in this vllm version are skipped silently.
    """
    by_section: dict[str, dict[str, dict[str, Any]]] = {
        "engine_params": {},
        "sampling_params": {},
    }
    for module_path, section in _VALIDATION_COLLECTION_MODULES:
        try:
            module = importlib.import_module(module_path)
        except (ImportError, AttributeError):
            continue
        try:
            collections = discover_validation_collections(module)
        except Exception:
            continue
        for field_name, spec in collections.items():
            by_section[section].setdefault(field_name, spec)
    return by_section


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover vLLM engine and sampling schemas.

    engine_params:   dataclasses.fields(EngineArgs)
    sampling_params: msgspec.json.schema(SamplingParams)
    """
    import vllm  # type: ignore[import-not-found]
    from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []
    # Shared $defs accumulator: see v0_7_3 introspector comment.
    all_defs: dict[str, Any] = {}
    engine_params = dataclass_fields_to_specs(EngineArgs, defs_acc=all_defs)

    sampling_params: dict[str, Any] = {}
    sampling_defs: dict[str, Any] = {}
    try:
        import msgspec  # type: ignore[import-not-found]

        raw_schema = msgspec.json.schema(vllm.SamplingParams)
        # Capture the upstream $defs block (nested config classes referenced
        # via $ref from sampling fields). Excluded from the envelope's $defs
        # at canonicalize time: ``SamplingParams`` itself - it's the root we
        # unpacked into ``sampling_params`` below; keeping it would be a
        # redundant duplicate.
        sampling_defs = raw_schema.get("$defs") or raw_schema.get("definitions") or {}
        props = raw_schema.get("properties")
        if not props:
            sp_def: Any = sampling_defs.get("SamplingParams") or next(iter(sampling_defs.values()), {})
            props = sp_def.get("properties", {}) if isinstance(sp_def, dict) else {}
        for name, spec in (props or {}).items():
            sampling_params[name] = jsonschema_property_to_canonical(spec)
    except Exception as exc:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": [],
                "reason": f"msgspec.json.schema(SamplingParams) failed: {exc!r}",
            }
        )

    limitations.append(
        {
            "section": "sampling_params",
            "fields": [],
            "reason": "constraints (e.g. temperature>=0, top_p in (0,1]) live in imperative "
            "_verify_args() and are not introspectable from field metadata",
        }
    )
    limitations.append(
        {
            "section": "engine_params",
            "fields": [],
            "reason": "per-field descriptions unavailable (vLLM EngineArgs has only a class docstring)",
        }
    )

    # PR-0.5: lift module-scope validation-collection constants (e.g.
    # ``_STR_DTYPE_TO_TORCH_DTYPE`` keys, ``FI_SUPPORTED_WORLD_SIZES``)
    # into ``enum`` entries on matching fields.
    collections = _collect_validation_collections()
    merge_validation_collections(engine_params, collections["engine_params"])
    merge_validation_collections(sampling_params, collections["sampling_params"])

    # Merge msgspec sampling_defs into the shared accumulator (dataclass
    # walker wins on duplicate keys; it ran first).
    for name, spec in sampling_defs.items():
        all_defs.setdefault(name, spec)

    return make_envelope(
        engine="vllm",
        engine_version=vllm.__version__,
        engine_commit_sha=getattr(vllm, "__commit__", None),
        image_ref=image_ref,
        base_image_ref=None,
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
        defs=canonicalize_defs(all_defs, exclude=["SamplingParams"]),
    )
