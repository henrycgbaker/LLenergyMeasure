"""Per-field section classifier for mined rule paths (D2).

Mined invariants address engine fields by a dotted path
``{engine}.{section}.{field}`` where ``section`` is one of ``engine_params`` /
``sampling_params``. That split is what runtime resolution
(``config.engine_rules.loader.resolve_field_path``) walks against the generated
Config, whose shape is driven by each pin's ``curated.yaml``. The classifier
decides the section for a given field by these rules, in order:

1. **Curated is the SSOT.** If the field is listed under a section in the pin's
   ``curated.yaml`` ``exposed_fields``, use that section. Curation drives the
   generated Config, which is exactly what runtime resolution walks, so a
   curated field's section is authoritative.

2. **Native-class-origin fallback.** Otherwise route by the native class the
   field was mined from (per-engine table below). A field that the engine
   accepts but that curation does not expose still needs a stable section so
   its rule path is well-formed.

3. **The discovered schema's section is non-authoritative** for rule paths -
   only curation + native origin decide. (The schema records where discovery
   *found* a field, which can disagree with how the Config exposes it.)

4. **Fail loud on an unknown native class.** A walked native class with no table
   entry raises :class:`UnknownNativeClassError`. New native surfaces are
   conscious additions, not silent ``engine_params`` defaults.

The tables key on the native class's short name (rightmost dotted component),
which is the stable identifier across the qualified forms the miners emit
(``vllm.config.ModelConfig`` and ``vllm.SamplingParams`` both reduce cleanly).
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from scripts.engine_producers._current import current_outputs_dir

ENGINE_PARAMS = "engine_params"
SAMPLING_PARAMS = "sampling_params"


class UnknownNativeClassError(ValueError):
    """A field's native class has no native-origin table entry for its engine."""


class UnreachableMatchPathError(ValueError):
    """A mined match path addresses a leaf the engine plugin never routes.

    The plugin-routing reachability model (:data:`_REACHABILITY`) determined the
    leaf has no sound runtime path: either its native class is not routed at all,
    or it is a blob-interior / flat leaf outside the plugin's read allowlist. A
    rule on such a leaf is silently dead - ``resolve_field_path`` returns ``None``
    and the predicate never fires - so the miner fails loud rather than emitting it.
    """


# Native-class-origin tables, keyed on the class short name per engine. A class
# absent from its engine's table fails loud (rule 4) - additions are conscious.
_NATIVE_ORIGIN: dict[str, dict[str, str]] = {
    # transformers: BitsAndBytesConfig + from_pretrained-side kwargs land in the
    # model-construction (engine) params; GenerationConfig + CompileConfig are
    # generation-time (sampling) params. Watermarking configs are nested under
    # GenerationConfig's generation surface, so they route sampling-side too.
    "transformers": {
        "BitsAndBytesConfig": ENGINE_PARAMS,
        "GenerationConfig": SAMPLING_PARAMS,
        "CompileConfig": SAMPLING_PARAMS,
        "WatermarkingConfig": SAMPLING_PARAMS,
        "SynthIDTextWatermarkingConfig": SAMPLING_PARAMS,
    },
    # vllm: SamplingParams (incl. nested guided/structured-output params) are
    # sampling-side; EngineArgs + the vllm.config.* sub-configs are engine-side.
    "vllm": {
        "SamplingParams": SAMPLING_PARAMS,
        "GuidedDecodingParams": SAMPLING_PARAMS,
        "StructuredOutputsParams": SAMPLING_PARAMS,
        "EngineArgs": ENGINE_PARAMS,
        "ParallelConfig": ENGINE_PARAMS,
        "CacheConfig": ENGINE_PARAMS,
        "ModelConfig": ENGINE_PARAMS,
        "MultiModalConfig": ENGINE_PARAMS,
        "SchedulerConfig": ENGINE_PARAMS,
        "LoRAConfig": ENGINE_PARAMS,
        "DecodingConfig": ENGINE_PARAMS,
        "SpeculativeConfig": ENGINE_PARAMS,
    },
    # tensorrt: the TrtLlmArgs / llm_args-side classes (incl. their sub-configs
    # and the StrEnum policy types) are engine-side; sampling-side natives are
    # sampling params.
    "tensorrt": {
        "BaseLlmArgs": ENGINE_PARAMS,
        "TrtLlmArgs": ENGINE_PARAMS,
        "CalibConfig": ENGINE_PARAMS,
        "LookaheadDecodingConfig": ENGINE_PARAMS,
        "BatchingType": ENGINE_PARAMS,
        "CapacitySchedulerPolicy": ENGINE_PARAMS,
        "ContextChunkingPolicy": ENGINE_PARAMS,
        # 1.0.0 additions: the pytorch-backend config classes the C++ -> pydantic
        # migration moved into Python (TorchLlmArgs + its nested CudaGraphConfig /
        # TorchCompileConfig). All engine-side; purely additive for the 0.21.0 pin.
        "TorchLlmArgs": ENGINE_PARAMS,
        "CudaGraphConfig": ENGINE_PARAMS,
        "TorchCompileConfig": ENGINE_PARAMS,
        "SamplingParams": SAMPLING_PARAMS,
    },
}


# ---------------------------------------------------------------------------
# Plugin-routing reachability model
# ---------------------------------------------------------------------------
#
# The native-origin table above decides the SECTION of a leaf; this table
# decides whether the leaf is REACHABLE at all and, if so, the SHAPE of its
# match path. The runtime resolver (``resolve_field_path``) walks a dotted path
# against the ExperimentConfig and silently returns ``None`` for a segment that
# is not a declared field - so a path the plugin never routes the value to is a
# dead rule that never fires. This table is mined from each plugin's
# ``_build_llm_kwargs`` / ``_model_load_kwargs`` (line-refs in each entry's
# comment), keyed - like ``_NATIVE_ORIGIN`` - on the native class short name.
#
# Each value is one of the reachability sentinels / shapes below:
#
# - ``FLAT``: the leaf is read flat by the plugin, or EngineArgs flat-hoists it
#   onto the top-level engine-args surface. Match path ``{engine}.{section}.{leaf}``
#   (BYTE-IDENTICAL to the historical lossy namespace-drop for these classes).
# - ``("flat", allowlist)``: FLAT but only for leaves in ``allowlist``; any other
#   leaf of the class is UNREACHABLE (the plugin reads a CLOSED flat set - e.g.
#   transformers BitsAndBytesConfig reads load_in_4bit/bnb_4bit_* flat but never
#   the llm_int8_* knobs).
# - ``("blob", container)``: the user sets ``engine_params.<container> = {dict}``
#   and the plugin forwards it WHOLESALE; every interior leaf is reachable. Match
#   path ``{engine}.{section}.{container}.{leaf}``.
# - ``("blob", container, allowlist)``: a blob container whose plugin reads only a
#   FIXED subset of interior leaves; leaves outside ``allowlist`` are UNREACHABLE.
# - ``UNREACHABLE``: the native class is not routed by the plugin at all - no
#   sound path exists for any of its leaves.

FLAT = "__flat__"
UNREACHABLE = "__unreachable__"

# Reachability rule = FLAT | UNREACHABLE | (kind, *args) tuple.
_Reach = object

_REACHABILITY: dict[str, dict[str, _Reach]] = {
    # transformers plugin (engines/transformers/plugin.py):
    "transformers": {
        # BitsAndBytesConfig: _model_load_kwargs (~522-544) builds BnB from a
        # CLOSED flat set read off engine_params; llm_int8_* are NOT read (fall
        # through to from_pretrained, a non-validating sink) -> UNREACHABLE.
        "BitsAndBytesConfig": (
            "flat",
            frozenset(
                {
                    "load_in_4bit",
                    "load_in_8bit",
                    "bnb_4bit_compute_dtype",
                    "bnb_4bit_quant_type",
                    "bnb_4bit_use_double_quant",
                }
            ),
        ),
        # GenerationConfig: _build_generate_kwargs (~699-736) forwards the whole
        # sampling_params model_dump + select engine_params fields flat into
        # GenerationConfig(**generate_kwargs) -> every GenerationConfig leaf is
        # flat-reachable.
        "GenerationConfig": FLAT,
        # The nested generation sub-configs are forwarded WHOLESALE as the
        # sampling_params blob fields compile_config / watermarking_config (part
        # of the sampling model_dump in _build_generate_kwargs ~711-712); the
        # GenerationConfig constructor rebuilds them from the dict, so interior
        # leaves are blob-reachable under their container field.
        "CompileConfig": ("blob", "compile_config"),
        "WatermarkingConfig": ("blob", "watermarking_config"),
        "SynthIDTextWatermarkingConfig": ("blob", "watermarking_config"),
    },
    # vllm plugin (engines/vllm/plugin.py _build_llm_kwargs ~442-483):
    # model_dump(exclude_none=True) then kwargs.update(dumped) forwards the whole
    # EngineArgs surface flat to LLM(); EngineArgs flat-hoists its scalar
    # sub-configs (ModelConfig.dtype etc.) onto the top level, so those classes
    # are FLAT. speculative_config / compilation_config are Any-typed blob fields
    # forwarded WHOLESALE (no allowlist) -> nested-blob reachable.
    "vllm": {
        "SamplingParams": FLAT,
        "GuidedDecodingParams": FLAT,
        "StructuredOutputsParams": FLAT,
        "EngineArgs": FLAT,
        "ParallelConfig": FLAT,
        "CacheConfig": FLAT,
        "ModelConfig": FLAT,
        "MultiModalConfig": FLAT,
        "SchedulerConfig": FLAT,
        "LoRAConfig": FLAT,
        "DecodingConfig": FLAT,
        "SpeculativeConfig": ("blob", "speculative_config"),
        "CompilationConfig": ("blob", "compilation_config"),
    },
    # tensorrt plugin (engines/tensorrt/plugin.py _build_llm_kwargs ~540-592):
    # scalar/extra fields forward flat via kwargs.update(dumped); three blob
    # containers are popped and rebuilt into native classes, reading only fixed
    # interior keys. LookaheadDecodingConfig / TorchCompileConfig are NOT handled
    # at all -> UNREACHABLE.
    "tensorrt": {
        "BaseLlmArgs": FLAT,
        "TrtLlmArgs": FLAT,
        "TorchLlmArgs": FLAT,
        "CalibConfig": FLAT,
        "CudaGraphConfig": FLAT,
        "BatchingType": FLAT,
        "CapacitySchedulerPolicy": FLAT,
        "ContextChunkingPolicy": FLAT,
        "SamplingParams": FLAT,
        # quant_config blob (~548-560): only quant_algo + kv_cache_quant_algo read.
        "QuantConfig": ("blob", "quant_config", frozenset({"quant_algo", "kv_cache_quant_algo"})),
        # scheduler_config blob (~580-590): only capacity_scheduling_policy read.
        "SchedulerConfig": (
            "blob",
            "scheduler_config",
            frozenset({"capacity_scheduling_policy"}),
        ),
        # kv_cache_config blob (~569-575): ALL non-None keys forwarded wholesale.
        "KvCacheConfig": ("blob", "kv_cache_config"),
        # Native sub-configs the plugin NEVER routes -> no sound path.
        "LookaheadDecodingConfig": UNREACHABLE,
        "TorchCompileConfig": UNREACHABLE,
    },
}


def _short_name(native_type: str) -> str:
    """Rightmost dotted component of a native-type string."""
    return native_type.rsplit(".", 1)[-1]


def _reachability_rule(engine: str, native_type: str, field: str) -> _Reach:
    """Return the reachability rule for ``native_type`` on ``engine``.

    Fails loud (``UnknownNativeClassError``) when the native class has no
    ``_REACHABILITY`` entry for its engine - new native surfaces are conscious
    additions (mirrors :func:`classify_section`'s native-origin discipline).
    """
    table = _REACHABILITY.get(engine, {})
    rule = table.get(_short_name(native_type))
    if rule is None:
        raise UnknownNativeClassError(
            f"No reachability rule for {engine} class {native_type!r} (field {field!r}). "
            "Add it to _REACHABILITY in _section_classifier.py - new native surfaces "
            "are conscious additions, not silent defaults."
        )
    return rule


def load_curated_sections(engine: str) -> dict[str, str]:
    """Map each curated exposed field to its section for ``engine``.

    Reads ``exposed_fields`` from the pin's ``outputs/curated.yaml`` (the same
    outputs/ directory the producers already read). Returns
    ``{field_name: section}``; a field listed under both sections (a curation
    error) resolves to the last section seen, which the per-engine emission
    tests would catch as a path mismatch.
    """
    curated_path = current_outputs_dir(engine) / "curated.yaml"
    data = yaml.safe_load(curated_path.read_text())
    exposed = data.get("exposed_fields", {}) if isinstance(data, dict) else {}
    sections: dict[str, str] = {}
    for section in (ENGINE_PARAMS, SAMPLING_PARAMS):
        for field in exposed.get(section, []) or []:
            sections[field] = section
    return sections


def classify_section(
    engine: str,
    native_type: str,
    field: str,
    curated_sections: dict[str, str],
) -> str:
    """Return the section (``engine_params`` / ``sampling_params``) for a field.

    ``curated_sections`` is the ``{field: section}`` map from
    :func:`load_curated_sections`. Curation wins (rule 1); otherwise the field's
    native class decides (rule 2). An unknown native class fails loud (rule 4).
    """
    curated = curated_sections.get(field)
    if curated is not None:
        return curated
    table = _NATIVE_ORIGIN.get(engine, {})
    section = table.get(_short_name(native_type))
    if section is None:
        raise UnknownNativeClassError(
            f"No native-origin section for {engine} class {native_type!r} "
            f"(field {field!r}). Add it to _NATIVE_ORIGIN in _section_classifier.py "
            "- new native surfaces are conscious additions, not silent defaults."
        )
    return section


def field_path(
    engine: str,
    native_type: str,
    field: str,
    curated_sections: dict[str, str],
) -> str:
    """Return the full ``{engine}.{section}.{field}`` rule path for a field."""
    section = classify_section(engine, native_type, field, curated_sections)
    return f"{engine}.{section}.{field}"


def _emit_path(engine: str, native_type: str, leaf: str, curated_sections: dict[str, str]) -> str:
    """Return the reachability-aware match path for ``leaf``, or raise.

    Consults :data:`_REACHABILITY` by the native class short name + leaf:

    - FLAT -> ``{engine}.{section}.{leaf}`` (byte-identical to the historical
      namespace-drop for flat-hoisted classes).
    - ``("flat", allowlist)`` -> FLAT when ``leaf`` is in the allowlist, else
      :class:`UnreachableMatchPathError`.
    - ``("blob", container[, allowlist])`` -> ``{engine}.{section}.{container}.{leaf}``
      when reachable; a leaf outside an allowlist is unreachable.
    - UNREACHABLE -> :class:`UnreachableMatchPathError`.

    The SECTION is the curated + native-origin classification of the LEAF (for
    flat) or the CONTAINER field (for a blob, since that is the field the user
    sets on the engine section). An unknown native class fails loud upstream.
    """
    rule = _reachability_rule(engine, native_type, leaf)

    if rule is FLAT:
        return field_path(engine, native_type, leaf, curated_sections)

    if isinstance(rule, tuple) and rule[0] == "flat":
        allowlist: frozenset[str] = rule[1]
        if leaf in allowlist:
            return field_path(engine, native_type, leaf, curated_sections)
        raise UnreachableMatchPathError(
            f"{engine} {native_type} leaf {leaf!r} is outside the plugin's flat "
            f"read allowlist {sorted(allowlist)} - no sound runtime path."
        )

    if isinstance(rule, tuple) and rule[0] == "blob":
        container = rule[1]
        if len(rule) == 3:
            blob_allow: frozenset[str] = rule[2]
            if leaf not in blob_allow:
                raise UnreachableMatchPathError(
                    f"{engine} {native_type} blob leaf {leaf!r} is outside the "
                    f"{container!r} read allowlist {sorted(blob_allow)} - no sound path."
                )
        section = classify_section(engine, native_type, container, curated_sections)
        return f"{engine}.{section}.{container}.{leaf}"

    # UNREACHABLE sentinel (or any unrecognised shape, defensively).
    raise UnreachableMatchPathError(
        f"{engine} {native_type} is not routed by the plugin - leaf {leaf!r} has no "
        "sound runtime path (UNREACHABLE)."
    )


def relabel_match_fields(
    match_fields: dict[str, object],
    *,
    engine: str,
    native_type: str,
    curated_sections: dict[str, str],
    allow_unreachable: str | None = None,
    verify_resolves: bool = False,
) -> dict[str, object]:
    """Re-key a mined ``match.fields`` dict onto reachability-aware match paths.

    Each existing key's bare field name is its rightmost dotted component (the
    miners address fields as ``{namespace}.{field}``, possibly nested). The
    emitted path is decided by :func:`_emit_path` (plugin-routing reachability +
    curated/native-origin section): flat-hoisted leaves stay
    ``{engine}.{section}.{leaf}`` (byte-identical to before), routed blob interiors
    become ``{engine}.{section}.{container}.{leaf}``, and unreachable leaves are
    handled by ``allow_unreachable``. Values (including ``@<name>`` cross-field rhs
    refs nested inside spec dicts) are passed through untouched - the loader
    resolves a bare ``@ref`` as a same-section sibling.

    ``allow_unreachable`` defaults to ``None`` = FAIL LOUD: an unreachable leaf
    raises :class:`UnreachableMatchPathError`. Pass ``"drop"`` to instead SKIP an
    unreachable leaf; callers needing the dropped ``(stale_key, reason)`` pairs
    should use :func:`relabel_match_fields_with_drops`.
    """
    relabelled, _dropped = relabel_match_fields_with_drops(
        match_fields,
        engine=engine,
        native_type=native_type,
        curated_sections=curated_sections,
        allow_unreachable=allow_unreachable,
        verify_resolves=verify_resolves,
    )
    return relabelled


def relabel_match_fields_with_drops(
    match_fields: dict[str, object],
    *,
    engine: str,
    native_type: str,
    curated_sections: dict[str, str],
    allow_unreachable: str | None = None,
    verify_resolves: bool = False,
) -> tuple[dict[str, object], list[tuple[str, str]]]:
    """:func:`relabel_match_fields` returning ``(relabelled, dropped)``.

    ``dropped`` is a list of ``(stale_key, reason)`` for leaves skipped under
    ``allow_unreachable="drop"`` (empty otherwise). The default
    (``allow_unreachable=None``) raises :class:`UnreachableMatchPathError`.

    ``verify_resolves=True`` additionally runs the host-only structural
    :func:`assert_path_resolves` net on every emitted path - the durable backstop
    that raises :class:`UnresolvableMatchPathError` for a path that does not
    resolve against the committed generated Config + discovered schema, even if
    the reachability table drifts. (A ``drop``-ped unreachable leaf is never
    emitted, so it is not RESOLVES-checked.)
    """
    relabelled: dict[str, object] = {}
    dropped: list[tuple[str, str]] = []
    for key, value in match_fields.items():
        leaf = key.rsplit(".", 1)[-1]
        try:
            emitted = _emit_path(engine, native_type, leaf, curated_sections)
        except UnreachableMatchPathError as exc:
            if allow_unreachable == "drop":
                dropped.append((key, str(exc)))
                continue
            raise
        if verify_resolves:
            assert_path_resolves(engine, emitted)
        relabelled[emitted] = value
    return relabelled, dropped


def relabel_or_skip_dead_path(
    match_fields: dict[str, object],
    *,
    engine: str,
    native_type: str,
    curated_sections: dict[str, str],
) -> dict[str, object] | None:
    """Relabel a static-mined candidate's match fields, or return ``None`` to SKIP it.

    A fresh re-mine re-discovers source constraints whose config-time match path
    is DEAD - either unreachable per the routing table (a field the engine plugin
    does not route, e.g. transformers ``BitsAndBytesConfig.llm_int8_*`` or tensorrt
    ``LookaheadDecodingConfig`` knobs) or unresolvable against the generated
    Config (e.g. a nested ``watermarking_config.*`` leaf that is not a declared
    field). Such a path can never fire, so the curated corpus omits it; emitting
    it (or emitting the candidate with the dead leaf silently dropped, which would
    only BROADEN its firing condition) would diverge the corpus from a fresh
    re-mine. So the WHOLE candidate is skipped (``None``) when any leaf is dead.

    Drift backstop: a *previously routable, in-corpus* rule whose leaf turns dead
    (a genuine Config regression) is dropped here too, which surfaces as a corpus
    byte-diff in ``regen_engine_corpus --check`` / ``build_corpus --check`` - the
    durable CI gate that already guards the corpus. Mine-time crash-on-dead-path
    is redundant with that gate and only aborts the re-mine on the first dead
    candidate; the byte-diff names exactly which rule moved. The LLM/Tier-D path
    keeps :func:`assert_path_resolves` fail-loud directly (it has no such gate).
    """
    try:
        relabelled, dropped = relabel_match_fields_with_drops(
            match_fields,
            engine=engine,
            native_type=native_type,
            curated_sections=curated_sections,
            allow_unreachable="drop",
            verify_resolves=True,
        )
    except UnresolvableMatchPathError:
        return None
    return None if dropped else relabelled


# ---------------------------------------------------------------------------
# Host-only structural RESOLVES check (the durable dead-path net)
# ---------------------------------------------------------------------------
#
# The reachability table above is the mine-time intent model; this is the
# independent structural net that proves an EMITTED path actually resolves
# against the committed generated Config model + discovered schema. It is pure
# and host-only: it imports the torch-free generated pydantic config module and
# reads the JSON schema envelope - NO library construction, NO GPU.


class UnresolvableMatchPathError(ValueError):
    """An emitted match path does not structurally resolve against the engine Config.

    The terminal segment is not a declared field on the resolved typed container,
    nor (for an Any-typed blob container) a property under the container's native
    class in the discovered schema ``$defs`` or section. The runtime resolver
    would silently return ``None`` for it - a dead path - so the miner fails loud.
    """


def _load_generated_config_model(engine: str) -> Any:
    """Import the engine's torch-free generated ``Config`` pydantic model (host-only)."""
    import importlib

    module = importlib.import_module(f"llenergymeasure.engines.{engine}.config")
    return module.Config


def _load_discovered_schema(engine: str) -> dict[str, Any]:
    """Load the committed ``schema.discovered.json`` envelope for the active pin."""
    path = current_outputs_dir(engine) / "schema.discovered.json"
    return json.loads(path.read_text())


def _model_fields(model: Any) -> dict[str, Any]:
    """Return ``model_fields`` of a pydantic model class, or ``{}`` for a non-model."""
    fields = getattr(model, "model_fields", None)
    return fields if isinstance(fields, dict) else {}


def _is_pydantic_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and hasattr(annotation, "model_fields")


def _typed_submodels(annotation: Any) -> list[Any]:
    """Return the pydantic model classes inside ``annotation`` (unwrapping X | None)."""
    import typing

    if _is_pydantic_model(annotation):
        return [annotation]
    return [arg for arg in typing.get_args(annotation) if _is_pydantic_model(arg)]


def _leaf_in_discovered_blob(discovered: dict[str, Any], section: str, leaf: str) -> bool:
    """True if ``leaf`` is a declared property of ANY class in the discovered ``$defs``
    or a field of the discovered ``section`` map (the Any-blob fallback target).
    """
    for cls_schema in (discovered.get("$defs") or {}).values():
        if isinstance(cls_schema, dict) and leaf in (cls_schema.get("properties") or {}):
            return True
    section_map = discovered.get(section)
    return isinstance(section_map, dict) and leaf in section_map


def assert_path_resolves(engine: str, emitted_path: str) -> None:
    """Assert ``emitted_path`` is structurally valid against the engine's Config + schema.

    Host-only + deterministic (no library construction, no GPU). Walks the dotted
    path against the generated ``Config`` model:

    - ``{engine}.{section}.{leaf}`` (flat): ``section`` must be a Config field and
      ``leaf`` a declared field on its typed sub-model (the resolver walks
      ``model_fields``); ``extra="allow"`` sub-models also accept a leaf present in
      the discovered section map.
    - ``{engine}.{section}.{container}.{leaf}`` (blob): when ``container`` is a typed
      sub-model the leaf must be its declared field; when ``container`` is an
      Any-typed blob field the leaf must exist under a discovered ``$defs`` class or
      the discovered section (the durable check for the forward-wholesale blobs).

    Raises :class:`UnresolvableMatchPathError` on a non-resolving path - the net
    that kills the dead-path class even if the reachability table drifts.
    """
    parts = emitted_path.split(".")
    if len(parts) < 3 or parts[0] != engine:
        raise UnresolvableMatchPathError(
            f"Malformed emitted path {emitted_path!r} for engine {engine!r}."
        )
    section = parts[1]
    config_model = _load_generated_config_model(engine)
    section_field = _model_fields(config_model).get(section)
    if section_field is None:
        raise UnresolvableMatchPathError(
            f"{emitted_path!r}: section {section!r} is not a field on the generated "
            f"{engine} Config model."
        )

    discovered = _load_discovered_schema(engine)
    # Sub-models of the section (engine_params / sampling_params), e.g. EngineParams.
    containers = _typed_submodels(section_field.annotation)
    remaining = parts[2:]

    def _resolve_against(models: list[Any], segs: list[str]) -> None:
        leaf = segs[-1]
        intermediate = segs[:-1]
        cur_models = models
        # Descend through intermediate (blob-container) segments.
        for seg in intermediate:
            next_models: list[Any] = []
            seg_is_blob = False
            for m in cur_models:
                fld = _model_fields(m).get(seg)
                if fld is None:
                    continue
                subs = _typed_submodels(fld.annotation)
                if subs:
                    next_models.extend(subs)
                else:
                    # Any-typed blob container: the leaf must exist in the
                    # discovered $defs / section (forward-wholesale target).
                    seg_is_blob = True
            if seg_is_blob and not next_models:
                if _leaf_in_discovered_blob(discovered, section, leaf):
                    return
                raise UnresolvableMatchPathError(
                    f"{emitted_path!r}: blob container {seg!r} interior leaf "
                    f"{leaf!r} is not a declared property in the discovered schema."
                )
            if not next_models:
                raise UnresolvableMatchPathError(
                    f"{emitted_path!r}: container segment {seg!r} is not a declared "
                    f"field on the {engine} {section} model."
                )
            cur_models = next_models
        # Terminal leaf must be a declared field on one of the resolved models,
        # or (extra="allow" / Any-blob debt fields) present in the discovered map.
        if any(leaf in _model_fields(m) for m in cur_models):
            return
        if _leaf_in_discovered_blob(discovered, section, leaf):
            return
        raise UnresolvableMatchPathError(
            f"{emitted_path!r}: leaf {leaf!r} is not a declared field on the "
            f"resolved {engine} {section} container."
        )

    _resolve_against(containers, remaining)
