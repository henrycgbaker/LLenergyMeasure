"""Adapt Opus-established ground-truth artefacts into the shape the locked
``trial_scoring.py`` scorer expects.

Why this module exists
----------------------
The ground-truth (GT) corpus under
``research/mining-substrate-trial/findings/ground_truth/<engine>/<version>/``
was authored by hand to a richer, human-legible shape than the canonical
mined-catalogue envelope. The scorer in ``trial_scoring.py`` is LOCKED
(its identity functions are the trial's pre-registered comparison axis), so
rather than touch it we translate GT -> canonical envelope here, then feed
the canonical form to the unmodified scorer.

Two artefacts, two shape gaps
-----------------------------

SCHEMA (``schema_ground_truth.json``):
  * ``engine_params`` / ``sampling_params`` exist but carry ``_``-prefixed
    meta-keys (``_source``, ``_note``, ``_from_pipeline`` ...) interleaved
    with real fields, and each field is ``{type|type_annotation, default|
    default_expr, source, ...}`` rather than a bare JSON-schema entry.
  * tensorrt splits engine params across ``engine_params_base`` /
    ``engine_params_trt`` / ``engine_params_torch`` (no flat
    ``engine_params``).
  * Nested config classes live under a ``subconfigs`` dict
    (``subconfigs.<ClassName>.<field>``) for vllm + tensorrt, but
    transformers already ships them under ``$defs.<ClassName>.properties``.
  The scorer's ``schema_field_identities`` reads ONLY ``engine_params``,
  ``sampling_params`` and ``$defs.<Class>.properties.<field>``. So we
  flatten tensorrt's split params, strip meta-keys, normalise the field
  body to carry a ``type``, and lift ``subconfigs`` into ``$defs``.

INVARIANTS (``invariants_ground_truth.yaml``):
  * Each entry is FLAT: ``{id, severity, native_type, predicate_kind,
    native_field, predicate_value, ...}`` and ``match`` is absent.
  * The scorer's ``invariant_identity`` derives its 4-tuple from
    ``inv.match.fields`` (dotted keys like ``vllm.sampling.seed``). With
    ``match`` absent every GT invariant collapses to the
    ``("", id, "unknown", "")`` fallback and intersects nothing.
  We synthesise ``match.fields = {<namespace>.<native_field>:
  <predicate-encoding>}`` where ``namespace`` is derived per-engine to
  match the baseline/cell convention (learned from a survey of
  ``engine_versions/<e>/<v>/outputs/invariants.proposed.yaml``), and the
  predicate encoding is chosen so the scorer's ``_predicate_kind_for``
  classifies it into the right ``PREDICATE_KINDS`` bucket.

Namespace convention (learned from baselines)
---------------------------------------------
Survey of baseline ``match.fields`` keys vs ``native_type``:

  vllm:
    vllm.SamplingParams                  -> vllm.sampling
    vllm.sampling_params.GuidedDecodingParams -> vllm.sampling
    vllm.config.LoRAConfig               -> vllm.engine.lora
    vllm.config.PromptAdapterConfig      -> vllm.engine.prompt_adapter
    vllm.config.TokenizerPoolConfig      -> vllm.engine.tokenizer
    vllm.config.<anything-else>Config    -> vllm.engine
  transformers:
    (native_field already dotted, e.g. transformers.sampling.early_stopping)
    -> namespace = native_field up to last dot, leaf = last segment.
  tensorrt:
    (native_field is ClassName.field, native_type absent)
    -> namespace = tensorrt (flat), leaf = last segment.

Predicate-kind mapping (GT vocab -> scorer PREDICATE_KINDS)
----------------------------------------------------------
The scorer classifies an invariant's predicate from the SHAPE of the
synthesised match-field value (``_predicate_kind_for``):
  dict has key "not_in" -> not_in; "not_equal" -> not_equal; ">" -> gt;
  "<" -> lt; ">=" -> ge; "<=" -> le; "type_is_not" -> type_is_not;
  "present" -> present; non-dict scalar -> exact; else -> unknown.

We therefore EMIT a match-field value whose shape lands the GT
predicate_kind in the intended bucket. The GT vocab is large and
research-descriptive; we collapse it onto the scorer's small closed set:

  type_check / type_is / type_is_not          -> type_is_not  ({"type_is_not": v})
  not_in / not_in_range_* / strenum_in /
    literal_in / allowlist_constant / enum     -> not_in       ({"not_in": v})
  lt / lt_or_eq / lt_either / cross_field_lt   -> lt           ({"<": v})
  gt / cross_field_gt / assert_positive /
    gt_fraction_of_host_memory                 -> gt           ({">": v})
  le                                           -> le           ({"<=": v})
  ge                                           -> ge           ({">=": v})
  eq / identity / decode_dispatch /
    model_property_check                       -> exact        (scalar v)
  is_none / is_not_none / required /
    file_exists / presence_conflict /
    mutual_exclusion / mutually_exclusive /
    cross_field* / cross_config* / cross_*combo /
    *_combo / range / numeric_range / in_*range /
    presence / any_falsy_in_list               -> present      ({"present": v})

Anything unrecognised -> present (a conservative, non-crashing bucket;
"present" still produces a stable, intersecting identity tuple rather than
the "unknown" fallback). The exact bucket choice is a research-protocol
deviation and is logged in ``findings/wave2_deviations.md``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Meta-key handling
# ---------------------------------------------------------------------------

# Field-shape meta-keys that may appear interleaved with real fields inside
# engine_params / sampling_params / subconfig dicts. Always ``_``-prefixed in
# the GT corpus, but we enumerate the known ones for clarity and skip ANY
# ``_``-prefixed key defensively.
_KNOWN_META_KEYS = frozenset(
    {
        "_source",
        "_note",
        "_decorators",
        "_from_pipeline",
        "_from_auto",
        "_commit_hash",
        "_fast_init",
    }
)


def _is_meta_key(key: str) -> bool:
    return key.startswith("_")


def _real_fields(section: dict[str, Any]) -> dict[str, Any]:
    """Return only the real field entries of a GT param/subconfig dict
    (drop ``_``-prefixed meta-keys; keep entries whose value is a dict)."""
    if not isinstance(section, dict):
        return {}
    return {
        name: body
        for name, body in section.items()
        if not _is_meta_key(name) and isinstance(body, dict)
    }


# ---------------------------------------------------------------------------
# Field-body normalisation (carry a best-effort `type` for type_accuracy)
# ---------------------------------------------------------------------------


def _coerce_type(body: dict[str, Any]) -> Any:
    """Best-effort extraction of a JSON-schema-ish ``type`` from a GT field.

    GT fields carry the type under either ``type`` (already JSON-schema
    form, e.g. ``"integer"`` or ``["integer", "null"]``) or
    ``type_annotation`` (a Python-source string, e.g. ``"AttentionBackendEnum
    | None"``). The scorer's ``schema_type_at`` compares ``type`` slots
    exactly, so we pass ``type`` through verbatim and surface
    ``type_annotation`` as the type string when ``type`` is absent. This is
    deliberately NOT a Python->JSON-schema type mapper: type_accuracy is
    only ever scored on the intersection of identities, and the cell/baseline
    outputs use their own type convention, so cross-convention type matches
    are expected to be rare regardless. Carrying SOMETHING keeps the metric
    well-defined rather than always-None.
    """
    if "type" in body:
        return body["type"]
    if "type_annotation" in body:
        return body["type_annotation"]
    return None


def _normalise_field(body: dict[str, Any]) -> dict[str, Any]:
    """Map a GT field body to a canonical JSON-schema-ish entry."""
    t = _coerce_type(body)
    out: dict[str, Any] = {}
    if t is not None:
        out["type"] = t
    return out


# ---------------------------------------------------------------------------
# Schema canonicalisation
# ---------------------------------------------------------------------------

# Top-level GT sections that map to the flat ``engine_params`` namespace
# (tensorrt splits its engine params three ways).
_ENGINE_PARAM_SECTIONS = (
    "engine_params",
    "engine_params_base",
    "engine_params_trt",
    "engine_params_torch",
)

# Top-level GT sections that are conceptually sampling-adjacent and that the
# scorer can only see if lifted into sampling_params or $defs. We fold the
# clear sampling-family sections into sampling_params; the rest become $defs
# pseudo-classes so they are at least scoreable (they will simply not
# intersect a baseline that lacks them, which is the correct low-recall
# signal, not a silent drop).
_SAMPLING_PARAM_SECTIONS = (
    "sampling_params",
    "beam_search_params",
)


def canonicalise_gt_schema(gt_schema: dict[str, Any]) -> dict[str, Any]:
    """Translate a GT ``schema_ground_truth.json`` dict into the canonical
    envelope shape consumed by ``trial_scoring.score_schema``.

    Emits ``engine_params``, ``sampling_params`` (meta-keys stripped, field
    bodies normalised) and ``$defs.<ClassName>.properties.<field>`` built
    from GT ``subconfigs`` (or passed through from a GT that already uses
    ``$defs``). Other top-level param sections are folded in so nothing is
    silently dropped.
    """
    out: dict[str, Any] = {"engine_params": {}, "sampling_params": {}, "$defs": {}}

    # --- engine_params (flatten tensorrt's three-way split) ---
    for section_name in _ENGINE_PARAM_SECTIONS:
        section = gt_schema.get(section_name)
        for name, body in _real_fields(section or {}).items():
            out["engine_params"][name] = _normalise_field(body)

    # --- sampling_params (fold beam_search into sampling) ---
    for section_name in _SAMPLING_PARAM_SECTIONS:
        section = gt_schema.get(section_name)
        for name, body in _real_fields(section or {}).items():
            out["sampling_params"][name] = _normalise_field(body)

    # --- $defs from native GT $defs (transformers) ---
    native_defs = gt_schema.get("$defs")
    if isinstance(native_defs, dict):
        for class_name, class_def in native_defs.items():
            if _is_meta_key(class_name) or not isinstance(class_def, dict):
                continue
            props = class_def.get("properties", {})
            props_out = {name: _normalise_field(body) for name, body in _real_fields(props).items()}
            out["$defs"][class_name] = {"type": "object", "properties": props_out}

    # --- $defs from subconfigs (vllm + tensorrt) ---
    subconfigs = gt_schema.get("subconfigs")
    if isinstance(subconfigs, dict):
        for class_name, class_def in subconfigs.items():
            if _is_meta_key(class_name) or not isinstance(class_def, dict):
                continue
            props_out = {
                name: _normalise_field(body) for name, body in _real_fields(class_def).items()
            }
            out["$defs"][class_name] = {"type": "object", "properties": props_out}

    # --- other top-level config-class sections -> $defs pseudo-classes ---
    # transformers ships quantization_configs / watermarking_configs /
    # cache_classes as dicts of {ClassName: {properties...}} or flat field
    # dicts; we lift each as a $defs class so it is scoreable. Sections we
    # have already consumed are skipped.
    _consumed = (
        set(_ENGINE_PARAM_SECTIONS)
        | set(_SAMPLING_PARAM_SECTIONS)
        | {
            "subconfigs",
            "$defs",
        }
    )
    _scalar_meta_top = {
        "schema_version",
        "engine",
        "engine_version",
        "engine_git_tag",
        "venv_src_root",
        "established_at",
        "established_by",
        "methodology_ref",
        "scope_notes",
        "namespaces",
        "engine_envs",
        "env_vars",
        "ground_truth_built_at",
        "ground_truth_method",
        "ground_truth_reviewer_check",
        "version_delta_ref",
    }
    for top_name, top_val in gt_schema.items():
        if top_name in _consumed or top_name in _scalar_meta_top:
            continue
        if _is_meta_key(top_name) or not isinstance(top_val, dict):
            continue
        _lift_config_section(top_name, top_val, out["$defs"])

    return out


def _lift_config_section(
    section_name: str, section: dict[str, Any], defs_out: dict[str, Any]
) -> None:
    """Lift a top-level config-class container section into ``$defs``.

    Two observed shapes:
      * ``{ClassName: {properties...} | {field: body, ...}}`` (e.g.
        transformers ``quantization_configs``) -> one $defs class per
        ClassName.
      * a flat ``{field: body, ...}`` group (e.g.
        ``guided_decoding_params``) -> one $defs class named after the
        section.
    """
    # Heuristic: if every real value is itself a dict-of-fields whose own
    # values are dicts, treat the section as a container of classes.
    real = {k: v for k, v in section.items() if not _is_meta_key(k) and isinstance(v, dict)}
    if not real:
        return
    looks_like_class_container = all(
        any(isinstance(fv, dict) for fk, fv in v.items() if not _is_meta_key(fk))
        for v in real.values()
    )
    if looks_like_class_container:
        for class_name, class_def in real.items():
            props = class_def.get("properties") if "properties" in class_def else class_def
            props_out = {
                name: _normalise_field(body) for name, body in _real_fields(props or {}).items()
            }
            if props_out:
                defs_out.setdefault(class_name, {"type": "object", "properties": props_out})
    else:
        props_out = {name: _normalise_field(body) for name, body in _real_fields(section).items()}
        if props_out:
            defs_out.setdefault(section_name, {"type": "object", "properties": props_out})


# ---------------------------------------------------------------------------
# Invariant namespace derivation (per-engine, matches baseline convention)
# ---------------------------------------------------------------------------

# vllm: native_type -> namespace. Specific config classes carry a sub-
# namespace; SamplingParams + guided-decoding go to vllm.sampling; every
# other vllm.config.* lands in vllm.engine.
_VLLM_NATIVE_TYPE_NS = {
    "vllm.SamplingParams": "vllm.sampling",
    "vllm.sampling_params.SamplingParams": "vllm.sampling",
    "vllm.sampling_params.GuidedDecodingParams": "vllm.sampling",
    "vllm.config.LoRAConfig": "vllm.engine.lora",
    "vllm.config.PromptAdapterConfig": "vllm.engine.prompt_adapter",
    "vllm.config.TokenizerPoolConfig": "vllm.engine.tokenizer",
}


def _vllm_namespace(native_type: str | None) -> str:
    if not native_type:
        return "vllm.engine"
    if native_type in _VLLM_NATIVE_TYPE_NS:
        return _VLLM_NATIVE_TYPE_NS[native_type]
    if "SamplingParams" in native_type or "GuidedDecodingParams" in native_type:
        return "vllm.sampling"
    # vllm.config.<X>Config and anything else config-ish -> vllm.engine
    return "vllm.engine"


def derive_namespace_and_field(
    engine: str, native_type: str | None, native_field: str
) -> tuple[str, str]:
    """Return ``(namespace, leaf_field)`` for a GT invariant, matching the
    baseline/cell convention for the engine.

    * transformers: ``native_field`` is already dotted
      (``transformers.sampling.early_stopping``); split on the last dot.
    * tensorrt: ``native_field`` is ``ClassName.leaf`` (native_type absent);
      everything collapses to the flat ``tensorrt`` namespace, leaf is the
      last segment.
    * vllm: ``native_type`` carries the class; the namespace comes from the
      per-class table, leaf is the last segment of ``native_field``.
    """
    native_field = str(native_field or "")
    leaf = native_field.rpartition(".")[2] if "." in native_field else native_field

    if engine == "transformers":
        namespace = native_field.rpartition(".")[0] if "." in native_field else "transformers"
        return namespace, leaf

    if engine == "tensorrt":
        return "tensorrt", leaf

    if engine == "vllm":
        return _vllm_namespace(native_type), leaf

    # Unknown engine: best-effort dotted split.
    if "." in native_field:
        return native_field.rpartition(".")[0], leaf
    return engine, leaf


# ---------------------------------------------------------------------------
# Predicate-kind mapping (GT vocab -> scorer PREDICATE_KINDS via value shape)
# ---------------------------------------------------------------------------

# Each entry maps a GT predicate_kind to a callable that, given the GT
# predicate_value, returns the match-field VALUE whose SHAPE the scorer's
# ``_predicate_kind_for`` will classify into the intended bucket.

_NOT_IN_KINDS = frozenset(
    {
        "not_in",
        "not_in_range_inclusive",
        "not_in_range_half_open",
        "strenum_in",
        "literal_in",
        "allowlist_constant",
        "enum",
    }
)
_LT_KINDS = frozenset({"lt", "lt_or_eq", "lt_either", "cross_field_lt"})
_GT_KINDS = frozenset({"gt", "cross_field_gt", "assert_positive", "gt_fraction_of_host_memory"})
_TYPE_KINDS = frozenset({"type_check", "type_is", "type_is_not"})
_EXACT_KINDS = frozenset({"eq", "identity", "decode_dispatch", "model_property_check"})
# Everything cross-field / presence / combo / range collapses to "present":
# these are multi-field or set-membership predicates that the scorer's flat
# single-field shape cannot represent precisely, so we bucket them into the
# stable "present" identity rather than the "unknown" fallback.
_PRESENT_KINDS = frozenset(
    {
        "is_none",
        "is_not_none",
        "required",
        "file_exists",
        "presence_conflict",
        "mutual_exclusion",
        "mutually_exclusive",
        "presence",
        "any_falsy_in_list",
        "range",
        "numeric_range",
        "in_open_range",
        "in_open_range_inclusive",
        "cross_field",
        "cross_field_combo",
        "cross_config_combo",
        "cross_subconfig_combo",
        "cross_runtime_combo",
        "env_var_combo",
        "platform_combo",
    }
)


def encode_predicate(predicate_kind: str | None, predicate_value: Any) -> Any:
    """Return the match-field value encoding a GT predicate so the scorer's
    ``_predicate_kind_for`` lands it in the intended bucket.

    The returned value's SHAPE is what matters (the scorer ignores the
    payload); we carry ``predicate_value`` inside for human legibility.
    """
    pk = (predicate_kind or "").strip()
    if pk in _TYPE_KINDS:
        return {"type_is_not": predicate_value}
    if pk in _NOT_IN_KINDS:
        return {"not_in": predicate_value}
    if pk in _LT_KINDS:
        return {"<": predicate_value}
    if pk in _GT_KINDS:
        return {">": predicate_value}
    if pk == "le":
        return {"<=": predicate_value}
    if pk == "ge":
        return {">=": predicate_value}
    if pk in _EXACT_KINDS:
        # scalar -> scorer classifies as "exact"
        return (
            predicate_value
            if not isinstance(predicate_value, dict)
            else {"present": predicate_value}
        )
    if pk in _PRESENT_KINDS:
        return {"present": predicate_value}
    # Unrecognised: conservative "present" bucket (stable, intersecting).
    return {"present": predicate_value}


def coarse_predicate_bucket(predicate_kind: str | None) -> str:
    """Map a GT predicate_kind to a coarse bucket name for the convention-
    TOLERANT matcher in ``gt_scoring`` (namespace-insensitive).

    The buckets are deliberately coarser than the scorer's PREDICATE_KINDS:
    they group all numeric comparisons, all set-membership checks and all
    cross-field/presence predicates so a cell that picked e.g. ``range``
    where GT said ``lt`` still matches on the tolerant axis.
    """
    pk = (predicate_kind or "").strip()
    if pk in _TYPE_KINDS:
        return "type"
    if pk in _NOT_IN_KINDS:
        return "membership"
    if pk in _LT_KINDS or pk in _GT_KINDS or pk in {"le", "ge"}:
        return "numeric"
    if pk in {"range", "numeric_range", "in_open_range", "in_open_range_inclusive"}:
        return "numeric"
    if pk in _EXACT_KINDS:
        return "exact"
    # cross-field / presence / combo families
    return "presence"


# ---------------------------------------------------------------------------
# Invariant canonicalisation
# ---------------------------------------------------------------------------


def canonicalise_gt_invariants(gt_invariants: dict[str, Any]) -> dict[str, Any]:
    """Translate a GT ``invariants_ground_truth.yaml`` dict into the
    ``{invariants: [...]}`` envelope consumed by
    ``trial_scoring.score_invariants``.

    Each output invariant gains a synthesised ``match`` block so the locked
    ``invariant_identity`` produces a sane 4-tuple. All original GT fields
    are preserved (severity, id, message_template, ...) so severity-accuracy
    scoring still works.
    """
    engine = str(gt_invariants.get("engine", "")).strip()
    raw = gt_invariants.get("invariants")
    out_invs: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return {"invariants": out_invs}

    for inv in raw:
        if not isinstance(inv, dict):
            continue
        native_type = inv.get("native_type")
        native_field = inv.get("native_field", "")
        predicate_kind = inv.get("predicate_kind")
        predicate_value = inv.get("predicate_value")

        namespace, leaf = derive_namespace_and_field(engine, native_type, native_field)
        match_value = encode_predicate(predicate_kind, predicate_value)
        match_key = f"{namespace}.{leaf}"

        canonical = dict(inv)  # preserve all GT fields
        canonical["match"] = {
            "engine": engine,
            "fields": {match_key: match_value},
        }
        # Stash the coarse bucket + canonical namespace for the tolerant
        # matcher (gt_scoring reads these; the locked scorer ignores them).
        canonical["_gt_coarse_bucket"] = coarse_predicate_bucket(predicate_kind)
        canonical["_gt_leaf_field"] = leaf
        canonical["_gt_namespace"] = namespace
        out_invs.append(canonical)

    return {"invariants": out_invs}
