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

import yaml

from scripts.engine_producers._current import current_outputs_dir

ENGINE_PARAMS = "engine_params"
SAMPLING_PARAMS = "sampling_params"


class UnknownNativeClassError(ValueError):
    """A field's native class has no native-origin table entry for its engine."""


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


def _short_name(native_type: str) -> str:
    """Rightmost dotted component of a native-type string."""
    return native_type.rsplit(".", 1)[-1]


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


def relabel_match_fields(
    match_fields: dict[str, object],
    *,
    engine: str,
    native_type: str,
    curated_sections: dict[str, str],
) -> dict[str, object]:
    """Re-key a mined ``match.fields`` dict onto classified ``{engine}.{section}.{field}`` paths.

    Each existing key's bare field name is its rightmost dotted component (the
    miners address fields as ``{namespace}.{field}``, possibly nested); the
    section is recomputed from curation + native origin and the namespace is
    dropped. Values (including ``@<name>`` cross-field rhs refs nested inside
    spec dicts) are passed through untouched - the loader resolves a bare
    ``@ref`` as a same-section sibling.
    """
    return {
        field_path(engine, native_type, key.rsplit(".", 1)[-1], curated_sections): value
        for key, value in match_fields.items()
    }
