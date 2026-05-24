"""HuggingFace Transformers schema introspector - vendored for v4.57.3.

Runs inside ``llenergymeasure:transformers-<tag>``. Introspects
``AutoModelForCausalLM.from_pretrained``, ``PreTrainedModel.from_pretrained``
and ``GenerationConfig`` and writes a JSON schema file with the common
envelope.

This module is the version-pinned vendored copy of the introspector body
that previously lived in ``scripts/engine_producers/transformers_introspector.py``.
The global module is now a dispatcher stub that delegates to this
per-version archive via the SSOT's ``library.current_version``.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    TRANSFORMERS_DOCKERFILE,
    annotation_to_type_str,
    discover_validation_collections,
    jsonable,
    make_envelope,
    merge_validation_collections,
    parse_sphinx_kwargs,
    read_dockerfile_from,
    runtime_value_to_spec,
    signature_param_to_spec,
)

# Schema-introspector LANDMARKS for Transformers 4.57.3.
#
# The introspector runs inside ``llenergymeasure:transformers-<tag>`` and
# lifts engine parameter specs via ``inspect.signature(from_pretrained)``
# and sampling parameter specs via ``GenerationConfig().to_dict()``. The
# drift tool (``scripts/_drift.py``) reads this tuple via the dispatcher
# and resolves each dotted path under the installed library before
# discovery runs; a missing landmark flips the probe verdict to ``fail``.
LANDMARKS: tuple[str, ...] = (
    "transformers.AutoModelForCausalLM",
    "transformers.AutoModelForCausalLM.from_pretrained",
    "transformers.PreTrainedModel",
    "transformers.PreTrainedModel.from_pretrained",
    "transformers.GenerationConfig",
    "transformers.GenerationConfig.to_dict",
    # Move 1 (2026-05-24): BitsAndBytesConfig companion-config landmark.
    # Provides docstring source for load_in_4bit / load_in_8bit /
    # bnb_4bit_* fields lifted via the Sphinx-kwargs walker.
    "transformers.BitsAndBytesConfig",
)

# Validation-collection source modules. Each entry is a (module_path,
# section) tuple where ``section`` is ``"engine_params"`` or
# ``"sampling_params"``. The walker (PR-0.5) lifts module-scope
# set/frozenset/tuple/list/dict constants that are referenced in
# ``if v [not] in <Name>:`` from validator bodies in the SAME module. Per-
# version filter: a module that doesn't exist in this version raises
# ``ImportError`` / ``AttributeError`` and is skipped silently.
_VALIDATION_COLLECTION_MODULES: tuple[tuple[str, str], ...] = (
    # Cache-implementation gates (cache_implementation lives on GenerationConfig).
    ("transformers.generation.configuration_utils", "sampling_params"),
    # Quantizer / quantization-config registries (engine-side fields).
    ("transformers.quantizers.auto", "engine_params"),
)


def _collect_validation_collections() -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``{section: {field_name: enum_spec}}`` lifted from validator gates.

    See :data:`_VALIDATION_COLLECTION_MODULES` for sources. Modules absent
    in the installed transformers version are skipped silently (per-version
    filter).
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
            # Walker is best-effort; any AST surprise just yields nothing for
            # this module rather than tanking the whole introspection.
            continue
        for field_name, spec in collections.items():
            by_section[section].setdefault(field_name, spec)
    return by_section


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover HuggingFace Transformers schemas.

    engine_params:   best-effort inspect.signature(from_pretrained) scrape;
                     **kwargs are opaque and recorded as a limitation
    sampling_params: GenerationConfig().to_dict() (~69 fields); None defaults
                     yield empty schemas (no ``type`` key) and are listed in
                     discovery_limitations
    """
    import transformers  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        GenerationConfig,
        PreTrainedModel,
    )

    limitations: list[dict[str, Any]] = []
    engine_params: dict[str, Any] = {}
    kwargs_points: list[str] = []

    for cls_name, cls in (
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("PreTrainedModel", PreTrainedModel),
    ):
        try:
            sig = inspect.signature(cls.from_pretrained)
        except (TypeError, ValueError) as exc:
            limitations.append(
                {
                    "section": "engine_params",
                    "fields": [cls_name],
                    "reason": f"inspect.signature({cls_name}.from_pretrained) failed: {exc!r}",
                }
            )
            continue

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "pretrained_model_name_or_path"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                kwargs_points.append(f"{cls_name}.from_pretrained.**{name}")
                continue
            if name in engine_params:
                continue  # prefer AutoModelForCausalLM over PreTrainedModel
            engine_params[name] = signature_param_to_spec(param)

    # Move 1 (2026-05-24): lift kwargs documented in the class docstring
    # via parse_sphinx_kwargs. HuggingFace from_pretrained accepts most
    # of its interesting kwargs (dtype, attn_implementation, device_map,
    # tp_plan, ...) via **kwargs documented in PreTrainedModel's
    # Sphinx-formatted Args block. skip_names = the params we ALREADY
    # discovered via inspect.signature so we don't double-count.
    sig_field_names = set(engine_params.keys())
    docstring_sources = (
        (
            "transformers.PreTrainedModel.from_pretrained.__doc__",
            PreTrainedModel.from_pretrained.__doc__ or "",
        ),
        (
            "transformers.AutoModelForCausalLM.from_pretrained.__doc__",
            AutoModelForCausalLM.from_pretrained.__doc__ or "",
        ),
    )
    for source_ref, doc in docstring_sources:
        for field_name, spec in parse_sphinx_kwargs(
            doc, skip_names=sig_field_names | set(engine_params.keys()), source_ref=source_ref
        ).items():
            # First docstring wins (PreTrainedModel is more authoritative);
            # AutoModel's pass only adds fields that PreTrainedModel didn't.
            engine_params.setdefault(field_name, spec)

    # BitsAndBytesConfig companion-config: lift load_in_4bit / load_in_8bit
    # / bnb_4bit_* via its own Sphinx Args block. These live inside
    # `quantization_config` semantically; flatten into engine_params so
    # the generated Config class exposes them at the engine level (matches
    # llem's existing hand-written TransformersConfig shape).
    try:
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
    except ImportError:
        BitsAndBytesConfig = None  # type: ignore[assignment]
    if BitsAndBytesConfig is not None:
        bnb_doc = BitsAndBytesConfig.__doc__ or ""
        for field_name, spec in parse_sphinx_kwargs(
            bnb_doc,
            skip_names={"kwargs"} | set(engine_params.keys()),
            source_ref="transformers.BitsAndBytesConfig.__doc__",
        ).items():
            engine_params.setdefault(field_name, spec)

    # `kwargs_points` still records that **kwargs accept arbitrary keys
    # beyond what we documented; surface as informational not blocking.
    if kwargs_points:
        limitations.append(
            {
                "section": "engine_params",
                "fields": kwargs_points,
                "reason": (
                    "from_pretrained accepts **kwargs; "
                    "documented entries lifted via parse_sphinx_kwargs; "
                    "this entry tracks the catchall."
                ),
            }
        )

    sampling_params: dict[str, Any] = {}
    none_default_fields: list[str] = []
    gc = GenerationConfig()
    for name, value in gc.to_dict().items():
        sampling_params[name] = runtime_value_to_spec(value)
        if value is None:
            none_default_fields.append(name)

    if none_default_fields:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": none_default_fields,
                "reason": (
                    "GenerationConfig has no type annotations; None defaults "
                    "yield empty schemas (no 'type' key)"
                ),
            }
        )

    # PR-0.5: lift module-scope validation-collection constants (e.g.
    # ``ALL_CACHE_IMPLEMENTATIONS``) into ``enum`` entries on the matching
    # fields. Walker only lifts constants that are actually referenced in a
    # validator-body ``if v [not] in <Name>:`` check, so unused
    # ``_VALID_X``-style naming heuristics don't produce false positives.
    collections = _collect_validation_collections()
    merge_validation_collections(engine_params, collections["engine_params"])
    merge_validation_collections(sampling_params, collections["sampling_params"])

    base_image_ref = read_dockerfile_from(repo_root / TRANSFORMERS_DOCKERFILE)
    return make_envelope(
        engine="transformers",
        engine_version=transformers.__version__,
        engine_commit_sha=getattr(transformers, "__commit__", None),
        image_ref=image_ref or base_image_ref,
        base_image_ref=base_image_ref,
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
