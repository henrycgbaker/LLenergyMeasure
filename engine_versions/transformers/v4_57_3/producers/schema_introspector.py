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
    read_dockerfile_from,
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
                     get type='unknown' and are listed in discovery_limitations
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
            default = None if param.default is inspect.Parameter.empty else param.default
            engine_params[name] = {
                "type": annotation_to_type_str(param.annotation),
                "default": jsonable(default),
            }

    if kwargs_points:
        limitations.append(
            {
                "section": "engine_params",
                "fields": kwargs_points,
                "reason": "from_pretrained accepts **kwargs; kwargs are not in the signature "
                "(documented kwargs live in the class docstring only)",
            }
        )

    sampling_params: dict[str, Any] = {}
    none_default_fields: list[str] = []
    gc = GenerationConfig()
    for name, value in gc.to_dict().items():
        if value is None:
            sampling_params[name] = {"type": "unknown", "default": None}
            none_default_fields.append(name)
        elif isinstance(value, (list, tuple)):
            sampling_params[name] = {"type": type(value).__name__, "default": jsonable(value)}
        elif isinstance(value, dict):
            sampling_params[name] = {"type": "dict", "default": jsonable(value)}
        else:
            sampling_params[name] = {
                "type": type(value).__name__,
                "default": jsonable(value),
            }

    if none_default_fields:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": none_default_fields,
                "reason": "GenerationConfig has no type annotations; None defaults yield type='unknown'",
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
        discovery_method="inspect.signature(from_pretrained) + GenerationConfig().to_dict() "
        "+ discover_validation_collections()",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
