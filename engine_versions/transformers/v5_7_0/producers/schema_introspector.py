"""HuggingFace Transformers schema introspector - vendored for v5.7.0.

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

import inspect
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    TRANSFORMERS_DOCKERFILE,
    annotation_to_type_str,
    docstring_arg_types,
    jsonable,
    make_envelope,
    merge_source_constraints,
    read_dockerfile_from,
)

# Schema-introspector LANDMARKS for Transformers 5.7.0.
#
# The introspector runs inside ``llenergymeasure:transformers-<tag>`` and
# lifts engine parameter specs via ``inspect.signature(from_pretrained)``
# and sampling parameter specs via ``GenerationConfig().to_dict()``. The
# probe (``scripts/_probe.py``) reads this tuple via the dispatcher and
# resolves each dotted path under the installed library before discovery
# runs; a missing landmark flips the probe verdict to ``fail``.
LANDMARKS: tuple[str, ...] = (
    "transformers.AutoModelForCausalLM",
    "transformers.AutoModelForCausalLM.from_pretrained",
    "transformers.PreTrainedModel",
    "transformers.PreTrainedModel.from_pretrained",
    "transformers.GenerationConfig",
    "transformers.GenerationConfig.to_dict",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover HuggingFace Transformers schemas.

    engine_params:   best-effort inspect.signature(from_pretrained) scrape;
                     **kwargs are opaque and recorded as a limitation
    sampling_params: GenerationConfig().to_dict() (~69 fields); type comes from
                     the docstring Args block (annotation surface) then the
                     runtime value's type; None-defaulted + undocumented fields
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

    # Type source priority: documented type (GenerationConfig's docstring Args
    # block - the only machine-readable annotation surface, since the class is
    # __init__(self, **kwargs) with no field annotations) THEN the runtime value's
    # type. transformers 5.x moved most GenerationConfig defaults to None, so
    # value-inference alone regresses previously-typed scalars (num_beams,
    # temperature, ...) to type='unknown'; reading the docstring type first keeps
    # them typed across the None-default bump.
    sampling_params: dict[str, Any] = {}
    unknown_fields: list[str] = []
    doc_types = docstring_arg_types(GenerationConfig)
    gc = GenerationConfig()
    for name, value in gc.to_dict().items():
        documented = doc_types.get(name)
        if documented is not None:
            sampling_params[name] = {"type": documented, "default": jsonable(value)}
        elif value is None:
            sampling_params[name] = {"type": "unknown", "default": None}
            unknown_fields.append(name)
        elif isinstance(value, (list, tuple)):
            sampling_params[name] = {"type": type(value).__name__, "default": jsonable(value)}
        elif isinstance(value, dict):
            sampling_params[name] = {"type": "dict", "default": jsonable(value)}
        else:
            sampling_params[name] = {
                "type": type(value).__name__,
                "default": jsonable(value),
            }

    if unknown_fields:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": unknown_fields,
                "reason": "GenerationConfig field is None-defaulted and undocumented in the "
                "docstring Args block; type='unknown'",
            }
        )

    # C9: fold source-text Field(...) bounds + Literal[...] membership onto the
    # discovered GenerationConfig sampling fields (mirrors the v4_57_3 cut). The
    # newest producer cut silently lost this call; without it a re-mine cannot
    # surface declarative constraints the walker reads from class source. Near-
    # zero on a pin whose GenerationConfig stuffs fields via **kwargs self-
    # assigns, but the plumbing is load-bearing once a pin moves to class-body
    # Field()/Literal declarations.
    gen_source = inspect.getsourcefile(GenerationConfig)
    if gen_source is not None:
        merge_source_constraints(sampling_params, [Path(gen_source)])

    base_image_ref = read_dockerfile_from(repo_root / TRANSFORMERS_DOCKERFILE)
    return make_envelope(
        engine="transformers",
        engine_version=transformers.__version__,
        engine_commit_sha=getattr(transformers, "__commit__", None),
        image_ref=image_ref or base_image_ref,
        base_image_ref=base_image_ref,
        discovery_method="inspect.signature(from_pretrained) + GenerationConfig().to_dict() "
        "with docstring-Args type recovery",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
