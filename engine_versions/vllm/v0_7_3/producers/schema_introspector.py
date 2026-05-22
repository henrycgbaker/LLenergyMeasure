"""Schema-introspector machinery for vLLM 0.7.3.

Vendored machinery: the full ``discover`` body lives here, version-pinned.
The global ``scripts/engine_producers/vllm_introspector.py`` is a
thin dispatcher that resolves ``library.current_version`` from the SSOT
and delegates to this module.

vLLM 0.7.3 exposes the same runtime-introspection surface contract as
later versions: ``vllm.engine.arg_utils.EngineArgs`` is a stdlib
dataclass (~104 fields) and ``vllm.SamplingParams`` is msgspec-typed.
``msgspec.json.schema(SamplingParams)`` returns a ``$ref`` envelope with
the concrete definition under ``$defs.SamplingParams``; the property-
discovery loop below handles both shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    make_envelope,
    msgspec_properties_to_specs,
)

LANDMARKS: tuple[str, ...] = (
    "vllm.SamplingParams",
    "vllm.engine.arg_utils.EngineArgs",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover vLLM 0.7.3 engine and sampling schemas.

    engine_params:   dataclasses.fields(EngineArgs) (Literal annotations
                     surface as ``enum``; per-field descriptions remain
                     absent because EngineArgs uses positional defaults
                     rather than field(metadata={'description': ...}))
    sampling_params: msgspec.json.schema(SamplingParams) preserved with
                     nested ``$defs``; per-leaf ``enum`` / ``minimum`` /
                     ``maximum`` / ``exclusiveMinimum`` etc kept as-is
    """
    import vllm  # type: ignore[import-not-found]
    from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []
    engine_params = dataclass_fields_to_specs(EngineArgs)

    sampling_params: dict[str, Any] = {}
    sampling_defs: dict[str, Any] = {}
    try:
        # Preserve private-prefixed fields (e.g. ``_all_stop_token_ids``,
        # ``_real_n``) for v1-shape parity. msgspec's own JSON Schema lists
        # them as msgspec exposes them.
        sampling_params, sampling_defs = msgspec_properties_to_specs(
            vllm.SamplingParams, skip_private=False
        )
    except Exception as exc:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": [],
                "reason": f"msgspec.json.schema(SamplingParams) failed: {exc!r}",
            }
        )

    # The msgspec schema captures field-level enum + min/max where present;
    # constraints that live solely in _verify_args() (e.g. mutual-exclusion
    # checks across fields) remain undiscoverable from field metadata, so
    # the cross-field caveat stays.
    limitations.append(
        {
            "section": "sampling_params",
            "fields": [],
            "reason": "cross-field constraints (e.g. top_p + top_k interactions) live in "
            "imperative _verify_args() and are not introspectable from field metadata",
        }
    )
    limitations.append(
        {
            "section": "engine_params",
            "fields": [],
            "reason": "per-field descriptions unavailable (vLLM EngineArgs has only a class docstring)",
        }
    )

    return make_envelope(
        engine="vllm",
        engine_version=vllm.__version__,
        engine_commit_sha=getattr(vllm, "__commit__", None),
        image_ref=image_ref,
        base_image_ref=None,
        discovery_method="dataclasses.fields(EngineArgs) + msgspec.json.schema(SamplingParams)",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
        sampling_params_defs=sampling_defs,
    )
