"""Per-experiment provenance formatting for the ``config.json`` sidecar.

Formats the provenance labels EMITTED by the merges that resolved an experiment:
the CLI-override merge records the paths it overlaid
(``LoadedStudyRaw.cli_override_paths``) and the sweep expansion records the paths
it varied (``LoadedStudyRaw.swept_paths``). This module does no resolving and no
inferring - it pairs each effective field with the label its merge emitted, using
the shared source vocabulary (``call_site`` / ``sweep`` / ``yaml``).

The comparison against Pydantic defaults that happens here is presentation only:
fields whose effective value equals the built-in default are trimmed from the log
to keep the sidecar small - EXCEPT fields a merge actually labelled (a call-site
override or a swept axis), which are kept even at the default value: an override
that happens to equal the default is still an override, and erasing its label
would lose real provenance. The default comparison never decides a field's
source.

The result is folded into the per-experiment ``config.json`` sidecar as its
``provenance`` section (there is no standalone ``_resolution.json`` file).

Usage::

    log = format_experiment_provenance(
        config.model_dump(), cli_override_paths={"task.model"}, swept_paths={"engine"}
    )
    # -> {"task.model": {"effective": "gpt2", "source": "call_site", "default": ...}, ...}
"""

from __future__ import annotations

from collections.abc import Set
from typing import Any

from pydantic import BaseModel
from pydantic.fields import PydanticUndefined  # type: ignore[attr-defined]

from llenergymeasure.config.ssot import SOURCE_CALL_SITE, SOURCE_YAML

#: Source label for a field the sweep expansion varied. Not a precedence-chain
#: layer (a swept value is still declared in the study file); it marks the study's
#: independent variables in the per-experiment record.
SOURCE_SWEEP = "sweep"


def format_experiment_provenance(
    config_dict: dict[str, Any],
    *,
    cli_override_paths: Set[str] = frozenset(),
    swept_paths: Set[str] = frozenset(),
) -> dict[str, Any]:
    """Format the provenance section for a single experiment config.

    Labels each field with the source its merge emitted - ``call_site`` when the
    CLI-override merge set it, ``sweep`` when the sweep expansion varied it,
    ``yaml`` otherwise - and trims fields whose effective value equals the
    Pydantic default (presentation only; a trimmed field was simply never
    overridden by anything). A field the merges labelled is never trimmed: a
    call-site override or swept axis sitting at the default value keeps its
    entry, because its provenance is real.

    Args:
        config_dict: Fully resolved experiment config as a dict (from model_dump()).
        cli_override_paths: Dotted paths the CLI-override merge overlaid, as
            recorded by that merge.
        swept_paths: Dotted paths the sweep expansion varied, as emitted by the
            expansion.

    Returns:
        Provenance map keyed by dotted field path: ``{path: {"effective": value,
        "source": ..., "default": ...}}`` (``default`` present only when the
        field has a Pydantic default). Consumed as the ``config.json``
        ``provenance`` section.
    """
    from llenergymeasure.config.models import ExperimentConfig

    flat_effective = _flatten_dict(config_dict)
    flat_defaults = _get_defaults_flat(ExperimentConfig)

    overrides: dict[str, dict[str, Any]] = {}
    for key, value in sorted(flat_effective.items()):
        # Skip None values (unset optional sub-configs)
        if value is None:
            continue

        # Presentation trim: a value equal to the built-in default was never
        # overridden by any layer, so it carries no provenance worth recording -
        # unless a merge DID label it (an override or swept axis that happens to
        # equal the default is still an override), in which case it stays.
        labelled = key in cli_override_paths or key in swept_paths
        if not labelled and key in flat_defaults and value == flat_defaults[key]:
            continue

        if key in cli_override_paths:
            source = SOURCE_CALL_SITE
        elif key in swept_paths:
            source = SOURCE_SWEEP
        else:
            source = SOURCE_YAML

        entry: dict[str, Any] = {"effective": value, "source": source}
        if key in flat_defaults:
            entry["default"] = flat_defaults[key]

        overrides[key] = entry

    return overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted keys, skipping None sub-dicts."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, f"{key}."))
        elif isinstance(v, list):
            # Store lists as-is (e.g. gpu_indices, measurement_warnings)
            items[key] = v
        else:
            items[key] = v
    return items


def _get_defaults_flat(model_cls: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    """Extract flattened Pydantic defaults from a model, recursing into sub-models."""
    defaults: dict[str, Any] = {}
    for name, field_info in model_cls.model_fields.items():
        key = f"{prefix}{name}" if prefix else name

        # Get default value
        if field_info.default is not PydanticUndefined:
            val = field_info.default
        elif field_info.default_factory is not None:
            val = field_info.default_factory()  # type: ignore[call-arg]
        else:
            continue  # Required field (no default) - any value is explicitly set

        # Recurse into BaseModel sub-config defaults
        if isinstance(val, BaseModel):
            defaults.update(_get_defaults_flat(type(val), f"{key}."))
        else:
            defaults[key] = val

    return defaults
