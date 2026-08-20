"""Principled config-resolution core: an UNSET-sentinel precedence chain.

The settings a run resolves against arrive in layers - built-in defaults, the
tool-wide user config, the study YAML, environment overrides, and explicit
call-site arguments - and each layer must be able to say either "set this value"
OR "defer to the layer below" WITHOUT that second choice colliding with an
explicit ``None`` (which means "the value is genuinely null"). This module is the
small, principled core that expresses exactly that:

- :data:`UNSET` is a distinct sentinel meaning "use the layer below". It is NOT
  ``None`` (an explicit null is a real, overriding value).
- :func:`prune_unset` drops the UNSET keys from a layer so they cannot overwrite a
  lower layer during the merge.
- :func:`resolve_layers` deep-merges the pruned layers in ascending precedence
  (lowest first), reusing :func:`llenergymeasure.config._dict_utils.deep_merge`.
- :class:`PrecedenceChain` names the precedence order - call-site > env > study YAML >
  user config > pydantic defaults - so a resolution reads as the chain it is.
- :func:`resolve_labelled_layers` merges the layers AND emits per-value provenance,
  so which layer won is recorded by the merge rather than guessed afterwards. The
  labels are the ``SOURCE_*`` vocabulary in :mod:`llenergymeasure.config.ssot`, the
  same one ``RunnerSpec.source`` uses.

Two resolvers are wired on top of the core:

- :func:`resolve_study_settings` resolves the study-wide settings in one merge: the
  results directory, the execution block (cycles, ordering, thermal gaps), and the
  per-engine runner and image pins. Every entry point resolves these here, before
  the study is dispatched, so nothing downstream re-reads the user config or
  re-implements a fall-through (#886).
- :func:`resolve_server_warmup` resolves the server warmup protocol, wired through
  :func:`apply_server_warmup_overlay`. The ``UserConfig`` carries a
  ``server.warmup`` home (``config.user_config.UserServerConfig``), and the
  resolved-vs-declared split lives on ``ExperimentConfig``: the overlay output is
  attached as side-channel state (``attach_resolved_server_warmup``) that
  ``mode_section_identity`` projects into the resolved/observed hashes, while the
  declared hash stays a wholesale dump of the DECLARED fields (user intent, no
  user-config leak). The overlay is applied during study resolution, before dedup,
  so dedup binds on the realised protocol.

Not to be confused with :mod:`llenergymeasure.config.resolution`, which formats
provenance for the per-experiment record - this module is the forward resolution
that decides which value wins in the first place.

Env-layer coverage is per-setting, not universal: the runner and image pins read
their ``LLEM_*`` overrides, and settings with no env var simply have no env layer.
Widening that coverage, and the offline warmup protocol, stay with #886.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from llenergymeasure.config._dict_utils import deep_merge
from llenergymeasure.config.ssot import (
    ALL_ENGINES,
    ENV_IMAGE_PREFIX,
    ENV_RUNNER_PREFIX,
    SOURCE_CALL_SITE,
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_USER_CONFIG,
    SOURCE_YAML,
    engine_str,
)

#: Engine names in a stable order, for the per-engine runner and image layers.
ENGINE_NAMES: Final[tuple[str, ...]] = tuple(sorted(engine_str(e) for e in ALL_ENGINES))

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, ServerWarmupConfig
    from llenergymeasure.config.user_config import UserConfig

__all__ = [
    "DEFAULT_RESULTS_DIR",
    "UNSET",
    "Layer",
    "PrecedenceChain",
    "Resolution",
    "ResolvedStudySettings",
    "apply_server_warmup_overlay",
    "fields_set_layer",
    "is_unset",
    "prune_unset",
    "resolve_labelled_layers",
    "resolve_layers",
    "resolve_server_warmup",
    "resolve_study_settings",
]


class _Unset:
    """Singleton sentinel meaning 'use the layer below' (distinct from ``None``).

    A dedicated type (not ``None``, not a bare ``object()``) so ``is_unset`` is an
    identity check, the repr is legible, and it survives a deep copy as the SAME
    object - a pruned layer must never smuggle a copied sentinel past an identity
    prune.
    """

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> _Unset:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _Unset:
        return self


#: The 'use the layer below' sentinel. Distinct from ``None`` (an explicit null).
UNSET: Final[_Unset] = _Unset()


def is_unset(value: Any) -> bool:
    """True iff ``value`` is the :data:`UNSET` sentinel (identity check)."""
    return value is UNSET


def prune_unset(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with every :data:`UNSET` value dropped, recursively.

    Nested mappings are pruned too, so a layer can defer individual leaf keys while
    still setting its siblings. A key whose value is UNSET simply does not appear in
    the result, so it cannot overwrite a lower layer during :func:`resolve_layers`.
    An explicit ``None`` is kept - it is a real, overriding null.
    """
    pruned: dict[str, Any] = {}
    for key, value in data.items():
        if is_unset(value):
            continue
        if isinstance(value, Mapping):
            pruned[key] = prune_unset(value)
        else:
            pruned[key] = value
    return pruned


def resolve_layers(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``layers`` in ASCENDING precedence (first = lowest), UNSET-pruned.

    Each layer is pruned of its UNSET keys, then deep-merged onto the accumulator so
    a higher layer's set keys win while its deferred (UNSET) keys leave the lower
    value in place. Reuses the canonical ``deep_merge`` - the only new behaviour is
    the UNSET prune.
    """
    result: dict[str, Any] = {}
    for layer in layers:
        result = deep_merge(result, prune_unset(layer))
    return result


@dataclass(frozen=True)
class Layer:
    """One precedence layer: the values it supplies, and what to call their source.

    ``source`` is a provenance tag from
    :mod:`llenergymeasure.config.ssot`'s ``SOURCE_*`` vocabulary - the same
    vocabulary ``RunnerSpec.source`` uses, so one label names one layer everywhere.
    """

    source: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class Resolution:
    """A resolved mapping plus the provenance of every value in it.

    ``provenance`` maps each dotted leaf path in ``values`` to the ``source`` of the
    layer that supplied the winning value. It is emitted BY the merge, so it says
    which layer actually won rather than inferring it afterwards by comparing the
    result against defaults.
    """

    values: dict[str, Any]
    provenance: dict[str, str]


def resolve_labelled_layers(*layers: Layer) -> Resolution:
    """Merge ``layers`` in ASCENDING precedence, emitting per-value provenance.

    The values are merged exactly as :func:`resolve_layers` merges them. Alongside,
    each layer records itself as the source of every leaf it supplies, so the
    highest layer to set a leaf is the one named in the result's provenance. A leaf
    that replaces a lower layer's whole subtree (or vice versa) takes that subtree's
    provenance with it, so the provenance keys always match the merged values.
    """
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for layer in layers:
        pruned = prune_unset(layer.values)
        # Record against the accumulator as it stands BEFORE this layer merges, so
        # the walk sees the same dict-versus-value shapes deep_merge will see.
        _record_provenance(pruned, values, layer.source, provenance)
        values = deep_merge(values, pruned)
    return Resolution(values=values, provenance=provenance)


def _record_provenance(
    layer: Mapping[str, Any],
    accumulated: Mapping[str, Any],
    source: str,
    provenance: dict[str, str],
    prefix: str = "",
) -> None:
    """Mark every value ``layer`` contributes with ``source``, mirroring deep_merge.

    ``deep_merge`` recurses only where BOTH sides are dicts; everywhere else the
    layer's value replaces what was there. This walk follows exactly that rule, so a
    key the layer merges into (rather than replaces) keeps the provenance of the
    siblings it did not touch.
    """
    for key, value in layer.items():
        path = f"{prefix}{key}"
        existing = accumulated.get(key) if isinstance(accumulated, Mapping) else None
        if isinstance(value, dict) and isinstance(existing, Mapping):
            _record_provenance(value, existing, source, provenance, f"{path}.")
            continue
        # A replacement: drop whatever was recorded for the subtree it overwrites,
        # and for any ancestor value it is now nested under.
        for stale in [p for p in provenance if p == path or p.startswith(f"{path}.")]:
            del provenance[stale]
        parts = path.split(".")
        for depth in range(1, len(parts)):
            provenance.pop(".".join(parts[:depth]), None)
        if isinstance(value, dict) and value:
            for leaf in _leaf_paths(value, f"{path}."):
                provenance[leaf] = source
        else:
            provenance[path] = source


def _leaf_paths(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths of every leaf in ``data`` (a non-dict, or an empty dict)."""
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(_leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


@dataclass(frozen=True)
class PrecedenceChain:
    """The precedence chain, named by layer (call-site highest, defaults lowest).

    Precedence order (high -> low): call-site override > env > study YAML > user config >
    pydantic defaults. Each layer is a mapping that may carry :data:`UNSET` values
    ('defer to the layer below'); :meth:`resolve` prunes them and deep-merges low ->
    high. Secrets are env-only by policy and never travel a study/user layer.

    Use :meth:`resolve_labelled` when the caller needs to record WHICH layer won.
    """

    defaults: Mapping[str, Any]
    user_config: Mapping[str, Any] = field(default_factory=dict)
    study_yaml: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, Any] = field(default_factory=dict)
    call_site: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self) -> dict[str, Any]:
        """Resolve the chain to a single dict (ascending-precedence merge)."""
        return resolve_layers(
            self.defaults,
            self.user_config,
            self.study_yaml,
            self.env,
            self.call_site,
        )

    def resolve_labelled(self) -> Resolution:
        """Resolve the chain, tagging each value with the layer that supplied it."""
        return resolve_labelled_layers(*self._layers())

    def _layers(self) -> tuple[Layer, ...]:
        """The chain's layers in ascending precedence, each with its SOURCE_* tag."""
        return (
            Layer(SOURCE_DEFAULT, self.defaults),
            Layer(SOURCE_USER_CONFIG, self.user_config),
            Layer(SOURCE_YAML, self.study_yaml),
            Layer(SOURCE_ENV, self.env),
            Layer(SOURCE_CALL_SITE, self.call_site),
        )


@dataclass(frozen=True)
class ResolvedStudySettings:
    """The study-wide settings the precedence chain owns, plus their provenance.

    ``provenance`` is keyed by study-file path (``output.results_dir``,
    ``study_execution.n_cycles``, ``runners.vllm``, ``images.vllm``) so a label can
    be read back against the file a researcher actually wrote.

    ``runners`` and ``images`` carry only the engines something explicitly pinned.
    An engine absent from them has no pin at any layer, which is what lets the
    runner auto-detection and the smart image default run: absence is the signal.
    """

    results_dir: str
    execution: dict[str, Any]
    runners: dict[str, str]
    images: dict[str, str]
    provenance: dict[str, str]


#: Built-in default results directory, the bottom of the ``output.results_dir`` chain.
DEFAULT_RESULTS_DIR: Final = "./results"


def resolve_study_settings(
    *,
    study_output: Mapping[str, Any],
    study_execution: Mapping[str, Any],
    study_runners: Mapping[str, str] | None,
    study_images: Mapping[str, str] | None,
    execution_defaults: Mapping[str, Any] | None = None,
    user_config: UserConfig | None = None,
    call_site: Mapping[str, Any] | None = None,
    env: Mapping[str, Any] | None = None,
) -> ResolvedStudySettings:
    """Resolve every study-wide setting the chain owns, in ONE merge.

    Precedence (high to low): call-site override > env > study file > user config >
    caller-supplied effective defaults > built-in defaults. The caller's effective
    defaults sit BELOW the study file on purpose - they fill what the file omitted
    (this is how the CLI applies 3 cycles and shuffle) rather than overriding what
    it declared - while a call-site override sits above everything (a ``-o``
    results directory, an explicit API argument).

    The settings resolved here are the results directory, the whole execution block
    (cycles, ordering, thermal gaps), the per-engine runner pins and the per-engine
    image pins. Resolving them together in one merge is the point: there is one
    precedence order for study settings, not one per field, and the provenance the
    merge emits says which layer actually won.

    Args:
        study_output: The study file's ``output`` block, as explicitly-set fields.
        study_execution: The study file's ``study_execution`` block, as
            explicitly-set fields.
        study_runners: The study file's ``runners`` section, or None.
        study_images: The study file's ``images`` section, or None.
        execution_defaults: Effective defaults for the execution block, applied
            beneath the study file.
        user_config: Tool-wide user config. ``None`` supplies no user layer, so a
            unit test resolving without one gets the built-in defaults untouched.
        call_site: Study-file-shaped overrides from the caller (highest layer).
        env: Environment mapping to read the ``LLEM_*`` overrides from. Defaults to
            ``os.environ``; tests pass their own.

    Returns:
        :class:`ResolvedStudySettings`.
    """
    resolution = resolve_labelled_layers(
        Layer(SOURCE_DEFAULT, _defaults_layer()),
        # The caller's effective defaults are the caller's values, so they carry the
        # call-site label; they simply sit below the study file rather than above it.
        Layer(SOURCE_CALL_SITE, {"study_execution": dict(execution_defaults or {})}),
        Layer(SOURCE_USER_CONFIG, _user_config_layer(user_config)),
        Layer(
            SOURCE_YAML,
            _study_file_layer(study_output, study_execution, study_runners, study_images),
        ),
        Layer(SOURCE_ENV, _env_layer(env)),
        Layer(SOURCE_CALL_SITE, dict(call_site or {})),
    )

    values = resolution.values
    # Only the four sections the chain owns are read back out; anything else a
    # call-site layer carried (an experiment-level key, say) is not this chain's
    # business and is left to the loader that merged it into the study file.
    return ResolvedStudySettings(
        results_dir=values["output"]["results_dir"],
        execution=values["study_execution"],
        runners={k: v for k, v in values.get("runners", {}).items() if isinstance(v, str)},
        images={k: v for k, v in values.get("images", {}).items() if isinstance(v, str)},
        provenance={
            path: source
            for path, source in resolution.provenance.items()
            if path.split(".")[0] in {"output", "study_execution", "runners", "images"}
        },
    )


def _defaults_layer() -> dict[str, Any]:
    """The built-in bottom layer: pydantic defaults plus the results-dir default."""
    from llenergymeasure.config.models import ExecutionConfig

    return {
        "output": {"results_dir": DEFAULT_RESULTS_DIR},
        "study_execution": ExecutionConfig().model_dump(mode="python"),
        "runners": {},
        "images": {},
    }


def _user_config_layer(user_config: UserConfig | None) -> dict[str, Any]:
    """The user config's contribution: results dir, thermal gaps, runner and image pins."""
    if user_config is None:
        return {}
    return {
        "output": {"results_dir": _deferring(user_config.output.results_dir)},
        "study_execution": {
            # The machine-local thermal defaults. The execution block documents an
            # unset gap as deferring to these, which is why they live in this layer
            # and not in the built-in defaults.
            "experiment_gap_seconds": user_config.execution.experiment_gap_seconds,
            "cycle_gap_seconds": user_config.execution.cycle_gap_seconds,
        },
        # "auto" is the user config's way of saying "no preference" for a runner, so
        # it defers to auto-detection rather than pinning the engine.
        "runners": {
            engine: value
            for engine in ENGINE_NAMES
            if (value := getattr(user_config.runners, engine, "auto")) != "auto"
        },
        "images": dict(user_config.images),
    }


def _study_file_layer(
    study_output: Mapping[str, Any],
    study_execution: Mapping[str, Any],
    study_runners: Mapping[str, str] | None,
    study_images: Mapping[str, str] | None,
) -> dict[str, Any]:
    """The study file's contribution, with its documented absences mapped onto UNSET."""
    execution = dict(study_execution)
    for gap in ("experiment_gap_seconds", "cycle_gap_seconds"):
        if gap in execution and execution[gap] is None:
            # The execution block documents an unset gap as "use the machine
            # default", so a null here defers rather than pinning zero seconds.
            execution[gap] = UNSET
    return {
        "output": {"results_dir": _deferring(study_output.get("results_dir"))},
        "study_execution": execution,
        "runners": {k: v for k, v in (study_runners or {}).items() if v is not None},
        "images": {k: v for k, v in (study_images or {}).items() if v is not None},
    }


def _env_layer(env: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-engine runner and image overrides from ``LLEM_RUNNER_*`` / ``LLEM_IMAGE_*``."""
    environ = os.environ if env is None else env
    runners = {}
    images = {}
    for engine in ENGINE_NAMES:
        if value := environ.get(f"{ENV_RUNNER_PREFIX}{engine.upper()}"):
            runners[engine] = value
        if value := environ.get(f"{ENV_IMAGE_PREFIX}{engine.upper()}"):
            images[engine] = value
    return {"runners": runners, "images": images}


def _deferring(value: Any) -> Any:
    """Map a documented "absent means defer to the layer below" value onto UNSET.

    ``None`` and the empty string both mean "not set" for the settings that use this
    (the results directory), matching the behaviour of the ``or`` fall-throughs this
    chain replaces. A real value passes through unchanged.
    """
    return UNSET if value is None or value == "" else value


def resolve_server_warmup(
    *,
    study_yaml: Mapping[str, Any] | _Unset = UNSET,
    user_config: Mapping[str, Any] | _Unset = UNSET,
    env: Mapping[str, Any] | _Unset = UNSET,
    call_site: Mapping[str, Any] | _Unset = UNSET,
) -> ServerWarmupConfig:
    """Resolve the effective server warmup protocol through the precedence chain.

    The v0.7 RESOLVER for the warmup-protocol overlay, wired into production
    through :func:`apply_server_warmup_overlay` (study loading and orchestration).
    The built-in ``ServerWarmupConfig`` defaults are the lowest layer; the study
    YAML's ``server.warmup`` block, an optional tool-wide user default, an env
    overlay, and an explicit call-site override stack above it in precedence order
    (the env and call-site layers stay supported-but-unfed at v0.7 - see the module
    docstring). Any layer may be :data:`UNSET` ('not supplied - defer'). The resolved
    dict is validated back into a
    :class:`~llenergymeasure.config.models.ServerWarmupConfig`, so the identity
    discipline holds: the DECLARED hash still names user intent (the study config),
    while this resolved protocol is what the run actually realises.
    """
    from llenergymeasure.config.models import ServerWarmupConfig

    chain = PrecedenceChain(
        defaults=ServerWarmupConfig().model_dump(mode="python"),
        user_config=_as_layer(user_config),
        study_yaml=_as_layer(study_yaml),
        env=_as_layer(env),
        call_site=_as_layer(call_site),
    )
    return ServerWarmupConfig.model_validate(chain.resolve())


def _as_layer(value: Mapping[str, Any] | _Unset) -> dict[str, Any]:
    """A supplied mapping becomes a layer dict; :data:`UNSET` becomes an empty layer."""
    return dict(value) if isinstance(value, Mapping) else {}


def apply_server_warmup_overlay(config: ExperimentConfig, user_config: UserConfig) -> None:
    """Overlay a tool-wide user-config server warmup onto a server-mode config.

    The production wiring of :func:`resolve_server_warmup`. For a server-mode
    config, resolves the effective warmup protocol through the precedence chain - built-in
    defaults < user config < study YAML - and attaches the OUTPUT as side-channel
    state (:meth:`~llenergymeasure.config.models.ExperimentConfig.attach_resolved_server_warmup`).
    Precedence is per-field: a warmup field the study YAML wrote always wins (study
    YAML is the higher chain layer); a field the study left unset takes the
    user-config value when present, else the built-in default.

    The DECLARED config hash is untouched (the overlay never enters the pydantic
    fields), while ``mode_section_identity`` projects the attached protocol into the
    resolved/observed hashes, so dedup binds on the realised protocol - two runs of
    one study under different user-config warmups are distinct measurements.

    No-op outside server mode, and when the user config supplies no warmup layer, so
    a config with no tool-wide warmup override is byte-identical to today.

    Mutates ``config`` in place (attaching private side-channel state); the declared
    fields are left exactly as parsed.
    """
    if config.serving_mode != "server" or config.server is None:
        return
    user_layer = _user_server_warmup_layer(user_config)
    if not user_layer:
        return
    # UNSET-awareness: only the fields the study YAML actually wrote enter the study
    # layer, so a field it left unset defers to the user/default layers below while a
    # study-set field always wins. See fields_set_layer.
    study_layer = fields_set_layer(config.server.warmup)
    resolved = resolve_server_warmup(study_yaml=study_layer, user_config=user_layer)
    config.attach_resolved_server_warmup(resolved)


def _user_server_warmup_layer(user_config: UserConfig) -> dict[str, Any]:
    """The user config's explicitly-set server-warmup fields (the per-field overlay layer).

    Empty when the user config declares no ``server`` section, so no overlay is
    applied; otherwise the fields-set layer of its ``server.warmup`` block.
    """
    server = user_config.server
    if server is None:
        return {}
    return fields_set_layer(server.warmup)


def fields_set_layer(model: BaseModel) -> dict[str, Any]:
    """A model's explicitly-set fields as a precedence layer (the per-field overlay).

    Only the fields actually written (``model_fields_set``) appear - a field set to
    the built-in default's VALUE still counts as set, so it enters the layer and
    wins over the layers below. This is the UNSET-aware projection every caller uses
    to turn a parsed pydantic model into a layer that defers on what it did not say:
    the declared study warmup, the user-config warmup, and the study execution block.
    """
    return {name: getattr(model, name) for name in model.model_fields_set}
