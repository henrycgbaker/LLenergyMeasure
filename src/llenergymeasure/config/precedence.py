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

Not to be confused with :mod:`llenergymeasure.config.resolution`, which is the
post-hoc provenance LOG (which value won, and why) - this module is the forward
resolution that decides which value wins in the first place.

v0.7 SCOPE: this is the resolution CORE, the warmup-protocol
RESOLVER (:func:`resolve_server_warmup`), and its WIRING for the server warmup
overlay (:func:`apply_server_warmup_overlay`). The ``UserConfig`` now carries a
``server.warmup`` home (``config.user_config.UserServerConfig``), and the
resolved-vs-declared split lives on ``ExperimentConfig``: the overlay output is
attached as side-channel state (``attach_resolved_server_warmup``) that
``mode_section_identity`` projects into the resolved/observed hashes, while the
declared hash stays a wholesale dump of the DECLARED fields (user intent, no
user-config leak). The overlay is applied during study resolution, before dedup, so
dedup binds on the realised protocol. Env/call-site chain layers stay
supported-but-unfed at v0.7. The full setup UX (llem init flow, .env
rationalisation, routing every user-config field through the chain) stays with the
setup-and-user-config workstream (#886).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from llenergymeasure.config._dict_utils import deep_merge

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, ServerWarmupConfig
    from llenergymeasure.config.user_config import UserConfig

__all__ = [
    "UNSET",
    "PrecedenceChain",
    "apply_server_warmup_overlay",
    "fields_set_layer",
    "is_unset",
    "prune_unset",
    "resolve_layers",
    "resolve_server_warmup",
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
class PrecedenceChain:
    """The precedence chain, named by layer (call-site highest, defaults lowest).

    Precedence order (high -> low): call-site override > env > study YAML > user config >
    pydantic defaults. Each layer is a mapping that may carry :data:`UNSET` values
    ('defer to the layer below'); :meth:`resolve` prunes them and deep-merges low ->
    high. Secrets are env-only by policy and never travel a study/user layer.
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
