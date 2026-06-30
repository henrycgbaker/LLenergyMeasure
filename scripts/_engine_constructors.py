"""Per-engine native-type class resolution for the validation gate.

The gate constructs probe configs by name: a mined invariant carries a
short-form ``native_type`` (``vllm.CacheConfig``, ``tensorrt_llm.CalibConfig``)
matching the AST symbol the walker saw. Resolving that name to a live class is
not always ``getattr(root_module, class_name)``: engines refactor their config
surface into subpackages between bumps. The clearest case is vllm's
post-0.19.1 move - ``CacheConfig``, ``ParallelConfig``, ``SchedulerConfig`` and
friends migrated from ``vllm.config`` (a module) into ``vllm.config.*``
submodules, so ``getattr(vllm, "CacheConfig")`` raises ``AttributeError`` even
though the class is alive one import away.

The study (audit P10 / D7) measured the cost of not handling this: the vllm
gate logged ``infra_error`` for 100 carried entries that were really just
unresolved class names; adding module-probe resolution dropped that to 4.

Design contract (audit "design it so adding an engine is a table entry"):
the resolution policy for each engine is one entry in
:data:`_ENGINE_RESOLUTION`. A new engine adds a row, not a code path. Each row
declares (a) explicit short-name -> deep-import-path overrides and (b) an
ordered list of candidate submodules to probe when the bare name misses.

This module deliberately holds NO construction kwargs logic (required-field
placeholders, strictness routing) - that stays in the per-engine runners in
:mod:`scripts.validate_rules`. This module only answers "given this
``native_type`` string, what class object does the live library expose?".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


class NativeTypeResolutionError(Exception):
    """The ``native_type`` could not be resolved to a live class.

    Raised when neither the explicit overrides nor any probed submodule yields
    the named class. The gate treats this as ``infra_error`` (the probe will
    not construct) rather than a rule-change finding - constructor drift is not
    evidence that an invariant stopped holding.
    """


@dataclass(frozen=True)
class EngineResolution:
    """Resolution policy for one engine's ``native_type`` strings.

    ``overrides`` maps a short-form ``native_type`` verbatim to a fully
    qualified ``module.Class`` import path; consulted first.

    ``probe_modules`` is an ordered list of submodule import paths to search
    for the bare class name when the root ``getattr`` misses. The first module
    that exposes the name wins. This is the module-probe resolution that
    survives subpackage refactors without pinning a per-version path.
    """

    root_module: str
    overrides: dict[str, str] = field(default_factory=dict)
    probe_modules: tuple[str, ...] = ()


# Per-engine resolution table. Adding an engine = adding a row here.
#
# vllm: the config classes live under ``vllm.config.*`` submodules post-0.19.1
# (cache.py, parallel.py, scheduler.py, ...). The walker emits ``vllm.<Class>``
# or ``vllm.config.<Class>``; probing the ``vllm.config`` package and its known
# submodules resolves both forms without a per-version path pin.
#
# tensorrt: the llmapi args classes live deep under
# ``tensorrt_llm.llmapi.llm_args``. The existing short-name overrides
# (previously ``_TRTLLM_NATIVE_TYPE_MAP`` in validate_rules) move here so
# all native-type resolution lives in one table.
#
# transformers: monolithic - classes resolve off the package root, so no
# overrides or probe modules are needed (audit: "transformers needs none").
_ENGINE_RESOLUTION: dict[str, EngineResolution] = {
    "transformers": EngineResolution(root_module="transformers"),
    "vllm": EngineResolution(
        root_module="vllm",
        probe_modules=(
            "vllm.config",
            "vllm.config.cache",
            "vllm.config.parallel",
            "vllm.config.scheduler",
            "vllm.config.model",
            "vllm.config.lora",
            "vllm.config.load",
            "vllm.config.compilation",
            "vllm.config.device",
            "vllm.sampling_params",
        ),
    ),
    "tensorrt": EngineResolution(
        root_module="tensorrt_llm",
        overrides={
            "tensorrt_llm.BaseLlmArgs": "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
            "tensorrt_llm.TrtLlmArgs": "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
            "tensorrt_llm.LookaheadDecodingConfig": (
                "tensorrt_llm.llmapi.llm_args.LookaheadDecodingConfig"
            ),
            "tensorrt_llm.CalibConfig": "tensorrt_llm.llmapi.llm_args.CalibConfig",
        },
        probe_modules=(
            "tensorrt_llm.llmapi.llm_args",
            "tensorrt_llm.llmapi",
            "tensorrt_llm.plugin.plugin",
            "tensorrt_llm.sampling_params",
        ),
    ),
}


def _import_class(import_path: str) -> Any:
    """Import ``module.Class`` and return the class, or raise Import/AttributeError."""
    module_path, _, class_name = import_path.rpartition(".")
    if not module_path:
        raise NativeTypeResolutionError(f"native_type {import_path!r} has no module component")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve_native_type(engine: str, native_type: str) -> Any:
    """Resolve a mined ``native_type`` string to a live class object.

    Resolution order:

    1. Explicit override (table ``overrides``) - a verbatim short-name to
       deep-path mapping.
    2. The native_type as a dotted import path, when it has a module component
       (e.g. ``vllm.config.SchedulerConfig``, ``transformers.GenerationConfig``).
    3. Module-probe: the bare class name searched across the engine's
       ``probe_modules`` in order, taking the first hit.

    Raises :class:`NativeTypeResolutionError` when no strategy resolves the
    name - the caller treats this as ``infra_error``, not a rule change.

    Unknown engines fall back to plain dotted-path import (no probe table),
    preserving the prior generic behaviour for any engine not yet tabled.
    """
    resolution = _ENGINE_RESOLUTION.get(engine)
    if resolution is None:
        # Unknown engine: best-effort dotted-path import, matching the old
        # ``_construct_generic`` behaviour so nothing regresses.
        try:
            return _import_class(native_type)
        except (ImportError, AttributeError, ValueError) as exc:
            raise NativeTypeResolutionError(
                f"could not resolve native_type {native_type!r} for engine {engine!r}: {exc}"
            ) from exc

    # 1. Explicit override.
    override = resolution.overrides.get(native_type)
    if override is not None:
        try:
            return _import_class(override)
        except (ImportError, AttributeError) as exc:
            raise NativeTypeResolutionError(
                f"override {native_type!r} -> {override!r} failed to import: {exc}"
            ) from exc

    # 2. Dotted-path import when native_type carries a module component beyond
    #    the bare root (e.g. ``vllm.config.SchedulerConfig``).
    module_path, _, class_name = native_type.rpartition(".")
    if module_path and module_path != resolution.root_module:
        try:
            return _import_class(native_type)
        except (ImportError, AttributeError):
            pass  # fall through to module-probe on the bare class name

    # 3. Module-probe: try the root then each candidate submodule for the bare
    #    class name. This is what survives subpackage refactors (audit P10).
    probe_targets = (resolution.root_module, *resolution.probe_modules)
    for module_name in probe_targets:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        candidate = getattr(module, class_name, None)
        if candidate is not None:
            return candidate

    raise NativeTypeResolutionError(
        f"could not resolve class {class_name!r} (native_type {native_type!r}) for "
        f"engine {engine!r}: not found on {resolution.root_module} or any probed "
        f"submodule {resolution.probe_modules}"
    )
