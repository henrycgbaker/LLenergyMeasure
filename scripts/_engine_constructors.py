"""Resolve and build engine-native config classes for probe execution.

The probe kernel (``scripts/probe_candidates.py``) proves candidate rules by
constructing the offending config inside the real engine and observing whether
the engine raises. Candidates address fields by CANONICAL dotted path
(``vllm.sampling_params.frequency_penalty``, ``vllm.engine_params.all2all_backend``,
``transformers.sampling_params.temperature``). This module answers two questions:

1. Given a canonical path, which native constructor class(es) validate that
   field? A path's ``(engine, section)`` prefix maps to an ordered list of
   candidate classes; the kernel picks the first that accepts every leaf.
2. Given a class and probe kwargs, how do we build an instance so the engine's
   own field validation fires (msgspec needs ``convert``; tensorrt needs a
   placeholder model; transformers needs an explicit ``.validate()`` call)?

It runs INSIDE the engine container and imports the live engine library lazily
(only when a probe actually needs it), so importing this module on a plain host
- as the unit tests do - pulls in no engine. Stdlib only.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import typing
from typing import Any

# Sentinel returned by :func:`resolved_value` when a field is absent on the
# constructed instance (distinct from a real ``None`` resolved value).
MISSING = object()


class ConstructorResolutionError(Exception):
    """No native constructor class could be resolved for a canonical path.

    The kernel treats this as ``unprobeable``: an unknown engine, an unknown
    section prefix, or a section whose classes cannot be imported in this
    container. It is never evidence that a rule stopped holding.
    """


# (engine, section) -> ordered (module, class_name) import specs. The kernel
# tries these in order and keeps the first class that accepts all the rule's
# leaf names. vllm.engine_params spans several config classes (a field lives on
# exactly one), so its list is the full config surface; a rule's fields all
# validate together on one class, so "first class accepting every leaf" wins.
_CONSTRUCTOR_TABLE: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("vllm", "sampling_params"): (("vllm", "SamplingParams"),),
    ("vllm", "engine_params"): (
        ("vllm.config.scheduler", "SchedulerConfig"),
        ("vllm.config.parallel", "ParallelConfig"),
        ("vllm.config.cache", "CacheConfig"),
        ("vllm.config.model", "ModelConfig"),
        ("vllm.config.load", "LoadConfig"),
        ("vllm.config.device", "DeviceConfig"),
        ("vllm.config.lora", "LoRAConfig"),
        ("vllm.config", "SchedulerConfig"),
        # Nested-object surface: engine_params.compilation_config.* leaves
        # validate on CompilationConfig. Appended last so every flat
        # engine_params leaf keeps resolving to its original class.
        ("vllm.config.compilation", "CompilationConfig"),
    ),
    ("transformers", "sampling_params"): (("transformers", "GenerationConfig"),),
    ("transformers", "sampling"): (("transformers", "GenerationConfig"),),
    ("transformers", "engine_params"): (("transformers", "GenerationConfig"),),
    ("tensorrt", "sampling_params"): (("tensorrt_llm", "SamplingParams"),),
    ("tensorrt", "engine_params"): (
        ("tensorrt_llm.llmapi.llm_args", "TrtLlmArgs"),
        # Nested-object surface (mirrors the vllm CompilationConfig entry):
        # engine_params.kv_cache_config.* leaves validate on KvCacheConfig;
        # engine_params.speculative_config.* leaves resolve on
        # LookaheadDecodingConfig (DecodingBaseConfig validators are
        # inherited). Appended last so every flat engine_params leaf keeps
        # resolving to TrtLlmArgs.
        ("tensorrt_llm.llmapi.llm_args", "KvCacheConfig"),
        ("tensorrt_llm.llmapi.llm_args", "LookaheadDecodingConfig"),
    ),
}

# transformers quantization fields are validated by BitsAndBytesConfig, not
# GenerationConfig - GenerationConfig silently stashes unknown kwargs, so it
# would falsely "accept" these and never fire their validation.
_BNB_FIELDS: frozenset[str] = frozenset(
    {
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "bnb_4bit_compute_dtype",
        "llm_int8_threshold",
    }
)

# Top-level import module per engine - used to decide whether this container can
# probe the engine at all (a missing engine is an infra fact, not a rule verdict).
_ROOT_MODULE = {"vllm": "vllm", "transformers": "transformers", "tensorrt": "tensorrt_llm"}

# *LlmArgs declare ``model`` required; the probe kwargs only carry the field
# under test, so an injected placeholder lets construction reach the validators
# under test. It is never resolved to a real checkpoint.
_TRTLLM_MODEL_PLACEHOLDER = "/tmp/llem-probe-model-placeholder"

# Minimal-valid stand-in per builtin type for scaffolding required constructor
# args the probe kwargs do not carry (see :func:`_scaffold_required`).
_SCAFFOLD_TYPE_DEFAULTS: dict[type, Any] = {bool: False, int: 1, float: 1.0, str: "x"}


def section_of(canonical_path: str) -> str:
    """Return the ``section`` segment of ``engine.section.leaf`` (or "")."""
    parts = canonical_path.split(".")
    return parts[1] if len(parts) >= 2 else ""


def leaf_of(canonical_path: str) -> str:
    """Return the constructor-kwarg name: the path's last segment."""
    return canonical_path.rsplit(".", 1)[-1]


def _import(module: str, name: str) -> Any:
    return getattr(importlib.import_module(module), name)


def engine_importable(engine: str) -> bool:
    """True iff this container can import ``engine``'s top-level library.

    A ``False`` here means every candidate is ``infra_error`` (wrong container),
    which is categorically distinct from an individual field being unprobeable.
    """
    module = _ROOT_MODULE.get(engine)
    if module is None:
        return False
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def candidate_classes(engine: str, canonical_path: str, leaves: list[str]) -> list[type]:
    """Ordered native config classes that might validate ``canonical_path``.

    ``leaves`` is every leaf the rule constrains; used to route transformers
    quantization fields to ``BitsAndBytesConfig``. Raises
    :class:`ConstructorResolutionError` when the engine/section is unknown or no
    class in the table can be imported here.
    """
    if engine == "transformers":
        target = (
            "BitsAndBytesConfig" if any(x in _BNB_FIELDS for x in leaves) else "GenerationConfig"
        )
        try:
            return [_import("transformers", target)]
        except (ImportError, AttributeError) as exc:  # pragma: no cover - container-only
            raise ConstructorResolutionError(f"cannot import transformers.{target}: {exc}") from exc

    specs = _CONSTRUCTOR_TABLE.get((engine, section_of(canonical_path)))
    if not specs:
        raise ConstructorResolutionError(
            f"no constructor table entry for engine {engine!r} section {section_of(canonical_path)!r}"
        )
    classes: list[type] = []
    seen: set[int] = set()
    for module, name in specs:
        try:
            cls = _import(module, name)
        except (ImportError, AttributeError):
            continue
        if id(cls) not in seen:
            seen.add(id(cls))
            classes.append(cls)
    if not classes:
        raise ConstructorResolutionError(
            f"none of the {engine}/{section_of(canonical_path)} classes could be imported"
        )
    return classes


def accepts(cls: type, leaf: str) -> bool:
    """True iff ``cls(**{leaf: ...})`` is a legal constructor call.

    Uses the constructor signature so dataclass ``InitVar`` params and
    ``**kwargs`` catch-alls (GenerationConfig) are handled uniformly.
    """
    try:
        params = inspect.signature(cls).parameters
    except (ValueError, TypeError):  # pragma: no cover - exotic C constructors
        return True
    if leaf in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def construct(engine: str, cls: type, kwargs: dict[str, Any], *, validate: bool) -> Any:
    """Build a ``cls`` instance from ``kwargs`` so its field validation fires.

    ``validate`` requests the strict/raising path where an engine separates
    construction from validation (transformers ``GenerationConfig.validate``);
    the identity probe passes ``validate=False`` to read resolved state without
    raising. Any exception raised here is the engine's own - the caller
    interprets it as the probe signal.
    """
    if engine == "transformers":
        obj = cls(**kwargs)
        if validate and hasattr(obj, "validate"):
            obj.validate(strict=True)
        return obj
    if engine == "tensorrt":
        kw = dict(kwargs)
        if cls.__name__.endswith("LlmArgs") and "model" not in kw:
            kw["model"] = _TRTLLM_MODEL_PLACEHOLDER
        return _build_config(cls, kw)
    if _is_msgspec_struct(cls):
        import msgspec  # type: ignore[import-not-found]

        return msgspec.convert(kwargs, cls)
    return _build_config(cls, kwargs)


def resolved_value(instance: Any, leaf: str) -> Any:
    """Return the engine-resolved value of ``leaf`` (or :data:`MISSING`)."""
    return getattr(instance, leaf, MISSING)


# ---------------------------------------------------------------------------
# Construction helpers (msgspec dispatch + required-arg scaffolding)
# ---------------------------------------------------------------------------


def _build_config(cls: type, kwargs: dict[str, Any]) -> Any:
    scaffold = _scaffold_required(cls, kwargs)
    return cls(**{**scaffold, **kwargs})


def _is_msgspec_struct(cls: Any) -> bool:
    """True iff ``cls`` is a ``msgspec.Struct`` (validates only on ``convert``).

    Guarded so a non-msgspec engine container never needs msgspec installed.
    """
    try:
        import msgspec  # type: ignore[import-not-found]
    except ImportError:
        return False
    return isinstance(cls, type) and issubclass(cls, msgspec.Struct)


def _scaffold_required(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Neutral stand-ins for mandatory constructor args ``kwargs`` omits.

    Some config classes declare required args the probe does not name (vLLM's
    ``SchedulerConfig`` makes ``max_model_len`` a no-default ``InitVar``), so a
    bare construction raises "field required" before the rule's own validator
    runs. The scaffold prefers the class's real default_factory over a synthetic
    value so a sibling a validator reads lands on its true default, and the
    probe kwargs always win on conflict, so a scaffolded arg never changes which
    branch the rule tests. Any introspection failure yields an empty scaffold -
    always safe, since this only ADDS args the probe omitted.
    """
    try:
        scaffold: dict[str, Any] = {}
        pyd_fields = getattr(cls, "__pydantic_fields__", None)
        if pyd_fields is not None:
            for name, info in pyd_fields.items():
                if name in kwargs or not getattr(info, "is_required", lambda: False)():
                    continue
                if getattr(info, "init", True) is False:
                    continue
                factory = getattr(info, "default_factory", None)
                scaffold[name] = (
                    factory() if callable(factory) else _scaffold_value(info.annotation)
                )
            return scaffold
        if dataclasses.is_dataclass(cls):
            for f in dataclasses.fields(cls):
                if not f.init or f.name in kwargs or f.default is not dataclasses.MISSING:
                    continue
                if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    scaffold[f.name] = f.default_factory()
                    continue
                scaffold[f.name] = _scaffold_value(f.type)
            return scaffold
    except Exception:
        return {}
    return {}


def _scaffold_value(annotation: Any) -> Any:
    """Minimal-valid value for a required field's annotation (fallback ``1``)."""
    inner = annotation
    if isinstance(inner, dataclasses.InitVar):
        inner = inner.type
    if typing.get_origin(inner) is typing.Literal:
        members = typing.get_args(inner)
        return members[0] if members else 1
    if isinstance(inner, type) and inner in _SCAFFOLD_TYPE_DEFAULTS:
        return _SCAFFOLD_TYPE_DEFAULTS[inner]
    return 1
