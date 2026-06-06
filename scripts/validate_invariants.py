#!/usr/bin/env python3
"""Run each validation invariant through the real library inside its engine container.

The validation step is the **observe half** of the "observe, don't re-encode"
design in :doc:`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`.
The YAML corpus at ``src/llenergymeasure/engines/{engine}/invariants.proposed.yaml`` declares each
invariant's ``expected_outcome``; this script executes the invariant through the
library and records what *actually* happened. Divergence between declared and
observed fails CI.

Usage (inside the engine's Docker container)::

    python scripts/validate_invariants.py \\
        --engine transformers \\
        --corpus src/llenergymeasure/engines/transformers/invariants.proposed.yaml \\
        --out src/llenergymeasure/engines/transformers/invariants.validated.yaml

Exit codes:

    0 - all invariants confirmed (positive + negative + expected matches observed)
    1 - one or more divergences; envelope still written
    2 - hard error (corpus malformed, engine not importable, etc.)

The envelope structure mirrors the parameter-discovery envelope in
``src/llenergymeasure/engines/{engine}/schema.discovered.json`` (same field shape,
YAML serialisation here so the proposed and validated corpora share a single
human-readable format).
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from dataclasses import asdict
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Ensure sibling module imports resolve when run via ``python scripts/...``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._invariant_validation_common import (  # noqa: E402  (late import after sys.path)
    TENSORRT_PRIVATE_FIELD_ALLOWLIST,
    TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
    CaptureBuffers,
    CaseResult,
    Divergence,
    classify_emission_channel,
    classify_outcome,
    compare_expected_vs_observed,
    diff_input_vs_state,
    message_matches_template,
    run_case,
    strip_warning_once_sentinel,
)

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Base class for validation-step errors."""


class ValidationCorpusError(ValidationError):
    """Corpus YAML is malformed or missing required fields."""


class ValidationEngineNotImportable(ValidationError):
    """The engine library is not importable in this environment."""


class ValidationDivergenceError(ValidationError):
    """One or more invariants diverged from their declared expected_outcome."""

    def __init__(self, divergences: list[Divergence]) -> None:
        super().__init__(
            f"{len(divergences)} invariant(s) diverged from expected_outcome. "
            "See the validated YAML 'divergences' array for details."
        )
        self.divergences = divergences


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text()
    except FileNotFoundError as exc:
        raise ValidationCorpusError(f"Corpus not found at {path}") from exc
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict) or "invariants" not in data:
        raise ValidationCorpusError(
            f"Corpus at {path} must be a mapping with a top-level 'invariants' key."
        )
    return data


# ---------------------------------------------------------------------------
# Per-engine native-type runners
# ---------------------------------------------------------------------------


def _run_transformers(
    native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
) -> CaptureBuffers:
    """Execute one invariant's kwargs through the transformers library.

    Handles both ``GenerationConfig`` (uses ``.validate()``) and
    ``BitsAndBytesConfig`` (construction itself raises). Other
    ``transformers.*`` native types are reached via a fallback import.

    ``strict_validate`` routes the GenerationConfig call: ``True`` raises a
    composed ValueError listing every issue (corresponds to corpus
    ``severity=error``); ``False`` emits dormant/announced issues via
    ``logger.warning_once`` (corresponds to corpus ``severity=dormant``). The
    caller picks the mode based on the invariant's declared severity.
    """
    logger_names = (
        "transformers",
        "transformers.generation",
        "transformers.generation.configuration_utils",
    )
    if native_type == "transformers.GenerationConfig":
        return run_case(
            lambda: _construct_and_validate_generation_config(kwargs, strict=strict_validate),
            logger_names=logger_names,
            private_allowlist=TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
        )
    if native_type == "transformers.BitsAndBytesConfig":
        return run_case(
            lambda: _construct_bitsandbytes_config(kwargs),
            logger_names=logger_names,
            private_allowlist=TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
        )
    # Fallback: treat native_type as a dotted import path.
    return run_case(
        lambda: _construct_generic(native_type, kwargs),
        logger_names=logger_names,
        private_allowlist=TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
    )


def _construct_and_validate_generation_config(kwargs: dict[str, Any], *, strict: bool) -> Any:
    # Corpus kwargs pass through verbatim - the raw YAML shape IS the invariant
    # under test (e.g. compile_config receives a raw dict on purpose).
    from transformers import GenerationConfig  # type: ignore

    gc = GenerationConfig(**kwargs)
    gc.validate(strict=strict)
    return gc


def _construct_bitsandbytes_config(kwargs: dict[str, Any]) -> Any:
    from transformers import BitsAndBytesConfig  # type: ignore

    return BitsAndBytesConfig(**kwargs)


def _construct_generic(native_type: str, kwargs: dict[str, Any]) -> Any:
    module_path, _, class_name = native_type.rpartition(".")
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# TensorRT-LLM runner
# ---------------------------------------------------------------------------
#
# Runs inside the ``llenergymeasure:tensorrt`` Docker image on the self-hosted
# GPU runner - TRT-LLM 0.21.0 cannot be imported on a CPU host (loads CUDA
# bindings on import), so this codepath is only exercised in CI by the
# ``validate-tensorrt`` job in ``.github/workflows/engine-invariants.yml``.
#
# A native_type tag can arrive in three shapes: a deep import path
# (``tensorrt_llm.llmapi.llm_args.X``), the LLEM engine namespace
# (``tensorrt.X``), or a bare class name (``X``). All three reduce to the bare
# class name, which we resolve by probing the canonical TRT-LLM export modules
# below in order (first hit wins). TRT-LLM re-exports most config + args
# classes at the top-level package and under ``llmapi``; a handful live only in
# their own subpackage (``PluginConfig`` under ``tensorrt_llm.plugin``). An
# explicit deep-path override in :data:`_TRTLLM_NATIVE_TYPE_MAP` still wins.
_TRTLLM_RESOLVE_MODULES: tuple[str, ...] = (
    "tensorrt_llm",
    "tensorrt_llm.llmapi",
    "tensorrt_llm.llmapi.llm_args",
    "tensorrt_llm.plugin",
)

# Explicit deep-path overrides, keyed by bare class name, for cases where the
# module-probe order would resolve to the wrong (e.g. abstract) symbol. Empty
# today: in the current window the probe order resolves every observed class to
# a constructible concrete type (``BaseLlmArgs`` is itself constructible in
# 1.x). Kept as the escape hatch for future version drift.
_TRTLLM_NATIVE_TYPE_MAP: dict[str, str] = {}

# The ``*LlmArgs`` family declares ``model`` as a required field. The corpus's
# ``kwargs_positive`` / ``kwargs_negative`` only carry the field under test, so
# without an injected default Pydantic raises a "model: field required" error
# before any invariant-relevant validator runs. The placeholder is never
# resolved to a real checkpoint - construction stops at validator-pass time on
# either the invariant's positive raise (intended) or the negative success
# (intended), both before the loader would try to read from disk.
_TRTLLM_MODEL_PLACEHOLDER = "/tmp/llem-validation-gate-model-placeholder"


def _run_tensorrt(
    native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
) -> CaptureBuffers:
    """Execute one TRT-LLM invariant's kwargs through the live library.

    ``strict_validate`` is accepted for parity with the transformers runner
    but is not consulted - TRT-LLM has no `.validate(strict=...)` analogue;
    the constructor itself runs all `model_validator` passes.
    """
    del strict_validate  # signature parity only
    logger_names = (
        "tensorrt_llm",
        "tensorrt_llm.llmapi",
        "tensorrt_llm.llmapi.llm_args",
    )
    return run_case(
        lambda: _construct_trtllm(native_type, kwargs),
        logger_names=logger_names,
        private_allowlist=TENSORRT_PRIVATE_FIELD_ALLOWLIST,
    )


def _construct_trtllm(native_type: str, kwargs: dict[str, Any]) -> Any:
    """Construct a TRT-LLM type from its native_type tag.

    Reduces the tag to a bare class name (stripping any ``tensorrt`` /
    ``tensorrt_llm`` / deep-path namespace) and resolves it by probing the
    canonical TRT-LLM export modules in :data:`_TRTLLM_RESOLVE_MODULES`. An
    explicit deep-path override in :data:`_TRTLLM_NATIVE_TYPE_MAP` (keyed by
    bare name) wins when present. Injects the required ``model`` placeholder
    for any ``*LlmArgs`` class when the corpus kwargs don't set it.
    """
    class_name = native_type.rpartition(".")[2]
    override = _TRTLLM_NATIVE_TYPE_MAP.get(class_name)
    if override is not None:
        module_path, _, class_name = override.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
    else:
        cls = None
        for mod_path in _TRTLLM_RESOLVE_MODULES:
            try:
                module = importlib.import_module(mod_path)
            except ImportError:
                continue
            cls = getattr(module, class_name, None)
            if cls is not None:
                break
        if cls is None:
            raise AttributeError(
                f"TRT-LLM native_type {native_type!r} (class {class_name!r}) not "
                f"resolvable in any of {_TRTLLM_RESOLVE_MODULES}"
            )
    use_kwargs = dict(kwargs)
    if class_name.endswith("LlmArgs") and "model" not in use_kwargs:
        use_kwargs["model"] = _TRTLLM_MODEL_PLACEHOLDER
    return cls(**use_kwargs)


# ---------------------------------------------------------------------------
# vLLM runner
# ---------------------------------------------------------------------------

# Regex bundles for stripping vLLM bootstrap noise out of captured warnings
# and log streams (see ``_run_vllm`` for the full rationale). Defined above
# the call site so the reading order matches the call order.

_VLLM_IMPORT_NOISE = re.compile(
    # Torch / ROCm import probe noise.
    r"_SixMetaPathImporter|SwigPy|swigvarlink|VendorImporter|"
    r"libamd_smi|amd-smi|distutils|sysconfig|"
    r"builtin type \w+ has no __module__ attribute|"
    # vLLM startup compile-config note.
    r"`compile_config` is set to .* but `mode`",
)

_VLLM_BOOTSTRAP_NOISE = re.compile(
    r"^Resolved architecture:|"
    r"^Using max model len|"
    r"^Using cuda graph capture|"
    r"^Defaulting to use mp for distributed|"
    r"^The 'pplx' all2all backend|"
    r"^Using external launcher|"
    r"^Disabling V1 multiprocessing|"
    r"^max_parallel_loading_workers|"
    r"^Setting LD_LIBRARY_PATH|"
    r"^load_general_plugins|"
    r"^Initializing distributed environment|"
    r"^This model supports multiple tasks"
)


def _run_vllm(native_type: str, kwargs: dict[str, Any], *, strict_validate: bool) -> CaptureBuffers:
    """Execute one invariant's kwargs through the vLLM library.

    ``strict_validate`` is unused - vLLM has no analog to HF's ``strict``
    flag; sampling-param errors raise during construction and dormancy
    surfaces via ``logger.warning_once`` from inside ``__post_init__`` /
    ``_verify_args``. Both paths run on plain construction.

    Dispatch: ``vllm.SamplingParams``, ``vllm.config.<X>``, and the dotted
    ``vllm.<module>.<Class>`` fallback all flow through the generic
    ``_construct_generic`` path. Logger names are vLLM's module loggers
    (vLLM uses ``init_logger(__name__)`` per ``vllm/config/cache.py:20``).

    INFO-level vLLM logs are pre-suppressed before delegating to
    ``run_case``: vLLM logs ``INFO 04-26 [model.py] Resolved architecture``
    on every ``ModelConfig`` construction, which the capture handler would
    otherwise classify as ``dormant_announced`` and trip ``negative_confirms``.
    The invariant's intended channel is ``logger.warning`` / ``warning_once`` /
    ``warnings.warn`` only; INFO is plumbing.

    A handful of import-time warnings (``_SixMetaPathImporter``, SWIG type
    metadata, ROCm probe failures) leak into ``warnings.catch_warnings``
    on the first run-case invocation per process. These are torch/vLLM
    bootstrap noise unrelated to the invariant under test, so we strip them
    from the captured warnings tuple before returning.
    """
    del strict_validate  # currently unused for vLLM
    logger_names = (
        "vllm",
        "vllm.config",
        "vllm.sampling_params",
        "vllm.engine",
    )
    result = run_case(
        lambda: _construct_generic(native_type, kwargs),
        logger_names=logger_names,
        # vLLM doesn't expose a strict private-field allowlist concept; the
        # corpus loader will re-validate per-field anyway.
        private_allowlist=frozenset(),
    )

    # Drop INFO-level vLLM bootstrap messages from the captured log stream:
    # ``Resolved architecture``, ``Using max model len`` etc. all surface
    # via ``vllm.config.model`` / ``vllm.config.scheduler`` and would
    # otherwise classify the negative case as ``dormant_announced``,
    # tripping ``negative_confirms`` for every invariant that constructs one of
    # these classes. The captured records carry no level prefix in the
    # plain-message handler (``%(message)s`` formatter), so we filter on
    # known-bootstrap message substrings instead - vLLM's
    # ``init_logger(__name__)`` uses standard logging without a level
    # marker baked into the message.
    filtered_logs = tuple(m for m in result.logger_messages if not _VLLM_BOOTSTRAP_NOISE.search(m))
    if filtered_logs != result.logger_messages:
        result = dataclass_replace(result, logger_messages=filtered_logs)
    # Strip torch/SWIG/ROCm bootstrap noise that has nothing to do with the
    # invariant under test.
    filtered_warnings = tuple(
        w for w in result.warnings_captured if not _VLLM_IMPORT_NOISE.search(w)
    )
    if filtered_warnings != result.warnings_captured:
        result = dataclass_replace(result, warnings_captured=filtered_warnings)
    return result


_ENGINE_RUNNERS = {
    "transformers": _run_transformers,
    "tensorrt": _run_tensorrt,
    "vllm": _run_vllm,
}


def get_native_type_runner(engine: str):
    """Return the per-engine dispatcher. Raises if engine unsupported."""
    runner = _ENGINE_RUNNERS.get(engine)
    if runner is None:
        raise ValidationError(
            f"No validation runner registered for engine {engine!r}. "
            f"Known engines: {sorted(_ENGINE_RUNNERS)}"
        )
    return runner


# ---------------------------------------------------------------------------
# Probe synthesis - derive positive/negative kwargs from a declared predicate
# ---------------------------------------------------------------------------
#
# A mined invariant describes its constraint declaratively: as a
# ``predicate_kind`` + ``predicate_value`` pair (the GT / Opus-pass schema) or
# as a single-field ``match.fields`` operator block (the static-miner /
# mechanical schema). When the corpus does NOT also carry hand-authored
# ``kwargs_positive`` / ``kwargs_negative``, the gate SYNTHESISES them: a
# positive probe engineered to VIOLATE the constraint (must fire) and a
# negative probe engineered to SATISFY it (must construct clean).
#
# Synthesis is SAFE: the gate still constructs both probes against the live
# engine and only confirms an entry when the positive fires AND the negative
# passes. A mis-synthesised probe therefore fails to confirm (the entry stays
# unverified) - it can never produce a false-confirmed entry. So we synthesise
# liberally; the runtime check is the arbiter. Predicate families that are not
# deterministically single-field-probeable (cross-field presence/exclusion,
# file existence, backend/decode dispatch, bare type checks) return None and
# are left unverified rather than guessed.

_PROBE_STRING_SENTINEL = "__llem_invalid_probe_value__"


def _leaf_field(invariant: dict[str, Any]) -> str | None:
    """Leaf field name to probe: from ``native_field`` or a single match key."""
    nf = invariant.get("native_field")
    if isinstance(nf, str) and nf:
        return nf.rpartition(".")[2]
    mf = (invariant.get("match") or {}).get("fields") or {}
    if len(mf) == 1:
        return next(iter(mf)).rpartition(".")[2]
    return None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_step(v: Any) -> Any:
    return 1 if isinstance(v, int) and not isinstance(v, bool) else 1.0


def _range_bounds(pv: Any) -> tuple[Any, Any]:
    if isinstance(pv, (list, tuple)) and len(pv) == 2:
        return pv[0], pv[1]
    if isinstance(pv, dict):
        return pv.get("min", pv.get("lo")), pv.get("max", pv.get("hi"))
    return None, None


def synthesize_probe_kwargs(invariant: dict[str, Any]) -> tuple[dict, dict] | None:
    """Return ``(kwargs_positive, kwargs_negative)`` synthesised from the
    invariant's declared predicate, or ``None`` when the family isn't
    deterministically synthesisable. positive should FIRE; negative should PASS.
    """
    leaf = _leaf_field(invariant)
    if not leaf:
        return None
    pk = (invariant.get("predicate_kind") or "").strip()
    pv = invariant.get("predicate_value")
    from_pk = _synth_from_predicate_kind(leaf, pk, pv)
    if from_pk is not None:
        return from_pk
    mf = (invariant.get("match") or {}).get("fields") or {}
    if len(mf) == 1:
        return _synth_from_operator(leaf, next(iter(mf.values())))
    return None


def _synth_from_predicate_kind(leaf: str, pk: str, pv: Any) -> tuple[dict, dict] | None:
    if not pk:
        return None
    if pk in {"literal_in", "strenum_in", "allowlist_constant"}:
        if isinstance(pv, list) and pv:
            return ({leaf: _PROBE_STRING_SENTINEL}, {leaf: pv[0]})
        return None
    if pk in {"range", "numeric_range", "in_open_range", "in_open_range_inclusive"}:
        lo, hi = _range_bounds(pv)
        if _is_num(lo) and _is_num(hi) and lo == hi:
            return None  # degenerate single-point range: no interior negative
        if _is_num(hi):
            neg = (lo + hi) / 2 if _is_num(lo) else hi
            return ({leaf: hi + _num_step(hi)}, {leaf: neg})
        if _is_num(lo):
            return ({leaf: lo - _num_step(lo)}, {leaf: lo})
        return None
    if pk == "assert_positive":
        return ({leaf: -1}, {leaf: 1})
    return None  # type_is / file_exists / *_conflict / *_dispatch -> not synthesisable


def _synth_from_operator(leaf: str, pred: Any) -> tuple[dict, dict] | None:
    if not isinstance(pred, dict) or len(pred) != 1:
        return None
    op, val = next(iter(pred.items()))
    if op == "not_in" and isinstance(val, list) and val:
        return ({leaf: _PROBE_STRING_SENTINEL}, {leaf: val[0]})
    if op == "not_equal":
        return ({leaf: (val + 1) if _is_num(val) else _PROBE_STRING_SENTINEL}, {leaf: val})
    if not _is_num(val):
        return None
    if op in {"<", "lt"}:
        return ({leaf: val - _num_step(val)}, {leaf: val})
    if op in {"<=", "le"}:
        return ({leaf: val}, {leaf: val + _num_step(val)})
    if op in {">", "gt"}:
        return ({leaf: val + _num_step(val)}, {leaf: val})
    if op in {">=", "ge"}:
        return ({leaf: val}, {leaf: val - _num_step(val)})
    return None  # present / type_is_not / scalar -> not synthesisable here


# ---------------------------------------------------------------------------
# Per-invariant driver
# ---------------------------------------------------------------------------


def validate_invariant(engine: str, invariant: dict[str, Any], *, gpu_mode: str) -> CaseResult:
    """Run one invariant's positive + negative kwargs and assemble the case result.

    ``gpu_mode`` is ``"all" | "skip" | "only"`` - hardware-dependent invariants
    are skipped unless ``gpu_mode`` permits them.
    """
    case, _pos, _neg = _validate_invariant_with_captures(engine, invariant, gpu_mode=gpu_mode)
    return case


def _validate_invariant_with_captures(
    engine: str, invariant: dict[str, Any], *, gpu_mode: str
) -> tuple[CaseResult, CaptureBuffers | None, CaptureBuffers | None]:
    """Run one invariant and return the case plus the raw positive/negative captures.

    The captures are needed by the gate-soundness checks added per
    Decision #12 of the invariant-miner adversarial review - they look at
    severity-specific behaviour (positive must raise for ``severity=error``)
    and the raised exception's message text, neither of which fit the
    public ``CaseResult`` shape. ``validate_invariant`` keeps its existing return
    type for backward compatibility with downstream tests; this internal
    helper exposes what the gate needs.

    Returns ``(case, pos, neg)``. ``pos`` / ``neg`` are ``None`` when the
    invariant was skipped.
    """
    invariant_id = invariant["id"]
    requires_gpu = bool(invariant.get("requires_gpu", False))
    hardware_dependent = bool(invariant.get("hardware_dependent", False))

    if gpu_mode == "skip" and (requires_gpu or hardware_dependent):
        return (
            CaseResult(
                id=invariant_id,
                outcome="skipped_hardware_dependent",
                emission_channel="none",
                skipped_reason="requires_gpu_and_gpu_mode_skip",
            ),
            None,
            None,
        )
    if gpu_mode == "only" and not requires_gpu:
        return (
            CaseResult(
                id=invariant_id,
                outcome="skipped_hardware_dependent",
                emission_channel="none",
                skipped_reason="cpu_rule_and_gpu_mode_only",
            ),
            None,
            None,
        )

    native_type = invariant.get("native_type")
    if not native_type:
        return (
            CaseResult(
                id=invariant_id,
                outcome="skipped_no_native_type",
                emission_channel="none",
                skipped_reason="no_native_type_to_construct",
            ),
            None,
            None,
        )
    runner = get_native_type_runner(engine)
    severity = str(invariant.get("severity", "")).lower()
    # Per-engine strictness routing: transformers' GenerationConfig has a
    # non-strict path (logger.warning for dormant/announced) and a strict
    # path (composed ValueError for errors). Dispatch by declared severity
    # so the validation observation matches the corpus's expected outcome shape.
    strict_validate = severity == "error"

    # Use hand-authored probes when present; otherwise synthesise them from the
    # declared predicate so the gate can validate predicate-only mined entries.
    synthesized = False
    if (
        invariant.get("kwargs_positive") is not None
        and invariant.get("kwargs_negative") is not None
    ):
        kwargs_positive = dict(invariant["kwargs_positive"])
        kwargs_negative = dict(invariant["kwargs_negative"])
    else:
        synth = synthesize_probe_kwargs(invariant)
        if synth is None:
            return (
                CaseResult(
                    id=invariant_id,
                    outcome="skipped_unsynthesizable",
                    emission_channel="none",
                    skipped_reason="predicate_not_deterministically_probeable",
                ),
                None,
                None,
            )
        kwargs_positive, kwargs_negative = synth
        synthesized = True

    pos = runner(native_type, kwargs_positive, strict_validate=strict_validate)
    neg = runner(native_type, kwargs_negative, strict_validate=strict_validate)

    # Silent self-assignments are only meaningful on the positive path and
    # only when construction succeeded.
    silent_normalisations: dict[str, dict[str, Any]] = {}
    if pos.observed_state is not None:
        silent_normalisations = diff_input_vs_state(kwargs_positive, pos.observed_state)

    outcome = classify_outcome(pos, silent_normalisations)
    emission = classify_emission_channel(pos)

    expected = dict(invariant.get("expected_outcome") or {})
    positive_confirmed = _positive_confirms(expected, outcome)
    neg_silent = (
        diff_input_vs_state(kwargs_negative, neg.observed_state) if neg.observed_state else {}
    )
    negative_confirmed = _negative_confirms(neg, neg_silent)

    observed_messages = list(pos.warnings_captured) + list(
        strip_warning_once_sentinel(pos.logger_messages)
    )
    observed_exception: dict[str, str] | None = None
    if pos.exception_type is not None:
        observed_exception = {
            "type": pos.exception_type,
            "message": pos.exception_message or "",
        }

    # Hardening for SYNTHESISED probes: only confirm when the positive firing is
    # ATTRIBUTABLE to the field under test - the leaf field name must appear in
    # the raised exception / captured messages. The pos/neg pair already differ
    # only in this field, so an attributable raise is strong evidence the field
    # itself enforces the constraint; this rejects the case where a synthesised
    # value trips an unrelated validator and "confirms" for the wrong reason.
    # Hand-authored probes keep the original (looser) confirmation rule.
    if synthesized and positive_confirmed:
        probe_leaf = _leaf_field(invariant) or ""
        haystack = (pos.exception_message or "") + " " + " ".join(observed_messages)
        if probe_leaf and probe_leaf not in haystack:
            positive_confirmed = False

    case = CaseResult(
        id=invariant_id,
        outcome=outcome,
        emission_channel=emission,
        observed_messages=observed_messages,
        observed_silent_normalisations=silent_normalisations,
        observed_exception=observed_exception,
        positive_confirmed=positive_confirmed,
        negative_confirmed=negative_confirmed,
        duration_ms=pos.duration_ms + neg.duration_ms,
    )
    return case, pos, neg


_FIRING_OUTCOMES = frozenset({"error", "warn", "dormant_announced", "dormant_silent"})

# Gate-soundness check names - exposed as constants so tests + downstream
# tooling can reference them by symbol rather than string-literal.
CHECK_POSITIVE_RAISES = "positive_raises"
CHECK_NEGATIVE_DOES_NOT_RAISE = "negative_does_not_raise"
CHECK_MESSAGE_TEMPLATE_MATCH = "message_template_match"
CHECK_MESSAGE_TEMPLATE_TOO_DYNAMIC = "message_template_too_dynamic"


def compute_gate_soundness_divergences(
    invariant: dict[str, Any], pos: CaptureBuffers, neg: CaptureBuffers
) -> list[Divergence]:
    """Return divergences from the three gate-soundness checks.

    Decision #12 of the invariant-miner adversarial review
    (`.product/designs/adversarial-review-invariant-miner-2026-04-26.md`)
    surfaced a soundness gap: the existing ``compare_expected_vs_observed``
    only checks fields *present* in ``expected_outcome``, so a typo in the
    corpus YAML silently passes. These three checks bind the invariant's
    declared shape to the live library's behaviour:

    1. ``positive_raises`` - for ``severity=error`` invariants, ``kwargs_positive``
       MUST raise. For ``severity=dormant`` invariants, it MUST emit (warn or
       dormant-announce) - i.e., not be a no-op and not raise unexpectedly.
    2. ``message_template_match`` - when the positive raised, the exception's
       ``str()`` must contain the static fragment of ``message_template``
       (case-insensitive). When the template has no useful static fragment
       (almost all placeholders), record ``message_template_too_dynamic``
       so the invariant's author knows to add a more specific template.
    3. ``negative_does_not_raise`` - ``kwargs_negative`` MUST construct
       successfully (i.e., not raise).

    Each divergence carries ``check_failed`` so downstream tooling can
    filter by check type.
    """
    invariant_id = invariant["id"]
    severity = str(invariant.get("severity", "")).lower()
    divergences: list[Divergence] = []

    # 1. Positive must fire (raise for severity=error, emit for severity=dormant).
    pos_silent = (
        diff_input_vs_state(dict(invariant.get("kwargs_positive") or {}), pos.observed_state)
        if pos.observed_state
        else {}
    )
    pos_outcome = classify_outcome(pos, pos_silent)
    if severity == "error":
        if pos.exception_type is None:
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="kwargs_positive",
                    expected="raises",
                    observed=pos_outcome,
                    check_failed=CHECK_POSITIVE_RAISES,
                )
            )
    else:
        # dormant invariants emit but don't raise - anything other than no_op or error.
        if pos_outcome in {"no_op", "error"}:
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="kwargs_positive",
                    expected="emits_warning_or_announce",
                    observed=pos_outcome,
                    check_failed=CHECK_POSITIVE_RAISES,
                )
            )

    # 2. Message-template substring match (only when positive raised, since
    #    the message_template specifically describes the raised string).
    template = str(invariant.get("message_template") or "")
    if pos.exception_type is not None and template:
        matched, fragment = message_matches_template(pos.exception_message or "", template)
        if not fragment:
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="message_template",
                    expected="contains_static_fragment",
                    observed=template,
                    check_failed=CHECK_MESSAGE_TEMPLATE_TOO_DYNAMIC,
                )
            )
        elif not matched:
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="message_template",
                    expected=fragment,
                    observed=pos.exception_message or "",
                    check_failed=CHECK_MESSAGE_TEMPLATE_MATCH,
                )
            )

    # 3. Negative must not raise.
    if neg.exception_type is not None:
        divergences.append(
            Divergence(
                invariant_id=invariant_id,
                field="kwargs_negative",
                expected="does_not_raise",
                observed={"type": neg.exception_type, "message": neg.exception_message or ""},
                check_failed=CHECK_NEGATIVE_DOES_NOT_RAISE,
            )
        )

    return divergences


def _positive_confirms(expected: dict[str, Any], observed_outcome: str) -> bool:
    """True iff the invariant fired on the positive kwargs as declared.

    When the corpus declares a specific outcome, positive_confirmed requires
    an exact match. When the corpus leaves ``outcome`` unset, we accept any
    non-``no_op`` observation as confirmation.
    """
    expected_outcome = expected.get("outcome")
    if expected_outcome in _FIRING_OUTCOMES:
        return observed_outcome == expected_outcome
    return observed_outcome != "no_op"


def _negative_confirms(neg: CaptureBuffers, silent_normalisations: dict[str, Any]) -> bool:
    """True iff the invariant did NOT fire on the negative kwargs.

    Delegates to :func:`classify_outcome` so the definition of "fired"
    lives in one place: anything other than ``no_op`` counts as firing,
    which would be a dead miner entry.
    """
    return classify_outcome(neg, silent_normalisations) == "no_op"


# ---------------------------------------------------------------------------
# Envelope assembly
# ---------------------------------------------------------------------------


def assemble_envelope(
    *,
    engine: str,
    engine_version: str,
    image_ref: str,
    base_image_ref: str,
    validation_commit: str,
    cases: list[CaseResult],
    divergences: list[Divergence],
) -> dict[str, Any]:
    """Build the validated invariants envelope (parallel to the parameter-discovery envelope)."""
    now = os.environ.get("LLENERGY_VALIDATION_FROZEN_AT") or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_version": engine_version,
        "image_ref": image_ref,
        "base_image_ref": base_image_ref,
        "validated_at": now,
        "validation_commit": validation_commit,
        "cases": [_case_to_dict(c) for c in cases],
        "divergences": [d.as_dict() for d in divergences],
    }


def _case_to_dict(case: CaseResult) -> dict[str, Any]:
    d = asdict(case)
    # Drop nullable optional fields when unset for a quieter envelope.
    for optional_key in ("observed_exception", "skipped_reason"):
        if d.get(optional_key) is None:
            d.pop(optional_key, None)
    # duration_ms is wall-clock noise (±1 ms run-to-run); excluding it from
    # the envelope keeps successive validation runs on unchanged source byte-
    # identical, which breaks the commit-back re-trigger loop.
    d.pop("duration_ms", None)
    return d


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------


def validate_engine(
    *,
    engine: str,
    corpus_path: Path,
    out_path: Path,
    gpu_mode: str = "all",
    image_ref: str | None = None,
    base_image_ref: str | None = None,
    validation_commit: str = "unknown",
) -> tuple[dict[str, Any], list[Divergence]]:
    """Run the full validation loop for one engine; write YAML envelope to ``out_path``.

    Returns ``(envelope, divergences)``. Raises :class:`ValidationEngineNotImportable`
    if the engine library can't be imported. Does NOT raise on divergence -
    the caller inspects the returned list and decides.
    """
    corpus = _load_corpus(corpus_path)
    engine_version = _resolve_engine_version(engine)

    cases: list[CaseResult] = []
    divergences: list[Divergence] = []

    for invariant in corpus.get("invariants", []):
        # ValidationError (and subclasses) propagate - they indicate misconfig, not
        # a library behaviour finding. Any other Exception gets recorded as a
        # per-invariant error so one bad invariant doesn't abort the full validation run.
        pos: CaptureBuffers | None = None
        neg: CaptureBuffers | None = None
        try:
            case, pos, neg = _validate_invariant_with_captures(engine, invariant, gpu_mode=gpu_mode)
        except ValidationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            case = CaseResult(
                id=invariant.get("id", "<unknown>"),
                outcome="error",
                emission_channel="none",
                observed_exception={"type": type(exc).__name__, "message": str(exc)},
            )
        cases.append(case)

        if case.skipped_reason is not None:
            continue

        rule_divergences = compare_expected_vs_observed(
            invariant_id=invariant["id"],
            expected=invariant.get("expected_outcome") or {},
            observed_outcome=case.outcome,
            observed_emission=case.emission_channel,
            silent_normalisations=case.observed_silent_normalisations,
        )
        if not case.positive_confirmed:
            rule_divergences.append(
                Divergence(
                    invariant_id=invariant["id"],
                    field="positive_confirmed",
                    expected=True,
                    observed=False,
                )
            )
        if not case.negative_confirmed:
            rule_divergences.append(
                Divergence(
                    invariant_id=invariant["id"],
                    field="negative_confirmed",
                    expected=True,
                    observed=False,
                )
            )
        # Gate-soundness checks (Decision #12). Only run when both captures
        # are available - defensive guard for the bare-except fallback above.
        if pos is not None and neg is not None:
            rule_divergences.extend(compute_gate_soundness_divergences(invariant, pos, neg))
        divergences.extend(rule_divergences)

    envelope = assemble_envelope(
        engine=engine,
        engine_version=engine_version,
        image_ref=image_ref or f"llenergymeasure:{engine}",
        base_image_ref=base_image_ref or f"llenergymeasure:{engine}",
        validation_commit=validation_commit,
        cases=cases,
        divergences=divergences,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(envelope, sort_keys=False, default_flow_style=False))

    return envelope, divergences


def _resolve_engine_version(engine: str) -> str:
    """Best-effort: return the installed library's version or ``"unknown"``."""
    if engine == "transformers":
        try:
            import transformers  # type: ignore

            return transformers.__version__
        except ImportError as exc:
            raise ValidationEngineNotImportable(
                "transformers is not importable in this environment"
            ) from exc
    if engine == "tensorrt":
        try:
            import tensorrt_llm  # type: ignore

            return tensorrt_llm.__version__
        except ImportError as exc:
            raise ValidationEngineNotImportable(
                "tensorrt_llm is not importable in this environment "
                "(expected when running outside the llenergymeasure:tensorrt "
                "Docker image on a GPU host)"
            ) from exc
    if engine == "vllm":
        try:
            import vllm  # type: ignore

            return vllm.__version__
        except ImportError as exc:
            raise ValidationEngineNotImportable(
                "vllm is not importable in this environment"
            ) from exc
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        required=True,
        choices=sorted(_ENGINE_RUNNERS),
        help="Engine whose corpus to validate.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to the YAML corpus file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the validated YAML envelope.",
    )
    parser.add_argument(
        "--gpu-cases",
        choices=("all", "skip", "only"),
        default="all",
        help=(
            "Which invariants to run. 'skip' drops invariants with requires_gpu=true "
            "(for GH-hosted CPU jobs); 'only' runs only those (for self-hosted "
            "GPU jobs); 'all' runs everything (default, useful locally)."
        ),
    )
    parser.add_argument(
        "--image-ref",
        default=None,
        help="Image reference to record in envelope.image_ref.",
    )
    parser.add_argument(
        "--base-image-ref",
        default=None,
        help="Base image reference to record in envelope.base_image_ref.",
    )
    parser.add_argument(
        "--validation-commit",
        default=os.environ.get("GITHUB_SHA", "unknown"),
        help=(
            "Git commit SHA under which this validation run occurred. Defaults to "
            "$GITHUB_SHA when CI runs set it; otherwise 'unknown'."
        ),
    )
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help=(
            "Exit 1 if any invariant diverged from its expected_outcome. CI always "
            "passes this flag; locally it's off by default so developers can "
            "inspect the YAML without CI-style exit."
        ),
    )

    args = parser.parse_args(argv)

    try:
        _envelope, divergences = validate_engine(
            engine=args.engine,
            corpus_path=args.corpus,
            out_path=args.out,
            gpu_mode=args.gpu_cases,
            image_ref=args.image_ref,
            base_image_ref=args.base_image_ref,
            validation_commit=args.validation_commit,
        )
    except ValidationCorpusError as exc:
        print(f"[{args.engine}] corpus error: {exc}", file=sys.stderr)
        return 2
    except ValidationEngineNotImportable as exc:
        print(f"[{args.engine}] engine not importable: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"[{args.engine}] validation error: {exc}", file=sys.stderr)
        return 2

    print(f"[{args.engine}] wrote {args.out}", file=sys.stderr)
    if divergences:
        print(
            f"[{args.engine}] {len(divergences)} divergence(s) - see YAML 'divergences' array.",
            file=sys.stderr,
        )
        for d in divergences[:10]:
            print(
                f"  - {d.invariant_id}: {d.field} expected={d.expected!r} observed={d.observed!r}",
                file=sys.stderr,
            )
        if len(divergences) > 10:
            print(f"  ... and {len(divergences) - 10} more.", file=sys.stderr)
        if args.fail_on_divergence:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
