#!/usr/bin/env python3
"""Run each validation invariant through the real library inside its engine container.

The validation step is the **observe half** of the "observe, don't re-encode"
design in :doc:`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`.
The YAML corpus at ``src/llenergymeasure/engines/{engine}/rules.proposed.yaml`` declares each
invariant's ``expected_outcome``; this script executes the invariant through the
library and records what *actually* happened. Divergence between declared and
observed fails CI.

Usage (inside the engine's Docker container)::

    python scripts/validate_rules.py \\
        --engine transformers \\
        --corpus src/llenergymeasure/engines/transformers/rules.proposed.yaml \\
        --out src/llenergymeasure/engines/transformers/rules.validated.yaml

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
import json
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

from scripts._engine_constructors import (  # noqa: E402  (late import after sys.path)
    resolve_native_type,
)
from scripts._rules_validation_common import (  # noqa: E402  (late import after sys.path)
    TENSORRT_PRIVATE_FIELD_ALLOWLIST,
    TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
    CaptureBuffers,
    CaseResult,
    Divergence,
    classify_emission_channel,
    classify_outcome,
    compare_expected_vs_observed,
    diff_input_vs_state,
    invariant_claimed_fields,
    is_type_check_invariant,
    is_type_coercion_artifact,
    locus_confirms_invariant,
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
    # Fallback: resolve native_type against the live transformers package.
    return run_case(
        lambda: _construct_generic("transformers", native_type, kwargs),
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


def _construct_generic(engine: str, native_type: str, kwargs: dict[str, Any]) -> Any:
    cls = resolve_native_type(engine, native_type)
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
# The static miner emits short-form ``native_type`` values (``tensorrt_llm.X``)
# matching the AST symbol it walked. The short-name -> deep-import-path mapping
# (and the ``BaseLlmArgs`` -> ``TrtLlmArgs`` substitution, since ``BaseLlmArgs``
# is the abstract parent and invariants tagged on it apply to inherited fields
# on the concrete subclass) lives in the per-engine resolution table in
# :mod:`scripts._engine_constructors`.

# ``BaseLlmArgs`` / ``TrtLlmArgs`` declare ``model`` as a required field. The
# corpus's ``kwargs_positive`` / ``kwargs_negative`` only carry the field under
# test, so without an injected default Pydantic raises a "model: field
# required" error before any invariant-relevant validator runs. The placeholder is
# never resolved to a real checkpoint - construction stops at validator-pass
# time on either the invariant's positive raise (intended) or the negative success
# (intended), both before the loader would try to read from disk.
_TRTLLM_LLMARGS_TYPES: frozenset[str] = frozenset(
    {"tensorrt_llm.BaseLlmArgs", "tensorrt_llm.TrtLlmArgs"}
)
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
    """Construct a TRT-LLM type by short native_type name.

    Resolves the class via the per-engine table in
    :mod:`scripts._engine_constructors` (which holds the short-name -> deep-path
    overrides) and injects the required ``model`` placeholder for ``*LlmArgs``
    types when the corpus kwargs don't set it.
    """
    cls = resolve_native_type("tensorrt", native_type)
    use_kwargs = dict(kwargs)
    if native_type in _TRTLLM_LLMARGS_TYPES and "model" not in use_kwargs:
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
        lambda: _construct_generic("vllm", native_type, kwargs),
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
# Per-invariant driver
# ---------------------------------------------------------------------------


def _validate_invariant_with_captures(
    engine: str, invariant: dict[str, Any]
) -> tuple[CaseResult, CaptureBuffers, CaptureBuffers]:
    """Run one invariant and return the case plus the raw positive/negative captures.

    The captures are needed by the gate-soundness checks added per
    Decision #12 of the invariant-miner adversarial review - they look at
    severity-specific behaviour (positive must raise for ``severity=error``)
    and the raised exception's message text, neither of which fit the
    public ``CaseResult`` shape.

    Returns ``(case, pos, neg)``.
    """
    invariant_id = invariant["id"]
    native_type = invariant["native_type"]
    runner = get_native_type_runner(engine)
    severity = str(invariant.get("severity", "")).lower()
    # Per-engine strictness routing: transformers' GenerationConfig has a
    # non-strict path (logger.warning for dormant/announced) and a strict
    # path (composed ValueError for errors). Dispatch by declared severity
    # so the validation observation matches the corpus's expected outcome shape.
    strict_validate = severity == "error"

    kwargs_positive = dict(invariant["kwargs_positive"])
    kwargs_negative = dict(invariant["kwargs_negative"])

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
# Chunk C4 gate hardening: confirm-attribution by error locus + coercion guard.
CHECK_LOCUS_MISMATCH = "locus_mismatch"
CHECK_TYPE_COERCION_ARTIFACT = "type_coercion_artifact"


def _erroring_field_value_is_numeric(invariant: dict[str, Any], error_details: list[Any]) -> bool:
    """True iff the field the error fired on carries a numeric ``kwargs_positive`` value.

    Used to decide whether a ``literal_error`` is a numeric-allowlist coercion
    artefact (reject) versus a categorical-allowlist rule (keep). Keyed on the
    *erroring* field (the error locus), not the claimed fields: a literal error
    on an unrelated categorical field must not be reclassified as a numeric
    coercion artefact just because some other claimed field is numeric. A purely
    structural signal off the probe kwargs - no schema lookup needed.
    """
    kwargs = invariant.get("kwargs_positive") or {}
    if not isinstance(kwargs, dict):
        return False
    error_fields = {part for detail in error_details for part in detail.loc}
    return any(
        isinstance(kwargs.get(name), (int, float)) and not isinstance(kwargs.get(name), bool)
        for name in error_fields
    )


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

    # 4 + 5. Confirm-attribution by error locus and type-coercion-artefact
    #        rejection (chunk C4). Only meaningful when the positive raised -
    #        these refine *why* a raise counts as a confirm of THIS rule.
    if pos.exception_type is not None:
        claimed = invariant_claimed_fields(invariant)
        type_check = is_type_check_invariant(invariant)

        # Coercion artefact: a parse/coercion error standing in for the claimed
        # value rule. Checked before locus so the more specific cause wins.
        numeric_predicate = _erroring_field_value_is_numeric(invariant, pos.error_details)
        if is_type_coercion_artifact(
            pos.error_details, is_type_check=type_check, numeric_predicate=numeric_predicate
        ):
            artefact_types = sorted({d.error_type for d in pos.error_details})
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="kwargs_positive",
                    expected="rule_fires_not_type_coercion",
                    observed=artefact_types,
                    check_failed=CHECK_TYPE_COERCION_ARTIFACT,
                )
            )
        # Locus mismatch: the raise concerns a field the invariant does not
        # claim (an incidental sibling/Literal error firing first).
        elif not locus_confirms_invariant(claimed, pos.error_details):
            observed_loc = sorted({part for d in pos.error_details for part in d.loc})
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="kwargs_positive",
                    expected=sorted(claimed),
                    observed=observed_loc,
                    check_failed=CHECK_LOCUS_MISMATCH,
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
# Carried-catalogue re-gate (decay alarm, design § 6 signal 1)
# ---------------------------------------------------------------------------

VERDICT_CONFIRMED = "confirmed"
VERDICT_FAILED = "failed"
VERDICT_INFRA_ERROR = "infra_error"

REGATE_REPORT_SCHEMA_VERSION = "1.0.0"

# Construction-drift exception names: when the negative probe (which is supposed
# to construct cleanly) raises one of these, the probe no longer builds against
# the new library - constructor drift, not a rule change. Reported as
# infra_error so the maintainer knows to refresh the probe, not chase a rule.
_CONSTRUCTION_DRIFT_EXC_NAMES: frozenset[str] = frozenset(
    {"NativeTypeResolutionError", "TypeError", "ImportError", "ModuleNotFoundError"}
)


def _carried_verdict(
    invariant: dict[str, Any], case: CaseResult, pos: CaptureBuffers, neg: CaptureBuffers
) -> tuple[str, str]:
    """Classify one carried entry against the current container.

    Returns ``(verdict, reason)``. Verdict is one of
    :data:`VERDICT_CONFIRMED` / :data:`VERDICT_FAILED` / :data:`VERDICT_INFRA_ERROR`.

    - ``infra_error`` - the probe will not construct against the new library:
      the native_type no longer resolves (``NativeTypeResolutionError``), or the
      *negative* probe (which should build cleanly) raised a construction-drift
      error. Constructor drift, not necessarily a rule change.
    - ``failed`` - the probe constructs but the rule no longer holds: the
      positive case stopped confirming, the negative case now fires, or a
      gate-soundness check (locus / coercion / template / raise) flags.
    - ``confirmed`` - the rule still holds exactly as carried.

    Reuses the ``case``'s ``positive_confirmed`` / ``negative_confirmed`` (already
    computed by :func:`_validate_invariant_with_captures`) so the confirm
    definition lives in one place.
    """
    # Resolution failure on the positive probe -> infra_error outright.
    if pos.exception_type == "NativeTypeResolutionError":
        return VERDICT_INFRA_ERROR, f"native_type unresolved: {pos.exception_message or ''}"

    # Negative probe must construct cleanly; a construction-drift error there is
    # an infra signal (the probe is broken), not a rule failure.
    if neg.exception_type in _CONSTRUCTION_DRIFT_EXC_NAMES:
        return (
            VERDICT_INFRA_ERROR,
            f"negative probe will not construct ({neg.exception_type}): "
            f"{neg.exception_message or ''}",
        )

    if not case.positive_confirmed:
        return VERDICT_FAILED, f"positive no longer fires (outcome={case.outcome})"
    if not case.negative_confirmed:
        return VERDICT_FAILED, "negative now fires"

    soundness = compute_gate_soundness_divergences(invariant, pos, neg)
    if soundness:
        checks = sorted({str(d.check_failed) for d in soundness if d.check_failed})
        return VERDICT_FAILED, f"gate-soundness check(s) failed: {', '.join(checks)}"

    return VERDICT_CONFIRMED, ""


def regate_carried_catalogue(
    *,
    engine: str,
    carried_corpus_path: Path,
) -> dict[str, Any]:
    """Re-gate a previous pin's carried catalogue against the current container.

    Runs each carried invariant through the live gate and emits the decay-alarm
    report (design § 6 signal 1). The report shape is stable for a later CI step
    (chunk C7) to render as a PR comment::

        {
          "schema_version": "1.0.0",
          "engine": "vllm",
          "engine_version": "0.21.0",          # current (in-container) version
          "carried_corpus": "engine_versions/vllm/v0_19_1/.../rules.proposed.yaml",
          "total": 41,
          "counts": {"confirmed": 37, "failed": 2, "infra_error": 2},
          "acceptance_rate": 0.902,            # confirmed / total
          "entries": [
            {"id": "...", "verdict": "confirmed", "reason": ""},
            {"id": "...", "verdict": "failed", "reason": "positive no longer fires (...)"},
            {"id": "...", "verdict": "infra_error", "reason": "native_type unresolved: ..."}
          ]
        }

    ``acceptance_rate`` is ``confirmed / total`` (infra_error and failed both
    count against it); the split lets the report separate "rule changed" from
    "probe broke", which the design calls for explicitly.
    """
    corpus = _load_corpus(carried_corpus_path)
    engine_version = _resolve_engine_version(engine)

    entries: list[dict[str, str]] = []
    counts = {VERDICT_CONFIRMED: 0, VERDICT_FAILED: 0, VERDICT_INFRA_ERROR: 0}

    for invariant in corpus.get("invariants", []):
        invariant_id = str(invariant.get("id", "<unknown>"))
        try:
            case, pos, neg = _validate_invariant_with_captures(engine, invariant)
        except ValidationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            verdict, reason = VERDICT_INFRA_ERROR, f"{type(exc).__name__}: {exc}"
            entries.append({"id": invariant_id, "verdict": verdict, "reason": reason})
            counts[verdict] += 1
            continue

        verdict, reason = _carried_verdict(invariant, case, pos, neg)
        entries.append({"id": invariant_id, "verdict": verdict, "reason": reason})
        counts[verdict] += 1

    total = len(entries)
    acceptance_rate = (counts[VERDICT_CONFIRMED] / total) if total else 0.0
    return {
        "schema_version": REGATE_REPORT_SCHEMA_VERSION,
        "engine": engine,
        "engine_version": engine_version,
        "carried_corpus": str(carried_corpus_path),
        "total": total,
        "counts": counts,
        "acceptance_rate": round(acceptance_rate, 4),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Rung 0: deterministic reconciliation join (decay-alarm triage)
# ---------------------------------------------------------------------------
#
# A re-gate "failed" verdict means the carried rule did not re-confirm against
# the new container. But the SAME bump's fresh deterministic re-mine often
# already re-encoded that rule - with an updated message template (the engine
# reworded the same check) or under a renamed id (the walker named it
# differently). Those are not decay: they are the workflow adopting gate-truth,
# and the Opus triage that re-derived them by hand was redundant. This join
# reclassifies such failures against the fresh VALIDATED corpus produced in the
# same run - by id first, then by structural signature - leaving only the
# genuine residual as decay candidates. No tokens, no LLM: healing is adoption
# of gate-truth, not generation.

HEALED_TEMPLATE_DRIFT = "healed_template_drift"
HEALED_RULE_MORPHED = "healed_rule_morphed"
RECONCILE_REPORT_SCHEMA_VERSION = "1.0.0"


def _signature(invariant: dict[str, Any]) -> tuple[str, frozenset[str], str]:
    """Structural identity of a rule: (native_type, claimed bare fields, severity).

    The second-pass join key. Bare field names (last dotted segment) match the
    grain :func:`invariant_claimed_fields` uses, so a rule that survives a bump
    under a renamed id but the same construction surface still joins.
    """
    native_type = str(invariant.get("native_type", ""))
    fields = invariant_claimed_fields(invariant)
    severity = str(invariant.get("severity", "")).lower()
    return native_type, fields, severity


def _index_corpus(corpus: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[Any, list[str]]]:
    """Index a proposed-shape corpus by id and by structural signature."""
    by_id: dict[str, dict[str, Any]] = {}
    by_signature: dict[Any, list[str]] = {}
    for invariant in corpus.get("invariants", []):
        if not isinstance(invariant, dict):
            continue
        rule_id = str(invariant.get("id", ""))
        if not rule_id:
            continue
        by_id[rule_id] = invariant
        by_signature.setdefault(_signature(invariant), []).append(rule_id)
    return by_id, by_signature


def reconcile_regate_report(
    report: dict[str, Any],
    *,
    carried_corpus_path: Path,
    fresh_validated_ids: set[str],
    fresh_proposed_path: Path,
) -> dict[str, Any]:
    """Join the re-gate report's ``failed`` entries against the fresh corpus.

    ``fresh_validated_ids`` is the set of ids the same run's gate confirmed
    (the cases in the new pin's ``rules.validated.yaml``). ``fresh_proposed_path``
    is the new pin's ``rules.proposed.yaml`` - it carries the structural shape
    (native_type / match.fields / severity / message_template) the validated
    cases lack, so the second-pass signature join and the template diff read
    from it. Both are byte-committed in the same writeback.

    Returns a ``reconciliation`` block:

        {
          "schema_version": "1.0.0",
          "healed": [
            {"id": "...", "kind": "healed_template_drift",
             "old_template": "...", "new_template": "..."},
            {"id": "...", "kind": "healed_rule_morphed", "new_id": "..."}
          ],
          "residual": [{"id": "...", "reason": "..."}],
          "counts": {"healed": 4, "residual": 3}
        }

    Only ``failed`` entries are reconciled; ``infra_error`` (probe broke) is a
    distinct signal and stays out of the join.
    """
    carried_by_id, _ = _index_corpus(_load_corpus(carried_corpus_path))
    fresh_by_id, fresh_by_signature = _index_corpus(_load_corpus(fresh_proposed_path))

    healed: list[dict[str, str]] = []
    residual: list[dict[str, str]] = []

    failed_entries = [e for e in report.get("entries", []) if e.get("verdict") == VERDICT_FAILED]
    for entry in failed_entries:
        rule_id = str(entry.get("id", ""))
        carried = carried_by_id.get(rule_id)

        # First pass: same id still gate-confirmed at the new pin. The rule
        # holds; the carried probe failed only because the engine reworded the
        # message (template drift). Show the old -> new template move.
        if rule_id in fresh_validated_ids:
            healed.append(
                {
                    "id": rule_id,
                    "kind": HEALED_TEMPLATE_DRIFT,
                    "old_template": str((carried or {}).get("message_template") or ""),
                    "new_template": str(
                        (fresh_by_id.get(rule_id) or {}).get("message_template") or ""
                    ),
                }
            )
            continue

        # Second pass: the rule re-appears under a new id with the same
        # structural signature (native_type, claimed fields, severity) and that
        # new id is gate-confirmed. The walker renamed it; the check survived.
        morph_id = None
        if carried is not None:
            for candidate in fresh_by_signature.get(_signature(carried), []):
                if candidate != rule_id and candidate in fresh_validated_ids:
                    morph_id = candidate
                    break
        if morph_id is not None:
            healed.append({"id": rule_id, "kind": HEALED_RULE_MORPHED, "new_id": morph_id})
            continue

        # Residual: neither id nor signature recovered - a genuine decay candidate.
        residual.append({"id": rule_id, "reason": str(entry.get("reason", ""))})

    return {
        "schema_version": RECONCILE_REPORT_SCHEMA_VERSION,
        "healed": healed,
        "residual": residual,
        "counts": {"healed": len(healed), "residual": len(residual)},
    }


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
    if d.get("observed_exception") is None:
        d.pop("observed_exception", None)
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
            case, pos, neg = _validate_invariant_with_captures(engine, invariant)
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

        # One underlying failure must surface as ONE finding. The two divergence
        # vocabularies overlap (a positive that stopped firing reads as both an
        # expected-vs-observed outcome mismatch AND a gate-soundness
        # positive_raises miss), so they are layered, not summed: the coarse
        # expected-vs-observed comparison runs first; the finer gate-soundness
        # checks (locus attribution, coercion-artefact, message-template, and
        # the positive/negative confirm) only run when that comparison is clean,
        # refining a confirm rather than re-reporting a failure the coarse check
        # already named.
        rule_divergences = compare_expected_vs_observed(
            invariant_id=invariant["id"],
            expected=invariant.get("expected_outcome") or {},
            observed_outcome=case.outcome,
            observed_emission=case.emission_channel,
            silent_normalisations=case.observed_silent_normalisations,
        )
        if not rule_divergences and pos is not None and neg is not None:
            rule_divergences = compute_gate_soundness_divergences(invariant, pos, neg)
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


# Engine name -> importable distribution module. tensorrt's PyPI/import name
# is ``tensorrt_llm``; the other two match the engine label.
_ENGINE_IMPORT_NAMES = {
    "transformers": "transformers",
    "tensorrt": "tensorrt_llm",
    "vllm": "vllm",
}


def _resolve_engine_version(engine: str) -> str:
    """Best-effort: return the installed library's version or ``"unknown"``."""
    module_name = _ENGINE_IMPORT_NAMES.get(engine)
    if module_name is None:
        return "unknown"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValidationEngineNotImportable(
            f"{module_name} is not importable in this environment "
            f"(expected when running outside the llenergymeasure:{engine} container)"
        ) from exc
    return str(module.__version__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fresh_validated_ids(validated_path: Path) -> set[str]:
    """Read the case ids out of a fresh validated envelope (rules.validated.yaml)."""
    data = yaml.safe_load(validated_path.read_text())
    if not isinstance(data, dict):
        return set()
    return {str(c["id"]) for c in data.get("cases", []) if isinstance(c, dict) and "id" in c}


def _run_carried_mode(args: argparse.Namespace) -> int:
    """Run the carried-catalogue re-gate and emit the JSON report. Returns exit code."""
    try:
        report = regate_carried_catalogue(
            engine=args.engine,
            carried_corpus_path=args.carried,
        )
    except ValidationCorpusError as exc:
        print(f"[{args.engine}] carried corpus error: {exc}", file=sys.stderr)
        return 2
    except ValidationEngineNotImportable as exc:
        print(f"[{args.engine}] engine not importable: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"[{args.engine}] validation error: {exc}", file=sys.stderr)
        return 2

    # Rung 0: reconcile carried-failed against the same run's fresh corpus.
    if args.fresh_validated is not None and args.fresh_proposed is not None:
        report["reconciliation"] = reconcile_regate_report(
            report,
            carried_corpus_path=args.carried,
            fresh_validated_ids=_fresh_validated_ids(args.fresh_validated),
            fresh_proposed_path=args.fresh_proposed,
        )

    payload = json.dumps(report, indent=2)
    if args.regate_out is not None:
        args.regate_out.parent.mkdir(parents=True, exist_ok=True)
        args.regate_out.write_text(payload + "\n")
        print(f"[{args.engine}] wrote re-gate report {args.regate_out}", file=sys.stderr)
    else:
        print(payload)

    counts = report["counts"]
    print(
        f"[{args.engine}] carried re-gate: {report['total']} entries, "
        f"acceptance {report['acceptance_rate']:.1%} "
        f"(confirmed={counts[VERDICT_CONFIRMED]} "
        f"failed={counts[VERDICT_FAILED]} infra_error={counts[VERDICT_INFRA_ERROR]})",
        file=sys.stderr,
    )
    recon = report.get("reconciliation")
    if recon is not None:
        rc = recon["counts"]
        print(
            f"[{args.engine}] reconciliation: {rc['healed']} healed, "
            f"{rc['residual']} residual (decay candidates)",
            file=sys.stderr,
        )
    # The decay alarm is informational (design § 6: no auto-blocking thresholds
    # in v0.10.0); always exit 0 so the CI step that consumes the report decides.
    return 0


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
        default=None,
        help="Path to the YAML corpus file (required unless --carried is given).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to write the validated YAML envelope (required unless --carried is given).",
    )
    parser.add_argument(
        "--carried",
        type=Path,
        default=None,
        help=(
            "Carried-catalogue re-gate mode (decay alarm). Path to a PREVIOUS pin's "
            "invariant corpus (proposed shape, with kwargs). Runs it against the CURRENT "
            "in-container engine and emits a JSON verdict report instead of the normal "
            "validation envelope. Mutually exclusive with --corpus/--out."
        ),
    )
    parser.add_argument(
        "--regate-out",
        type=Path,
        default=None,
        help="Path to write the carried re-gate JSON report (default: stdout).",
    )
    parser.add_argument(
        "--fresh-validated",
        type=Path,
        default=None,
        help=(
            "Rung 0 reconciliation (carried mode): the SAME run's fresh "
            "rules.validated.yaml. Failed carried entries whose id (or "
            "structural signature) re-confirms here are reclassified as healed "
            "rather than decay. Requires --fresh-proposed."
        ),
    )
    parser.add_argument(
        "--fresh-proposed",
        type=Path,
        default=None,
        help=(
            "Rung 0 reconciliation (carried mode): the SAME run's fresh "
            "rules.proposed.yaml (carries the structural shape + templates the "
            "validated envelope lacks). Requires --fresh-validated."
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

    # Carried-catalogue re-gate mode (decay alarm) takes a different output path
    # entirely - a JSON verdict report, not the validation envelope.
    if args.carried is not None:
        if args.corpus is not None or args.out is not None:
            parser.error("--carried is mutually exclusive with --corpus/--out")
        return _run_carried_mode(args)

    if args.corpus is None or args.out is None:
        parser.error("--corpus and --out are required unless --carried is given")

    try:
        _envelope, divergences = validate_engine(
            engine=args.engine,
            corpus_path=args.corpus,
            out_path=args.out,
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
