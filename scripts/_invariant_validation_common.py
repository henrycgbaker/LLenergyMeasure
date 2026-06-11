"""Shared utilities for the validate-invariants pipeline.

Factored out of :mod:`scripts.validate_invariants` so that per-engine native-type
runners can live here while the CLI + loop driver stays in ``validate_invariants``.

This module is engine-agnostic. Per-engine behaviour lives behind
:func:`get_native_type_runner`, which dispatches on engine name.

Design contract: the validation step observes library behaviour concretely. It
never re-interprets the invariant's declared shape - if the library behaves
differently from what the corpus claims, CI fails. See
:doc:`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`
§4.3 for the full contract.

Type-coercion-artefact guard - msgspec coverage status
------------------------------------------------------
:func:`is_type_coercion_artifact` covers both pydantic and msgspec validation
errors (:func:`extract_error_details` reads pydantic's structured ``.errors()``
and parses msgspec's ``Expected X, got Y - at `$.field``` message shape). The
spike's version was pydantic-only; this closes that gap for the case where a
msgspec ValidationError actually surfaces.

One residual, documented rather than over-engineered: the gate constructs
probes by *direct* ``Struct(**kwargs)`` construction, and a msgspec ``Struct``
does NOT validate types or constraints on direct construction (it validates
only on ``decode`` / ``convert``). vllm's ``SamplingParams`` is a msgspec
Struct, but its value rules fire in ``__post_init__`` / ``_verify_args`` as
plain ``ValueError``s, not msgspec errors. So on the live construction path a
msgspec *parse* artefact is not reachable in the first place; the msgspec branch
here is a correctness backstop for any path that does surface one (e.g. a future
decode-based probe), not a hot path today.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Private-field allowlist
# ---------------------------------------------------------------------------
# The validation state-diff excludes engine-specific bookkeeping fields that would
# pollute the diff with non-deterministic state (commit hashes, cached derived
# flags, per-run tensors). Each engine declares its own allowlist; the default
# covers fields common across engines.

_DEFAULT_PRIVATE_FIELD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "_commit_hash",
        "_from_model_config",
        "_original_object_hash",
        "_all_stop_token_ids",
    }
)

TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST: frozenset[str] = _DEFAULT_PRIVATE_FIELD_ALLOWLIST | frozenset(
    {
        # HF-specific derived fields populated during __post_init__ that do
        # not constitute a user-facing normalisation.
        "_eos_token_tensor",
        "_pad_token_tensor",
        "_bos_token_tensor",
        "transformers_version",
    }
)

TENSORRT_PRIVATE_FIELD_ALLOWLIST: frozenset[str] = _DEFAULT_PRIVATE_FIELD_ALLOWLIST | frozenset(
    {
        # TRT-LLM `*LlmArgs` populate a handful of private bookkeeping fields
        # during `model_validator(mode='after')` passes; they are not
        # user-facing normalisations and would pollute the silent-normalisation
        # diff with non-deterministic state on every validation run.
        "_parallel_config",
        "_speculative_config",
        "_quant_config",
        "_build_config",
    }
)


# ---------------------------------------------------------------------------
# Observation dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorDetail:
    """One structured field-error extracted from a raised validation exception.

    ``loc`` is the (possibly nested) field path the error concerns; the last
    element is the field name. ``error_type`` is the validator's machine code
    (pydantic's ``int_parsing`` / ``literal_error`` / ``greater_than`` / ...;
    for msgspec and plain raises a synthesised label, see
    :func:`extract_error_details`). These two together let the gate attribute a
    confirm to the rule whose field(s) the error actually concerns and reject
    type-coercion artefacts - without bare substring matching on the message.
    """

    loc: tuple[str, ...]
    error_type: str


@dataclass(frozen=True)
class CaptureBuffers:
    """Container for everything a single ``native_type(**kwargs)`` call produced."""

    exception_type: str | None
    exception_message: str | None
    warnings_captured: tuple[str, ...]
    logger_messages: tuple[str, ...]
    observed_state: dict[str, Any] | None
    duration_ms: int
    error_details: tuple[ErrorDetail, ...] = ()


@dataclass
class CaseResult:
    """Per-invariant observed outcome, ready for JSON serialisation."""

    id: str
    outcome: str  # see _classify_outcome
    emission_channel: str  # mirrors corpus "emission_channel" tag
    observed_messages: list[str] = field(default_factory=list)
    observed_silent_normalisations: dict[str, dict[str, Any]] = field(default_factory=dict)
    observed_exception: dict[str, str] | None = None
    positive_confirmed: bool = False
    negative_confirmed: bool = False
    duration_ms: int = 0
    skipped_reason: str | None = None


@dataclass
class Divergence:
    """One observed-vs-expected mismatch.

    ``check_failed`` names the gate-soundness check that surfaced this
    divergence (one of ``positive_raises``, ``negative_does_not_raise``,
    ``message_template_match``, ``message_template_too_dynamic``) when the
    divergence came from the soundness checks added per Decision #12 of the
    invariant-miner adversarial review (`.product/designs/adversarial-review-invariant-miner-2026-04-26.md`).
    Pre-existing expected-vs-observed comparisons leave this ``None``.
    """

    invariant_id: str
    field: str
    expected: Any
    observed: Any
    check_failed: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "invariant_id": self.invariant_id,
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
        }
        if self.check_failed is not None:
            out["check_failed"] = self.check_failed
        return out


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def extract_state(
    obj: Any, *, private_allowlist: Iterable[str] = TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST
) -> dict[str, Any]:
    """Uniform dump of an arbitrary config object's public state.

    Handles Pydantic v2 (``model_dump``), dataclasses, ``__slots__`` classes
    and plain ``__dict__`` classes. Private attributes (``_foo``) are dropped
    unless they appear in ``private_allowlist`` - see module docstring for
    why the allowlist exists.
    """
    allowlist = frozenset(private_allowlist)

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return {k: v for k, v in dumped.items() if not k.startswith("_") or k in allowlist}
        except Exception:
            pass

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: getattr(obj, f.name)
            for f in dataclasses.fields(obj)
            if not f.name.startswith("_") or f.name in allowlist
        }

    collected: dict[str, Any] = {}
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        for name in slots:
            if (not name.startswith("_") or name in allowlist) and hasattr(obj, name):
                collected[name] = getattr(obj, name)
    if hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            if not name.startswith("_") or name in allowlist:
                collected.setdefault(name, value)
    return collected


def diff_input_vs_state(
    kwargs: dict[str, Any], observed_state: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Identify silent normalisations - fields the library changed post-construction.

    Returns ``{field: {"declared": <input>, "observed": <state>}}``.
    """
    diffs: dict[str, dict[str, Any]] = {}
    for field_name, declared in kwargs.items():
        if field_name not in observed_state:
            continue
        observed = observed_state[field_name]
        if declared != observed:
            diffs[field_name] = {
                "declared": _jsonable(declared),
                "observed": _jsonable(observed),
            }
    return diffs


# ---------------------------------------------------------------------------
# Capture primitives
# ---------------------------------------------------------------------------


_WARNING_ONCE_SENTINEL = "\x00LLEM_WARNING_ONCE\x00"
"""Prefix injected by :func:`_patch_warning_once` to distinguish
``logger.warning_once`` records from plain ``logger.warning`` at the
stdlib-record level (HF's ``warning_once`` is ``@lru_cache``-wrapped
``self.warning``, identical in the record stream otherwise)."""


def _attach_loggers(
    loggers: Iterable[str],
) -> tuple[logging.Handler, io.StringIO, list[tuple[logging.Logger, int]]]:
    """Attach a StringIO handler to each named logger.

    Returns the handler, its buffer and a list of ``(logger, previous_level)``
    pairs so the caller can restore levels afterwards.
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    restore: list[tuple[logging.Logger, int]] = []
    for name in loggers:
        logger = logging.getLogger(name)
        restore.append((logger, logger.level))
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return handler, buf, restore


def _detach_loggers(handler: logging.Handler, restore: list[tuple[logging.Logger, int]]) -> None:
    for logger, prev in restore:
        logger.removeHandler(handler)
        logger.setLevel(prev)


def _patch_warning_once() -> Callable[[], None]:
    """Install a sentinel-tagging spy over ``Logger.warning_once``.

    Returns a restore callable the caller must run in ``finally``. No-op when
    HF isn't importable (the attribute is attached at ``transformers.utils.logging``
    import time; outside the HF container the method is absent and there is
    nothing to patch).

    HF's ``warning_once`` is ``@functools.lru_cache``-wrapped at the module
    level - the cache survives across ``run_case`` calls in the same process.
    Without clearing it, a dormancy invariant that fires its message on invariant N
    would silently no-op on invariant N+1 reusing the same template, and the
    validation classifier would observe ``logger_warning`` (the underlying
    ``warning`` channel) instead of ``logger_warning_once`` for every invariant
    after the first hit. Clear the cache on every spy installation so each
    invariant sees a clean slate.
    """
    original = getattr(logging.Logger, "warning_once", None)
    if original is None:
        return lambda: None

    # Best-effort: clear HF's process-level lru_cache on warning_once / info_once
    # so successive run_case calls in one process don't trip the dedup wrapper.
    # The wrappers live on ``transformers.utils.logging``; if HF isn't importable
    # we already returned no-op above, so this branch is safe.
    try:
        from transformers.utils import logging as _hf_logging  # type: ignore

        for attr in ("warning_once", "info_once"):
            cached = getattr(_hf_logging, attr, None)
            cache_clear = getattr(cached, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    except ImportError:
        pass

    def spy(self: logging.Logger, msg: Any, *args: Any, **kwargs: Any) -> Any:
        tagged = f"{_WARNING_ONCE_SENTINEL}{msg}" if isinstance(msg, str) else msg
        return original(self, tagged, *args, **kwargs)

    logging.Logger.warning_once = spy  # type: ignore[attr-defined]

    def restore() -> None:
        logging.Logger.warning_once = original  # type: ignore[attr-defined]

    return restore


# Backtick-quoted field name in a plain-raise message, e.g.
# "`num_beams` is greater than 1". The engines (transformers, vllm, tensorrt)
# consistently backtick the field name in their validation messages, so this is
# a reliable structured locus for plain (non-pydantic) raises - far tighter
# than whole-message substring matching.
_BACKTICK_FIELD_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")

# msgspec encodes the failing field as a trailing JSON-pointer-ish locus,
# e.g. "Expected `int`, got `str` - at `$.temperature`". The leading `$.`
# is msgspec's document-root marker.
_MSGSPEC_LOC_RE = re.compile(r"at `\$\.([A-Za-z_][A-Za-z0-9_.\[\]]*)`")

# msgspec parse-error shape: "Expected `<type>`, got `<type>`" with no
# constraint operator. A constraint failure instead reads "Expected `int` >= 0".
_MSGSPEC_PARSE_RE = re.compile(r"Expected `[^`]+`, got `[^`]+`")


def extract_error_details(exc: BaseException) -> tuple[ErrorDetail, ...]:
    """Extract structured ``(loc, error_type)`` pairs from a raised exception.

    Three sources, in priority order:

    1. **pydantic ``ValidationError``** - the structured ``.errors()`` list is
       authoritative: each entry carries a ``loc`` tuple and a machine
       ``type`` (``int_parsing``, ``literal_error``, ``greater_than``, ...).
    2. **msgspec ``ValidationError``** - no ``.errors()`` API; the locus and a
       coarse type are parsed from the message (``... - at `$.field```; the
       ``Expected X, got Y`` shape marks a parse/coercion artefact, surfaced as
       error_type ``msgspec_parsing``, else ``msgspec_validation``).
    3. **plain exceptions** (``ValueError`` from ``__post_init__`` /
       ``_verify_args`` / ``GenerationConfig.validate``) - backtick-quoted
       field names in the message are the locus; error_type is
       ``plain`` (no machine code available).

    Returns ``()`` when no field locus can be recovered (the caller then falls
    back to permissive behaviour - it never *blocks* a confirm on absence of a
    locus, only refines attribution when one is present).
    """
    details = _extract_pydantic_details(exc)
    if details:
        return details
    details = _extract_msgspec_details(exc)
    if details:
        return details
    return _extract_plain_details(exc)


def _extract_pydantic_details(exc: BaseException) -> tuple[ErrorDetail, ...]:
    errors_fn = getattr(exc, "errors", None)
    if not callable(errors_fn):
        return ()
    # Guard: only pydantic ValidationError carries the (loc, type) error shape.
    # Duck-type on the structure rather than importing pydantic (this module is
    # engine-agnostic and pydantic may be absent in some test environments).
    try:
        raw_errors = errors_fn()
    except Exception:
        return ()
    if not isinstance(raw_errors, (list, tuple)):
        return ()
    collected: list[ErrorDetail] = []
    for err in raw_errors:
        if not isinstance(err, dict) or "loc" not in err or "type" not in err:
            return ()  # not a pydantic-shaped error list
        loc = tuple(str(part) for part in err["loc"])
        collected.append(ErrorDetail(loc=loc, error_type=str(err["type"])))
    return tuple(collected)


def _extract_msgspec_details(exc: BaseException) -> tuple[ErrorDetail, ...]:
    # msgspec.ValidationError is a subclass of ValueError; identify it by class
    # name to avoid a hard import dependency on msgspec.
    if type(exc).__name__ != "ValidationError" or type(exc).__module__.split(".")[0] != "msgspec":
        return ()
    message = str(exc)
    loc_match = _MSGSPEC_LOC_RE.search(message)
    if not loc_match:
        return ()
    loc = tuple(loc_match.group(1).split("."))
    error_type = "msgspec_parsing" if _MSGSPEC_PARSE_RE.search(message) else "msgspec_validation"
    return (ErrorDetail(loc=loc, error_type=error_type),)


def _extract_plain_details(exc: BaseException) -> tuple[ErrorDetail, ...]:
    fields = _BACKTICK_FIELD_RE.findall(str(exc))
    # De-dup preserving order; each backticked field becomes a one-element loc.
    seen: set[str] = set()
    details: list[ErrorDetail] = []
    for name in fields:
        if name not in seen:
            seen.add(name)
            details.append(ErrorDetail(loc=(name,), error_type="plain"))
    return tuple(details)


def run_case(
    callable_fn: Callable[[], Any],
    *,
    logger_names: Iterable[str] = (),
    private_allowlist: Iterable[str] = TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
) -> CaptureBuffers:
    """Run ``callable_fn()`` and capture exceptions / warnings / logger output / state.

    ``callable_fn`` is usually ``lambda: native_type(**kwargs)`` or
    ``lambda: native_type(**kwargs).validate(strict=True)``. Returns a
    :class:`CaptureBuffers` regardless of whether the call raised.
    """
    handler, buf, restore = _attach_loggers(logger_names)
    restore_warning_once = _patch_warning_once()
    start = time.perf_counter()
    exc_type: str | None = None
    exc_msg: str | None = None
    error_details: tuple[ErrorDetail, ...] = ()
    obj: Any = None
    captured_warnings: list[warnings.WarningMessage] = []

    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                obj = callable_fn()
            except Exception as exc:
                exc_type = type(exc).__name__
                exc_msg = str(exc)
                error_details = extract_error_details(exc)
            # Snapshot inside the catch_warnings scope so warnings captured
            # alongside an exception are preserved (dormant-then-raise paths).
            captured_warnings = list(recorded or [])
    finally:
        restore_warning_once()
        _detach_loggers(handler, restore)
        duration_ms = int((time.perf_counter() - start) * 1000)

    warnings_tuple = tuple(str(w.message) for w in captured_warnings)

    log_messages = _split_log_buffer(buf.getvalue())
    observed_state = (
        extract_state(obj, private_allowlist=private_allowlist) if obj is not None else None
    )

    return CaptureBuffers(
        exception_type=exc_type,
        exception_message=exc_msg,
        warnings_captured=warnings_tuple,
        logger_messages=log_messages,
        observed_state=observed_state,
        duration_ms=duration_ms,
        error_details=error_details,
    )


def _split_log_buffer(raw: str) -> tuple[str, ...]:
    """Split buffer text into one entry per record, dropping empty trailing."""
    if not raw:
        return ()
    lines = [line for line in raw.split("\n") if line.strip()]
    return tuple(lines)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def classify_outcome(capture: CaptureBuffers, silent_normalisations: dict[str, Any]) -> str:
    """Given captured behaviour, compute the observed outcome label.

    Preference order:

    1. Exception raised -> ``"error"``
    2. ``warnings.warn`` captured -> ``"warn"``
    3. Logger message captured -> ``"dormant_announced"``
    4. Silent state change detected -> ``"dormant_silent"``
    5. Nothing observed -> ``"no_op"``
    """
    if capture.exception_type is not None:
        return "error"
    if capture.warnings_captured:
        return "warn"
    if capture.logger_messages:
        return "dormant_announced"
    if silent_normalisations:
        return "dormant_silent"
    return "no_op"


def classify_emission_channel(capture: CaptureBuffers) -> str:
    """Return the corpus-compatible ``emission_channel`` tag.

    ``logger_warning_once`` is distinguished from plain ``logger_warning``
    via the sentinel prepended by :func:`_patch_warning_once`. Mixed
    batches (same invariant emitting both forms) classify as
    ``logger_warning_once`` - the dedup-wrapped form is the stricter claim
    on user visibility.
    """
    if capture.exception_type is not None:
        return "none"
    if capture.warnings_captured:
        return "warnings_warn"
    if capture.logger_messages:
        if any(_WARNING_ONCE_SENTINEL in m for m in capture.logger_messages):
            return "logger_warning_once"
        return "logger_warning"
    return "none"


def strip_warning_once_sentinel(messages: Iterable[str]) -> tuple[str, ...]:
    """Remove the ``warning_once`` sentinel from captured messages for envelope output.

    Classification (``classify_emission_channel``) needs the sentinel; downstream
    consumers do not. Call this right before serialising observed messages.
    """
    return tuple(m.replace(_WARNING_ONCE_SENTINEL, "") for m in messages)


# ---------------------------------------------------------------------------
# Expected vs observed comparison
# ---------------------------------------------------------------------------


def compare_expected_vs_observed(
    *,
    invariant_id: str,
    expected: dict[str, Any],
    observed_outcome: str,
    observed_emission: str,
    silent_normalisations: dict[str, Any],
) -> list[Divergence]:
    """Return the list of expected-vs-observed divergences for one invariant.

    Missing/extra fields on either side are *not* treated as divergence -
    only fields present in ``expected`` are checked. This keeps the
    comparison permissive while still catching drift in the tracked fields.
    """
    divergences: list[Divergence] = []
    expected_outcome = expected.get("outcome")
    if expected_outcome and expected_outcome != observed_outcome:
        divergences.append(
            Divergence(
                invariant_id=invariant_id,
                field="outcome",
                expected=expected_outcome,
                observed=observed_outcome,
            )
        )
    expected_channel = expected.get("emission_channel")
    if expected_channel and expected_channel != observed_emission:
        divergences.append(
            Divergence(
                invariant_id=invariant_id,
                field="emission_channel",
                expected=expected_channel,
                observed=observed_emission,
            )
        )

    expected_norm_fields = expected.get("normalised_fields") or []
    if expected_norm_fields:
        missing = [f for f in expected_norm_fields if f not in silent_normalisations]
        if missing:
            divergences.append(
                Divergence(
                    invariant_id=invariant_id,
                    field="normalised_fields",
                    expected=list(expected_norm_fields),
                    observed=sorted(silent_normalisations.keys()),
                )
            )

    return divergences


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Coerce a value so ``json.dumps`` can handle it without ``default=str``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return str(value)


# ---------------------------------------------------------------------------
# Gate-soundness helpers (Decision #12 of the invariant-miner adversarial review)
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")
"""Matches a single format-string placeholder like ``{declared_value}`` or ``{}``.

We deliberately do NOT match nested braces - Python's format-string grammar
permits them but the corpus's ``message_template`` strings do not use them
(verified by inspection of ``src/llenergymeasure/engines/transformers/rules.proposed.yaml``).
A non-greedy non-recursive regex is sufficient and simpler to reason about.
"""

_MIN_STATIC_FRAGMENT_LEN = 4
"""Minimum length for a static fragment to be useful for substring matching.

Below this we treat the template as ``too_dynamic`` - a 3-character substring
like "is " or " a " has too high a coincidental-match rate to be load-bearing.
"""


def message_template_to_substring(template: str) -> str:
    """Extract the longest static fragment from a ``message_template``.

    The corpus's ``message_template`` field is a Python format-string with
    placeholders like ``{declared_value}`` filled in at raise time. To
    compare it against a raised exception's ``str()``, we drop placeholders
    and pick the longest contiguous static run as the substring to match.

    The miner sometimes records the AST literal verbatim, so the template
    may arrive wrapped in ``f'...'`` / ``f"..."`` quoting. We strip the
    f-string prefix + trailing quote before extracting fragments so the
    longest-static-run heuristic doesn't pick up the leading ``f'``.

    Returns the empty string if the template has no static content
    longer than :data:`_MIN_STATIC_FRAGMENT_LEN` characters - the caller
    should treat this as ``message_template_too_dynamic`` and skip the
    substring check (recording a divergence so the invariant's author knows
    the template is too placeholder-heavy to verify).

    Examples
    --------
    >>> message_template_to_substring("`{flag}` is set to `{value}` but ...")
    '` is set to `'
    >>> message_template_to_substring("Invalid `cache_implementation` ({val}). Choose one of: ...")
    'Invalid `cache_implementation` ('
    >>> message_template_to_substring("{a}{b}")
    ''
    >>> message_template_to_substring("f'Greedy methods do not support {x}.'")
    'Greedy methods do not support '
    """
    if not template:
        return ""
    normalised = _strip_fstring_quoting(template)
    fragments = _PLACEHOLDER_RE.split(normalised)
    longest = max(fragments, key=len, default="")
    if len(longest.strip()) < _MIN_STATIC_FRAGMENT_LEN:
        return ""
    return longest


def _strip_fstring_quoting(template: str) -> str:
    """Strip a leading ``f'`` / ``f"`` and matching trailing quote, if present.

    Some corpus invariants record the AST source literal rather than the
    runtime format-string. ``"f'Greedy methods do not support {x}.'"``
    becomes ``"Greedy methods do not support {x}."`` so the placeholder
    splitter can do its job.
    """
    stripped = template.strip()
    if len(stripped) >= 3 and stripped[:2] in ('f"', "f'") and stripped[-1] == stripped[1]:
        return stripped[2:-1]
    return template


def message_matches_template(observed_message: str, template: str) -> tuple[bool, str]:
    """Check whether ``observed_message`` contains the static fragment of ``template``.

    Returns ``(matched, fragment)``. When the template is too dynamic to
    extract a useful fragment, returns ``(False, "")`` and the caller
    should record a ``message_template_too_dynamic`` divergence rather
    than a substring-mismatch one.

    Comparison is case-insensitive - corpus templates and runtime exception
    messages occasionally differ in capitalisation of opening words.
    """
    fragment = message_template_to_substring(template)
    if not fragment:
        return False, ""
    return fragment.lower() in (observed_message or "").lower(), fragment


# ---------------------------------------------------------------------------
# Locus attribution + type-coercion-artefact rejection (chunk C4 gate hardening)
# ---------------------------------------------------------------------------
#
# Two false-confirm classes the mining-substrate study hit (carry-forward audit
# D2). Both are guarded here against the structured ``error_details`` captured
# by :func:`run_case`, never against bare message substrings.


# pydantic machine error codes for value-parsing/coercion (D2 pitfall 2). When
# the positive probe raised one of these AND the invariant is not itself a
# type-check, the "confirm" is a coercion artefact, not the claimed rule firing.
_PARSING_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "int_parsing",
        "float_parsing",
        "bool_parsing",
        "decimal_parsing",
        # msgspec's synthesised parse label (see extract_error_details).
        "msgspec_parsing",
    }
)

# ``literal_error`` is a coercion artefact only when the predicate it guards is
# numeric-labelled - a Literal allowlist on a genuinely categorical field is a
# real rule. The caller passes whether the predicate is numeric.
_LITERAL_ERROR_TYPE = "literal_error"


def invariant_claimed_fields(invariant: dict[str, Any]) -> frozenset[str]:
    """Return the bare field names an invariant claims to govern.

    Reads ``match.fields`` (dotted keys like ``transformers.sampling.num_beams``)
    and returns the last dotted segment of each - the live library raises errors
    keyed on the bare field name (``num_beams``), so that is the grain to match
    against the captured error locus.
    """
    match = invariant.get("match")
    if not isinstance(match, dict):
        return frozenset()
    fields = match.get("fields")
    if not isinstance(fields, dict):
        return frozenset()
    return frozenset(str(key).rsplit(".", 1)[-1] for key in fields)


_TYPE_CHECK_MESSAGE_MARKERS: tuple[str, ...] = (
    "must be an instance of",
    "must be a boolean",
    "must be of type",
    "must be an int",
    "must be a float",
    "must be a str",
    "must be a string",
    "expected type",
    "is not a valid",
)


def is_type_check_invariant(invariant: dict[str, Any]) -> bool:
    """True iff the invariant itself asserts a field's *type* (not a value rule).

    Type-check invariants legitimately fire a parsing/coercion error on their
    positive probe - that IS the behaviour under test - so the coercion guard
    (:func:`is_type_coercion_artifact`) must exempt them. Detection is
    structural: the corpus may set ``type_check: true`` explicitly; otherwise we
    infer from the id segment ``_type_`` and the type-assertion phrasing in the
    ``message_template`` / ``invariant_under_test``.
    """
    if bool(invariant.get("type_check", False)):
        return True
    invariant_id = str(invariant.get("id", ""))
    if "_type_" in invariant_id:
        return True
    haystack = (
        str(invariant.get("message_template") or "")
        + " "
        + str(invariant.get("invariant_under_test") or "")
    ).lower()
    return any(marker in haystack for marker in _TYPE_CHECK_MESSAGE_MARKERS)


def locus_confirms_invariant(
    claimed_fields: Iterable[str], error_details: Iterable[ErrorDetail]
) -> bool:
    """True iff the captured error locus matches the invariant's claimed fields.

    Decision (audit D2 pitfall 1): a confirm must be attributed to the rule
    whose field(s) the raised error actually concerns. An incidental error
    (e.g. a Literal allowlist violation on field X firing before the claimed
    cross-field validator on fields Y/Z) must NOT confirm the claimed rule.

    Matching rule:

    - No captured locus (``error_details`` empty) -> ``True``. We never *block*
      a confirm purely because no structured locus was recoverable; locus
      attribution only *refines* when evidence is present (e.g. transformers'
      ``GenerationConfig.validate`` composes a single ValueError with no
      per-field loc, where the existing message-template check still applies).
    - No claimed fields (invariant omits ``match.fields``) -> ``True``. Nothing
      to attribute against; defer to the other gate checks.
    - Otherwise: at least one captured locus element must intersect the claimed
      fields. A cross-field rule claiming {Y, Z} is confirmed only if the error
      touches Y or Z, never if it touches an unrelated X.
    """
    claimed = frozenset(claimed_fields)
    if not claimed:
        return True
    details = tuple(error_details)
    if not details:
        return True
    error_fields = {part for detail in details for part in detail.loc}
    return bool(error_fields & claimed)


def is_type_coercion_artifact(
    error_details: Iterable[ErrorDetail],
    *,
    is_type_check: bool,
    numeric_predicate: bool,
) -> bool:
    """True iff a confirm should be rejected as a type-coercion artefact.

    Decision (audit D2 pitfall 2): reject a lenient/recall-mode confirm whose
    positive probe raised a pydantic PARSING error (int/float/bool/decimal
    parsing, or msgspec's parse equivalent), or a ``literal_error`` on a
    numeric-labelled predicate - UNLESS the invariant under test is itself a
    type-check rule (then the parse error is the intended behaviour).

    ``numeric_predicate`` distinguishes a ``literal_error`` that fired because a
    numeric value missed a numeric Literal allowlist (a coercion artefact) from
    one on a genuinely categorical field (a real rule). The caller derives this
    from the claimed field's declared predicate.
    """
    if is_type_check:
        return False
    for detail in error_details:
        if detail.error_type in _PARSING_ERROR_TYPES:
            return True
        if detail.error_type == _LITERAL_ERROR_TYPE and numeric_predicate:
            return True
    return False
