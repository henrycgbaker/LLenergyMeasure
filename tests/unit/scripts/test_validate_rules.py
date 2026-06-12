"""Tests for :mod:`scripts.validate_rules` and :mod:`scripts._rules_validation_common`.

The test strategy: exercise the validate_rules script via synthetic native types -
we construct small Pydantic / dataclass / ``__slots__`` fixtures and point
the validation step at them. Tests that touch the real transformers library
live in the workflow-smoke integration test; unit tests stay deterministic
and fast.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import _rules_validation_common, validate_rules  # noqa: E402
from scripts._rules_validation_common import (  # noqa: E402
    CaptureBuffers,
    ErrorDetail,
    classify_emission_channel,
    classify_outcome,
    compare_expected_vs_observed,
    diff_input_vs_state,
    extract_error_details,
    extract_state,
    invariant_claimed_fields,
    is_type_check_invariant,
    is_type_coercion_artifact,
    locus_confirms_invariant,
    message_matches_template,
    message_template_to_substring,
    run_case,
)

# ---------------------------------------------------------------------------
# Synthetic native types for fixture-based testing
# ---------------------------------------------------------------------------


@dataclass
class _DataclassConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    _private: int = 0


class _SlotsConfig:
    __slots__ = ("_internal", "alpha", "beta")

    def __init__(self, alpha: int = 1, beta: str = "x", _internal: bool = False) -> None:
        self.alpha = alpha
        self.beta = beta
        self._internal = _internal


class _DictConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@dataclass
class _NormalisingConfig:
    """Dataclass that silently strips ``temperature`` when ``do_sample=False``."""

    do_sample: bool = True
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.do_sample and self.temperature != 1.0:
            self.temperature = 1.0


# ---------------------------------------------------------------------------
# extract_state
# ---------------------------------------------------------------------------


class TestExtractState:
    def test_dataclass(self) -> None:
        obj = _DataclassConfig(temperature=0.5)
        state = extract_state(obj)
        assert state == {"temperature": 0.5, "top_p": 1.0}
        assert "_private" not in state

    def test_slots(self) -> None:
        obj = _SlotsConfig(alpha=2, beta="y", _internal=True)
        state = extract_state(obj)
        assert state["alpha"] == 2
        assert state["beta"] == "y"
        assert "_internal" not in state

    def test_dict_class(self) -> None:
        obj = _DictConfig(foo=1, bar="baz", _hidden=99)
        state = extract_state(obj)
        assert state["foo"] == 1
        assert state["bar"] == "baz"
        assert "_hidden" not in state

    def test_private_allowlist(self) -> None:
        obj = _DictConfig(foo=1, _commit_hash="abc123")
        state = extract_state(obj, private_allowlist={"_commit_hash"})
        assert state["_commit_hash"] == "abc123"


# ---------------------------------------------------------------------------
# diff_input_vs_state
# ---------------------------------------------------------------------------


class TestDiffInputVsState:
    def test_no_diff(self) -> None:
        kwargs = {"a": 1, "b": 2}
        state = {"a": 1, "b": 2, "c": 3}
        assert diff_input_vs_state(kwargs, state) == {}

    def test_silent_normalisation(self) -> None:
        kwargs = {"temperature": 0.9}
        state = {"temperature": 1.0}
        diffs = diff_input_vs_state(kwargs, state)
        assert diffs == {"temperature": {"declared": 0.9, "observed": 1.0}}

    def test_missing_from_state_ignored(self) -> None:
        kwargs = {"absent": "value"}
        state = {"other": "thing"}
        assert diff_input_vs_state(kwargs, state) == {}


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------


class TestRunCase:
    def test_captures_exception(self) -> None:
        def boom() -> None:
            raise ValueError("nope")

        buf = run_case(boom)
        assert buf.exception_type == "ValueError"
        assert buf.exception_message == "nope"
        assert buf.observed_state is None

    def test_captures_warnings(self) -> None:
        import warnings

        def warner() -> _DataclassConfig:
            warnings.warn("heads up", UserWarning, stacklevel=2)
            return _DataclassConfig()

        buf = run_case(warner)
        assert buf.exception_type is None
        assert any("heads up" in str(w) for w in buf.warnings_captured)

    def test_captures_state(self) -> None:
        def ok() -> _DataclassConfig:
            return _DataclassConfig(temperature=0.7)

        buf = run_case(ok)
        assert buf.exception_type is None
        assert buf.observed_state is not None
        assert buf.observed_state["temperature"] == 0.7

    def test_captures_logger_output(self) -> None:
        import logging

        logger_name = "llenergymeasure_test_validate_rules_capture"

        def emitter() -> _DataclassConfig:
            logging.getLogger(logger_name).warning("observed emission")
            return _DataclassConfig()

        buf = run_case(emitter, logger_names=(logger_name,))
        assert any("observed emission" in m for m in buf.logger_messages)

    def test_preserves_warnings_when_call_raises(self) -> None:
        # Dormant-then-raise paths (e.g. deprecation warning followed by a
        # strict-mode ValueError) must preserve the warning alongside the
        # exception - both are the invariant's fingerprint.
        import warnings

        def warn_then_raise() -> None:
            warnings.warn("about to fail", UserWarning, stacklevel=2)
            raise ValueError("strict mode")

        buf = run_case(warn_then_raise)
        assert buf.exception_type == "ValueError"
        assert buf.exception_message == "strict mode"
        assert any("about to fail" in str(w) for w in buf.warnings_captured)


# ---------------------------------------------------------------------------
# classify_outcome / classify_emission_channel
# ---------------------------------------------------------------------------


class TestClassify:
    def test_error_on_exception(self) -> None:
        buf = _rules_validation_common.CaptureBuffers(
            exception_type="ValueError",
            exception_message="x",
            warnings_captured=(),
            logger_messages=(),
            observed_state=None,
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "error"
        assert classify_emission_channel(buf) == "none"

    def test_warn_on_captured_warning(self) -> None:
        buf = _rules_validation_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=("heads up",),
            logger_messages=(),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "warn"
        assert classify_emission_channel(buf) == "warnings_warn"

    def test_dormant_announced_on_logger_only(self) -> None:
        buf = _rules_validation_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=("silent normalisation",),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "dormant_announced"
        assert classify_emission_channel(buf) == "logger_warning"

    def test_logger_warning_once_classified_when_sentinel_present(self) -> None:
        # Any sentinel-tagged line upgrades the classification from
        # logger_warning to logger_warning_once - the dedup-wrapped form is
        # the stricter claim on user visibility.
        sentinel = _rules_validation_common._WARNING_ONCE_SENTINEL
        buf = _rules_validation_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=(f"{sentinel}one-shot warning from HF", "regular warning"),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_emission_channel(buf) == "logger_warning_once"

    def test_strip_warning_once_sentinel_cleans_messages(self) -> None:
        sentinel = _rules_validation_common._WARNING_ONCE_SENTINEL
        messages = (f"{sentinel}deprecated kwarg", "plain log")
        cleaned = _rules_validation_common.strip_warning_once_sentinel(messages)
        assert cleaned == ("deprecated kwarg", "plain log")
        assert all(sentinel not in m for m in cleaned)

    def test_dormant_silent_on_state_change_only(self) -> None:
        buf = _rules_validation_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=(),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_outcome(buf, {"a": {"declared": 2, "observed": 1}}) == "dormant_silent"

    def test_no_op_when_nothing_observed(self) -> None:
        buf = _rules_validation_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=(),
            observed_state={},
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "no_op"


# ---------------------------------------------------------------------------
# compare_expected_vs_observed
# ---------------------------------------------------------------------------


class TestCompareExpectedVsObserved:
    def test_exact_match_no_divergence(self) -> None:
        divergences = compare_expected_vs_observed(
            invariant_id="r",
            expected={"outcome": "error", "emission_channel": "none"},
            observed_outcome="error",
            observed_emission="none",
            silent_normalisations={},
        )
        assert divergences == []

    def test_outcome_mismatch(self) -> None:
        divergences = compare_expected_vs_observed(
            invariant_id="r",
            expected={"outcome": "error"},
            observed_outcome="warn",
            observed_emission="warnings_warn",
            silent_normalisations={},
        )
        assert len(divergences) == 1
        assert divergences[0].field == "outcome"

    def test_normalised_fields_mismatch(self) -> None:
        divergences = compare_expected_vs_observed(
            invariant_id="r",
            expected={"outcome": "dormant_silent", "normalised_fields": ["x", "y"]},
            observed_outcome="dormant_silent",
            observed_emission="none",
            silent_normalisations={"x": {"declared": 1, "observed": 0}},
        )
        assert any(d.field == "normalised_fields" for d in divergences)


# ---------------------------------------------------------------------------
# _validate_invariant_with_captures - end-to-end on a synthetic corpus
# ---------------------------------------------------------------------------


class TestVendorRuleSynthetic:
    """Exercise ``_validate_invariant_with_captures`` via a synthetic engine runner.

    We monkeypatch the transformers runner to point at our synthetic configs.
    This covers the full validation loop without needing transformers installed.
    """

    @pytest.fixture
    def patched_runner(self, monkeypatch: pytest.MonkeyPatch):
        def synthetic_runner(
            native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
        ) -> _rules_validation_common.CaptureBuffers:
            if native_type == "test.raises":
                return run_case(lambda: (_ for _ in ()).throw(ValueError("expected")))
            if native_type == "test.normalises":
                return run_case(lambda: _NormalisingConfig(**kwargs))
            return run_case(lambda: _DataclassConfig(**kwargs))

        monkeypatch.setitem(validate_rules._ENGINE_RUNNERS, "transformers", synthetic_runner)
        return synthetic_runner

    def test_error_rule_positive_confirmed(self, patched_runner: Any) -> None:
        invariant = {
            "id": "test_raises",
            "severity": "error",
            "native_type": "test.raises",
            "kwargs_positive": {},
            "kwargs_negative": {},
            "expected_outcome": {"outcome": "error", "emission_channel": "none"},
        }
        result, _pos, _neg = validate_rules._validate_invariant_with_captures(
            "transformers", invariant
        )
        assert result.outcome == "error"
        assert result.positive_confirmed is True
        assert result.observed_exception is not None
        assert result.observed_exception["type"] == "ValueError"

    def test_dormant_silent_detected(self, patched_runner: Any) -> None:
        invariant = {
            "id": "test_normalises",
            "severity": "dormant",
            "native_type": "test.normalises",
            "kwargs_positive": {"do_sample": False, "temperature": 0.9},
            "kwargs_negative": {"do_sample": True, "temperature": 0.9},
            "expected_outcome": {
                "outcome": "dormant_silent",
                "emission_channel": "none",
                "normalised_fields": ["temperature"],
            },
        }
        result, _pos, _neg = validate_rules._validate_invariant_with_captures(
            "transformers", invariant
        )
        assert result.outcome == "dormant_silent"
        assert "temperature" in result.observed_silent_normalisations


# ---------------------------------------------------------------------------
# envelope writing
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_assemble_writes_expected_keys(self) -> None:
        envelope = validate_rules.assemble_envelope(
            engine="transformers",
            engine_version="4.56.0",
            image_ref="test:latest",
            base_image_ref="test:latest",
            validation_commit="abc",
            cases=[],
            divergences=[],
        )
        assert envelope["schema_version"] == "1.0.0"
        assert envelope["engine"] == "transformers"
        assert "validated_at" in envelope
        assert envelope["cases"] == []
        assert envelope["divergences"] == []

    def test_validate_engine_writes_envelope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_path = tmp_path / "t.yaml"
        corpus_path.write_text(
            "schema_version: 1.0.0\nengine: transformers\nengine_version: 4.56.0\n"
            "mined_at: 2026-01-01T00:00:00Z\n"
            "invariants: []\n"
        )
        out_path = tmp_path / "t.validated.yaml"

        monkeypatch.setattr(validate_rules, "_resolve_engine_version", lambda _e: "test-ver")

        envelope, divergences = validate_rules.validate_engine(
            engine="transformers",
            corpus_path=corpus_path,
            out_path=out_path,
        )
        assert out_path.exists()
        assert envelope["engine_version"] == "test-ver"
        assert divergences == []
        written = yaml.safe_load(out_path.read_text())
        assert written["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_main_exits_2_on_missing_corpus(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    out = tmp_path / "out.validated.yaml"
    exit_code = validate_rules.main(
        [
            "--engine",
            "transformers",
            "--corpus",
            str(missing),
            "--out",
            str(out),
        ]
    )
    assert exit_code == 2


def test_main_exits_0_on_no_divergence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_path = tmp_path / "t.yaml"
    corpus_path.write_text(
        "schema_version: 1.0.0\nengine: transformers\nengine_version: 4.56.0\n"
        "mined_at: 2026-01-01T00:00:00Z\ninvariants: []\n"
    )
    out_path = tmp_path / "t.validated.yaml"

    monkeypatch.setattr(validate_rules, "_resolve_engine_version", lambda _e: "test-ver")

    exit_code = validate_rules.main(
        [
            "--engine",
            "transformers",
            "--corpus",
            str(corpus_path),
            "--out",
            str(out_path),
            "--fail-on-divergence",
        ]
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# message_template_to_substring + message_matches_template
# ---------------------------------------------------------------------------


class TestMessageTemplateSubstring:
    def test_simple_placeholder_drop(self) -> None:
        assert (
            message_template_to_substring("`{flag}` is set to `{value}` but ...") == "` is set to `"
        )

    def test_picks_longest_static_run(self) -> None:
        assert (
            message_template_to_substring("Invalid `cache_implementation` ({val}). Choose one of:")
            == "Invalid `cache_implementation` ("
        )

    def test_only_placeholders_returns_empty(self) -> None:
        assert message_template_to_substring("{a}{b}") == ""

    def test_below_min_length_returns_empty(self) -> None:
        # "is" has only 2 non-whitespace chars - below the floor.
        assert message_template_to_substring("{a} is {b}") == ""

    def test_empty_template_returns_empty(self) -> None:
        assert message_template_to_substring("") == ""

    def test_strips_fstring_quoting_single_quote(self) -> None:
        assert (
            message_template_to_substring("f'Greedy methods do not support {x}.'")
            == "Greedy methods do not support "
        )

    def test_strips_fstring_quoting_double_quote(self) -> None:
        assert (
            message_template_to_substring('f"Greedy methods do not support {x}."')
            == "Greedy methods do not support "
        )

    def test_no_placeholders_returns_full_template(self) -> None:
        assert (
            message_template_to_substring("bnb_4bit_compute_dtype must be torch.dtype")
            == "bnb_4bit_compute_dtype must be torch.dtype"
        )


class TestMessageMatchesTemplate:
    def test_substring_match_case_insensitive(self) -> None:
        matched, fragment = message_matches_template(
            "INVALID `cache_implementation` (got 'foo'). Choose one of: ...",
            "Invalid `cache_implementation` ({val}). Choose one of: ...",
        )
        assert matched is True
        assert "cache_implementation" in fragment

    def test_no_match(self) -> None:
        matched, fragment = message_matches_template(
            "Some unrelated runtime message.",
            "Invalid `cache_implementation` ({val}). Choose one of: ...",
        )
        assert matched is False
        assert fragment != ""

    def test_too_dynamic_template(self) -> None:
        matched, fragment = message_matches_template("anything", "{a}{b}")
        assert matched is False
        assert fragment == ""

    def test_empty_observed_message(self) -> None:
        matched, _ = message_matches_template("", "expected fragment here")
        assert matched is False


# ---------------------------------------------------------------------------
# compute_gate_soundness_divergences
# ---------------------------------------------------------------------------


def _capture(
    *,
    exception_type: str | None = None,
    exception_message: str | None = None,
    warnings_captured: tuple[str, ...] = (),
    logger_messages: tuple[str, ...] = (),
    observed_state: dict[str, Any] | None = None,
    error_details: tuple[ErrorDetail, ...] = (),
) -> CaptureBuffers:
    """Convenience constructor for synthetic capture buffers."""
    return CaptureBuffers(
        exception_type=exception_type,
        exception_message=exception_message,
        warnings_captured=warnings_captured,
        logger_messages=logger_messages,
        observed_state=observed_state,
        duration_ms=0,
        error_details=error_details,
    )


class TestComputeGateSoundnessDivergences:
    """Decision #12 of the invariant-miner adversarial review."""

    def test_clean_error_rule_no_divergence(self) -> None:
        invariant = {
            "id": "r1",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(exception_type="ValueError", exception_message="field `a` must be non-zero")
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert divergences == []

    def test_positive_did_not_raise_for_error_severity(self) -> None:
        invariant = {
            "id": "r2",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "irrelevant",
        }
        pos = _capture(observed_state={"a": 1})  # construction succeeded
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(d.check_failed == validate_rules.CHECK_POSITIVE_RAISES for d in divergences)

    def test_dormant_severity_accepts_warning(self) -> None:
        invariant = {
            "id": "r3",
            "severity": "dormant",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "use `a=0` for stability",
        }
        pos = _capture(logger_messages=("use `a=0` for stability",))
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        # Dormant invariant fired (logger.warning) - no positive_raises divergence.
        assert all(d.check_failed != validate_rules.CHECK_POSITIVE_RAISES for d in divergences)

    def test_dormant_severity_no_op_is_divergence(self) -> None:
        invariant = {
            "id": "r4",
            "severity": "dormant",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "use `a=0` for stability",
        }
        pos = _capture(observed_state={"a": 1})  # nothing fired
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(d.check_failed == validate_rules.CHECK_POSITIVE_RAISES for d in divergences)

    def test_message_template_match_failure(self) -> None:
        invariant = {
            "id": "r5",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(exception_type="ValueError", exception_message="totally different message")
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(
            d.check_failed == validate_rules.CHECK_MESSAGE_TEMPLATE_MATCH for d in divergences
        )

    def test_message_template_too_dynamic(self) -> None:
        invariant = {
            "id": "r6",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "{a}{b}",
        }
        pos = _capture(exception_type="ValueError", exception_message="anything")
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(
            d.check_failed == validate_rules.CHECK_MESSAGE_TEMPLATE_TOO_DYNAMIC for d in divergences
        )

    def test_negative_raised_unexpectedly(self) -> None:
        invariant = {
            "id": "r7",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(exception_type="ValueError", exception_message="field `a` must be non-zero")
        neg = _capture(exception_type="TypeError", exception_message="oops")
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(
            d.check_failed == validate_rules.CHECK_NEGATIVE_DOES_NOT_RAISE for d in divergences
        )

    def test_divergence_dict_includes_check_failed_field(self) -> None:
        invariant = {
            "id": "r8",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(observed_state={"a": 1})  # no raise - should trip positive_raises
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        d = divergences[0].as_dict()
        assert "check_failed" in d
        assert d["check_failed"] == validate_rules.CHECK_POSITIVE_RAISES
        assert d["invariant_id"] == "r8"
        assert d["field"] == "kwargs_positive"


# ---------------------------------------------------------------------------
# extract_error_details - pydantic / plain (chunk C4)
# ---------------------------------------------------------------------------


class TestExtractErrorDetails:
    def test_pydantic_validation_error_structured_loc(self) -> None:
        pydantic = pytest.importorskip("pydantic")

        class M(pydantic.BaseModel):
            x: int = 0

        try:
            M(x="not-an-int")
        except pydantic.ValidationError as exc:
            details = extract_error_details(exc)
        assert len(details) == 1
        assert details[0].loc == ("x",)
        assert details[0].error_type == "int_parsing"

    def test_plain_raise_backtick_field_locus(self) -> None:
        details = extract_error_details(ValueError("`num_beams` is greater than 1"))
        assert details == (ErrorDetail(loc=("num_beams",), error_type="plain"),)

    def test_plain_raise_no_backtick_yields_empty(self) -> None:
        assert extract_error_details(ValueError("something went wrong")) == ()


# ---------------------------------------------------------------------------
# invariant_claimed_fields + is_type_check_invariant
# ---------------------------------------------------------------------------


class TestClaimedFields:
    def test_extracts_bare_field_names_from_match(self) -> None:
        invariant = {
            "match": {"engine": "x", "fields": {"transformers.sampling.num_beams": 1}},
        }
        assert invariant_claimed_fields(invariant) == frozenset({"num_beams"})

    def test_cross_field_multiple(self) -> None:
        invariant = {"match": {"fields": {"e.s.a": 1, "e.s.b": 2}}}
        assert invariant_claimed_fields(invariant) == frozenset({"a", "b"})

    def test_no_match_block_empty(self) -> None:
        assert invariant_claimed_fields({}) == frozenset()


class TestIsTypeCheckInvariant:
    def test_explicit_flag(self) -> None:
        assert is_type_check_invariant({"type_check": True}) is True

    def test_id_segment(self) -> None:
        assert is_type_check_invariant({"id": "transformers_compile_config_type_foo"}) is True

    def test_message_marker(self) -> None:
        assert is_type_check_invariant(
            {"id": "x", "message_template": "you provided it but it must be an instance of Y"}
        )

    def test_value_rule_not_type_check(self) -> None:
        assert (
            is_type_check_invariant({"id": "x", "message_template": "`a` must be non-zero"})
            is False
        )


# ---------------------------------------------------------------------------
# locus_confirms_invariant (D1) + is_type_coercion_artifact (D2) - unit
# ---------------------------------------------------------------------------


class TestLocusConfirms:
    def test_intersecting_locus_confirms(self) -> None:
        details = (ErrorDetail(loc=("num_beams",), error_type="plain"),)
        assert locus_confirms_invariant({"num_beams"}, details) is True

    def test_incidental_error_does_not_confirm_cross_field(self) -> None:
        # Claimed cross-field rule {a, b}; error fired on unrelated `mode`.
        details = (ErrorDetail(loc=("mode",), error_type="literal_error"),)
        assert locus_confirms_invariant({"a", "b"}, details) is False

    def test_empty_details_is_permissive(self) -> None:
        # No structured locus recoverable (composed ValueError) - do not block.
        assert locus_confirms_invariant({"a"}, ()) is True

    def test_no_claimed_fields_is_permissive(self) -> None:
        details = (ErrorDetail(loc=("x",), error_type="plain"),)
        assert locus_confirms_invariant(set(), details) is True


class TestIsTypeCoercionArtifact:
    def test_int_parsing_is_artifact(self) -> None:
        details = (ErrorDetail(loc=("a",), error_type="int_parsing"),)
        assert is_type_coercion_artifact(details, is_type_check=False, numeric_predicate=False)

    def test_type_check_invariant_exempt(self) -> None:
        details = (ErrorDetail(loc=("a",), error_type="int_parsing"),)
        assert not is_type_coercion_artifact(details, is_type_check=True, numeric_predicate=False)

    def test_literal_error_numeric_predicate_is_artifact(self) -> None:
        details = (ErrorDetail(loc=("a",), error_type="literal_error"),)
        assert is_type_coercion_artifact(details, is_type_check=False, numeric_predicate=True)

    def test_literal_error_categorical_predicate_kept(self) -> None:
        # Literal on a categorical field is a real rule, not a coercion artefact.
        details = (ErrorDetail(loc=("mode",), error_type="literal_error"),)
        assert not is_type_coercion_artifact(details, is_type_check=False, numeric_predicate=False)

    def test_greater_than_is_not_artifact(self) -> None:
        # A genuine bound violation is the rule firing, not coercion.
        details = (ErrorDetail(loc=("a",), error_type="greater_than"),)
        assert not is_type_coercion_artifact(details, is_type_check=False, numeric_predicate=False)


# ---------------------------------------------------------------------------
# compute_gate_soundness_divergences - locus + coercion integration (C4)
# ---------------------------------------------------------------------------


class TestGateSoundnessLocusAndCoercion:
    def test_locus_mismatch_recorded(self) -> None:
        # Cross-field rule claims {a, b} but the raise concerns `mode`.
        invariant = {
            "id": "cross",
            "severity": "error",
            "match": {"fields": {"e.a": 1, "e.b": 2}},
            "kwargs_positive": {"a": 1, "b": 2},
            "kwargs_negative": {"a": 0, "b": 0},
            "message_template": "incompatible `a` and `b`",
        }
        pos = _capture(
            exception_type="ValidationError",
            exception_message="bad mode",
            error_details=(ErrorDetail(loc=("mode",), error_type="literal_error"),),
        )
        neg = _capture(observed_state={"a": 0, "b": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(d.check_failed == validate_rules.CHECK_LOCUS_MISMATCH for d in divergences)

    def test_matching_locus_no_locus_divergence(self) -> None:
        invariant = {
            "id": "ok",
            "severity": "error",
            "match": {"fields": {"e.a": 1}},
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "incompatible `a`",
        }
        pos = _capture(
            exception_type="ValueError",
            exception_message="incompatible `a`",
            error_details=(ErrorDetail(loc=("a",), error_type="plain"),),
        )
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert all(d.check_failed != validate_rules.CHECK_LOCUS_MISMATCH for d in divergences)

    def test_coercion_artifact_recorded_for_value_rule(self) -> None:
        # Value rule (not a type-check) whose positive probe tripped int_parsing.
        invariant = {
            "id": "val",
            "severity": "error",
            "match": {"fields": {"e.a": 5}},
            "kwargs_positive": {"a": 5},
            "kwargs_negative": {"a": 0},
            "message_template": "`a` must be positive",
        }
        pos = _capture(
            exception_type="ValidationError",
            exception_message="int_parsing",
            error_details=(ErrorDetail(loc=("a",), error_type="int_parsing"),),
        )
        neg = _capture(observed_state={"a": 0})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert any(
            d.check_failed == validate_rules.CHECK_TYPE_COERCION_ARTIFACT for d in divergences
        )

    def test_coercion_artifact_exempt_for_type_check_invariant(self) -> None:
        invariant = {
            "id": "transformers_compile_config_type_check",
            "severity": "error",
            "match": {"fields": {"e.compile_config": 1}},
            "kwargs_positive": {"compile_config": 1},
            "kwargs_negative": {"compile_config": {}},
            "message_template": "must be an instance of CompileConfig",
        }
        pos = _capture(
            exception_type="ValidationError",
            exception_message="must be an instance",
            error_details=(ErrorDetail(loc=("compile_config",), error_type="int_parsing"),),
        )
        neg = _capture(observed_state={"compile_config": {}})
        divergences = validate_rules.compute_gate_soundness_divergences(invariant, pos, neg)
        assert all(
            d.check_failed != validate_rules.CHECK_TYPE_COERCION_ARTIFACT for d in divergences
        )


# ---------------------------------------------------------------------------
# Carried-catalogue re-gate (decay alarm, design § 6 signal 1) - chunk C4
# ---------------------------------------------------------------------------


def _carried_corpus(tmp_path: Path, invariants: list[dict[str, Any]]) -> Path:
    """Write a carried (proposed-shape) corpus to a temp file and return its path."""
    path = tmp_path / "carried.proposed.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0.0",
                "engine": "transformers",
                "engine_version": "old-ver",
                "mined_at": "2026-01-01T00:00:00Z",
                "invariants": invariants,
            },
            sort_keys=False,
        )
    )
    return path


def _confirmed_invariant() -> dict[str, Any]:
    return {
        "id": "still_holds",
        "severity": "error",
        "native_type": "test.raises",
        "match": {"fields": {"e.a": 1}},
        "kwargs_positive": {"a": 1},
        "kwargs_negative": {"a": 0},
        "expected_outcome": {"outcome": "error"},
        "message_template": "`a` must differ",
    }


class TestCarriedRegate:
    """Stub-engine fixtures proving the three verdict classes and report shape."""

    @pytest.fixture
    def patched_engine(self, monkeypatch: pytest.MonkeyPatch):
        """Synthetic runner + pinned engine version for deterministic re-gate."""

        def synthetic_runner(
            native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
        ) -> CaptureBuffers:
            if native_type == "test.raises":
                # Positive (a=1) raises citing `a`; negative (a=0) constructs.
                if kwargs.get("a"):
                    return run_case(
                        lambda: (_ for _ in ()).throw(ValueError("`a` must differ from 1"))
                    )
                return run_case(lambda: _DataclassConfig())
            if native_type == "test.no_longer_fires":
                # Rule decayed: positive no longer raises (constructs cleanly).
                return run_case(lambda: _DataclassConfig())
            if native_type == "test.unresolvable":
                # Constructor drift: native_type will not resolve.
                from scripts._engine_constructors import NativeTypeResolutionError

                return run_case(
                    lambda: (_ for _ in ()).throw(
                        NativeTypeResolutionError("class vanished post-refactor")
                    )
                )
            return run_case(lambda: _DataclassConfig(**kwargs))

        monkeypatch.setitem(validate_rules._ENGINE_RUNNERS, "transformers", synthetic_runner)
        monkeypatch.setattr(validate_rules, "_resolve_engine_version", lambda _e: "new-ver")
        return synthetic_runner

    def test_confirmed_verdict(self, tmp_path: Path, patched_engine: Any) -> None:
        corpus = _carried_corpus(tmp_path, [_confirmed_invariant()])
        report = validate_rules.regate_carried_catalogue(
            engine="transformers", carried_corpus_path=corpus
        )
        assert report["engine_version"] == "new-ver"
        assert report["total"] == 1
        assert report["counts"][validate_rules.VERDICT_CONFIRMED] == 1
        assert report["acceptance_rate"] == 1.0
        assert report["entries"][0]["verdict"] == validate_rules.VERDICT_CONFIRMED

    def test_failed_verdict_rule_decayed(self, tmp_path: Path, patched_engine: Any) -> None:
        decayed = {
            "id": "decayed",
            "severity": "error",
            "native_type": "test.no_longer_fires",
            "match": {"fields": {"e.a": 1}},
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "expected_outcome": {"outcome": "error"},
            "message_template": "`a` must differ",
        }
        corpus = _carried_corpus(tmp_path, [decayed])
        report = validate_rules.regate_carried_catalogue(
            engine="transformers", carried_corpus_path=corpus
        )
        assert report["counts"][validate_rules.VERDICT_FAILED] == 1
        assert report["entries"][0]["verdict"] == validate_rules.VERDICT_FAILED
        assert "no longer fires" in report["entries"][0]["reason"]

    def test_infra_error_verdict_unresolvable(self, tmp_path: Path, patched_engine: Any) -> None:
        unresolved = {
            "id": "gone",
            "severity": "error",
            "native_type": "test.unresolvable",
            "match": {"fields": {"e.a": 1}},
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "expected_outcome": {"outcome": "error"},
            "message_template": "`a` must differ",
        }
        corpus = _carried_corpus(tmp_path, [unresolved])
        report = validate_rules.regate_carried_catalogue(
            engine="transformers", carried_corpus_path=corpus
        )
        assert report["counts"][validate_rules.VERDICT_INFRA_ERROR] == 1
        assert report["entries"][0]["verdict"] == validate_rules.VERDICT_INFRA_ERROR
        assert "unresolved" in report["entries"][0]["reason"]

    def test_mixed_report_shape_and_acceptance_rate(
        self, tmp_path: Path, patched_engine: Any
    ) -> None:
        confirmed = _confirmed_invariant()
        decayed = {**_confirmed_invariant(), "id": "decayed", "native_type": "test.no_longer_fires"}
        gone = {**_confirmed_invariant(), "id": "gone", "native_type": "test.unresolvable"}
        corpus = _carried_corpus(tmp_path, [confirmed, decayed, gone])
        report = validate_rules.regate_carried_catalogue(
            engine="transformers", carried_corpus_path=corpus
        )
        # Stable report shape (consumed by chunk C7's PR comment step).
        assert set(report) == {
            "schema_version",
            "engine",
            "engine_version",
            "carried_corpus",
            "total",
            "counts",
            "acceptance_rate",
            "entries",
        }
        assert report["total"] == 3
        assert report["counts"] == {
            validate_rules.VERDICT_CONFIRMED: 1,
            validate_rules.VERDICT_FAILED: 1,
            validate_rules.VERDICT_INFRA_ERROR: 1,
        }
        # acceptance_rate = confirmed / total; infra_error + failed count against it.
        assert report["acceptance_rate"] == round(1 / 3, 4)
        assert all(set(e) == {"id", "verdict", "reason"} for e in report["entries"])

    def test_cli_carried_mode_writes_report(self, tmp_path: Path, patched_engine: Any) -> None:
        corpus = _carried_corpus(tmp_path, [_confirmed_invariant()])
        out = tmp_path / "regate.json"
        exit_code = validate_rules.main(
            [
                "--engine",
                "transformers",
                "--carried",
                str(corpus),
                "--regate-out",
                str(out),
            ]
        )
        assert exit_code == 0
        assert out.exists()
        written = __import__("json").loads(out.read_text())
        assert written["counts"]["confirmed"] == 1

    def test_cli_carried_rejects_corpus_combo(self, tmp_path: Path) -> None:
        corpus = _carried_corpus(tmp_path, [])
        with pytest.raises(SystemExit):
            validate_rules.main(
                [
                    "--engine",
                    "transformers",
                    "--carried",
                    str(corpus),
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(tmp_path / "o.yaml"),
                ]
            )


# ---------------------------------------------------------------------------
# Rung 0 reconciliation join (decay-alarm triage) - chunk R1
# ---------------------------------------------------------------------------


def _write_corpus(path: Path, invariants: list[dict[str, Any]]) -> Path:
    path.write_text(yaml.safe_dump({"invariants": invariants}, sort_keys=False))
    return path


def test_reconcile_unit_template_drift_and_morph(tmp_path: Path) -> None:
    """Unit join: one same-id template-drift heal, one signature-morph heal, one residual."""
    carried = _write_corpus(
        tmp_path / "carried.yaml",
        [
            {
                "id": "drifted",
                "severity": "error",
                "native_type": "test.X",
                "match": {"fields": {"e.a": 1}},
                "message_template": "old wording for `a`",
            },
            {
                "id": "old_name",
                "severity": "error",
                "native_type": "test.Y",
                "match": {"fields": {"e.b": 1}},
                "message_template": "`b` rule",
            },
            {
                "id": "gone",
                "severity": "error",
                "native_type": "test.Z",
                "match": {"fields": {"e.c": 1}},
                "message_template": "`c` rule",
            },
        ],
    )
    fresh_proposed = _write_corpus(
        tmp_path / "fresh.proposed.yaml",
        [
            {
                "id": "drifted",
                "severity": "error",
                "native_type": "test.X",
                "match": {"fields": {"e.a": 1}},
                "message_template": "new wording for `a`",
            },
            # Same signature as carried `old_name`, renamed.
            {
                "id": "new_name",
                "severity": "error",
                "native_type": "test.Y",
                "match": {"fields": {"e.b": 1}},
                "message_template": "`b` rule reworded",
            },
        ],
    )
    report = {
        "entries": [
            {"id": "drifted", "verdict": "failed", "reason": "template"},
            {"id": "old_name", "verdict": "failed", "reason": "template"},
            {"id": "gone", "verdict": "failed", "reason": "positive no longer fires"},
            {"id": "stable", "verdict": "confirmed", "reason": ""},
        ]
    }
    recon = validate_rules.reconcile_regate_report(
        report,
        carried_corpus_path=carried,
        fresh_validated_ids={"drifted", "new_name"},
        fresh_proposed_path=fresh_proposed,
    )
    assert recon["counts"] == {"healed": 2, "residual": 1}
    by_id = {h["id"]: h for h in recon["healed"]}
    assert by_id["drifted"]["kind"] == validate_rules.HEALED_TEMPLATE_DRIFT
    assert by_id["drifted"]["old_template"] == "old wording for `a`"
    assert by_id["drifted"]["new_template"] == "new wording for `a`"
    assert by_id["old_name"]["kind"] == validate_rules.HEALED_RULE_MORPHED
    assert by_id["old_name"]["new_id"] == "new_name"
    assert recon["residual"][0]["id"] == "gone"


# The bump-1 corpora live in the repo as committed evidence: the carried
# v4.57.3 proposed catalogue and the fresh v5.7.0 proposed + validated corpora.
_TF_V4 = _PROJECT_ROOT / "engine_versions" / "transformers" / "v4_57_3" / "outputs"
_TF_V5 = _PROJECT_ROOT / "engine_versions" / "transformers" / "v5_7_0" / "outputs"


def _bump1_failed_ids() -> list[str]:
    """Derive bump-1's 7 decay-alarm failures from the committed corpora.

    The carried v4.57.3 catalogue is re-gated against the v5.7.0 container; the
    failures are (a) same-id rules whose carried message-template fragment no
    longer appears in the v5.7.0 wording (the gate's message-template soundness
    check fails - template drift), and (b) the carried-only bnb dormant rules
    that the v5.7.0 re-mine dropped entirely (genuine decay). Reconstructed from
    the committed YAML so the test tracks the real corpora, not a frozen literal.
    """
    carried = yaml.safe_load((_TF_V4 / "rules.proposed.yaml").read_text())
    fresh = yaml.safe_load((_TF_V5 / "rules.proposed.yaml").read_text())
    carried_by_id = {i["id"]: i for i in carried["invariants"]}
    fresh_by_id = {i["id"]: i for i in fresh["invariants"]}

    drift_failures = []
    for rule_id in sorted(set(carried_by_id) & set(fresh_by_id)):
        old_t = carried_by_id[rule_id].get("message_template") or ""
        new_t = fresh_by_id[rule_id].get("message_template") or ""
        if old_t == new_t:
            continue
        fragment = message_template_to_substring(old_t)
        if fragment and fragment.lower() not in new_t.lower():
            drift_failures.append(rule_id)

    bnb_failures = [
        rule_id
        for rule_id in (set(carried_by_id) - set(fresh_by_id))
        if "bnb_4bit" in rule_id and "dormant_without_load_in_4bit" in rule_id
    ]
    return drift_failures + sorted(bnb_failures)


def test_reconcile_bump1_yields_four_healed_three_residual() -> None:
    """Replay bump-1 (carried v4.57.3 vs fresh v5.7.0): exactly 4 healed + 3 residual.

    The evidence target from the three-lens review: the deterministic
    reconciliation join turns the decay alarm's "7 failed" into "4 healed +
    3 decay candidates" with zero tokens. The 4 healers are same-id rules the
    v5.7.0 gate re-confirmed under a drifted template; the 3 residual are the
    bnb dormant rules the re-mine genuinely dropped.
    """
    failed_ids = _bump1_failed_ids()
    assert len(failed_ids) == 7, failed_ids

    report = {
        "entries": [
            {"id": rule_id, "verdict": "failed", "reason": "gate-soundness check(s) failed"}
            for rule_id in failed_ids
        ]
    }
    fresh_validated_ids = validate_rules._fresh_validated_ids(_TF_V5 / "rules.validated.yaml")
    recon = validate_rules.reconcile_regate_report(
        report,
        carried_corpus_path=_TF_V4 / "rules.proposed.yaml",
        fresh_validated_ids=fresh_validated_ids,
        fresh_proposed_path=_TF_V5 / "rules.proposed.yaml",
    )

    assert recon["counts"] == {"healed": 4, "residual": 3}
    assert all(h["kind"] == validate_rules.HEALED_TEMPLATE_DRIFT for h in recon["healed"])
    # Each healer shows a real old -> new template move.
    assert all(
        h["old_template"] and h["new_template"] and h["old_template"] != h["new_template"]
        for h in recon["healed"]
    )
    # The residual are the dropped bnb dormant rules.
    assert all("bnb_4bit" in r["id"] for r in recon["residual"])
