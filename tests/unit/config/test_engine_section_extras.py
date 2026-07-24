"""Tests for the ``validate_engine_section_extras`` model validator.

Covers the migration safety net for the generated nested engine shape
(``transformers: {engine_params: {...}, sampling_params: {...}}``):

- a pre-nested-shape flat config (knobs directly on the section wrapper) is an
  ERROR with a migration hint naming the correct nested location;
- a typo'd key on the section wrapper is an ERROR with a ``did you mean``
  suggestion;
- a typo'd key INSIDE ``engine_params`` / ``sampling_params`` (which would pass
  through to the engine) WARNS with a suggestion but still parses;
- a genuinely-new engine field inside a sub-section passes (no close match, no
  error) - passthrough of new engine kwargs is intended;
- the nested happy path is unaffected.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.warnings import ConfigValidationWarning

# ---------------------------------------------------------------------------
# (a) Flat legacy shape -> error with migration hint
# ---------------------------------------------------------------------------


def test_flat_legacy_engine_param_errors_with_migration_hint() -> None:
    """A flat ``load_in_4bit`` on the wrapper names the correct nested location."""
    with pytest.raises(
        ValidationError,
        match=r"transformers\.load_in_4bit moved to transformers\.engine_params\.load_in_4bit",
    ):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"load_in_4bit": True},  # type: ignore[arg-type]
        )


def test_flat_legacy_sampling_param_errors_with_migration_hint() -> None:
    """A flat ``temperature`` on the wrapper points at ``sampling_params``."""
    with pytest.raises(
        ValidationError,
        match=r"transformers\.temperature moved to transformers\.sampling_params\.temperature",
    ):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"temperature": 0.7},  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# (b) Wrapper-level typo -> error with did-you-mean suggestion
# ---------------------------------------------------------------------------


def test_wrapper_level_typo_errors_with_suggestion() -> None:
    """A typo on the wrapper that is not a known field errors with a suggestion."""
    with pytest.raises(
        ValidationError,
        match=r"unknown field 'dtypee' on transformers; did you mean engine_params\.dtype\?",
    ):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"dtypee": "float16"},  # type: ignore[arg-type]
        )


def test_wrapper_level_unknown_with_no_close_match_still_errors() -> None:
    """A wrapper-level extra with no close field still errors (no passthrough)."""
    with pytest.raises(ValidationError, match=r"unknown field 'zzz_nonsense' on transformers"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"zzz_nonsense": 1},  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# (c) In-params typo -> warn with suggestion, still parses
# ---------------------------------------------------------------------------


def test_in_params_typo_warns_with_suggestion_and_parses() -> None:
    """A close typo inside engine_params warns but still parses (passthrough)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        cfg = ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"engine_params": {"dtypee": "float16"}},
        )

    matched = [
        w
        for w in caught
        if issubclass(w.category, ConfigValidationWarning) and "dtypee" in str(w.message)
    ]
    assert matched, "expected a soft-validation warning naming dtypee"
    assert "did you mean dtype" in str(matched[0].message)
    # The typo'd key still passes through to the engine (extra='allow').
    assert cfg.transformers is not None
    assert cfg.transformers.engine_params.model_extra == {"dtypee": "float16"}


def test_in_params_typo_in_sampling_warns() -> None:
    """A close typo inside sampling_params also warns (same vocabulary path)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"sampling_params": {"temperatur": 0.7}},
        )
    matched = [
        w
        for w in caught
        if issubclass(w.category, ConfigValidationWarning) and "temperatur" in str(w.message)
    ]
    assert matched, "expected a soft-validation warning naming temperatur"
    assert "did you mean temperature" in str(matched[0].message)


# ---------------------------------------------------------------------------
# (d) Legitimate unknown new engine field -> passes, no error
# ---------------------------------------------------------------------------


def test_new_engine_field_inside_params_passes() -> None:
    """A genuinely-new engine field (no close match) parses with no error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        cfg = ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            serving_mode="offline",
            transformers={"engine_params": {"brand_new_hf_kwarg_xyz": 1}},
        )
    soft = [
        w
        for w in caught
        if issubclass(w.category, ConfigValidationWarning)
        and "brand_new_hf_kwarg_xyz" in str(w.message)
    ]
    assert not soft, "a genuinely-new field must not trigger a typo warning"
    assert cfg.transformers is not None
    assert cfg.transformers.engine_params.model_extra == {"brand_new_hf_kwarg_xyz": 1}


# ---------------------------------------------------------------------------
# (e) Nested happy path unaffected
# ---------------------------------------------------------------------------


def test_nested_happy_path_unaffected() -> None:
    """A well-formed nested config parses cleanly with no extras warning."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
        transformers={
            "engine_params": {"dtype": "float16", "load_in_4bit": True},
            "sampling_params": {"top_p": 0.9},
        },
    )
    assert cfg.transformers is not None
    assert cfg.transformers.engine_params.dtype == "float16"
    assert cfg.transformers.engine_params.load_in_4bit is True
    assert cfg.transformers.sampling_params.top_p == 0.9


def test_no_engine_section_is_unaffected() -> None:
    """A config with no engine section parses (nothing to check)."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers", serving_mode="offline")
    assert cfg.transformers is None
