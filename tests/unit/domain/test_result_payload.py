"""Unit tests for the shared ExperimentResult payload parser.

The parser is the one seam both read paths share: the docker exchange-read
(cross-version IPC, tolerant) and the bundle-read (our own strict contract).
These tests pin the tolerance parity between the two shapes and the version-skew
handshake.
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from llenergymeasure.domain.result_payload import parse_experiment_result_payload
from tests.conftest import make_result

_LOGGER = "llenergymeasure.domain.result_payload"


def _payload(**overrides) -> dict:
    """A valid result payload as a raw dict (JSON round-tripped for fidelity)."""
    raw = json.loads(make_result(**overrides).model_dump_json())
    return raw


# ---------------------------------------------------------------------------
# Tolerance parity: exchange (tolerant) vs bundle (strict)
# ---------------------------------------------------------------------------


def test_both_paths_parse_a_clean_payload_identically() -> None:
    """A clean payload parses to the same result via tolerant and strict modes."""
    raw_a = _payload()
    raw_b = _payload()
    exchange = parse_experiment_result_payload(raw_a, tolerant=True)
    bundle = parse_experiment_result_payload(raw_b, tolerant=False)
    assert exchange.model_dump_json() == bundle.model_dump_json()


def test_tolerant_strips_unknown_fields() -> None:
    """The exchange path drops fields the host schema does not know (version skew)."""
    raw = _payload()
    raw["a_field_from_a_newer_container"] = {"nested": 1}
    result = parse_experiment_result_payload(raw, tolerant=True)
    assert result.experiment_id == "test-001"


def test_strict_rejects_unknown_fields() -> None:
    """The bundle path is strict: an unknown field is real drift, not skew."""
    raw = _payload()
    raw["a_field_from_a_newer_container"] = {"nested": 1}
    with pytest.raises(ValidationError):
        parse_experiment_result_payload(raw, tolerant=False)


def test_both_paths_tolerate_the_retired_schema_version_key() -> None:
    """Legacy ``schema_version`` is dropped by the model before-validator on both paths."""
    raw_a = _payload()
    raw_a.pop("bundle_version", None)
    raw_a["schema_version"] = "5.0"
    raw_b = dict(raw_a)
    # Even strict mode reads it: the retired key is dropped before validation.
    assert parse_experiment_result_payload(raw_a, tolerant=False).experiment_id == "test-001"
    assert parse_experiment_result_payload(raw_b, tolerant=True).experiment_id == "test-001"


# ---------------------------------------------------------------------------
# Version-skew handshake (expected_version)
# ---------------------------------------------------------------------------


def test_version_warning_when_expected_version_differs(caplog) -> None:
    """A version mismatch against expected_version warns (the docker handshake)."""
    raw = _payload(llenergymeasure_version="0.1.0")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        parse_experiment_result_payload(raw, tolerant=True, expected_version="9.9.9")
    assert any("rebuild Docker images" in r.message for r in caplog.records)


def test_version_warning_when_result_version_none(caplog) -> None:
    """A None result version also warns when a version is expected."""
    raw = _payload()  # llenergymeasure_version defaults to None
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        parse_experiment_result_payload(raw, tolerant=True, expected_version="1.0.0")
    assert any("rebuild Docker images" in r.message for r in caplog.records)


def test_no_version_warning_when_versions_match(caplog) -> None:
    """No warning when the result version matches expected_version."""
    raw = _payload(llenergymeasure_version="1.2.3")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        parse_experiment_result_payload(raw, tolerant=True, expected_version="1.2.3")
    assert not any("rebuild Docker images" in r.message for r in caplog.records)


def test_no_version_handshake_on_the_bundle_path(caplog) -> None:
    """The bundle path passes expected_version=None: no handshake, no warning."""
    raw = _payload()  # version None
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        parse_experiment_result_payload(raw, tolerant=False, expected_version=None)
    assert not any("rebuild Docker images" in r.message for r in caplog.records)
