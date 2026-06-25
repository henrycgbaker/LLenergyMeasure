"""Engine invariants: loader, matcher, predicate engine.

The corpus at ``src/llenergymeasure/engines/{engine}/rules.proposed.yaml``
is the SSOT for what runtime config validation tells users about their configs.
This package consumes that corpus and exposes a typed matcher API used by the
generic ``@model_validator`` on :class:`llenergymeasure.config.models.ExperimentConfig`
and (eventually) by the library resolution mechanism.
"""

from llenergymeasure.config.engine_rules.loader import (
    VALID_ADDED_BY,
    VALID_EMISSION_CHANNEL,
    VALID_OUTCOME,
    VALID_SEVERITY,
    AddedBy,
    EmissionChannel,
    EngineInvariants,
    EngineRulesLoader,
    Invariant,
    InvariantMatch,
    LLMProposedSeverityError,
    Outcome,
    Severity,
    UnknownAddedByError,
    UnknownEmissionChannelError,
    UnknownEnumValueError,
    UnknownOutcomeError,
    UnknownSeverityError,
    UnsupportedSchemaVersionError,
    evaluate_predicate,
    resolve_field_path,
)

__all__ = [
    "VALID_ADDED_BY",
    "VALID_EMISSION_CHANNEL",
    "VALID_OUTCOME",
    "VALID_SEVERITY",
    "AddedBy",
    "EmissionChannel",
    "EngineInvariants",
    "EngineRulesLoader",
    "Invariant",
    "InvariantMatch",
    "LLMProposedSeverityError",
    "Outcome",
    "Severity",
    "UnknownAddedByError",
    "UnknownEmissionChannelError",
    "UnknownEnumValueError",
    "UnknownOutcomeError",
    "UnknownSeverityError",
    "UnsupportedSchemaVersionError",
    "evaluate_predicate",
    "resolve_field_path",
]
