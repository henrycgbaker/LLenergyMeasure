"""Engine validation rules: loader, matcher, predicate engine.

The corpus at ``src/llenergymeasure/engines/{engine}/rules.yaml`` is the single
shipped source of engine-correctness knowledge. This package parses that file
and exposes a typed matcher API used by the generic ``@model_validator`` on
:class:`llenergymeasure.config.models.ExperimentConfig` and by the sweep
library-resolution dedup.
"""

from llenergymeasure.config.engine_rules.loader import (
    VALID_SEVERITY,
    VALID_SOURCE,
    VALID_VERIFIED,
    EngineRules,
    EngineRulesLoader,
    Provenance,
    Rule,
    RuleCorpusError,
    RuleMatch,
    Severity,
    Source,
    UnknownEnumValueError,
    UnknownSeverityError,
    UnknownSourceError,
    UnknownVerifiedError,
    UnsupportedSchemaVersionError,
    Verified,
    evaluate_predicate,
    resolve_field_path,
)

__all__ = [
    "VALID_SEVERITY",
    "VALID_SOURCE",
    "VALID_VERIFIED",
    "EngineRules",
    "EngineRulesLoader",
    "Provenance",
    "Rule",
    "RuleCorpusError",
    "RuleMatch",
    "Severity",
    "Source",
    "UnknownEnumValueError",
    "UnknownSeverityError",
    "UnknownSourceError",
    "UnknownVerifiedError",
    "UnsupportedSchemaVersionError",
    "Verified",
    "evaluate_predicate",
    "resolve_field_path",
]
