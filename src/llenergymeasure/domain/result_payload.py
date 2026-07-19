"""Shared ExperimentResult payload parser.

Lives in the domain layer (Layer 0) so BOTH read paths can share one parser
without a layering violation: the infra exchange-read path
(``infra.docker_runner._read_result``) is below ``results`` and could not import
a parser that lived there, and the bundle-read path
(``results.bundle.BundleReader`` / ``results.persistence.load_result``) is above
infra. Domain is below both, so it is the only home a shared parser can have.

The two call shapes differ only in tolerance:

- The docker exchange payload is cross-version IPC: the container may run an
  older or newer llenergymeasure than the host, so unknown fields are stripped
  before validation and a version-skew warning is logged when the container
  version does not match the host (``tolerant=True`` with ``expected_version``
  set to the host version).
- A persisted bundle is our own on-disk contract: no stripping, no version
  handshake (``tolerant=False``, ``expected_version=None``).

Tolerance for the retired per-artefact ``schema_version`` key is handled by the
``ExperimentResult`` before-validator, so both call shapes read legacy payloads.
"""

from __future__ import annotations

import logging
from typing import Any

from llenergymeasure.domain.experiment import ExperimentResult

logger = logging.getLogger(__name__)


def parse_experiment_result_payload(
    raw: dict[str, Any],
    *,
    tolerant: bool,
    expected_version: str | None = None,
) -> ExperimentResult:
    """Parse a raw result payload into an :class:`ExperimentResult`.

    Args:
        raw: The decoded JSON object for one result. Mutated in place when
            ``tolerant`` strips unknown fields (callers pass a fresh dict).
        tolerant: When True, strip fields unknown to the current
            ``ExperimentResult`` schema before validation (cross-version IPC).
            When False, validate strictly - our own persisted bundle contract,
            where an unexpected field signals real drift, not version skew.
        expected_version: When set, log a version-skew warning if the parsed
            result's ``llenergymeasure_version`` is None or differs from it (the
            docker host/container image handshake). None disables the check (the
            bundle path, which has no cross-version handshake).

    Returns:
        The validated ExperimentResult.
    """
    if tolerant:
        known = set(ExperimentResult.model_fields)
        extra = set(raw) - known
        if extra:
            for key in extra:
                raw.pop(key)
            logger.debug("Stripped unknown fields from result payload: %s", extra)

    result = ExperimentResult.model_validate(raw)

    if expected_version is not None:
        version = result.llenergymeasure_version
        if version is None or version != expected_version:
            logger.warning(
                "Container result version %s differs from host %s - rebuild Docker images",
                version,
                expected_version,
            )

    return result
