"""Unified runner-provenance domain model.

``RunnerProvenance`` records how one experiment was executed - as a host
process or inside a container - together with the image, its resolved
registry digest, and the precedence source that selected the runner. It mirrors
the config-layer ``RunnerSpec`` (config and domain are independent sibling
layers, so the config value object cannot itself live on a domain result).

It lives in its own domain module rather than on ``experiment`` or
``environment`` because both of those import it and ``experiment`` already
imports ``environment``; a shared low-level module is the only home that avoids
an import cycle. The one model is serialised into BOTH per-experiment bundle
artefacts (dual serialisation): ``result.json`` (via
``ExperimentResult.runner_provenance``, keeping result.json self-contained) and
``system.json`` (via ``EnvironmentSnapshot.runner``). It carries the
superset of the fields the two sinks used to split across the retired
``RunnerEnvironment`` sibling: ``image_source`` (the result.json image-resolution
provenance) and ``image_digest`` (the system.json reproducibility anchor).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunnerProvenance(BaseModel):
    """How an experiment was executed - host process or container.

    Persisted reproducibility metadata mirroring the fields of the config-layer
    ``RunnerSpec``. Serialised into both result.json and system.json.
    """

    mode: Literal["process", "container"] = Field(
        ...,
        description='Execution mode - "process" or "container". A closed vocabulary: a '
        "pre-v0.7 bundle carrying the renamed values fails validation loudly on read "
        "(clean break, no alias translation).",
    )
    image: str | None = Field(
        default=None, description="Container image used (None for process mode)"
    )
    source: str | None = Field(
        default=None,
        description='Precedence layer that produced the runner ("env", "yaml", '
        '"user_config", "auto_detected", "default", "multi_engine_elevation", "implicit" '
        "when no spec was resolved)",
    )
    image_source: str | None = Field(
        default=None,
        description="Where the container image was resolved from (None for process mode or when "
        "unresolved). The result.json image-resolution provenance.",
    )
    image_digest: str | None = Field(
        default=None,
        description="Resolved image registry digest ('repo@sha256:...'). None for process runs "
        "or when the digest could not be resolved (locally-built image, docker unavailable, "
        "inspect error) - resolution is best-effort and never fails a run. The system.json "
        "reproducibility anchor pinning the full software stack.",
    )

    model_config = {"frozen": True, "extra": "forbid"}
