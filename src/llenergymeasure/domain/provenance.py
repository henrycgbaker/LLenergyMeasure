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
``environment.json`` (via ``EnvironmentSnapshot.runner``). It carries the
superset of the fields the two sinks used to split across the retired
``RunnerEnvironment`` sibling: ``image_source`` (the result.json image-resolution
provenance) and ``image_digest`` (the environment.json reproducibility anchor).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

# Legacy runner-mode vocabulary -> canonical, for read-path alias tolerance.
# The runner mode was renamed local->process / docker->container within the SAME
# untagged bundle_version "2.0" stamp, so a reader keying on bundle_version alone
# cannot distinguish 09ec455e-era 2.0 blocks (local/docker) from post-rename 2.0
# blocks (process/container); this map lets the older blocks still parse.
_LEGACY_MODE_ALIASES: dict[str, str] = {"local": "process", "docker": "container"}

# The no-spec source sentinel was renamed "local" -> "implicit" alongside the mode
# rename: with "local" vacated from the mode vocabulary, a block reading
# {mode: "process", source: "local"} would invite exactly the stale-mode misreading
# the rename removes. The sentinel predates bundle 2.0 (#837), so 09ec455e-era 2.0
# blocks can carry source="local"; this map rewrites it on read.
_LEGACY_SOURCE_ALIASES: dict[str, str] = {"local": "implicit"}


class RunnerProvenance(BaseModel):
    """How an experiment was executed - host process or container.

    Persisted reproducibility metadata mirroring the fields of the config-layer
    ``RunnerSpec``. Serialised into both result.json and environment.json.
    """

    mode: str = Field(..., description='Execution mode - "process" or "container"')
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
        "inspect error) - resolution is best-effort and never fails a run. The environment.json "
        "reproducibility anchor pinning the full software stack.",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_vocabulary(cls, data: Any) -> Any:
        """Read a legacy (09ec455e-era bundle 2.0) runner block best-effort.

        The runner-mode vocabulary was renamed ``local`` -> ``process`` and
        ``docker`` -> ``container``, and the no-spec source sentinel ``local`` ->
        ``implicit``, within the same untagged bundle 2.0 break. Map an older
        block's ``mode`` and ``source`` onto the canonical values so those bundles
        still parse rather than failing (mode) or leaking a stale-mode string
        (source). Mirrors the ``CUDAEnvironment._map_legacy_cuda_version`` read-path
        precedent.
        """
        if not isinstance(data, dict):
            return data
        if data.get("mode") in _LEGACY_MODE_ALIASES or data.get("source") in _LEGACY_SOURCE_ALIASES:
            data = dict(data)
            if data.get("mode") in _LEGACY_MODE_ALIASES:
                data["mode"] = _LEGACY_MODE_ALIASES[data["mode"]]
            if data.get("source") in _LEGACY_SOURCE_ALIASES:
                data["source"] = _LEGACY_SOURCE_ALIASES[data["source"]]
        return data
