"""Canonical layout of the results-bundle artefacts.

These names are the on-disk contract between every writer (harness, study, api)
and every reader (results.persistence, report_gaps, resume). They were inlined
as string literals across five packages; centralising them here keeps the
bundle layout in one place and lets the layer contract expose them to all
writers (domain is the lowest layer above utils, so everything may import it).

Alongside the filenames this module owns two contract-level facts:

- ``BUNDLE_VERSION`` - the single version stamped into every JSON artefact of a
  per-experiment bundle. It versions the layout, the artefact set, and each
  artefact's schema as ONE contract (superseding the retired per-artefact
  ``schema_version`` counters). Bump it once per documented bundle break.
- ``ARTEFACTS`` - a declarative table describing each per-experiment artefact
  (whether it is required, whether its absence is worth a loud warning, and
  whether it is JSON or Parquet). It is the extension point: a new bundle
  artefact (e.g. a future server-mode per-request series) is added by
  registering one entry here plus one writer method, not by hand-gluing a fifth
  file into the writer. It is deliberately a data table, not plugin machinery.

Changing a filename or the version changes the on-disk layout, so treat them as
a stable format contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Study-level artefacts directory (per-study, holds the config copy, skipped
# configs, system overrides, and the study-level environment snapshot).
STUDY_ARTEFACTS_DIR = "_study-artefacts"

# Per-experiment bundle files.
RESULT_FILENAME = "result.json"
CONFIG_SIDECAR_FILENAME = "config.json"
ENVIRONMENT_FILENAME = "environment.json"
TIMESERIES_FILENAME = "timeseries.parquet"

# Study-level bundle files.
MANIFEST_FILENAME = "manifest.json"
EQUIVALENCE_GROUPS_FILENAME = "equivalence_groups.json"
SYSTEM_OVERRIDES_FILENAME = "system_overrides.json"

# Single version for the whole per-experiment bundle. Stamped into result.json,
# config.json, and environment.json (Parquet is self-describing and stays
# unversioned). One number to bump, one CHANGELOG line per break.
BUNDLE_VERSION = "1.0"


@dataclass(frozen=True)
class ArtefactSpec:
    """Declarative description of one per-experiment bundle artefact.

    Attributes:
        filename: On-disk name inside the experiment directory.
        required: The bundle is incomplete without it (result.json). Documents
            the contract for the reader (S5); the writer guarantees required
            artefacts by raising if it cannot write them.
        warn_if_missing: Whether the writer's ``finalize()`` sweep emits a loud
            warning when the artefact is absent. Used for the loudness backstops
            that make silent data loss visible (config-sidecar provenance,
            a declared-but-missing timeseries).
        kind: ``"json"`` or ``"parquet"`` - the on-disk encoding.
        missing_note: Human-readable reason the artefact's absence matters,
            appended to the ``finalize()`` warning. ``None`` for artefacts whose
            absence needs no explanation. Lives here so the registry is the sole
            source: adding an artefact needs one entry, nothing hand-synced.
    """

    filename: str
    required: bool
    warn_if_missing: bool
    kind: Literal["json", "parquet"]
    missing_note: str | None = None


# The per-experiment bundle artefact set. BundleWriter/BundleReader iterate this
# registry for existence checks, loudness backstops, and rescue sweeps. Add a
# new artefact by registering an entry here plus a writer method - the finalize
# sweep picks it up automatically.
ARTEFACTS: dict[str, ArtefactSpec] = {
    "result": ArtefactSpec(RESULT_FILENAME, required=True, warn_if_missing=False, kind="json"),
    "config": ArtefactSpec(
        CONFIG_SIDECAR_FILENAME,
        required=False,
        warn_if_missing=True,
        kind="json",
        missing_note=(
            "provenance and authoritative engine/model identity are missing from this result"
        ),
    ),
    "environment": ArtefactSpec(
        ENVIRONMENT_FILENAME, required=False, warn_if_missing=False, kind="json"
    ),
    "timeseries": ArtefactSpec(
        TIMESERIES_FILENAME,
        required=False,
        warn_if_missing=True,
        kind="parquet",
        missing_note=(
            "the result references a timeseries but the parquet did not land in the bundle"
        ),
    ),
}
