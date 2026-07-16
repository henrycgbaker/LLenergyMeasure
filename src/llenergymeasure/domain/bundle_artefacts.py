"""Canonical filenames for the results-bundle artefacts.

These names are the on-disk contract between every writer (harness, study, api)
and every reader (results.persistence, report_gaps, resume). They were inlined
as string literals across five packages; centralising them here keeps the
bundle layout in one place and lets the layer contract expose them to all
writers (domain is the lowest layer above utils, so everything may import it).

These are literal filenames only - no behaviour lives here. Changing a value
changes the on-disk layout, so treat them as a stable format contract.
"""

from __future__ import annotations

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
