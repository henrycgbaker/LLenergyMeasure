---
created: 2026-02-26T10:19:52.012Z
title: Consider study-level cross-config aggregation
area: measurement
files:
  - .product/preservation_audit/results/P-02-study-aggregation-grouping.md
  - src/llenergymeasure/results/aggregation.py
---

## Problem

v1.x has ~200 LOC for study-level cross-config statistical comparison: bootstrap CIs per config,
grouping by arbitrary field paths (dot-notation extraction across `effective_config`), and
`aggregate_campaign_with_grouping()`. This was cut from v2.0 (preservation audit P-02, 2026-02-26)
because the tool's primary purpose is measurement, not statistics — `StudyResult` is a manifest +
list of `ExperimentResult` paths, with analysis left to downstream notebooks.

Post-v2.0, revisit whether built-in grouping/CIs add enough value for researchers to justify
adding this back. The v1.x code could serve as a starting point.

## Solution

TBD — evaluate after v2.0 ships and researchers provide feedback on whether they want built-in
comparison or prefer notebook-based analysis. Consider:
- Is the grouping logic (arbitrary field-path extraction) something researchers repeatedly
  reimplement in notebooks?
- Would a separate `llem analyze` command or `llenergymeasure.analyze` module be appropriate?
- Should this live in the core library or as a companion package?
