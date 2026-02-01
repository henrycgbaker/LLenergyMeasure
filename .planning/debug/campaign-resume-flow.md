---
status: fixing
trigger: "Investigate and fix: campaign-resume-and-container-cleanup"
created: 2026-01-31T10:30:00Z
updated: 2026-01-31T10:30:00Z
---

## Current Focus

hypothesis: Root cause confirmed via code inspection
test: Execute fixes per plan sections A1-A4, B1-B3
expecting: Resume flow restructured, --fresh flag added, dead code deleted, docs updated
next_action: Execute plan - currently in read-only plan mode, need to exit to execution mode

## Symptoms

expected: When resuming a campaign with -y, the user should see a brief "Resuming Campaign" panel and only remaining experiments — no grid validation display, no full config panel, no confirmation prompt. Per-experiment resume prompts inside Docker should NOT appear. Dead ContainerManager code should be removed.

actual:
1. Resume shows ALL grid validation, full config, full 6-experiment plan, confirmation prompt BEFORE filtering to remaining 4
2. Per-experiment "Incomplete experiment detected / Resume this experiment?" prompt fires inside Docker despite --yes (because --fresh not passed, and config hash matches old state in shared named volume)
3. Early resume count says "3 remaining" instead of "4 remaining" (counts pending but not failed)
4. Dead ContainerManager code (308 lines) + tests (184 lines) still exist, never used
5. Planning docs (STATE.md, ROADMAP.md) still reference old docker compose up+exec approach

errors: No crashes — UX issues and dead code

reproduction:
- Run `lem campaign /tmp/uat_3backend.yaml -n 10 -d ai_energy_score -y`
- Ctrl+C during first experiment
- Re-run same command — observe broken resume flow

started: Discovered during Phase 2.1 UAT. ContainerManager was created in Phase 2-02 but never integrated — campaign uses subprocess + docker compose run --rm instead.

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-01-31T10:30:00Z
  checked: Plan file read
  found: Complete implementation plan with detailed steps
  implication: This is a guided implementation task, not an open-ended investigation

- timestamp: 2026-01-31T10:31:00Z
  checked: campaign.py lines 244-273, 276-314, 382-409
  found: Symptoms confirmed - early check at line 252 uses `pending_count` not `get_remaining()`, resume filter happens AFTER display (line 382+), no --fresh flag in Docker/local commands, index-based manifest linking at line 395-397
  implication: All described symptoms exist - ready to execute fix plan

## Resolution

root_cause: |
  CONFIRMED - Multiple UX and code issues in campaign resume flow:
  1. Line 252: early resume count uses pending_count (excludes failed) instead of get_remaining() (includes failed)
  2. Lines 276-314: resume=True still shows full grid validation, full config summary, full execution plan
  3. Lines 382-398: manifest load + resume filter happens AFTER all displays, should be BEFORE execution plan
  4. Lines 395-397: fragile index-based manifest linking (breaks on shuffled campaigns without seed)
  5. Lines 793, 905: no --fresh flag passed to experiment subprocess (causes per-experiment resume prompts)
  6. Dead code: container.py (308 lines) + test_container.py (184 lines) never used, campaign uses docker compose run --rm
  7. Planning docs still reference old up+exec approach

fix: Execute plan sections A1-A4 (campaign resume flow), B1-B3 (dead code cleanup)
verification: (pending - awaiting execution mode)
files_changed: []
