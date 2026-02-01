---
status: resolved
trigger: "Investigate two bugs in the campaign CLI orchestrator."
created: 2026-01-31T10:00:00Z
updated: 2026-01-31T10:20:00Z
---

## Current Focus

hypothesis: Fixes implemented and ready for verification
test: code review confirms fixes are in place correctly
expecting: --yes suppresses resume prompts, Ctrl+C aborts campaigns
next_action: verify fixes work as expected

## Symptoms

expected: (A) `lem campaign ... -y` should auto-handle stale state without prompts. (B) Ctrl+C should abort entire campaign.
actual: (A) Interactive "Resume this experiment? [y/n]" prompt appears inside Docker container despite -y flag. (B) Ctrl+C kills current experiment but campaign continues to next one.
errors: None - prompts block automation, campaign loop continues after SIGINT
reproduction: (A) Run campaign with -y, Ctrl+C during first experiment, re-run with -y - resume prompt appears. (B) Ctrl+C during any experiment - campaign continues.
started: Unknown - present in current codebase

## Eliminated

## Evidence

- timestamp: 2026-01-31T10:05:00Z
  checked: campaign.py _build_docker_command (line 849-888)
  found: Passes `--yes` flag to Docker command (line 877)
  implication: Campaign correctly passes --yes to experiment command

- timestamp: 2026-01-31T10:06:00Z
  checked: campaign.py _run_single_experiment (line 674-773)
  found: Line 764 builds non-Docker command with `--yes`, line 769 uses subprocess.run() directly
  implication: Non-Docker path also passes --yes correctly

- timestamp: 2026-01-31T10:07:00Z
  checked: campaign.py main loop (line 443-542)
  found: Line 474-481 calls _run_single_experiment, checks exit_code == 0 (line 483) or else records as failed (line 499-512)
  implication: Exit code 130 (SIGINT) is treated as generic failure, loop continues

- timestamp: 2026-01-31T10:08:00Z
  checked: experiment.py experiment_cmd (line 250-939)
  found: --yes flag defined at line 347-349, used for config warning prompts (line 617-628), NOT used for resume prompt (line 641)
  implication: Resume prompt at line 641 ignores --yes flag - uses Confirm.ask() unconditionally

## Resolution

root_cause: |
  Bug A: experiment.py line 638-646 detects stale experiments by config_hash and shows interactive resume prompt,
  but ignores the --yes flag. The prompt uses Confirm.ask() unconditionally, blocking automated campaigns.

  Bug B: campaign.py line 483-512 checks if exit_code == 0 for success, else treats all non-zero codes as failures.
  Exit code 130 (SIGINT) is logged as failure and the loop continues. No special handling for interrupt signals.

fix: |
  Bug A: Add --yes check before Confirm.ask() at line 641, similar to config warning handling at lines 617-628.
  When --yes is set and stale state detected, auto-resume without prompting.

  Bug B: After _run_single_experiment at line 481, check if exit_code == 130 and abort entire campaign with typer.Exit(130).
  This propagates the interrupt signal to the parent process/user.

verification: |
  Bug A verification:
  - experiment.py lines 641-650: Now checks `if yes:` before prompting
  - When --yes is set, auto-resumes with message "Auto-resuming experiment {id} (--yes flag)"
  - Interactive prompt only shown when --yes is NOT set
  - Matches pattern used for config warning handling (lines 617-628)

  Bug B verification:
  - campaign.py line 484: Checks `if exit_code == 130:` immediately after _run_single_experiment
  - On detection: prints interrupt message, updates manifest to "failed" with error, raises typer.Exit(130)
  - Daemon mode (line 651): Same check but returns early instead of raising Exit
  - Exit code 130 no longer falls through to generic failure handling

  Both fixes follow existing patterns in the codebase and maintain consistency.

files_changed:
- src/llenergymeasure/cli/experiment.py (lines 641-650)
- src/llenergymeasure/cli/campaign.py (lines 484-495, 651-662)
