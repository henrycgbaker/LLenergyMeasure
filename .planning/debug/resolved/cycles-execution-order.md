---
status: resolved
trigger: "cycles-execution-order - Campaign cycles execute in wrong order"
created: 2026-02-04T10:30:00Z
updated: 2026-02-04T10:55:00Z
---

## Current Focus

hypothesis: RESOLVED
test: Logic verified with unit test simulation
expecting: Campaign execution will now follow round-robin order as designed
next_action: N/A - ready for user testing

## Symptoms

expected: Round-robin execution order - cycle 1 of all experiments, then cycle 2 of all experiments, etc. (vllm_c1→pytorch_c1→vllm_c2→pytorch_c2→...). User also expected config options for mixing strategy (interleave, shuffle, etc.)
actual: Batch-per-experiment - all 3 cycles of vllm_example complete, THEN all 3 cycles of pytorch_example. Campaign plan showed correct interleading but execution didn't follow it.
errors: None - runs without errors but wrong execution order
reproduction: Run example campaign config with 2+ backends and 3+ cycles. The campaign YAML is at configs/examples/campaign_example.yaml
started: Observed in recent campaign testing. User thought round-robin was already implemented.

## Eliminated

## Evidence

- timestamp: 2026-02-04T10:32:00Z
  checked: campaign.py generate_execution_order() method
  found: Method correctly implements round-robin ordering - for structure="shuffled" or "interleaved", it iterates cycles FIRST (outer loop), then configs (inner loop). Lines 174-193.
  implication: The execution order list is correctly structured with interleaving

- timestamp: 2026-02-04T10:33:00Z
  checked: campaign.yaml example config
  found: Config specifies `execution.cycles: 5` and `structure: shuffled`. This should produce round-robin order.
  implication: Config is requesting the right behavior

- timestamp: 2026-02-04T10:34:00Z
  checked: campaign.py CLI execution loop (lines 526-638)
  found: Loop iterates through execution_order (which IS round-robin), calling _run_single_experiment for each. Each CampaignExperiment has cycle_index set correctly (0-indexed).
  implication: The loop structure is correct - it should run one experiment at a time in order

- timestamp: 2026-02-04T10:35:00Z
  checked: _run_single_experiment function (lines 834-968)
  found: Function builds metadata with cycle_id from experiment.cycle_index. Calls subprocess with config file. NO --cycles flag passed to subprocess. Uses --yes and --fresh flags.
  implication: Each subprocess call should run ONE execution, not multiple cycles

- timestamp: 2026-02-04T10:40:00Z
  checked: experiment.py experiment_cmd function, lines 658-909
  found: **ROOT CAUSE** - Experiment command has its own internal multi-cycle loop (line 814: `for cycle_idx in range(effective_cycles)`). The effective_cycles comes from `config.num_cycles` (line 660). When campaign calls experiment command, it passes the full config which includes num_cycles from the original YAML.
  implication: Each "experiment" in the campaign plan actually runs num_cycles iterations internally, causing the batch-per-experiment behavior

- timestamp: 2026-02-04T10:42:00Z
  checked: How campaign detects it's running vs standalone
  found: Campaign passes metadata via config._metadata with campaign_id, campaign_name, cycle_id (lines 875-881 in campaign.py). Experiment command reads this at line 647-656 and sets in_campaign flag. Line 666-674 shows it DISPLAYS different message when in_campaign, but DOESN'T disable internal cycling.
  implication: The campaign context detection exists but isn't used to prevent inner cycling

## Resolution

root_cause: The experiment command has an internal multi-cycle loop (line 814 in experiment.py) that runs `config.num_cycles` iterations. When campaign orchestration calls the experiment command, it passes the full experiment config (which includes num_cycles from the original YAML). This causes each "single experiment" in the campaign plan to actually run multiple cycles internally, resulting in batch-per-experiment execution instead of round-robin.

The campaign correctly detects it's running (via _metadata.campaign_id) and shows appropriate messages, but doesn't disable the internal cycling behavior.

fix: Modified experiment.py (lines 658-682) to force effective_cycles=1 when in_campaign=True. Restructured the logic:
1. Check if in_campaign first
2. If in campaign: set effective_cycles=1, skip validation, show campaign context
3. If standalone: use config.num_cycles with validation and appropriate messaging

This ensures that when campaign orchestrates execution, each experiment subprocess runs exactly one cycle, allowing the campaign's round-robin scheduling to work correctly.

verification:
- Logic verified with Python simulation - campaign mode correctly forces effective_cycles=1
- Standalone mode still respects config.num_cycles for backward compatibility
- Code change is minimal and targeted - only affects campaign execution path
- Ready for integration testing with actual campaign run

files_changed: [src/llenergymeasure/cli/experiment.py]
