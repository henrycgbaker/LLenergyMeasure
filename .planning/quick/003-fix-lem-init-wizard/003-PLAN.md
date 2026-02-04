---
phase: quick
plan: 003
type: execute
wave: 1
depends_on: []
files_modified:
  - src/llenergymeasure/cli/init_cmd.py
autonomous: true

must_haves:
  truths:
    - "All UserConfig fields are prompted in interactive mode"
    - "thermal_gaps.between_cycles is configurable via wizard"
    - "notifications.on_complete/on_failure toggles are configurable"
  artifacts:
    - path: "src/llenergymeasure/cli/init_cmd.py"
      provides: "Complete init wizard prompts"
      contains: "between_cycles"
---

<objective>
Fix lem init wizard to prompt for all configurable UserConfig fields.

Purpose: Currently init_cmd.py only prompts for 4 of 9 configurable fields.
Missing: between_cycles, on_complete, on_failure, warmup_delay, auto_teardown.
The description specifically calls out webhook toggles and thermal gaps.

Output: Updated init_cmd.py with complete prompts for all user-configurable fields.
</objective>

<execution_context>
@/home/h.baker@hertie-school.lan/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@src/llenergymeasure/config/user_config.py (UserConfig model - SSOT for fields)
@src/llenergymeasure/cli/init_cmd.py (current wizard implementation)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add missing prompts to init wizard</name>
  <files>src/llenergymeasure/cli/init_cmd.py</files>
  <action>
Add prompts for missing UserConfig fields in interactive mode:

1. After Q2 (thermal gap between experiments), add Q2b:
   - Prompt: "Thermal gap between cycles (seconds):"
   - Default: str(int(defaults.thermal_gaps.between_cycles))
   - Validation: float conversion with fallback to default

2. After webhook URL (Q4), add webhook toggles Q4b/Q4c:
   - Only show if webhook_url was provided (non-empty)
   - Q4b: "Send notification on completion?" (confirm, default=defaults.notifications.on_complete)
   - Q4c: "Send notification on failure?" (confirm, default=defaults.notifications.on_failure)

3. Update config construction (lines 276-292) to use new values:
   - ThermalGapConfig: use both thermal_gap and thermal_gap_cycles
   - NotificationsConfig: use on_complete and on_failure answers

4. Update non-interactive mode (lines 193-201) to also support new CLI flags if desired,
   OR keep it minimal (existing + defaults). Since task description focuses on wizard,
   keep non-interactive unchanged (uses existing values or defaults).

Do NOT add prompts for docker.warmup_delay or docker.auto_teardown - these are advanced
options better left as config-file-only (power users can edit .lem-config.yaml directly).
  </action>
  <verify>
Run: `python -c "from llenergymeasure.cli.init_cmd import init_cmd; print('import ok')"`
Grep for new prompts: `grep -n "between_cycles\|on_complete\|on_failure" src/llenergymeasure/cli/init_cmd.py`
  </verify>
  <done>
- init_cmd.py prompts for thermal_gaps.between_cycles
- init_cmd.py prompts for notifications.on_complete/on_failure (when webhook_url provided)
- All prompted values flow through to config construction
  </done>
</task>

</tasks>

<verification>
```bash
# Verify import works
python -c "from llenergymeasure.cli.init_cmd import init_cmd; print('ok')"

# Verify new prompts exist
grep -c "between_cycles" src/llenergymeasure/cli/init_cmd.py  # Should be >= 2

# Run ruff
ruff check src/llenergymeasure/cli/init_cmd.py
ruff format --check src/llenergymeasure/cli/init_cmd.py
```
</verification>

<success_criteria>
- init_cmd.py imports without error
- Wizard prompts for between_cycles thermal gap
- Wizard prompts for on_complete/on_failure (conditional on webhook_url)
- ruff passes
</success_criteria>

<output>
After completion, test manually: `lem init` (in a test directory)
</output>
