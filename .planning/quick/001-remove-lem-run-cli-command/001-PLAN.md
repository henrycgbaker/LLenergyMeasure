---
type: quick
plan: 001
wave: 1
depends_on: []
files_modified:
  - src/llenergymeasure/cli/__init__.py
  - src/llenergymeasure/cli/experiment.py
  - src/llenergymeasure/cli/listing.py
  - docs/cli.md
  - .planning/codebase/ARCHITECTURE.md
autonomous: true
---

<objective>
Remove the legacy `lem run` CLI command — a stub that doesn't actually work.

Purpose: Clean up dead code. The `run_cmd` function prints a "requires accelerate launch" message and exits without doing anything useful. The working command is `lem experiment`.

Output: Cleaner CLI with one clear entry point for experiments.
</objective>

<context>
@src/llenergymeasure/cli/__init__.py
@src/llenergymeasure/cli/experiment.py
@src/llenergymeasure/cli/listing.py
@docs/cli.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Remove run_cmd and all references</name>
  <files>
    - src/llenergymeasure/cli/__init__.py
    - src/llenergymeasure/cli/experiment.py
    - src/llenergymeasure/cli/listing.py
    - docs/cli.md
    - .planning/codebase/ARCHITECTURE.md
  </files>
  <action>
    1. In `src/llenergymeasure/cli/__init__.py`:
       - Remove line 100: `app.command("run")(experiment.run_cmd)  # Legacy command`

    2. In `src/llenergymeasure/cli/experiment.py`:
       - Delete the entire `run_cmd` function (lines ~155-236)
       - Remove `"run_cmd"` from the `__all__` export list (line ~910)

    3. In `src/llenergymeasure/cli/listing.py`:
       - Line 35: Change `lem run config.yaml` to `lem experiment config.yaml` in the usage hint

    4. In `docs/cli.md`:
       - Delete the entire `### run` section (lines ~213-232) including the "Legacy command" note

    5. In `.planning/codebase/ARCHITECTURE.md`:
       - Line 158-159: Remove `run_cmd` reference and `lem run` from the CLI description
  </action>
  <verify>
    - `lem --help` shows no `run` command
    - `lem run` returns "No such command" error
    - `lem experiment --help` still works
    - `grep -r "run_cmd\|lem run" src/ docs/` returns no matches (except batch_run_cmd which is unrelated)
  </verify>
  <done>
    - `run_cmd` function deleted from experiment.py
    - Command registration removed from __init__.py
    - All documentation updated to reference `lem experiment` only
    - No remaining references to `lem run` in codebase
  </done>
</task>

</tasks>

<verification>
```bash
# Verify run command is gone
lem --help | grep -v batch | grep run  # Should return nothing

# Verify experiment still works
lem experiment --help  # Should show help

# Verify no stale references
grep -r "run_cmd" src/llenergymeasure/cli/ --include="*.py" | grep -v batch_run_cmd  # Should be empty
grep -r "lem run" docs/ src/ | grep -v "docker compose run"  # Should be empty
```
</verification>

<success_criteria>
- `lem run` command no longer exists
- `lem experiment` remains the sole entry point for experiments
- All docs reference `lem experiment` only
- No dead code remains
</success_criteria>
