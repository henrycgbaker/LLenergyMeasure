---
phase: quick-002
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/deployment.md
autonomous: true

must_haves:
  truths:
    - "User understands tradeoffs between ephemeral and persistent container strategies"
    - "User knows how to configure container strategy via CLI flag or config file"
    - "User can choose appropriate strategy for their use case"
  artifacts:
    - path: "docs/deployment.md"
      provides: "Container Strategy section with tradeoffs, configuration, recommendations"
      contains: "Container Strategies"
  key_links:
    - from: "docs/deployment.md"
      to: ".lem-config.yaml"
      via: "configuration example"
      pattern: "docker:\\s+strategy:"
---

<objective>
Document ephemeral vs persistent container strategies in docs/deployment.md.

Purpose: Users running multi-backend campaigns need to understand the tradeoffs between container strategies and know how to configure their preference.

Output: New "Container Strategies" section in deployment.md explaining both modes, tradeoffs, configuration options, and recommendations.
</objective>

<execution_context>
@/home/h.baker@hertie-school.lan/.claude/get-shit-done/workflows/execute-plan.md
@/home/h.baker@hertie-school.lan/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@docs/deployment.md
@src/llenergymeasure/config/user_config.py
@.planning/debug/container-strategy-research.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add Container Strategies section to deployment.md</name>
  <files>docs/deployment.md</files>
  <action>
Add a new "Container Strategies" section after the "Docker Compose" section (before "VS Code Devcontainer").

Include:

1. **Overview** - Brief explanation that campaigns can use two container strategies for Docker execution

2. **Strategy Comparison Table** - Show key tradeoffs:
   | Strategy | Startup | Isolation | Use Case |
   |----------|---------|-----------|----------|
   | ephemeral (default) | 3-5s/experiment | Perfect (fresh GPU state) | Most campaigns, reproducibility |
   | persistent | Once at start | Shared (GPU memory accumulates) | Many short experiments, development |

3. **Ephemeral Mode (Default)** - Explain:
   - Uses `docker compose run --rm` per experiment
   - Fresh container = fresh GPU memory, CUDA context
   - Automatic cleanup on errors
   - Recommended for reproducible measurements
   - Negligible overhead (1-3% of typical experiment time)

4. **Persistent Mode** - Explain:
   - Uses `docker compose up -d` + `docker compose exec`
   - Containers stay running between experiments
   - Faster (no startup overhead)
   - Tradeoffs: GPU memory may accumulate, less isolation
   - Requires confirmation prompt (or --yes to skip)

5. **Configuration** - Show how to set:
   - CLI flag: `lem campaign config.yaml --container-strategy persistent`
   - User config (.lem-config.yaml):
     ```yaml
     docker:
       strategy: persistent  # or ephemeral (default)
       warmup_delay: 5.0     # seconds after container start
       auto_teardown: true   # cleanup after campaign
     ```
   - Precedence: CLI > user config > default (ephemeral)

6. **Recommendations** - When to use each:
   - Ephemeral: Production measurements, reproducibility, long experiments (>1 min)
   - Persistent: Development/debugging, many short experiments, iterating on configs

Keep the section concise (~60-80 lines). Reference the research document insight that overhead is typically 1.3% of campaign time.
  </action>
  <verify>
    - grep -q "Container Strategies" docs/deployment.md
    - Section appears between "Docker Compose" and "VS Code Devcontainer"
    - Examples show both CLI and config file usage
  </verify>
  <done>
    docs/deployment.md contains Container Strategies section with tradeoffs, configuration examples, and recommendations
  </done>
</task>

</tasks>

<verification>
- docs/deployment.md contains new "Container Strategies" section
- Section explains both ephemeral and persistent modes with tradeoffs
- Configuration via --container-strategy flag and .lem-config.yaml documented
- Recommendations help users choose appropriate strategy
</verification>

<success_criteria>
Users reading deployment.md can:
1. Understand what ephemeral vs persistent container strategies mean
2. Know the tradeoffs (isolation vs startup overhead)
3. Configure their preferred strategy via CLI or config file
4. Choose the right strategy for their use case
</success_criteria>

<output>
After completion, create `.planning/quick/002-doc-container-strategies/002-SUMMARY.md`
</output>
