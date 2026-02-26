# Progressive Disclosure & Complexity Ladder

**Status:** Proposed
**Date decided:** N/A — not yet decided
**Last updated:** 2026-02-20
**Research:** N/A

---

## Context

llem must serve two user types simultaneously:
- **Quick user**: "I want to measure this model, now, with no setup"
- **Research user**: "I want a rigorous, reproducible, multi-config study for a paper"

Progressive disclosure means complexity is always available but never in the way. Users
should be able to start at the simplest tier and graduate upward without hitting walls or
being forced to restart from scratch.

Several open questions remain about where the tier boundaries sit and how upgrade paths
are surfaced. These must be resolved before `designs/cli-commands.md` can be finalised.

---

## Considered Options

### The Complexity Ladder (Draft)

The proposed five-tier structure:

```
Tier 0 — Zero-config (no YAML, no user config)
  llem run --model meta-llama/Llama-3.1-8B
  → interactive prompts for missing required fields
  → sensible defaults everywhere else

Tier 1 — Single config YAML
  llem run experiment.yaml
  → all params explicit, reproducible
  → no sweep, no execution protocol

Tier 2 — Study YAML with sweep
  llem run study.yaml
  → Cartesian parameter sweep
  → execution block: n_cycles, cycle_order

Tier 3 — Study YAML + user config
  ~/.config/llenergymeasure/config.yaml
  → machine-specific defaults (runner, gaps, energy backend)
  → separates "what to measure" from "how this machine runs"

Tier 4 — Programmatic (library API)
  llem.run_experiment(config) → ExperimentResult
  llem.run_study(config) → StudyResult
  → full control, CI/CD integration, custom orchestration
```

### Peer Patterns

| Tool | Disclosure approach |
|------|--------------------|
| `git` | Full flag set exposed; `git commit` → sensible defaults; `--amend`, `--rebase` are progressive |
| `cargo` | `cargo run` works in any project; `Cargo.toml` for full control |
| `docker` | `docker run image` works instantly; `docker compose` for full orchestration |
| `gh pr create` | Interactive prompts for missing required fields; `--fill` auto-fills from branch |
| `pytest` | `pytest` with no args runs all tests in cwd; `pytest.ini` for full config |

### Q1 — Can users graduate tier-by-tier without rewriting?

When a user at Tier 0 wants to save their config (Tier 1):

| Option | Pros | Cons |
|--------|------|------|
| `llem run ... --save-config experiment.yaml` | Explicit user intent; familiar pattern (cf. `gh pr create --fill`) | Extra flag to remember |
| `llem run ... --emit-config` | Decoupled from save location | Non-standard naming; unclear if it writes or just prints |
| Automatic config echo after run | Zero extra effort from user | Clutters run output; unclear what to do with it |

Not yet decided.

### Q2 — Where does the complexity boundary live?

At what point should users be directed to a YAML vs inline flags?

| Option | Pros | Cons |
|--------|------|------|
| All flags mirrored in YAML (any YAML field = available as flag) | Maximum flexibility; no arbitrary limits | CLI becomes unwieldy; complex fields ill-suited to flags |
| **[Preferred — not confirmed]** Only "common" flags at CLI level (backend, model, precision, n) | Clean interface; forces YAML for complex config | Arbitrary boundary; power users frustrated |
| Complex fields (decoder, warmup, lora) require YAML only | Keeps CLI clean; YAML is natural for those fields | May block quick experiments with non-default decoder settings |

Not yet decided.

### Q3 — Interactive mode scope

At Tier 0, which missing fields trigger interactive prompts vs silent defaults?

| Option | Pros | Cons |
|--------|------|------|
| **[Preferred — not confirmed]** Required fields prompt (model, backend if >1 installed); optional fields use silent defaults | Mirrors `gh pr create` pattern; avoids surprise | Backend auto-select if only 1 installed may surprise users on their first multi-backend setup |
| All unspecified fields prompt | Maximum transparency | Tedious for experienced users |
| No interactive prompts; fail with helpful error | Predictable scripting behaviour | Poor experience for first-time users |

Not yet decided. See also [zero-config-defaults.md](zero-config-defaults.md) for the full
defaults table.

### Q4 — Does the complexity ladder need to be surfaced in the CLI?

| Option | Pros | Cons |
|--------|------|------|
| `llem help` shows the ladder explicitly ("Getting Started → Common Usage → Advanced") | Discoverable; reduces docs dependency | Help text bloat; most tools don't do this |
| `llem config` passive output includes "next step" guidance | Contextual; already a natural entry point | May feel out of place in a status/env display command |
| **[Preferred — not confirmed]** Docs-only: no CLI surfacing needed | Clean CLI; standard practice for mature tools | Relies on users finding docs |

Not yet decided.

---

## Decision

Not yet decided. The five-tier ladder structure is the working draft. All four questions
above (save-config UX, flag boundary, interactive scope, CLI surfacing) must be resolved
before this ADR can move to `Accepted`.

Rationale: N/A — pending decision session.

---

## Consequences

Positive: N/A — pending.

Negative / Trade-offs: N/A — pending.

Neutral / Follow-up decisions triggered:
- Resolution blocks finalisation of `designs/cli-commands.md`
- Zero-config defaults table (Q3) must align with [zero-config-defaults.md](zero-config-defaults.md)

---

## Related

- [zero-config-defaults.md](zero-config-defaults.md) — Tier 0 specifics and defaults table
- [cli-ux.md](cli-ux.md) — Confirmed CLI decisions (some Tier 0 defaults already confirmed there, row 19)
- [../designs/cli-commands.md](../designs/cli-commands.md) — Full command signatures (blocked on this decision)
