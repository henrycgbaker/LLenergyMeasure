# Zero-Config Defaults

**Status:** Proposed
**Date decided:** N/A — not yet decided
**Last updated:** 2026-02-20
**Research:** N/A

---

## Context

`llem run --model X` (or equivalent) should work with no YAML file, no user config, and no
prior setup. This decision defines exactly what defaults apply in that scenario, and how far
"zero-config" extends.

Related but separate: how users graduate from zero-config to full config is in
[progressive-disclosure.md](progressive-disclosure.md). This ADR covers only the defaults
themselves and the failure modes when dependencies are absent.

These decisions are prerequisite for finalising `designs/cli-commands.md` and
`designs/experiment-config.md`.

---

## Considered Options

### Q1 — What does `llem run --model X` resolve to?

When no YAML is provided and no user config exists, the effective values for each field:

| Field | Current default | Open questions |
|-------|----------------|---------------|
| `backend` | interactive prompt if >1 installed | Does it prompt even if only 1 backend installed? Auto-select if 1? |
| `precision` | `bf16` | Should it validate against GPU capability first? |
| `dataset` | `aienergyscore` | Does it auto-download on first run? With confirmation? |
| `n` | `100` | Is 100 prompts the right zero-config default? |
| `batch_size` | `1` | |
| `decoder.decoding_strategy` | `greedy` | |
| `n_cycles` | `1` | For quick single run; see execution model decision. |
| `output_dir` | `./results/` | Auto-creates on first run? |
| `energy_backend` | `nvml` | Falls back gracefully if NVML unavailable? |

Not yet decided — these defaults require confirmation.

### Q2 — What happens when dependencies are missing at zero-config time?

| Option | Pros | Cons |
|--------|------|------|
| **[Preferred — not confirmed]** Hard fail with exact install command | Predictable; consistent with pip extras model; no magic | Slightly less friendly for first-time users |
| Soft prompt — "no backend installed; run `pip install llenergymeasure[pytorch]`?" | More interactive guidance | Requires interactive prompt infrastructure at error time |
| Silent fallback to CPU-only mode | Never blocks | CPU-only is likely wrong for energy measurement; misleads users |

**Rejected (2026-02-20):** Silent CPU fallback. Energy measurement without a GPU produces
meaningless results for the tool's primary use case. A clear error is more honest.

### Q3 — Does zero-config work without GPU?

| Option | Pros | Cons |
|--------|------|------|
| **[Preferred — not confirmed]** Fail immediately with clear error | Honest; avoids silent incorrect results | No escape hatch for testing tool behaviour |
| Run in "dry-run-only" mode showing what would happen | Useful for CI/development | Adds complexity; dry-run mode needs its own design |
| Run with CPU for correctness testing (no energy measurement) | Allows integration testing without GPU | Two-mode execution complicates result schema |

Not yet decided.

### Q4 — What is the zero-config output filename format?

With no user config and no YAML, the result filename is auto-generated.

| Option | Pros | Cons |
|--------|------|------|
| `{model_slug}_{timestamp}.json` | Minimal; always unique | Doesn't encode backend or precision — hard to distinguish results |
| **[Preferred — not confirmed]** `{model_slug}_{backend}_{timestamp}.json` | Encodes most-differentiating parameter | Longer filename |
| `{model_slug}_{backend}_{precision}_{timestamp}.json` | Maximum descriptiveness | Very long; precision often inferred from backend |

Not yet decided.

### Peer Patterns

| Tool | Zero-config behaviour |
|------|-----------------------|
| `lm-eval` | Requires `--model` and `--tasks` at minimum; no defaults for those |
| `optimum-benchmark` | Requires launcher config YAML |
| `cargo run` | Runs project from `Cargo.toml` — project-scoped defaults |
| `gh pr create` | Interactive prompts for required fields (title, body) |
| `docker run nginx` | Pulls image, uses image defaults for everything else |

---

## Decision

Not yet decided. The defaults table above is a working draft based on existing v1.x
behaviour. All four questions must be confirmed before this ADR moves to `Accepted`.

Rationale: N/A — pending decision session.

---

## Consequences

Positive: N/A — pending.

Negative / Trade-offs: N/A — pending.

Neutral / Follow-up decisions triggered:
- Resolution is prerequisite for `designs/cli-commands.md` and `designs/experiment-config.md`
- Must align with interactive mode scope in [progressive-disclosure.md](progressive-disclosure.md) Q3

---

## Related

- [progressive-disclosure.md](progressive-disclosure.md) — how users graduate from zero-config
- [cli-ux.md](cli-ux.md) — existing zero-config defaults decision (row 19 has some confirmed values)
- [../designs/cli-commands.md](../designs/cli-commands.md) — full `llem run` signature (blocked on this decision)
- [../designs/experiment-config.md](../designs/experiment-config.md) — field definitions (blocked on this decision)
