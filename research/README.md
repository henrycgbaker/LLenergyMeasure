# research/

This directory holds long-running research artefacts (empirical trials, methodology explorations, decision-space surveys) that aren't part of the production codebase. The contents are exploratory: locked prompts, raw LLM transcripts, per-cell extraction outputs, scoring tables, and the narrative DECISIONS_LOG that captures how each trial unfolded.

The directory is included by default in fresh clones. Casual contributors who don't need the research corpus can opt out:

```bash
bash scripts/setup-research-optin.sh
```

To restore, run `git sparse-checkout disable`.

## Catalogue

| Trial | Status | Deliverable |
|---|---|---|
| [`mining-substrate-trial/`](mining-substrate-trial/) | ACTIVE (Phase 3c pending external API key) | `mining-substrate-trial/findings/empirical_trial_outcome.md` |

## Conventions

- Production code lives under `src/`. This directory is research/exploration only.
- Findings markdown is reviewer-facing prose; reproducibility scripts live alongside in each trial's `scripts/` directory.
- Per-trial sub-READMEs describe scope, lifecycle status, and reproducibility instructions.
- Lint / format / type-check gates do not run against this directory (see `pyproject.toml` `extend-exclude`).
