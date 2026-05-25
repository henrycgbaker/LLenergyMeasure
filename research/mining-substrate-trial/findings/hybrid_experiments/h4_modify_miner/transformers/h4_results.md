# H4 results: transformers v4.57.3

**Pattern:** H4 LLM-modifies-miner. Llama3.1:70b q4 reads the
1599-LoC mature transformers walker, defensive-import gap context, and
the inviarants.proposed.yaml output sample.

Run wall-clock: 58.3s.

## Counts

| Metric | Value |
|---|---|
| Canonical reference count | 41 |
| Baseline count (unpatched walker, system py3.12 has transformers 4.57.6) | 28 |
| Patched count | 28 (== baseline; patch didn't apply) |
| Patched run OK | True |
| Diagnoses produced | 1 |
| Patches proposed | 1 |
| Patches applied | 0 |

**Baseline-vs-canonical drift (28 vs 41) is the LARGEST of the three engines.**
The system transformers is 4.57.6, the canonical is 4.57.3 - a minor-version
bump that ALREADY changed which invariants the walker can extract. 22
canonical invariants are MISSING from baseline (i.e. these no longer
match the current source); 5 NEW invariants appear that the canonical
didn't have. This is real version-drift cost for (a), independent of H4.

The transformers WALKER itself is mature; the gap inventory's G-trf-1 is
about **brittleness at version bumps**, not about coverage gaps at the
active version. The H4 prompt invited the LLM to propose defensive-
import hardening.

## Diagnoses (1 of 1; CORRECT but RESTATING the prompt)

### G-trf-1: defensive imports for version bumps

> The walker's import block hits hard ImportError on bumped transformers
> versions (v4_55_4 / v5_9_0) when symbols renamed in `tokenizers` or
> `huggingface_hub`. Defensive `try ... except ImportError` would let
> the walker emit zero invariants gracefully instead of crashing the
> miner subprocess.

This is essentially the gap-inventory text echoed back. Not new insight,
but a correctly-framed restatement.

## Patches

### P-trf-1: wrap fragile imports in try/except

- **Status:** NOT applied (anchor not found in walker source).
- **LLM's anchor:** `from transformers import AutoConfig, AutoModel,
  AutoTokenizer, BitsAndBytesConfig\nfrom transformers.generation.
  configuration_utils import GenerationConfig\nfrom transformers.utils
  import logging, is_torch_available\nfrom huggingface_hub import HfApi`
- **Reality:** the actual walker DOESN'T have these imports. The
  transformers walker is pure-AST and uses `inspect.getsourcefile` on a
  single `transformers.generation.configuration_utils` module (line
  1462) plus a couple of utility imports. The LLM HALLUCINATED the
  import block from the engine-source EXCERPT I provided in the prompt
  (which was meta-illustrative, not from the walker).
- **Verdict:** the LLM conflated "imports the engine likely has" with
  "imports the walker has". The prompt didn't clearly distinguish
  these two surfaces.

## What worked

- The walker module ran cleanly via subprocess (transformers 4.57.6 is
  installed in the trial worktree's py3.12) without any monkey-patching.
  This validates the subprocess harness on the simplest engine.
- The LLM correctly produced ZERO patches that would damage the walker
  (anchor mismatch == defensive failure).

## What didn't work

- **Anchor hallucination**: the LLM treated the engine-excerpt block as
  an authoritative source of import-strings, even though it was an
  illustrative gloss. Lesson for future H4 prompts: be more explicit
  about which text is the patch target vs which is context.
- **No genuine coverage patches**: the transformers walker IS mature
  (1599 LoC, 41 canonical invariants). There were no obvious coverage
  gaps to invite a patch on. The single proposed patch is for
  brittleness-hardening, not coverage extension. As a result the H4
  recall-lift is zero for transformers.

## Spike-refactor value

- **Low** for transformers specifically. The walker is mature; H4
  doesn't surface mergeable patches.
- The DIAGNOSIS text restates `post_trial_a_gap_closure.md` G-trf-1
  but adds no new perspective.
- One side observation worth keeping: the 22-invariant LOSS from
  4.57.3 -> 4.57.6 (minor-version bump) confirms (a)'s brittleness
  signal even at minor bumps - relevant for Phase 4 substrate
  conclusions.

## Negative findings

- The engine excerpt I supplied in the prompt was treated by the LLM as
  the walker's import block. Future H4 prompt revisions should clearly
  partition "WALKER SOURCE" vs "ENGINE EXAMPLE" vs "BUG CONTEXT".
- For mature walkers, H4's value-add is limited: the gaps that DO
  exist are architectural (G-trf-1 is an env-hardening pattern that
  needs a small per-engine plumbing change, not an algorithmic walker
  fix). Single-shot LLM doesn't generate that kind of meta-patch
  cleanly.

## Artefacts

- `transformers/raw_llm_outputs/{prompt.txt, raw_response.txt,
  diagnoses.json, patches.json}` - LLM trail.
- `transformers/proposed_patches/P-trf-1.json` - the patch record.
- `transformers/patched_producer/static_invariant_miner.py` - patched
  walker (== baseline, since patch didn't apply).
- `transformers/baseline_unpatched_run/invariants.proposed.yaml` -
  28-invariant baseline.
- `transformers/h4_summary.json` - structured summary.
