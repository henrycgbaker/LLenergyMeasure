# Wave 2.6 deliverable: LLM-role matrix (quality per role per scale)

Axis 2 of WAVE2_PRIMITIVES: what is the LLM actually GOOD at, given the model
scales reachable this run (<=14B OSS)? Source: `wave2_llm_cells.json` (extract,
extend roles measured this wave) + Wave 1 evidence (diagnose, patch, agentic) +
the bump-survivability need for an external signal.

## Role x evidence

| LLM role | measured this wave? | quality at <=14B OSS | notes |
|---|---|---|---|
| `extract` (pure propose) | yes (pure-b) | POOR: 0.05-0.12 recall, 4-30x below floor | non-viable standalone; the ~50% Wave-1 ceiling was at 70B-q4 and does not survive to 7B |
| `extend` (propose residual on a floor) | yes (W-G) | WEAK: +0.00-0.04 recall, precision DROPS | net-negative value once precision + gate cost counted |
| `gate` (yes/no on candidates) | partial | UNMEASURED live | gate infra needs per-entry kwargs replay fields the prompts omitted; the runtime-validate gate itself is the necessary precision cleanup (hallucination proxy 0.87-1.0) |
| `diagnose` (categorical gaps) | no (Wave 1: strong) | promising | Wave 1: 0 fabrications across 8 diagnoses at 70B-q4; the lowest-risk role |
| `decide` / `curate` | no | untested | deferred |
| `patch-code` (W-D) | no (Wave 1: 0/3) | poor | model-scale-dependent; unlikely to improve at 7-14B |

## Findings

1. **At OSS-small scale the LLM is bad at PRODUCTION roles (extract, extend) and
   should be used only in JUDGMENT roles.** Every extraction/extension number this
   wave is at-or-below the deterministic floor with a precision penalty. The model
   does not know the engine's invariants well enough to enumerate them; it DOES
   (per Wave 1) reliably reason about a SPECIFIC candidate or gap.
2. **The highest-value LLM role for this problem is diagnose / diff-review**, not
   extraction. It directly serves the bump-survivability gap: the deterministic
   floor silently under-emits on a refactor bump (vllm cliff), and an LLM that
   reads the v_old->v_new diff and flags "these invariant CATEGORIES likely
   changed" is the external degradation signal the self-update workflow needs -
   and it is the role OSS models are good at.
3. **Any LLM-proposed entry must pass the runtime gate.** Hallucination proxy
   0.87-1.0 (over-counts, since GT is a minimum set, but the direction is
   unambiguous): small models emit mostly-unverifiable entries. The gate is
   non-optional for any LLM-touching path.

## Recommendation to the engineering session

Allocate the LLM budget to GATE / DIAGNOSE / DIFF-REVIEW roles, not extraction.
Specifically: an LLM diff-reviewer over (v_old catalogue, v_new source) to detect
silent floor degradation + propose categories to re-mine, with every concrete
proposal gated by runtime-validate. Reserve extraction/extend for a frontier-scale
re-test (Wave 3) before committing the LLM to a producing role.

## Deferred

- gate role as a live measured cell (needs kwargs_positive/negative-bearing
  prompts + the per-engine runtime gate containers).
- diagnose / diff-review live cells (W-F) at OSS + frontier scale.
- decide / curate (long-running maintenance simulation).
