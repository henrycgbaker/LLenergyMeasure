# H7 cross-cell summary - agentic-loop with tool use

**Pattern:** Tier 2, follow-up to the cheap-patterns batch (H2 + H3 + H9)
and H4. The LLM has access to tools (`read_file`, `list_validators`,
`run_miner`, `score_against`, `finalise`) and decides next action each
step. Budget: 30 tool calls OR 30 min wall-clock per cell.

**Question:** does CLOSED-LOOP FEEDBACK (LLM emits draft -> sees score
-> refines) shift the ceiling established by single-shot (b) at 70B-q4?
Or does the LLM still hit the diagnose-vs-synthesise asymmetry?

**Backend:** container Ollama @ port 11435, model `llama3.1:70b`
(q4_K_M), num_ctx=32768, temperature=0.

**Cells run:** transformers v4.57.3, vllm v0.7.3.

**Wall-clock:** 278.3 + 341.9 = 620.2s total (~10.3 min).

## Headline numbers

| Engine | Stop reason | Tool calls | Finalised? | Recall | Precision | (b) baseline recall | Delta |
|---|---|---:|:-:|---:|---:|---:|---:|
| transformers v4.57.3 | max_calls | 30 / 30 | NO | 0.000 | 0.000 | 0.538 (pure b) | **-53.8%** |
| vllm v0.7.3 | max_calls | 30 / 30 | NO | 0.000 | 0.000 | 0.385 (pure b) | **-38.5%** |

**Headline:** H7 produced ZERO finalised invariants on either cell. The
LLM exhausted the 30-call budget exploring source via tools but NEVER
synthesised an output. The trial-internal score is 0/0; closed-loop
feedback at 70B-q4 does NOT lift the ceiling - it COLLAPSES the ceiling
to zero because the LLM cannot complete the synthesis step under the
agentic protocol.

## Tool-usage patterns

| Tool | transformers calls | vllm calls | Both cells |
|---|---:|---:|---:|
| read_file | 12 | 15 | 27 |
| list_validators | 16 | 8 | 24 |
| score_against | 1 | 6 | 7 |
| run_miner | 1 | 1 | 2 |
| finalise | 0 | 0 | 0 |
| parse_error | 0 | 0 | 0 |

**Tool dispatch was 100% successful (zero parse errors).** The LLM
emitted well-formed JSON tool calls on every turn. The failure is NOT
in the harness; it's in the LLM's strategy.

### transformers tool-call sequence (paraphrased)

1. run_miner -> sees (a) covers 41 invariants
2. read_file at WRONG PATH (the LLM hallucinated `trial_transformers_v4_57_3_venv`
   instead of the actual `v4_57_6` venv) -> rejected by sandbox
3. read_file at WRONG PATH again -> rejected
4. list_validators(GenerationConfig) -> returns [] (the
   GenerationConfig validator method is named `validate`, not
   `_verify_*` or `__post_init__`, so AST helper returns empty)
5. read_file(...generation_config.py) -> rejected (wrong filename)
6. list_validators(ModelConfig) -> returns [] (not a transformers class)
7. read_file(generation/utils.py) -> ok
8. list_validators(SamplingConfig) -> returns [] (not a transformers class)
9-11. More wrong-path read attempts + one ok read.
12. list_validators(GenerationConfig) -> [] again
13-30. **LLM enters a degenerate loop**: list_validators(GenerationConfig)
   8 times in a row interspersed with sequential read_file on
   generation/utils.py. Never builds a draft. Never tries finalise.

### vllm tool-call sequence (paraphrased)

1. run_miner -> sees (a) covers 26 invariants
2. list_validators(SamplingParams) -> finds methods
3-6. Systematic read_file on config.py in 200-line slices (0-1200)
7. **score_against({"invariants": []})** -> recall=0/precision=0
8-10. Continues read_file slices (1200-2000)
11-12. list_validators(ModelConfig), score_against again with []
13-30. **LLM continues systematic config.py scan** (slices 2000-3000),
   calls score_against 6 total times - ALWAYS with empty invariants
   list. Never adds a single invariant to its draft. Never finalises.

## The synthesis-blindness pattern

Both cells show the same failure mode. The LLM:

1. **Successfully orchestrates exploration**: well-formed tool calls,
   systematic file traversal, mixes read + list operations.
2. **Calls score_against mid-loop**: the agentic primitive worked - the
   LLM understands score_against is the gate.
3. **Never writes invariants**: every score_against payload contains
   `"invariants": []`. The LLM never synthesises a draft entry between
   exploration and verification.

**This is the diagnose-vs-synthesise asymmetry from H4 manifesting
under closed-loop conditions.** H4's single-shot finding ("70B-q4
diagnoses correctly but writes pseudocode for the fix") generalises
to H7: even WITH feedback, the model doesn't bridge from "I read the
source" to "here is the structured output".

Notably, the LLM did NOT use the score_against feedback to improve.
It got recall=0 six times in a row and never adjusted strategy. The
score_against tool is a check, not a teacher; at this model scale
it doesn't translate negative feedback into corrective synthesis.

## Convergence pattern

- **No convergence on either cell.** Recall stayed at 0 across the
  full 30-call budget on both engines.
- **No plateau either** - the LLM kept exploring (sequential file
  slices on vllm config.py from 0 to 3000 lines) but never moved into
  output synthesis mode.
- **No early-exit signal**: the LLM did not call finalise after some
  arbitrary draft; it simply ran out of budget while still in
  exploration mode.

## Budget consumption

| Cell | Tool-calls used | Wall sec | LLM sec | Hit cap by |
|---|---:|---:|---:|---|
| transformers | 30 / 30 | 278.3 | 143.4 | max_calls (after 4.6 min) |
| vllm | 30 / 30 | 341.9 | 329.8 | max_calls (after 5.7 min) |

Both cells hit the tool-call cap, not the wall-clock cap. The wall-
clock budget (30 min) was generous - both cells finished well under
6 minutes. If the LLM had been finalising-ready, it would have had
plenty of headroom.

## LLM-role insight: closed-loop feedback does NOT shift the ceiling

H4 + cheap-patterns batch established the LLM-role split at 70B-q4:

| Role | Pattern | Quality | H7 impact |
|---|---|---|---|
| diagnose-only | H4 text + H9 | excellent | H7 didn't test pure diagnosis |
| validate / subtract | H2 | inconsistent (3 false-drops on vllm) | not tested in H7 |
| extend / propose | (b) + H3 verify | mixed; 0.538 recall ceiling on transformers | **H7 collapsed to 0.0 recall** |
| modify-miner / synthesise patches | H4 patches | poor (broken anchors, undefined helpers) | **H7 confirms synthesis weakness** |

The H7 finding is **STRONGER** than the H4 finding: even with iterative
tool feedback, the model doesn't synthesise. It uses tools for
exploration (which is a passive READ activity) but never enters the
SYNTHESIS mode where it must produce structured YAML output.

**Hypothesis on why:** at 70B-q4, "I have read a lot of source" and
"I have a complete invariant catalogue" feel SIMILAR to the model
internally. Without a strong-enough prior on "stop reading, start
emitting", the model defaults to reading more. Single-shot patterns
(b) FORCE synthesis by the prompt shape (one call -> emit). Agentic
patterns make synthesis optional, and the model defers it indefinitely.

## What works / what doesn't in agentic mode

### Works

- **Well-formed tool calls**: 0 parse errors across 60 turns.
- **Reasonable tool selection**: the LLM picked tools sensibly (start
  with run_miner, then list_validators, then read_file slices).
- **Systematic file traversal**: vllm config.py scan was 0-3000 in
  200-line steps, no jumps or repetition.
- **score_against discovery**: the LLM understood score_against was
  the gate; called it 6 times on vllm.

### Doesn't work

- **Invariant synthesis**: zero invariants drafted across both cells.
  The model never bridged from "I have read source" to "I emit
  structured catalogue".
- **Feedback assimilation**: 6 score_against calls returning recall=0
  did not trigger any strategy change. The model kept exploring as if
  the score was metadata to be ignored.
- **Path hallucination**: transformers cell's first read attempts used
  a non-existent venv path (`v4_57_3` instead of `v4_57_6`). Although
  the sandbox correctly rejected with a clear error message, the
  model burned several turns on this before adapting.
- **Class-name guessing**: transformers cell called
  `list_validators(ModelConfig)`, `list_validators(SamplingConfig)` -
  classes that don't exist in transformers. AST tool returned `[]`
  but the model couldn't tell whether "empty list" meant "wrong
  class name" or "this class has no validators".
- **finalise never called**: 0/2 cells finalised. The LLM has no
  internal model of "budget pressure -> emit what I have".

## Recommendation: H7-style for 70B-q4 is NOT viable

H7 produced zero useful catalogue output on both cells. The trial-
internal score is decisively worse than every other strategy tested
(pure (b), d-ab, H4 with patches applied, etc.).

**Investment recommendations:**

1. **Do NOT invest more in H7-style for Phase 4 production substrate
   at this model scale.** The empirical finding is sharp: the model
   doesn't synthesise under agentic conditions.

2. **A stronger model (claude-opus / claude-sonnet-4.5) MIGHT make
   H7 work.** Anthropic models with native tool-use and stronger
   long-form synthesis can complete the read-then-emit loop. This is
   a hypothesis - not tested in this run because the cost would be
   ~$10-20 per cell with 30 turns.

3. **Single-shot (b) IS the right shape for 70B-q4 substrate**: it
   forces synthesis by the prompt structure. The agentic flexibility
   that helps stronger models hurts the q4 model.

4. **The harness IS reusable infrastructure** even though H7-as-
   strategy is not viable. Future hybrid patterns that wrap the LLM
   in different ways (e.g. forced-synthesis-per-class with tool
   support for verification only) can use the same harness with
   different system prompts.

## Specific findings on harness behaviour

The harness itself functioned as designed. Five observations on the
tool implementations that bear on future variants:

1. **`run_miner` artefact-replay was correct**: returns count +
   namespaces + IDs in 1 call. The LLM used it. No subprocess-flakiness
   risk (vllm/tensorrt's dep wall would have blown up an actual re-run).

2. **`list_validators(<wrong_class>)` returns `[]` not an error**: this
   may have contributed to the model's path-guessing loop. An
   alternative semantic would be to return an error like "class not
   found in any source file under sandbox roots" so the model gets a
   clearer signal. Defer to future H7-variant runs.

3. **`read_file` MAX_READ_LINES=200 + truncation hint worked**: vllm's
   model walked through 3000 lines of config.py in 200-line slices
   without difficulty. The truncation hint did the right job.

4. **Sandbox path rejection worked**: transformers cell's path
   hallucinations (`v4_57_3` venv) were rejected with clear errors.
   The model adapted by trying other paths (though it adapted slowly).

5. **`score_against` with `invariants: []` returns recall=0**: this is
   correct behaviour but a different prompt that explicitly tells the
   LLM "you MUST submit a non-empty draft before score_against returns
   useful information" might shift behaviour. Speculative; not tested.

## Artefacts

Per-cell:
- `_spike/findings/hybrid_experiments/h7_agentic/<engine>/agent_trace.json` (~30k each; full tool-call log).
- `_spike/findings/hybrid_experiments/h7_agentic/<engine>/finalised_invariants.yaml` (empty in both cells).
- `_spike/findings/hybrid_experiments/h7_agentic/<engine>/score.json`.
- `_spike/findings/hybrid_experiments/h7_agentic/<engine>/observations.md` (per-cell tool sequence + tally).

Cross-cell: this file + `h7_aggregate.json` (machine-readable).

Harness + runner:
- `_spike/scripts/strategies/agentic_tool_harness.py` (~570 LoC; reusable).
- `_spike/scripts/strategies/run_h7_agentic.py` (~310 LoC; H7-specific).
- `_spike/scripts/strategies/test_agentic_tool_harness.py` (25 tests; all pass).

## Implications for Phase 4 synthesis

H7 contributes a sharp negative finding to the strategy decision space:

- **Agentic patterns at 70B-q4 are not viable for SYNTHESIS tasks.**
  The model cannot complete the read-then-emit loop without strong
  prompt scaffolding.
- **Single-shot (b) wins on output completeness.** Even with 30 tool
  calls and a generous wall-clock budget, the model produced zero
  output. The (b) baseline at 0.538/0.385 recall is the more useful
  artefact.
- **The synthesis-blindness pattern is reproducible.** H4 (single-
  shot patches) + H7 (agentic-loop) both show 70B-q4 prefers
  EXPLORATION over PRODUCTION when given the choice.
- **The harness itself unlocks future Anthropic-model variants.**
  H7 with claude-opus / claude-sonnet-4.5 is worth one cell-pair if
  budget allows; the harness is ready.

For Phase 4: H7 fails as a STRATEGY-internal pattern. It succeeds as
**negative evidence for the LLM-as-orchestrator hypothesis at 70B-q4
scale.**
