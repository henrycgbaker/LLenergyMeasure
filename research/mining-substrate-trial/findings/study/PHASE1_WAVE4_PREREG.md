# Phase 1, Wave 4 - pre-registration (pure-LLM assembly x call-shapes x roster)

Status: 4a RUNNING; 4b/4c built, run after 4a. Builds on wave-3
(`PHASE1_WAVE3_FINDINGS.md`): the LLM rung sweet spot for det-then-llm-extend is
a mid code-tuned ~32B. This wave drops the deterministic floor entirely
(llm-only assembly) and varies the CALL-SHAPE, to find shape x model sweet spots.

## Objective

Map what each model achieves at the PURE-LLM (llm-only, no floor) assembly across
three call-shapes, scored by the real runtime gate. Headline contrasts:
- shape: pure-LLM vs wave-3's det-then-llm-extend (does the det floor's
  scaffolding matter, or can the LLM extract the full catalogue unaided?);
- agency: prompt (4a) vs agentic source-exploration (4b) vs agentic +
  gate-feedback (4c);
- model: a broadened, ollama-verified roster (code-tuned + a general anchor),
  incl the MoE cost-frontier datapoint (qwen3-coder:30b = 30B-A3B, ~3B active).

## Call-shapes (all llm-only; kwargs-emission carried into every shape)

- 4a PROMPT: read source + locked prompt, chunked, emit invariants + kwargs. No
  floor in prompt; dedup INTERNALLY (not vs floor); gate; recall vs GT.
  Locked prompt: `phase1_wave4/pure_llm_kwargs_prompt.md` (wave-2 kwargs prompt
  minus the floor input + floor-dedup rule). Runner: `wave4_pure.py`.
- 4b AGENTIC extract-only (LangGraph): agent with tools list/grep/read source +
  emit_invariant; builds the catalogue by exploring source; gate scores AFTER.
  Tests whether agentic source-navigation beats single-shot chunked prompting.
  Runner: `wave4_agentic.py --mode extract_only`.
- 4c AGENTIC gate-as-tool (LangGraph): adds `gate_probe` so the agent tests each
  probe against the REAL gate and self-corrects before emitting. Tests whether
  gate-feedback rescues the "emit correct kwargs" bottleneck. EXPENSIVE (one
  container gate call per probe = the per-bump CI-cost concern - that cost IS the
  finding). Runner: `wave4_agentic.py --mode gate_tool`.

## Roster (ollama-verified tags; served by containerized ollama :11435)

4a/4b across the code roster: qwen2.5-coder:14b, qwen2.5-coder:32b (wave-3
winner / anchor), qwen3-coder:30b (MoE 30B-A3B), deepseek-coder-v2:16b (2nd
family, MoE-lite), llama3.1:70b (general anchor). 4b/4c require tool-calling
(qwen-coder, qwen3-coder, llama3.1 confirmed tool-capable; devstral:24b /
command-r:35b available if a tool-tuned datapoint is wanted). Opus ceiling via
langchain-anthropic optional later. codellama:70b (stale 2023 large-code anchor)
deferred.

## Cells / metrics

- Cells: vllm 0.19.1 + tensorrt 1.2.1 (same as waves 1-3, for comparability).
- Metrics per (shape, model, cell): full-catalogue RECALL vs GT (tolerant);
  verified-real cross-field count; gate-confirmed precision; verdict breakdown;
  cost = wall-sec (gen) + (4c) gate-call count. 4a/4b sequential on one ollama so
  wall-sec stays a clean cost signal.
- Headline tables: shape contrast (4a vs det-then-llm-extend at qwen-32b);
  agency ladder (4a -> 4b -> 4c at fixed model); MoE efficiency (qwen3-coder:30b
  3B-active vs qwen2.5-coder:32b dense).

## Mandatory soundness + guard (unchanged, load-bearing)

Every cross-field confirm at every (shape, model) is adversarially source-verified
before counting (the wave-2/3 inflation class). The internals-guard (drop
private/underscore fields, type-validation trivia, observability, runtime-launch
state) is applied to the candidate-config tally. Headline number = verified-real
candidate-config, never raw confirms.

## Discipline / deviations

- Locked: pure-LLM prompt body (`pure_llm_kwargs_prompt.md`); agentic system
  prompts (SYSTEM_4B/4C in `wave4_agentic.py`); models = ollama digests recorded
  at run; engine container digests as wave-3.
- All runs on the containerized ollama (:11435) for consistent memory/GPU
  (sidesteps the shared :11434 cgroup cap). Generation temp=0, num_ctx 16384.
- 4b/4c run AFTER 4a (not concurrent) so 4a wall-times are clean.
- N=2 cells, single-shot per chunk (4a). DIRECTIONAL.
