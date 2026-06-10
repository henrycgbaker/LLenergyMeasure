# Handoff - next session (engine-config mining spike)

Self-contained kickoff. Branch `spike/engine-knowledge-as-data`. Confirm HEAD with
`git log -1` (`git fetch origin` first if anything looks off; `git log -1` after
commits - the sync hook can replay a stale snapshot). Use Opus for all subagents.
ASCII only, no Co-Authored-By, no "CLAUDE" mentions in committed files (hooks reject).

## THE ONE JOB LEFT: consolidate (per the user's exact spec)

The spike is sprawling. The user wants ONE consolidated body of writeups that is the
single source of truth, with everything old/stale archived (NOT deleted). Exact spec
(from the user, load-bearing - do this precisely):

1. WRITE consolidated findings + report + writeup docs that capture ALL important
   signal. Must include, integrated into ONE coherent narrative (not disconnected):
   - the FULL experimental design: north-star, design space (roles x assemblies x
     call-shapes x tiers; the W-A..W-G workflow taxonomy in `WAVE2_WORKFLOWS.md`),
     the runtime-gate method ("observe, don't re-encode"), prereg discipline +
     mandatory adversarial verification + internals-guard;
   - the PREVIOUS PoC findings consolidated TOGETHER with the new ones: the
     deterministic-baseline study (Round 0/0b, the 15-cell GT, `FULL_MATRIX.md`,
     `ROUND0B_BASELINE.md`), the bump-survivability cliff
     (`findings/wave2_bump_survivability.md`), the h2-h9 hybrid experiments
     (`findings/hybrid_experiments/`);
   - the NEW Phase-1 wave findings (waves 1-4) + the cross-bump work;
   - the LANGCHAIN CAVEAT prominently: we likely did NOT set up the langchain cells
     properly (the chain underperformed due to a concrete design flaw - stage 2 was
     decoupled from the source; the fix is marginal so far), so the langchain result
     is PROVISIONAL and warrants future inspection - NOT a verdict that chains fail.
2. MOVE everything old/stale into `_archive/` (a real `git mv`, preserved not
   deleted) - superseded drafts (`STUDY_SYNTHESIS.md`), per-wave docs once subsumed,
   stale planning (`LLM_PATTERNS_NEXT_SESSION.md`, dead-PoC `WAVE2_*.md` planning,
   `README.md`, `DECISIONS_LOG.md`), and the reproducible intermediates
   (`phase1_wave*/results/*_raw.txt`, `*_corpus.yaml`).
3. CROSS-REFERENCE the archive from the consolidated writeups (e.g. "per-wave detail:
   `_archive/PHASE1_WAVE3_FINDINGS.md`").
RULE: fold a doc's signal into the consolidated writeup BEFORE archiving it; show the
user the archive/keep list before doing the git mv.

The consolidation agent already drafted `CANONICAL_FINDINGS.md` (a 1-pager) +
`CONSOLIDATION_plan.md` (inventory + plan) - START from those, expand per the spec
above (they predate the head-to-head / Opus / hybrid-inversion / langchain-fix /
cross-bump results - add those).

## IN-FLIGHT when handed off (check + finish these first)

- Fixed multistage qwen2.5-coder:32b: DONE. THE FIX WORKED: infra 143->47, conf
  0->28, recall 0->20 - now COMPETITIVE with single-shot construct (30/21). The
  catastrophic 143/143-infra was purely our bad chain setup (stage 2 blind to source).
  (14b fix was marginal - 14b just isn't the catastrophic case.) So the langchain
  result is: a PROPERLY-FORMED chain matches single-shot; whether it can EXCEED it is
  OPEN (the hybrid-chain still trailed: 49 vs single-shot 55). Re-run the hybrid-chain
  with the source fix to check.
- BUMP-DIAGNOSE NOT YET RUN. Run it (needs the ollama free):
  `WAVE_OLLAMA=http://localhost:11435 PY=/tmp/round0b_venv/bin/python
   $PY scripts/phase1/wave5_bump_diagnose.py --engine vllm --old-vslug v0_19_1
   --new-vslug v0_20_0 --old-version 0.19.1 --new-version 0.20.0
   --model qwen2.5-coder:32b`
  (vllm 0.20.0 source is at /tmp/trial_vllm_v0_20_0_venv/src/vllm). It scores the LLM
  diff-reviewer (broke/new) vs the GT diff - the north-star "does the LLM catch the
  silent recall cliff" test. CAVEAT: 0.19.1->0.20.0 is a MINOR bump (94% survival),
  so the diff is small; consider also a wider/major bump for a real cliff.

## KEY RESULTS (so they are not lost) - all committed except the langchain-fix runs

THE METHOD WORKS / the strategy frontier (waves 1-4, committed, in PHASE1_WAVE{1-4}_FINDINGS):
- Wave 1: the bottleneck is the VALIDATION path, not LLM recall (single-field
  auto-synth gate can't probe the cross-field tail).
- Wave 2: kwargs-emission lever unlocks the cross-field tail (Opus 0->8 verified-real).
- Wave 3 (tier 2x2): SCALE is the threshold to reach cross-field (12B->32B);
  CODE-TUNING sharpens (32B-code beat 70B-general).
- Wave 4: CONSTRUCTION-GROUNDING (inject AST ctor signatures) is the OSS lever -
  breaks the tensorrt infra wall (0->20 verified-real); HYBRID (det floor +
  construct-grounded 32B-coder) = vllm 69% / tensorrt 59% recall. Agentic (LangGraph
  ReAct) FAILS for OSS (tool-call flakiness). Two gate fixes (cross-field locus +
  type-coercion-artifact, `_positive_is_type_coercion_artifact` in
  validate_invariants.py). Residual is a STUDY-FLOOR artifact (production
  pydantic-lift already covers PluginConfig).

70B-vs-32B + OPUS head-to-head (construct-grounding, vllm 0.19.1 single-version):
| model | recall | conf | precision | infra |
|---|---|---|---|---|
| qwen2.5-coder:32b (code) | 21 | 30 | 0.21 | 37 |
| qwen2.5:32b (general) | 15 | 22 | ~0.20 | 31 |
| llama3.1:70b (general) | 15 | 20 | ~0.19 | 26 |
| qwen2.5:72b (general) | 0 (format-failed) | - | - | - |
| Opus 4.8 (Agent tool) | 29 | 53 | 0.80 | 4 |
=> CODE-TUNING BEATS SCALE (32B-code 21 > 70B-general 15). Opus dominates STANDALONE
   (0.80 prec, 4 infra). But the HYBRID INVERSION: Opus hybrid = floor+4 net-new =
   48/80 (60%) LOSES to qwen-coder-32b hybrid = floor+11 = 55/80 (69%) - breadth
   beats precision for the floor-extension job. CAVEAT: N=1 cell, small net-new
   counts, tolerant-key over-credits - flagged for more cells.

LANGCHAIN cells (vllm, PROVISIONAL - likely bad setup):
- single-shot construct hybrid (qwen-coder-32b): hybrid 55, lift 11.
- langchain hybrid-chain (qwen-coder-32b): hybrid 49, lift 5 (worse).
- langchain multistage (qwen-coder-32b): BROKEN was 143/143 infra, 0 conf (stage 2
  decoupled from source). FIX (STAGE2 now takes `source=chunk` in `wave4_multistage.py`
  + `wave4_hybrid_chain.py`) WORKED: infra 143->47, conf 0->28, recall 0->20 -
  COMPETITIVE with single-shot construct (30/21). So the chain was badly formed, not
  fundamentally worse. OPEN: can a chain EXCEED single-shot? (hybrid-chain still
  trailed at 49 vs 55 - re-run it with the fix; try a separate repair stage; fix the
  AST-inheritance signature gaps the methodology review flagged.)

CROSS-BUMP (the north-star pivot, NEW):
- gate-acceptance degradation signal WORKS (`wave5_gate_acceptance.py`): carry old
  catalogue -> gate vs new container -> acceptance DROP = the alarm the silent
  deterministic substrate cannot raise. vllm minor bumps: 0.18.1->0.22.0 = 0.931,
  0.19.1->0.20.0 = 0.943 (5-8 broke). Survival high for MINOR bumps; the dramatic
  cliff (-0.366) needs a MAJOR refactor (v0.7.3->0.19.1 imperative->declarative, per
  wave2_bump_survivability). TWO degradation modes: (1) old invariants break
  [gate-acceptance catches]; (2) miner can't find the new surface [the silent recall
  cliff - the LLM bump-diagnose targets this].
- bump-diagnose (`wave5_bump_diagnose.py`) BUILT, NOT YET RUN (see above).

THREE ADVERSARIAL REVIEWS (committed, in findings/study/REVIEW_*.md + CONSOLIDATION_plan.md):
- north-star: spike optimised single-version recall, not cross-bump CURRENCY. USER
  REFRAMED: single-version mining IS the re-mine response to degradation, so not
  wasted; "no runtime consumer yet" is FINE - this is an info-gathering spike to
  decide what to wire up. So keep the single-version axis AND do cross-bump.
- methodology: the gate guards are PYDANTIC-ONLY and silently skip the vllm msgspec
  SamplingParams surface (so those confirms rest on manual review); recall-vs-GT
  partly circular (GT built from same sources); verified-real counts not auditable
  (prose only); AST extractor IGNORES INHERITANCE (may confound construct-grounding
  model-specificity). Cheap high-value fixes: make gate error-extraction
  msgspec-aware; commit chunked inputs + verify records. (NOT yet done.)
- consolidation: the cleanup plan + the CANONICAL_FINDINGS draft.

## INFRA POINTERS

- Container ollama: `ollama_w3` on host :11435, `--gpus all`, volume `/tmp/ollama_w3`
  (has qwen2.5-coder 14b/32b, qwen3-coder:30b, deepseek-coder-v2:16b, llama3.1:70b,
  qwen2.5:32b/72b, devstral:24b, command-r:35b). Set `WAVE_OLLAMA=http://localhost:11435`.
  GOTCHA: the shared host REAPS the container when idle - if `docker ps` shows it
  gone, `docker run -d --gpus all -v /tmp/ollama_w3:/root/.ollama -p 11435:11434
  --name ollama_w3 ollama/ollama` (volume persists, no re-pull). If ollama hangs
  (stuck generation), `docker restart ollama_w3`.
- Driver venv: `/tmp/round0b_venv/bin/python` (tree-sitter, pyyaml, langgraph,
  langchain-ollama). NO pip - use `uv pip install --python /tmp/round0b_venv/bin/python`.
- Gate: `scripts/study_gt_pilot.py` (load+gate) -> `scripts/validate_invariants.py`
  in the engine container. Floor = improved-det-v2
  (`findings/trial_runs/wave2/w2-a-improved-det-v2/`). GT:
  `findings/study/ground_truth/<engine>/<vslug>/invariants/PILOT_GT.yaml`.
- Sources (EPHEMERAL /tmp): `/tmp/trial_<engine>_<vslug>_venv/src/<pkg>`. vllm 0.19.1
  + 0.20.0 present; others may need re-extraction (`docker create <image>` +
  `docker cp` the package out, per the wave5 extraction).
- Opus rung = the Agent tool (no API key for langchain-anthropic). Spawn an Opus
  subagent to extract -> writes a corpus -> gate it with study_gt_pilot.
- Runners: scripts/phase1/wave{1,4_pure,4_construct,4_multistage,4_hybrid_chain,
  4_selfconsistency}.py + wave5_{gate_acceptance,bump_diagnose}.py + the *_sweep.sh
  + wave3_dump_confirmed.py (re-gate a corpus -> CONFIRMED.yaml). Result schema in
  the JSONs; corpora are the deduped proposals; CONFIRMED.yaml has per-confirm detail.
- GOTCHAS: nested `nohup &` inside a background bash gets orphaned (launch runs as
  direct background tasks). `pgrep -f wave4_multistage` matches a script whose
  command line contains that string (self-kill) - target the python explicitly.

## GOTCHAS / discipline

- Mandatory adversarial source-verification of EVERY cross-field confirm before
  counting it real (the inflation class recurs). Internals-guard (drop private/
  underscore fields, type-trivia, observability, launch-state).
- Mining is COMPREHENSIVE by design (expose a subset downstream); allowlist is
  exposure-time not mining-time.
- Spawn an adversarial Opus subagent at checkpoints to verify correctness AND
  north-star alignment; don't self-certify long autonomous runs.
- After committing, `git log -1` to confirm HEAD is yours (sync hook).
