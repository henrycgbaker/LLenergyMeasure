# Wave 2 batch 2 ground-truth prompts

**Status:** Ready to fire 2026-06-05 ~3am. 3 Opus agents, run in parallel, xhigh effort. Refinements vs batch 1 are baked in (see DECISIONS_LOG 2026-06-05 batch-1-synthesis entry).

## Launch instructions

Launch via 3 parallel `Agent` tool calls with `subagent_type=general-purpose`, `model=opus`, `run_in_background=true`. Use the prompts below verbatim per engine. Soft cap ~150k tokens per agent.

After launch: wait for completion notifications. When all 3 in, repeat the batch-1 synthesis pattern (per-result short report + cross-engine table + decisions log entry).

---

## Prompt 1: transformers v5.6.2

```
Establish PROPER GROUND TRUTH for `transformers` at version 5.6.2 for the LLEM mining-substrate trial.

**Worktree:** /home/h.baker@hertie-school.lan/workspace/llenergymeasure-trial

**Engine source (build if missing):**
ls /tmp/trial_transformers_v5_6_2_venv/src/transformers/ 2>/dev/null || \
  uv run python research/mining-substrate-trial/scripts/venv_setup.py --engine transformers --version-slug v5_6_2

**Existing baseline (read so you know the delta):**
- engine_versions/transformers/v5_6_2/outputs/  (if present)
- engine_versions/transformers/v5_6_2/producers/

**Prior ground truth for the BUMP-PAIR (read this — your job includes per-bump delta):**
- research/mining-substrate-trial/findings/ground_truth/transformers/v4_57_3/ (just landed; the v_old GT)

**Output locations (create dirs):**
- research/mining-substrate-trial/findings/ground_truth/transformers/v5_6_2/schema_ground_truth.json
- research/mining-substrate-trial/findings/ground_truth/transformers/v5_6_2/invariants_ground_truth.yaml
- research/mining-substrate-trial/findings/ground_truth/transformers/v5_6_2/methodology.md
- research/mining-substrate-trial/findings/ground_truth/transformers/v5_6_2/delta.md ← PRIMARY DELIVERABLE
- research/mining-substrate-trial/findings/ground_truth/transformers/v5_6_2/version_delta.md ← also primary: explicit list of what changed between v4.57.3 and v5.6.2 (you have both GTs by the time you finish)

**Scope — enumerate exhaustively. Categories batch 1 v4.57.3 agent FOUND MISSING vs baseline (don't repeat that mistake):**

1. from_pretrained(**kwargs) — PreTrainedModel + AutoModelForCausalLM. v5 may have added new kwargs (tp_plan, cache_implementation refactors).
2. GenerationConfig — every field + every validate() invariant. v5 may have reorganised this.
3. **18 quantization configs** beyond BNB (AwqConfig, GPTQConfig, HqqConfig, EetqConfig, QuantoConfig, AqlmConfig, QuarkConfig, FbgemmFp8Config, TorchAoConfig, SpQRConfig, CompressedTensorsConfig + any v5-new ones).
4. CacheConfig FAMILY (this is a v5+ surface — DynamicCache, StaticCache, SlidingWindowCache, HybridCache, MambaCache, etc.). Enumerate per-cache fields + validators.
5. CompileConfig — v5+ addition for torch.compile.
6. WatermarkingConfig, SynthIDTextWatermarkingConfig.
7. **TRANSFORMERS_* + HF_* env vars** — batch 1 v4.57.3 found 38 of these missing entirely. Enumerate from transformers.utils.hub + huggingface_hub.constants. ALSO INCLUDE THE IMPORT-TIME-BINDING GOTCHA (e.g. HF_HUB_OFFLINE bound at import; setting after import is no-op).
8. pipeline() kwargs.
9. **9 generate-only kwargs rejected by GenerationConfig.validate** (logits_processor, stopping_criteria, streamer, assistant_model, negative_prompt_ids, ...). Batch 1 found these missing despite 2-line walk. Confirm they're still present in v5.
10. Cross-package peer configs (bitsandbytes, accelerate device_map semantics).

**Method (xhigh effort)**

1. Read existing baseline + read v4.57.3 GT to know your starting point.
2. Read v5.6.2 source thoroughly. Walk entry-points + references. Use ast / tree-sitter for systematic coverage.
3. Cross-reference docs online: hf.co/docs/transformers at v5.6.2 + github at v5.6.2 tag.
4. EXPLICIT VERSION DELTA: compare to v4.57.3 GT. What's added / removed / renamed?
5. Emit canonical envelope shape.

**Quality bar (per batch 1)**

- Every entry has citation (file, line, qualname).
- Every invariant: id, native_field, predicate_kind, severity, message_template, kwargs_positive/negative, citation.
- Every env var: name, default, citation, behaviour-impact.
- delta.md: itemised additions vs baseline.
- version_delta.md: itemised diff vs v4.57.3 GT.

**Discipline**

- ASCII only. No em-dashes / en-dashes. No emojis. No Co-Authored-By: Claude footer (project hook rejects).
- Do NOT commit; main agent reviews.

**Report back (under 250 words)**

(1) total fields, (2) total invariants, (3) total env vars, (4) delta count vs baseline, (5) delta count vs v4.57.3 GT (added + removed + renamed), (6) 2-3 high-value v5-specific findings, (7) any low-confidence sections.
```

---

## Prompt 2: vllm v0.19.1

```
Establish PROPER GROUND TRUTH for `vllm` at version 0.19.1 for the LLEM mining-substrate trial.

**Worktree:** /home/h.baker@hertie-school.lan/workspace/llenergymeasure-trial

**Engine source (build if missing):**
ls /tmp/trial_vllm_v0_19_1_venv/src/vllm/ 2>/dev/null || \
  uv run python research/mining-substrate-trial/scripts/venv_setup.py --engine vllm --version-slug v0_19_1

**Existing baseline:**
- engine_versions/vllm/v0_19_1/outputs/ (if present)
- engine_versions/vllm/v0_19_1/producers/

**Prior ground truth for bump-pair (read for version delta):**
- research/mining-substrate-trial/findings/ground_truth/vllm/v0_7_3/

**Output locations (create dirs):**
- research/mining-substrate-trial/findings/ground_truth/vllm/v0_19_1/{schema_ground_truth.json,invariants_ground_truth.yaml,methodology.md,delta.md,version_delta.md}

**Scope. Important: vllm 0.19.1 restructured config.py into a SUBPACKAGE (vllm/config/*). Per-concern modules. Don't assume the 0.7.3 layout.**

Categories batch 1 v0.7.3 agent FOUND MISSING vs baseline (cover all of these):

1. EngineArgs — every field, type, default.
2. SamplingParams (msgspec) — every field + validators.
3. BeamSearchParams (mode-exclusive with SamplingParams).
4. PoolingParams — likely out of LLEM scope but enumerate + flag.
5. Subconfig classes at v0.19.1 (SUBPACKAGE layout): ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, LoadConfig, LoRAConfig, PromptAdapterConfig, ObservabilityConfig, MultiModalConfig, SpeculativeConfig, DecodingConfig, GuidedDecodingParams, VllmConfig, CompilationConfig, KVTransferConfig + any others present at v0.19.1.
6. **vllm.envs — entire env-var surface (87 entries at v0.7.3; may have grown).** This was missed entirely by baseline; enumerate every VLLM_*. Note source-vs-stub footguns (var name documented vs var name actually read).
7. **Silent-normalisation invariants at VllmConfig.__post_init__** (MLA disables prefix_caching + chunked_prefill; LoRA disables torch.compile; cpu_offload disables torch.compile). Check v0.19.1 equivalents.
8. **Per-platform check_and_update_config under vllm/platforms/*** (v0.7.3 agent flagged this as low-confidence gap). Walk per-platform module for the platform-conditional invariants.
9. **Quantization sub-config tree** (AWQ / GPTQ / FP8 / ...) — v0.7.3 agent flagged this as opaque-treated. Enumerate per-quantizer config classes.
10. GuidedDecodingParams / StructuredOutputsConfig — constrained-decoding parameters.

**Method (xhigh effort)**

1. Read existing baseline + read v0.7.3 GT.
2. Read v0.19.1 source thoroughly. The subpackage layout (vllm/config/*) replaces the 0.7.x monolithic config.py.
3. Cross-reference docs.vllm.ai (0.19.x) + github vllm-project at v0.19.1 tag.
4. EXPLICIT VERSION DELTA vs v0.7.3 GT: what's added / removed / renamed.
5. Emit canonical envelope shape.

**Quality bar / Discipline:** same as v0.7.3 agent (cite source for every entry; ASCII only; no AI footer; don't commit).

**Report back (under 250 words)**

(1) total fields, (2) total invariants, (3) total env vars (gap probe), (4) delta vs baseline, (5) delta vs v0.7.3 GT, (6) 2-3 high-value v0.19-specific findings (e.g. did SpeculativeConfig grow, did the subpackage refactor add/remove invariants), (7) low-confidence sections.
```

---

## Prompt 3: tensorrt-llm v1.2.1

```
Establish PROPER GROUND TRUTH for `tensorrt-llm` at version 1.2.1 for the LLEM mining-substrate trial.

**Worktree:** /home/h.baker@hertie-school.lan/workspace/llenergymeasure-trial

**Engine source (build if missing):**
ls /tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm/ 2>/dev/null || \
  uv run python research/mining-substrate-trial/scripts/venv_setup.py --engine tensorrt --version-slug v1_2_1

**Existing baseline:**
- engine_versions/tensorrt/v1_2_1/outputs/ (if present)
- engine_versions/tensorrt/v1_2_1/producers/

**Prior ground truth for bump-pair:**
- research/mining-substrate-trial/findings/ground_truth/tensorrt/v0_21_0/

**Output locations (create dirs):**
- research/mining-substrate-trial/findings/ground_truth/tensorrt/v1_2_1/{schema_ground_truth.json,invariants_ground_truth.yaml,methodology.md,delta.md,version_delta.md}

**Scope. Important: TRT-LLM 1.x represents a major-version jump from 0.x. Class structure likely reorganised. Don't assume 0.21 layout.**

Categories batch 1 v0.21.0 agent FOUND MISSING vs baseline:

1. LlmArgs / TrtLlmArgs / BaseLlmArgs / TorchLlmArgs — every field + validators. v1.x may have unified or further-split these.
2. SamplingParams (TRT's own) + per-mode validators.
3. KvCacheConfig.
4. **PluginConfig (43 fields + Blackwell SM-100 killswitches at v0.21)** — entire metaclass-generated property tree. Confirm + extend at v1.2.1.
5. **Full speculative-decoding tree**: MedusaDecodingConfig, EagleDecodingConfig (with Eagle3Config replacement), NGramDecodingConfig, DraftTargetDecodingConfig, MTPDecodingConfig. Plus LookaheadDecodingConfig (v0.21 agent flagged C++-side defaults as low-confidence).
6. **BuildConfig (27 fields at v0.21)** — was opaque in baseline; v1.2.1 likely expanded further.
7. CalibConfig (6/7 fields missed at v0.21).
8. SchedulerConfig.
9. **TLLM_*/TRTLLM_* env vars (44 at v0.21).** Confirm + extend. Note TRTLLM_DG_* DeepGEMM JIT vars (v0.21 partial).
10. LoraConfig, PromptAdapterConfig.
11. **C++ pybind boundary**: at v0.21 the agent treated tensorrt_llm.bindings classes (ModelConfig, WorldConfig, TrtGptModelOptionalParams, raw ExecutorConfig - 17 of 19 classes) as out-of-scope. **Make an explicit scope call for v1.2.1**: include them or document why excluded.

**Method (xhigh effort)**

1. Read existing baseline + read v0.21.0 GT.
2. Read v1.2.1 source thoroughly. Note that LlmArgs may have unified or refactored.
3. Cross-reference nvidia/TensorRT-LLM github at v1.2.1 + nvidia docs.
4. EXPLICIT VERSION DELTA vs v0.21.0 GT.
5. Emit canonical envelope.

**Quality bar / Discipline:** same as v0.21 agent. ASCII only. No AI footer. Don't commit.

**Report back (under 250 words)**

(1) total fields, (2) total invariants, (3) total env vars, (4) delta vs baseline, (5) delta vs v0.21.0 GT (note major-version structural changes), (6) 2-3 high-value v1.x findings, (7) low-confidence sections + your C++ pybind scope call.
```

---

## Post-batch-2 plan

After all 3 v_new agents return:

1. Per-result short report to user (same pattern as batch 1).
2. Cross-engine + per-bump-pair synthesis: deltas, multipliers, what changed across versions.
3. Append all to DECISIONS_LOG.md.
4. Consider follow-up gap-fill agents for the v0.21 LookaheadDecodingConfig C++ + vllm quantization subconfig tree + vllm/platforms/* — but those decisions are user's call.
5. Update `WAVE2_PRIMITIVES.md` Axis 8 (version situation) with empirical bump-survivability data.
6. Write `WAVE2_NEXT_SESSION.md` — the handoff doc.
7. Optionally implement `a_improved_det.py` per `findings/wave2_improved_det_primitives.md` so next session can run it as a Wave 2.0 cell.
