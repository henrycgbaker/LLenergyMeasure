# Phase 2 - LLM infrastructure design + calibration

**Status:** Phase 2 closed; Phase 2.5 follow-on closed. Schema recall
hit target at 83.0%; invariant recall at 53.8% (under the Phase 2.5
4-tuple identity rubric; +12.8pp over the rubric-fix-only baseline of
41.0%; 21pp under the 75% target).

Calibrated recall hit 75%? **Schema YES** (83.0%); **invariants NO**
(53.8%). Phase 3 launches with the gap documented; see § 11 verdict.

**Authored:** 2026-05-25 (Phase 2); updated 2026-05-25 with Phase 2.5
section.
**Cross-refs:**
- `.planning/mining-substrate-empirical-trial.md` § Per-strategy
  infrastructure needs / (b) Pure OSS LLM (the plan this phase
  operationalises).
- `research/mining-substrate-trial/findings/trial_epistemic_framing.md` § Pure vs hybrid (the
  discipline rule that prompts lock at Phase 2 end).
- `research/mining-substrate-trial/findings/bakeoff_B_local_llm.md` (the prior POC that informed
  the four design lessons).
- `research/mining-substrate-trial/findings/phase1_trial_runner_design.md` § 8 Concrete next-
  steps for Phase 2 (the contract this phase satisfies).
- `research/mining-substrate-trial/findings/phase2_locked_prompts/` (the locked-prompts archive
  this doc summarises).

---

## 1. Container Ollama setup

### Commands

```bash
# One-time:
docker volume create trial-ollama-models
docker run -d --name trial-ollama \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -p 11435:11434 \
    -v trial-ollama-models:/root/.ollama \
    ollama/ollama serve

# Pull both models:
docker exec trial-ollama ollama pull llama3.1:70b   # 42 GB
docker exec trial-ollama ollama pull llama3.1:8b    # 4.9 GB

# Verify GPUs:
docker exec trial-ollama nvidia-smi -L   # expect 4 A100-PCIE-40GB

# Verify Ollama API:
curl -s http://localhost:11435/api/tags | jq '.models[].name'
```

### Port mapping

- **11434**: host Ollama (existing; another user; don't touch).
- **11435**: container Ollama (trial-only; mapped from container 11434
  to host 11435 to avoid collision).

### Model inventory

| Model | Quantisation | Disk | GPU footprint @ 32k ctx | Primary use |
|---|---|---|---|---|
| `llama3.1:70b` | Q4_K_M | 42 GB | ~52 GB across 2 A100-40GBs | Strategy (b) primary |
| `llama3.1:8b` | Q4_K_M | 4.9 GB | ~10 GB single A100 | M5 speed-quality probe |

### Context-window setting

- Default Ollama context: **128k** → 70B spills 35% to CPU. Catastrophic
  for throughput (Bake-off B's 35min schema call was this slow).
- Trial setting: **num_ctx=32768**. Keeps llama3.1:70b fully on GPU
  across 2x A100-40GB. Per-chunk wall-clock drops from ~5-10min to
  ~30-120s.
- Trial setting: **num_predict=4096**. Caps generation length; most
  chunks need <2k output tokens.
- Trial setting: **temperature=0**. Deterministic for repro.
- Trial setting: **keep_alive="30m"**. Keeps model loaded across
  chunks (cold-load is ~20s, would dominate per-chunk wall-clock).

### Container management

```bash
docker ps --filter name=trial-ollama        # status
docker logs --tail 50 trial-ollama          # log tail
docker exec trial-ollama ollama list        # model inventory
docker exec trial-ollama nvidia-smi         # GPU usage inside container
docker stop trial-ollama && docker rm trial-ollama   # tear down (data persists in volume)
docker volume rm trial-ollama-models        # destroy model cache (45GB+)
```

### Resource footprint observed (Phase 2 calibration)

- 70B loads on GPUs 0+1 at 32k ctx (~52 GB total across the two cards).
- GPUs 2+3 untouched during single-cell calibration; available for
  Phase 3 parallel cell runs.
- Model disk footprint: 45 GB total in the `trial-ollama-models`
  volume.

---

## 2. (b) infrastructure design

### Chunking strategy

Implemented in `research/mining-substrate-trial/scripts/strategies/transformers_chunker.py`.

**Schema (3 chunks)**:
1. `from_pretrained_engine_params` - PreTrainedModel.from_pretrained
   signature + docstring (8000-char truncation; the body's kwarg-pop
   block is included for internal-plumbing recognition).
2. `bitsandbytes_compile_configs` - BitsAndBytesConfig +
   CompileConfig + WatermarkingConfig source inlined. **Companion-
   class lesson** from Bake-off B: the LLM doesn't follow imports;
   inline them.
3. `generation_config_sampling_params` - GenerationConfig.__init__
   + docstring (the docstring documents fields that have no type
   annotation in the function signature).

**Invariants (1 + ~12 chunks)**:
1. `generation_config_init_invariants` - GenerationConfig.__init__
   construct-time type checks + CompileConfig reference.
2. `validate_section_NN_<label>` - validate() body split by the
   `# 1.x` / `# 2.x` comment markers. Sections include decoding
   attrs, cache attrs, performance attrs, watermarking, sampling-
   only-when-greedy, beam-only-when-greedy, num_return_sequences
   cross-field, cache-related cross-field, etc.

Each chunk is <15k chars (~3.5k tokens). With 32k ctx + ~4k output
budget, there's ample headroom even with the prompt's few-shot
examples.

### Structured output strategy

- **Schema**: Ollama `format: "json"` mode + JSON Schema validation
  via `jsonschema` library. The schema (in
  `research/mining-substrate-trial/scripts/strategies/prompts.py::SCHEMA_JSON_SCHEMA`) requires
  `engine`, `engine_version`, `chunk_fields` at top level; each field
  entry has a `namespace` (enum-constrained) plus optional `type`,
  `default`, `enum`, `anyOf`, `description`.
- **Invariants**: plain-text mode + post-parse YAML shape check. YAML
  doesn't lend itself to JSON Schema validation; we accept any shape
  that parses + merges.

### Retry strategy

`llm_extractor.extract_with_retry()`:
- Send prompt; parse response.
- On parse failure: re-prompt with a corrective preamble that
  includes the parse error + the previous (truncated) output.
- On schema-validation failure (schema task only): same as above.
- `max_retries=2` per chunk (3 attempts total). After exhaustion,
  the chunk is logged as `failure_modes=['parse_failure_after_retries']`
  in observations; the executor continues with other chunks.

### Code-fence stripping

`llm_extractor.strip_code_fences()` handles:
- ```yaml / ```yml / ```json opening fences.
- ``` closing fences (including missing closing).
- Leading commentary prose ("Here is the YAML:") - the YAML parser
  steps past the prose to the first `key:` line.
- Multi-fence outputs (very rare; takes first block).

Bake-off B lost the ENTIRE invariants score to fence wrapping; this
parser is the recovery.

### Companion-class handling

The schema chunker (#2) inlines `BitsAndBytesConfig` source DIRECTLY
in the prompt rather than relying on the LLM to "follow the import"
from `from_pretrained`. Same for `CompileConfig` (used in sampling
params) and `WatermarkingConfig`.

### Internal-plumbing filter

Two-line defence:
1. **Prompt rule** (line 3 of CRITICAL RULES in
   `SCHEMA_PROMPT_TEMPLATE`): "Skip internal-plumbing fields: any
   name starting with `_` and these explicit names: `adapter_kwargs`,
   `model_kwargs`, `torch_dtype`."
2. **Post-filter** (`filter_internal_plumbing`): defensive belt-and-
   braces in case the LLM emits one anyway.

The blocklist is intentionally narrow - over-blocking would hide
legitimate fields like `_commit_hash` (which IS internal but
sometimes appears in user-facing places).

### Few-shot strategy

Schema prompt: 6 examples covering primitive bool, PathLike anyOf,
BitsAndBytesConfig field, sampling_params with enum from validate(),
$defs entry, sampling_params with docstring-inferred default. Drawn
from the v4_57_3 reference.

Invariants prompt: 3 examples covering ERROR (enum violation),
DORMANT (sampling-only when greedy), CROSS-FIELD ERROR
(num_return_sequences > num_beams).

### Output deduplication + merging

After all chunks complete:
- **Schema**: merge `chunk_fields` by `namespace`; same-namespace
  field-name collisions last-write-win (rare; mostly defensive). Drop
  internal-plumbing fields (post-filter).
- **Invariants**: concatenate per-chunk `invariants: [...]` arrays;
  deduplicate by `id` AND by `(namespace, native_field,
  predicate_kind)` identity tuple (mirrors `trial_scoring`'s
  canonical identity).

---

## 3. Calibration results

### Round 1 (baseline; 3 schema chunks + 13 invariants chunks)

See `research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/calibration_round_1_report.md`.

| Metric | Round 1 result |
|---|---|
| Schema recall | **51.8%** (58/112) |
| Schema precision | 92.1% (58/63) |
| Schema type accuracy | 60.3% |
| Invariant recall | **32.1%** (9/28 identities) |
| Invariant precision | 64.3% (9/14) |
| Invariant severity accuracy | 77.8% |
| Wall-clock | 480s (8 min) |
| Energy | 21.7 Wh |

**Headline:** Schema recall in the same range as Bake-off B's 52.3%
single-prompt baseline; chunking + 32k context delivered the same
quality at **4.4x speed-up** (Bake-off B was 35 min wall-clock).
Invariant recall lower than Bake-off B's projected 40-60% - artifact
of the canonical identity rubric collapsing multi-field invariants
(28 unique identities vs 41 reference invariants).

**Failure modes observed:**
- 21 engine_params fields missed - documented via the kwargs.pop
  pattern + docstring, but chunk 1's docstring truncation at 8000
  chars left the latter half of from_pretrained's docstring (where
  `device_map`, `dtype`, `attn_implementation`, `tp_plan`, etc. are
  documented) out of the prompt.
- 33 sampling_params missed - GenerationConfig.__doc__ truncated at
  8000 chars of 19000 total; the latter half (where most
  kwargs.pop'd fields are documented with type + default) was elided.
- 8 BitsAndBytesConfig type-check invariants missed - the invariants
  chunker did NOT have a BitsAndBytesConfig chunk in round 1; only
  GenerationConfig.__init__ + validate() sections.
- Multiple sampling-only-when-greedy dormancy invariants collapsed
  into one - the LLM emitted ONE invariant keyed by `do_sample` instead
  of one-per-field. Round 2 prompt fix: explicit "one per FIELD" rule
  in chunk-specific extra_context for sections 06 and 07.

### Round 2 (docstring expansion + BNB invariants chunk)

See `research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/calibration_round_2_report.md`.

Prompt deltas vs round 1:
- Schema chunks expanded from 3 to 5; added separate
  `from_pretrained_docstring_kwargs` (latter half of docstring) and
  `generation_config_docstring` (FULL doc) chunks.
- Invariants chunks expanded from 13 to 14; added
  `bitsandbytes_config_invariants` chunk explicitly targeting BNB type
  checks.
- Section 06/07 invariants prompts got an "emit one per field, not
  one per gate" extra_context.

| Metric | Round 2 result | Delta |
|---|---|---|
| Schema recall | **83.0%** (93/112) | **+31.2pp** |
| Schema precision | 93.9% (93/99) | +1.8pp |
| Schema type accuracy | 57.0% | -3.3pp |
| Invariant recall | **32.1%** (9/28) | 0pp |
| Invariant precision | 64.3% (9/14) | 0pp |
| Invariant severity accuracy | 77.8% | 0pp |
| Wall-clock | 1118s (18.6 min) | +638s |
| Energy | 50.3 Wh | +28.6 |

**Headline:** Schema recall **CLEARED the 75% target** with the docstring-
expansion fix (83.0%). Invariants unchanged - the BNB chunk had a
YAML parse failure after 3 retries because the LLM emitted unquoted
`torch.dtype` in a `type_is_not` list (interpreted as nested YAML
key). The salvage-parse fix (round 3) attempts to rescue partial
output from these failures.

**Failure modes observed:**
- BNB invariants chunk: `parse_failure_after_retries` due to
  unquoted dotted-name `torch.dtype` in YAML list. Output was
  partially extractable (8+ valid invariants visible in raw text).
- Schema type accuracy DROP (60.3% -> 57.0%): the new docstring
  chunks produced more permissive `anyOf` types (e.g. `anyOf: [null,
  object, string]`) where the reference has primitive types. This
  is a scoring-rubric strictness artefact, not a quality regression.
- Dormancy expansion (one-per-field): the LLM correctly produced
  one invariant per field in section 06 (7 invariants for
  temperature/top_p/min_p/typical_p/top_k/epsilon_cutoff/eta_cutoff)
  but the canonical identity rubric deduplicates by FIRST-FIELD;
  since all 7 are keyed by `do_sample: False`, they collapse to one
  identity. Reference catalogue has the SAME structure (28 unique
  identities vs 41 invariants) so the rubric is consistent but the
  recall ceiling is artificially low.

### Round 3 (YAML parse salvage + namespace param)

See `research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/calibration_round_3_report.md`.

Prompt + code deltas vs round 2:
- `parse_yaml_block()` salvage logic: on parse failure, try
  parsing successively-shorter prefixes ending at the LAST complete
  `- id:` entry boundary. Defensive against truncated invariants.
- `INVARIANTS_PROMPT_TEMPLATE` parameterised on `field_namespace`:
  per-chunk namespace (e.g. `transformers` for BNB engine_params
  invariants, `transformers.sampling` for validate() invariants)
  flows into the prompt's predicate-form examples + match-field
  template. This unblocked the bnb chunk's intended namespace.

| Metric | Round 3 result | Delta vs round 2 |
|---|---|---|
| Schema recall | **83.0%** (93/112) | 0pp |
| Schema precision | 93.9% (93/99) | 0pp |
| Schema type accuracy | 57.0% | 0pp |
| Invariant recall | **60.7%** (17/28) | **+28.6pp** |
| Invariant precision | 65.4% (17/26) | +1.1pp |
| Invariant severity accuracy | 88.2% | +10.4pp |
| Wall-clock | 890s (14.8 min) | -228s |
| Energy | 37.0 Wh | -13.3 |

**Headline:** Invariant recall **nearly doubled** (32.1% → 60.7%) -
the bnb chunk + namespace parameterisation unlocked 8 more
identities. Schema unchanged (already locked at round 2's 83.0%).
The non-deterministic Ollama run gave us a different output this
time on the bnb chunk - first attempt parsed cleanly without
needing the salvage logic. The salvage is still load-bearing as a
fallback for the next non-determinism.

**Failure modes observed:**
- 11 of 28 reference identities still missed - mostly multi-field
  invariants where the LLM emits the right CONTENT but the canonical
  identity rubric collapses them (e.g. all sampling-only-when-greedy
  dormancies map to `(transformers.sampling, do_sample, exact)`).
  Reference catalogue has 41 invariants → 28 identities; LLM
  produces 26 invariants → ~17 unique identities.
- Severity accuracy improved to 88.2% (from 77.8%) - the namespace
  fix incidentally improved severity classification.

### Locked round

Per plan rule "cap prompt-iteration cycles at 3 per strategy; lock
prompts at Phase 2 end".

**Prompts locked at end of round 3.**

Archive: `research/mining-substrate-trial/findings/phase2_locked_prompts/`:
- `schema_prompt.md` - the full text of `SCHEMA_PROMPT_TEMPLATE`.
- `invariants_prompt.md` - the full text of `INVARIANTS_PROMPT_TEMPLATE`.
- `hybrid_extension_prompt.md` - the hybrid extension prompt.
- `hybrid_diagnose_prompt.md` - the hybrid diagnose prompt.
- `prompts.py.lockedref` - source snapshot.
- `transformers_chunker.py.lockedref` - chunker snapshot.

### Target assessment

Plan target was **75% recall** on both schema and invariants.

| Axis | Final (Round 3) | Target | Hit? |
|---|---|---|---|
| Schema recall | 83.0% | 75% | **YES** |
| Invariant recall | 60.7% | 75% | NO (missed by 14.3pp) |
| Schema precision | 93.9% | (no target; very high) | n/a |
| Invariant precision | 65.4% | (no target) | n/a |
| Severity accuracy | 88.2% | (no target) | n/a |

**Honest gap report on invariant recall**:

The canonical identity rubric (namespace, primary_field, predicate_kind)
caps the achievable recall artificially. Reference catalogue has 41
invariants but only 28 unique identities - 13 logical invariants
collapse into the same tuple. The LLM round-3 produced 26 invariants
mapping to 17 identities → 60.7% on rubric.

If we score by INVARIANT COUNT (not identity), the LLM hit
approximately 26 of 41 = **63%**. Either way, we missed the 75%
target on invariants.

Plausible Phase 2.5 fixes to close the gap (not in Phase 2 scope):
1. **Revise the canonical identity** to include `secondary_field`
   for multi-field invariants - this would let the dormancy-per-field
   invariants score as 7+ distinct identities, not 1.
2. **Multi-pass refinement** (subsume what hybrid d-ab does for pure
   (b)): after the chunked extraction, do ONE more pass asking the
   LLM to find anything it missed.
3. **Per-section prompt tuning** for the validate() dormancy block -
   explicitly enumerate the expected per-field invariants in the
   few-shot examples.

---

## 4. Failure-modes catalogue

### Round 1 failure modes observed

| Failure | Frequency | Round 2/3 mitigation |
|---|---|---|
| Truncated docstring in from_pretrained chunk | 21 engine_params missed | Round 2: separate `from_pretrained_docstring_kwargs` chunk |
| Truncated GenerationConfig docstring | 33 sampling_params missed | Round 2: full 16k-char chunk for docstring |
| BNB type-check invariants not extracted | 8 invariants missed | Round 2: dedicated `bitsandbytes_config_invariants` chunk |
| Per-field dormancy collapsed to one entry | ~6 invariants collapsed | Round 2: per-section prompt hint to emit one-per-field |
| Code-fence wrapping in JSON/YAML | None blocked | `strip_code_fences()` worked from round 1 |
| Internal-plumbing leakage | 0 (filtered) | Prompt rule + post-filter worked |

### Round 2 failure modes observed

| Failure | Frequency | Round 3 mitigation |
|---|---|---|
| BNB chunk YAML parse failure (unquoted dotted name?) | 1 chunk (bnb) | Round 3: `parse_yaml_block()` salvage logic; non-determinism gave clean parse |
| Schema namespace confusion in invariants prompt | Bnb identities used wrong namespace | Round 3: `field_namespace` template parameter |
| Type accuracy regression (60.3% → 57.0%) | Many anyOf vs primitive mismatches | Scoring-rubric strictness; not a real regression |

### Round 3 failure modes observed

| Failure | Frequency | Phase 2.5 / Phase 3 follow-on |
|---|---|---|
| Multi-field invariant identity collapse | 13 logical invariants → 5 identities | Revise canonical identity to include secondary_field |
| ~13 invariants still missing | LLM doesn't emit all reference items | Multi-pass refinement (Phase 2.5 #2) |
| Reference catalogue has spurious-looking entries | ~5 from LLM look right but reference disagrees | Phase 3 runtime-validation feedback (already designed in trial_scoring) |

### Failure modes designed AROUND (no longer reproducible)

| Failure | Mitigation |
|---|---|
| Markdown-fence wrapping | `strip_code_fences()` parser; retry-on-fail |
| Long-signature truncation | Chunked by class/method |
| BitsAndBytesConfig under-walking | Companion-class inlined in chunk #2 + dedicated invariants chunk |
| Internal-plumbing leakage (`_commit_hash`, `adapter_kwargs`) | Prompt rule + post-filter |
| Output truncation under high context | num_ctx=32k keeps model on GPU |
| Long generation timeout | streaming + per-chunk 1800s budget |
| YAML parse failure mid-chunk | `parse_yaml_block()` salvage |

---

## 5. 8B vs 70B probe (M5)

Trial ran the locked round-3 prompts through `llama3.1:8b` on the
SAME transformers v4_57_3 cell. Recorded under
`research/mining-substrate-trial/findings/trial_runs/b_8b/transformers/v4_57_3/`.

Plan check: "If 8B is 80%+ quality at 10x speed, that materially
changes (b)'s economics for the full matrix."

### M5 result

| Metric | 70B (round 3) | 8B (locked prompts) | Delta |
|---|---|---|---|
| Schema recall | 83.0% | **85.7%** | +2.7pp (8B narrowly WINS) |
| Schema precision | 93.9% | 93.2% | -0.7pp |
| Invariant recall | 60.7% | 35.7% | -25.0pp |
| Invariant precision | 65.4% | 16.1% | -49.3pp |
| Wall-clock | 890s | **413s** | 2.2x faster |
| Energy | 37.0 Wh | **4.9 Wh** | 7.5x less |

### Verdict

8B is NOT a drop-in replacement for 70B in (b):
- **Schema**: 8B is comparable (slightly better recall, slightly
  worse precision; type accuracy similar). For pure schema
  extraction the 7.5x energy reduction is compelling.
- **Invariants**: 8B drops sharply on recall (~59% of 70B quality)
  AND blows up on precision (~25% of 70B - generates 62 invariants
  vs 70B's 26, mostly spurious or hallucinated).
- **Severity accuracy**: 100% on 8B's overlap (10/10) but only 10
  identities overlap.
- **Speed**: 2.2x faster, not the 10x the plan threshold required.

**Operationally**: 8B is viable if the trial's economics dominate
(low-budget rerun on a new engine version, schema-only). For the
empirical trial's main matrix, 70B is the right primary model.

8B's role in Phase 3: SCHEMA-ONLY probe per (engine, version) cell
- cheap parallel schema discovery to confirm `engine_versions/...`
references, even when the full (b) cell uses 70B.

### Failure modes observed (8B-specific)

- One chunk (`validate_section_03_1.3._Performance_attributes`)
  hit `parse_failure_after_retries`.
- One chunk (`validate_section_08_2.4._check_num_return_sequences`)
  parsed but yielded 0 unique invariants (deduplicated to nothing).
- 8B's higher cell_count (62 invariants emitted vs 70B's 26) is the
  precision-blowup cause - the smaller model is more permissive
  about emitting candidate invariants.

---

## 6. (d-ab) hybrid scaffolding

Module: `research/mining-substrate-trial/scripts/strategies/hybrid_extractor.py`.

### Architecture

```
(a) deterministic output  -->  engine_versions/<e>/v<v>/outputs/invariants.proposed.yaml
                                                  |
                              +-------------------+--------------+
                              |                                  |
                              v                                  v
                  hybrid extension prompt           reconciliation merge
                  (LLM reads (a)'s output + source)
                              |
                              v
                   YAML with three sections:
                   1. added_by_llm_verifier
                   2. flagged_spurious_in_deterministic
                   3. missed_diagnosis
                              |
                              v
                   merge into proposed.yaml (with x-conflict + flagged_for_review)
                   write reconciliation.yaml (LLM annotations alone)
```

### Contract

`hybrid_extractor.run_d_ab_on_transformers_active(out_dir, ...)`:
- Reads (a)'s `invariants.proposed.yaml` for the cell.
- Reads validate() + __init__ source via `transformers_chunker`.
- Sends the FULL (a) output + a curated source summary in ONE prompt
  (Phase 2 doesn't chunk hybrid; Phase 3 may revisit for engines
  where (a)'s output exceeds context budget).
- Parses the three YAML sections.
- Emits:
  - `invariants.proposed.yaml`: (a)'s entries + LLM extension
    entries (with `added_by: llm_verifier`, `flagged_for_review:
    true`). Flagged-spurious entries get `x-conflict` annotation;
    NOT removed.
  - `reconciliation.yaml`: LLM annotations only (extension /
    spurious / diagnosis).
  - `schema.json`: copy of (a)'s schema (hybrid Phase 2 is
    invariants-only).

### (d-ac) variant

Same module; swap `backend=OllamaBackend()` for `AnthropicBackend()`
via `run_d_ac_on_transformers_active`. Same prompt verbatim.

---

## 7. (c) Claude SDK contract stub

Module: `research/mining-substrate-trial/scripts/strategies/claude_extractor.py`.

### Activation path

`run_c_on_transformers_active(out_dir, ...)`:
- Instantiates `AnthropicBackend()`; raises `KeyAbsentError` if
  `ANTHROPIC_API_KEY` not in env.
- Otherwise reuses (b)'s prompt + chunking machinery verbatim.

When the key arrives:
1. `export ANTHROPIC_API_KEY=...`
2. `uv add anthropic` if not installed.
3. Phase 3 cells start producing (c) outputs alongside (b).

### Cost cap mechanism

- `AnthropicBackend.per_cell_usd_cap = 5.0` (default; matches plan).
- Logged after each call; trial_runner can choose to halt the cell
  if exceeded. Phase 2 doesn't enforce hard halts (logs only) - the
  coordinator does trial-wide enforcement.
- Trial-wide cap **$75** per the plan.

### Prompt caching

The `AnthropicBackend.call` heuristic splits the prompt into
INSTRUCTIONS (before `=== SOURCE:`) and SOURCE (everything from
`=== SOURCE:` onward); the SOURCE block is marked `cache_control:
ephemeral` so retries hit the cache (~90% input-token savings on
hit).

---

## 8. Phase 3 readiness checklist

### Trial_runner contract

`trial_runner.py` orchestrates per-cell execution. The 5 dispatchers
are wired:

- [x] `run_strategy_a(spec)` - returns canonical
  `engine_versions/<e>/v<v>/outputs/` paths.
- [x] `run_strategy_b(spec)` - dispatches to
  `llm_b_oss.run_b_on_transformers_active` for transformers active;
  raises NotImplementedError for other engines/versions (Phase 3
  builds vllm/tensorrt chunkers + venv plumbing).
- [x] `run_strategy_c(spec)` - dispatches to
  `claude_extractor.run_c_on_transformers_active`; raises
  KeyAbsentError until ANTHROPIC_API_KEY arrives.
- [x] `run_strategy_d(spec)` - dispatches to
  `hybrid_extractor.run_d_ab` / `run_d_ac` based on
  `spec.strategy`.
- [x] `measure_energy_during(thunk)` - NVML via
  `select_energy_sampler('auto')`.
- [x] `run_cell(spec)` - end-to-end orchestrator with crash record.
- [x] `_emit_crash_record(spec, exc)` - placeholder CellScore for
  crashed cells; special-cases KeyAbsentError → failure_mode
  `key_absent`.

### Scoring contract

`trial_scoring.py` is wired:

- [x] `score_cell()` - dispatches to score_schema + score_invariants;
  produces CellScore + diff artefacts.
- [x] `score_schema()` - recall/precision/type-accuracy + ItemDiff
  per missed/spurious/type-mismatch.
- [x] `score_invariants()` - recall/precision/severity-accuracy + ItemDiff.
- [x] Self-scoring the reference yields perfect scores (test:
  `test_score_cell_self_scores_perfectly`).

### Cell registry

`CELL_REGISTRY` seeded with:
- Strategy (a) on all 3 active-version cells.
- Strategies (b), (c), (d-ab), (d-ac) on transformers v4_57_3
  active.

Phase 3 appends:
- vllm/tensorrt active cells for (b)/(c)/(d-ab)/(d-ac).
- All bumped cells (v-2, v-1, v+1, v+major) per the Day 3 PyPI
  version lock.

### What Phase 3 needs that Phase 2 doesn't provide

Phase 2.5 status notes:
- **[x] vllm chunker** - `research/mining-substrate-trial/scripts/strategies/vllm_chunker.py`
  (Phase 2.5 P25-1; 7 schema + 10 invariants chunks).
- **[x] tensorrt chunker** - `research/mining-substrate-trial/scripts/strategies/tensorrt_chunker.py`
  (Phase 2.5 P25-2; 7 schema + 6 invariants chunks).
- **[x] Per-engine-per-version venvs** - `research/mining-substrate-trial/scripts/venv_setup.py`
  source-only venv scaffolding (Phase 2.5 P25-6). Phase 3 threads
  `lazy_build=True` through `trial_runner.resolve_cell_config`.

Still on Phase 3:
- Reference catalogues for non-active cells (Day 4 deferred per
  Phase 1 trial-runner-design Q1 recommendation: defer; score only
  active cells in Phase 3).
- Aggregator implementation (`trial_aggregate.build_matrix`,
  `emit_markdown`, `emit_csv` are still stubbed per Phase 1 design
  Q5; Phase 3 fills these).
- The schema JSON Schema's `namespace` enum is transformers-specific
  (`engine_params`, `sampling_params`, `$defs.CompileConfig`, ...).
  Phase 3 must extend the enum for vllm + tensorrt namespaces when
  those cells run.

### Phase 3 invocation patterns (verified)

All five strategies execute cleanly via the trial_runner CLI:

```bash
# Strategy a (uses canonical engine_versions outputs)
uv run python -m _spike.scripts.trial_runner --strategy a --engine transformers --version-slug v4_57_3
# -> schema_recall=1.0 inv_recall=1.0 (self-score by design)

# Strategy b (full LLM extraction; ~15 min wall-clock)
uv run python -m _spike.scripts.trial_runner --strategy b --engine transformers --version-slug v4_57_3
# -> schema_recall=0.83 inv_recall=0.61 (round 3 locked)

# Strategy c (Claude; raises KeyAbsentError without ANTHROPIC_API_KEY)
uv run python -m _spike.scripts.trial_runner --strategy c --engine transformers --version-slug v4_57_3
# -> failure_modes=['key_absent']

# Strategy d-ab (hybrid; ~20s wall-clock for the LLM extension call)
uv run python -m _spike.scripts.trial_runner --cell-spec d-ab/transformers/v4_57_3
# -> schema_recall=1.0 inv_recall=1.0 (carries (a)'s output forward + 2 extensions)
```

---

## 9. Phase 2.5 follow-on items

Listed in priority order. Each is testable; none blocks Phase 3
start. Status added 2026-05-25 after Phase 2.5 closure; see § 10
for details on the DONE items.

1. **8B vs 70B probe (M5)** [DONE in Phase 2; § 5]
2. **Multi-pass refinement** [DONE in Phase 2.5; § 10.2] - the plan
   calls for "extract -> verify -> extend". Phase 2 shipped extract-only.
   Phase 2.5 adds verify + extend prompts as new LLM calls per chunk;
   merged via `extract_invariants_multipass`.
3. **Wider engine chunkers** [DONE in Phase 2.5; § 10.4] - the
   calibration is transformers-only; vllm + tensorrt chunkers built
   in `research/mining-substrate-trial/scripts/strategies/{vllm_chunker, tensorrt_chunker}.py`.
4. **Streaming + per-chunk progress UI** [DEFERRED] - currently
   per-chunk wall-clock is silent until the chunk finishes. Phase 3
   can revisit if matrix execution wall-clock becomes a bottleneck.
5. **Runtime-validation feedback** [DONE-scaffolding in Phase 2.5;
   § 10.6] - per Phase 1 design Q3: `trial_scoring.runtime_validate_invariants`
   replays kwargs_positive/kwargs_negative against the live library.
   Transformers-only for Phase 2.5; other engines need their canonical
   container (Phase 4).
6. **Anthropic SDK prompt-caching tuning** [DEFERRED] - the cache_control
   heuristic in `AnthropicBackend.call` is approximate; tuning needs
   (c) actually running. Phase 3 follow-on.
7. **Parallel cell execution** [DEFERRED] - per Phase 1 design Q5:
   the LLM stack supports concurrent Ollama requests; trial_runner
   could `--parallel N` for (b/c) cells. Phase 3 runtime concern.
8. **Rubric fix (Phase 2.5 addition)** [DONE; § 10.1] - the canonical
   invariant identity was a 3-tuple; Phase 2.5 extends to a 4-tuple
   `(namespace, native_field, predicate_kind, secondary_field)` to
   disambiguate multi-field invariants.
9. **Per-engine source-only venvs (Phase 2.5 addition)** [DONE; § 10.5]
   - `research/mining-substrate-trial/scripts/venv_setup.py::ensure_source_only_venv` builds
   `/tmp/trial_<engine>_<version_slug>_venv/` lazily for non-active
   cells.

---

## 10. Phase 2.5 follow-on closure

Phase 2.5 closed the deferred items from § 9. Status summary:

| Item | Status | Notes |
|---|---|---|
| Multi-pass refinement | DONE | `research/mining-substrate-trial/scripts/strategies/llm_b_oss.py::extract_invariants_multipass` |
| Rubric fix (identity 3-tuple -> 4-tuple) | DONE | `research/mining-substrate-trial/scripts/trial_scoring.py::invariant_identity` |
| vllm chunker | DONE | `research/mining-substrate-trial/scripts/strategies/vllm_chunker.py` (7 schema + 10 invariants chunks) |
| tensorrt chunker | DONE | `research/mining-substrate-trial/scripts/strategies/tensorrt_chunker.py` (7 schema + 7 invariants chunks) |
| Per-engine venvs | DONE (source-only) | `research/mining-substrate-trial/scripts/venv_setup.py::ensure_source_only_venv` |
| Runtime-validation feedback | DONE (transformers only) | `research/mining-substrate-trial/scripts/trial_scoring.py::runtime_validate_invariants` |
| Streaming progress UI | DEFERRED | non-blocker; Phase 3 may revisit |
| Anthropic cache tuning | DEFERRED | needs (c) running first |
| Parallel cell execution | DEFERRED | Phase 3 runtime concern |

### 10.1 Rubric fix impact

The original 3-tuple identity ``(namespace, native_field, predicate_kind)``
collapsed multi-field invariants into single identities. Reference
catalogue had 41 invariants but only 28 unique identities; round-3 b
output had 26 invariants -> 17 unique identities -> 60.7% recall.

Phase 2.5 extends the identity to a 4-tuple:
``(namespace, native_field, predicate_kind, secondary_field)``.

Empirical impact on the round-3 (b) output (re-scored with the fixed
rubric, no other changes):

| Metric | Round 3 BEFORE fix | Round 3 AFTER fix | Delta |
|---|---|---|---|
| Reference identities | 28 | 39 | +11 (multi-field disambiguation) |
| Cell identities | 17 | 26 | +9 (1:1 with raw invariant count) |
| Intersection | 17 | 16 | -1 |
| Invariant recall | 60.7% | **41.0%** | -19.7pp |
| Invariant precision | 65.4% | 61.5% | -3.9pp |
| Severity accuracy | 88.2% | 87.5% | -0.7pp |

**Interpretation**: the original 60.7% was inflated by identity
collapse. The 41.0% is the HONEST recall under the strict rubric.
The rubric fix REVEALS rather than HIDES the gap.

The plan's projected ~7pp recall gain was incorrect - the rubric fix
alone is a recall ceiling DROP (more identities to find), not a
recall floor LIFT. The actual recall recovery requires multi-pass
refinement (next sub-section).

### 10.2 Multi-pass refinement pipeline

Architecture (per chunk):
1. **Pass 1 (extract)** - unchanged from Phase 2 round 3.
   Prompt: `INVARIANTS_PROMPT_TEMPLATE` (locked).
2. **Pass 2 (verify)** - re-prompt with pass-1 output + source.
   Prompt: `INVARIANTS_VERIFY_PROMPT_TEMPLATE` (new).
   Output: `confirmed:` + `flagged:` (with `fix:` actions).
   Reconciliation: `fix: drop` removes the invariant; other fixes
   recorded as observations (pass-1 prompt is locked; we don't
   auto-modify it).
3. **Pass 3 (extend)** - re-prompt with confirmed pass-1 + flags + source.
   Prompt: `INVARIANTS_EXTEND_PROMPT_TEMPLATE` (new).
   Output: invariants that pass 1 missed.
   Reconciliation: new entries tagged `added_by: llm_miner_pass3`;
   identities clashing with pass-1 dropped.

Rule 3 of `INVARIANTS_EXTEND_PROMPT_TEMPLATE` explicitly enumerates
pass-1's known failure modes (per-field collapse, multi-clause
collapse, type-check skip, predicate-kind regression). This is
structured-CoT prompting - the LLM does the recall lift, the prompt
orchestrates which gaps to fill.

Locked prompts archive:
- `research/mining-substrate-trial/findings/phase2_locked_prompts/invariants_verify_prompt.md`
- `research/mining-substrate-trial/findings/phase2_locked_prompts/invariants_extend_prompt.md`

### 10.3 Multi-pass + rubric fix calibration

Run command (reproducible):

```bash
uv run python -m _spike.scripts.strategies.run_multipass_calibration \
    --out-dir research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/phase2_5 \
    --schema-source research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/calibration_round_3_schema.json
```

Outputs:
- `phase2_5/invariants.proposed.yaml`
- `phase2_5/phase2_5_score.json`
- `phase2_5/phase2_5_report.md`
- `phase2_5/raw_llm_transcripts/{pass1,pass2_verify,pass3_extend}__<chunk>.md`

Reuses round-3 schema (locked at 83.0% recall); multi-pass changes are
invariants-only.

Results (multi-pass + rubric fix, transformers v4_57_3):

| Metric | Round 3 (old rubric) | Rubric fix only | Multi-pass + rubric fix | Delta vs rubric fix |
|---|---|---|---|---|
| Reference identities | 28 | 39 | 39 | 0 |
| Cell identities | 17 | 26 | 68 | +42 (pass-3 added 45 new) |
| Intersection | 17 | 16 | 21 | +5 |
| Invariant recall | 60.7% | 41.0% | **53.8%** (21/39) | **+12.8pp** |
| Invariant precision | 65.4% | 61.5% | 30.9% | -30.6pp |
| Severity accuracy | 88.2% | 87.5% | 76.2% | -11.3pp |
| Wall-clock | 890s | n/a (no re-extract) | 1250s | n/a |
| Energy | 37.0 Wh | n/a | 69.7 Wh | n/a |

Pass breakdown:
- Pass 1 (extract): 23 invariants emitted (matches the original round-3
  pattern; same prompt).
- Pass 2 (verify): 1 invariant flagged with `fix: drop` and removed
  (a pad_token_id severity correction was also flagged but not auto-
  applied per design - logged as observation).
- Pass 3 (extend): 45 NEW invariants added across the 14 chunks.

**Interpretation**:

- The recall lift is REAL (+12.8pp vs rubric fix alone). The pass-3
  prompt's structured CoT (rule 3 enumerating known blind spots)
  triggered per-field expansion for the sampling-only-when-greedy
  dormancies (+7 identities) and various type-check additions.
- The precision drop is the cost of structured-CoT prompting. Pass-3
  emits MANY candidate invariants (45 new across 14 chunks); only 5
  of them landed on a reference identity. The other 40 are
  near-misses (wrong primary field, hallucinated cross-field combos,
  type-checks that exist but on a different surface).
- The severity drop is correlated - 11pp of severity-accuracy loss
  comes from pass-3 entries with `severity: error` where the reference
  has `severity: dormant` (the LLM defaults to error in ambiguous
  cases).

**Failure modes observed**:
- 1 chunk (`validate_section_01_1.1._Decoding_attributes`) had pass-3
  parse failure after retries. Pass-1 + pass-2 outputs survived; only
  the extension entries from that chunk were lost.
- Pass-2 flagged 1 invariant with `fix: correct_severity:error` (a
  pad_token_id warning the LLM tagged as warning but the source uses
  minor_issues). Phase 2.5's design is conservative: only `fix: drop`
  is auto-applied; the correction was recorded as an observation for
  Phase 4 review.

**Verdict on 75% target**:
- Invariant recall 53.8% is below the 75% target by 21pp.
- Schema recall 83.0% is ABOVE target (unchanged by Phase 2.5).
- Per Phase 2.5 spec rubric: "If 70-75%: ship with caveat; partial
  close is real progress. If < 70%: document residual gap; ship
  anyway." We are in the latter bucket.
- The matrix-discipline principle (epistemic framing) says ship and
  proceed to Phase 3. The HONEST measurement under the fixed rubric
  is more valuable than further single-cell optimisation.

**Phase 2.6 follow-on suggestions** (NOT blockers):
1. Tighten pass-3 precision: add a `temperature=0` re-prompt with
   "if you're unsure, emit empty list" emphasis. Trade some recall
   for precision.
2. Auto-apply non-drop fixes from pass-2 (correct_severity,
   correct_predicate) - currently conservative. Would help severity
   accuracy.
3. Investigate the 1 parse-failure chunk - the pass-3 YAML output
   may be too long for the salvage parser's `- id:` boundary
   detection.

### 10.4 vllm + tensorrt chunkers

#### vllm 0.7.3

Source: `/tmp/vllm-unpacked/vllm/` (pre-staged Phase 1 Day 1).

Schema chunks (7):
- `sampling_params_class` (SamplingParams + GuidedDecodingParams companion)
- `engine_args_class` (EngineArgs ~125 fields)
- `model_config_class` (ModelConfig)
- `cache_config_class` (CacheConfig)
- `scheduler_config_class` (SchedulerConfig)
- `parallel_config_class` (ParallelConfig)
- `small_config_classes` (LoRAConfig + PromptAdapterConfig + TokenizerPoolConfig + DecodingConfig)

Invariants chunks (10):
- `sampling_params_invariants` (post_init + _verify_args + _verify_greedy_sampling)
- `guided_decoding_params_invariants` (mutual exclusion)
- `model_config_verify_*` (4 separate validator chunks)
- `cache_config_invariants` (3 validators combined)
- `scheduler_config_invariants`
- `parallel_config_invariants`
- `lora_prompt_adapter_invariants`

Companion classes inlined: GuidedDecodingParams referenced by SamplingParams.

Smoke test PASSED: `vllm_chunker.{schema_chunks, invariants_chunks}` return
the expected counts; all chunks have <14k char source (32k ctx headroom).

#### tensorrt_llm 0.21.0

Source: `/tmp/trt-llm-0.21.0/tensorrt_llm/` (pre-staged Phase 1 Day 2).

Key design: TRT uses Pydantic v2 validators (`@field_validator`,
`@model_validator`) NOT classic `if X: raise`. Chunks MUST include
decorator + body together for the LLM to recognise Pydantic-style
validation.

Schema chunks (7):
- `base_llm_args_class` (BaseLlmArgs - parent of TrtLlmArgs/TorchLlmArgs)
- `trt_llm_args_class` (TrtLlmArgs - TRT-specific subclass)
- `calib_kv_cache_configs` (CalibConfig with Literal['cuda', 'cpu'] + KvCacheConfig)
- `scheduler_peft_configs` (SchedulerConfig + PeftCacheConfig + DynamicBatchConfig)
- `build_cache_config_class` (BuildCacheConfig + BuildCache - classic dataclass)
- `enum_allowlists` (BatchingType + CapacitySchedulerPolicy + ContextChunkingPolicy)
- `decoding_configs` (Lookahead/Medusa/Eagle/NGram/MTP)

Invariants chunks (7):
- `base_llm_args_validators_top` (first 4 @field_validator + first 8 @model_validator decorators)
- `base_llm_args_validators_bottom` (remaining 2 @model_validator decorators)
- `trt_llm_args_validators` (validate_enable_build_cache + 3 others)
- `lookahead_validator` (one @field_validator on 3 fields -> 3 invariants)
- `calib_config_literal` (Pydantic-enforced enum on `device`)
- `enum_allowlist_invariants` (StrEnum-bound fields -> 1 invariant each)
- `build_cache_invariants` (BuildCache.__init__ has the only classic `if X: raise`)

Note: BaseLlmArgs has 14 validator methods total (~720 lines of source).
Splitting into two chunks keeps each under ~14k chars while preserving
all 14 validator decorator + body blocks. A validator-only extractor
strips the field declarations to focus the LLM on the validation logic.

Smoke test PASSED: chunker emits the expected counts; Pydantic validator
decorators correctly captured by the source extractor.

### 10.5 Per-engine source-only venvs

`research/mining-substrate-trial/scripts/venv_setup.py::ensure_source_only_venv` builds
`/tmp/trial_<engine>_<version_slug>_venv/` with a `src/<engine>/`
symlink to either:
- A pre-staged tree (Phase 1 miner-lift artefact at `/tmp/vllm-unpacked/`
  or `/tmp/trt-llm-0.21.0/`).
- A `pip download --no-deps` unpack of the wheel.

The venv is SOURCE-ONLY (no Python install). Strategies (b)/(c)/(d-*)
read source via chunkers; they do NOT import the engine package.
Strategy (a) needs a CUDA-bearing container and is NOT served by this
scaffolding.

Trial_runner integration: `trial_runner.resolve_cell_config(spec,
lazy_build=True)` triggers a build on first hit for non-active cells
with strategies (b)/(c)/(d-*).

Phase 3 must thread `lazy_build=True` through its cell execution loop.

### 10.6 Runtime-validation feedback (scaffolding only)

`research/mining-substrate-trial/scripts/trial_scoring.py::runtime_validate_invariants` replays
each emitted invariant's `kwargs_positive` / `kwargs_negative` against
the live library (transformers GenerationConfig / BitsAndBytesConfig
only; other engines need their canonical container).

Returns `RuntimeValidation` records with `positive_confirmed`,
`negative_confirmed`, `observed_outcome` per invariant. Phase 2.5 does
NOT fold these into precision (per plan recommendation: separate
metric).

Smoke test result on round-3 cell output (transformers):
- 26 invariants tested
- 13 positive-confirmed (live library raised/warned on kwargs_positive)
- 9 negative-confirmed (live library DID NOT raise on kwargs_negative)
- 8 both-confirmed
- 12 validation-infra errors (mostly missing native_type from LLM output)

Phase 4 may use this as a tie-break signal for cells with similar
recall/precision.

---

## 11. Calibrated recall verdict (final)

| Axis | Round 3 (Phase 2; old rubric) | Phase 2.5 (multi-pass + new rubric) | Target | Hit? |
|---|---|---|---|---|
| Schema recall | 83.0% | 83.0% (unchanged) | 75% | **YES** |
| Invariant recall | 60.7% | **53.8%** (21/39) | 75% | NO (21pp under) |
| Schema precision | 93.9% | 93.9% (unchanged) | (no target) | n/a |
| Invariant precision | 65.4% | 30.9% | (no target) | n/a |
| Severity accuracy | 88.2% | 76.2% | (no target) | n/a |

**Note on comparability**: the 60.7% (Round 3) and 53.8% (Phase 2.5)
numbers are NOT directly comparable - they use different rubrics. The
3-tuple identity collapsed multi-field invariants into single
identities (28 vs 39 ref identities). The Phase 2.5 4-tuple is the
correct rubric going forward; under it, the round-3 output scores
**41.0%** and the multi-pass output scores **53.8%** - a real
+12.8pp lift from the multi-pass extend pass.

**The 75% invariant-recall target is NOT hit.** Per Phase 2.5 spec
rubric ("If < 70%: document residual gap; ship anyway"), Phase 3
launches with the gap documented in § 10.3 and a Phase 2.6 follow-on
list for precision-recovery work.

**Schema recall hit target at Round 3 and remains stable** under
Phase 2.5. The multi-pass pipeline did NOT touch schema extraction
(no schema regressions).

The multi-pass result is the CORRECT ceiling for "what one cell of
(b) can do without iterating the locked round-3 extract prompt
further." Phase 2.6 could trade some recall for precision (tighter
pass-3 prompt); Phase 4 synthesis will know whether the matrix shape
favours high-recall-low-precision or vice versa.

---

## 12. Container teardown / restart

If the trial pauses (e.g. system reboot), the container Ollama is
stateful via its volume - restart preserves model inventory.

```bash
docker start trial-ollama
# Confirm models still listed:
docker exec trial-ollama ollama list
```

If the container is destroyed but the volume remains, re-create
with the same `-v trial-ollama-models:/root/.ollama` flag to reuse
the model cache (no re-download).
