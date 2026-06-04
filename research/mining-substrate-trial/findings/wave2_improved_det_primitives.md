# Improved deterministic primitives proposal (post-GT)

**Status:** Sketched 2026-06-05 in response to user question "now that we have ground truths, can we design better deterministic tools than the ones we did already". Empirical input: batch-1 GT delta (300% baseline coverage shortfall, with consistent structural patterns).

## The empirical observation that drives this

Batch 1 ground truth across 3 engines showed baseline producers covered ~26-30% of actual invariants and 0% of env vars. The misses were not random — they clustered into 5 universal patterns:

1. **Env-var module surface** (0% baseline coverage, 169 entries across 3 engines)
2. **Per-quantizer / per-decoder class fan-out** (~0% baseline; transformers has 18 quantizers beyond BNB; tensorrt has 5 of 6 speculative-decode subclasses missed)
3. **Long-tail config classes** (vllm: SpeculativeConfig + KVTransfer + Compilation + Observability + Decoding; tensorrt: PluginConfig + TorchLlmArgs + CalibConfig)
4. **Silent-normalisation invariants at aggregator level** ("caller declares X, engine runs Y")
5. **Mechanical-walk gaps within "covered" surfaces** (e.g. transformers GenerationConfig.validate's 9 generate-only kwargs are a 2-line AST walk baseline missed)

None of these required LLM intelligence to catch. They required different deterministic primitives than the ones the existing producers used. With richer det primitives we likely close 60-80% of the baseline gap CHEAPLY with no LLM call.

## Proposed primitive set

Each primitive is engine-agnostic where possible. Tree-sitter Python grammar is the substrate (sub-second per engine per query). Hand-rolled Python AST walks where tree-sitter is too brittle.

### Primitive 1: Universal class-fanout walker

**What:** discover EVERY class matching structural patterns, not a hand-curated list.

**Patterns to match:**
- `@dataclass` decorator (covers transformers configs + vllm config classes)
- `class X(msgspec.Struct)` or `class X(Struct)` (vllm SamplingParams)
- `class X(BaseModel)`, `class X(BaseSettings)` (Pydantic - tensorrt)
- Classes with `__post_init__` method present
- Classes with `@field_validator` / `@model_validator` decorators
- Classes with `_verify_*` methods (vllm convention)
- Classes with `validate*` methods (transformers convention)

**Catches:** all the long-tail config classes baseline missed (vllm SpeculativeConfig etc; tensorrt PluginConfig + TorchLlmArgs; transformers per-quantizer fan-out).

**Cost:** sub-second tree-sitter scan per engine.

### Primitive 2: Env-var module enumerator

**What:** dedicated discoverer for env-var surfaces.

**Method:**
- Per-engine entry points: `vllm/envs.py`, `transformers/utils/hub.py`, `tensorrt_llm/_environments.py` (or whatever the canonical envs module is per engine - discoverable via grep on `os.environ`).
- Walk for `os.environ.get(...)`, `os.getenv(...)`, `lambda: os.environ[...]` patterns.
- Extract var name, default value, behavior comment if present (often in adjacent docstring).
- Also detect the source-vs-stub footgun pattern (var X is documented but the code reads var Y) flagged in batch 1.

**Catches:** all 169 env vars + the 2-3 source-vs-stub-divergence footguns.

**Cost:** ~100 ms per engine.

### Primitive 3: Silent-normalisation detector

**What:** dedicated walker for aggregator-level normalization invariants.

**Patterns:**
- `if self.X: self.Y = Z` (silently disables/enables Y when X)
- `if self.X is None: self.X = <default>` (defaults normalization)
- `self.X = self.X or <default>` (fallback normalization)
- Repeated pattern across multiple fields in one `__post_init__`

**Targets:** top-level aggregator classes (vllm `VllmConfig.__post_init__`, transformers `GenerationConfig.__post_init__`, tensorrt `TrtLlmArgs.__post_init__`).

**Catches:** the "caller declares X, engine runs Y" invariants baseline missed (MLA + LoRA + cpu_offload silent-disable patterns in vllm; equivalents in transformers + tensorrt).

**Cost:** sub-second.

### Primitive 4: Mechanical-gap probe (enum membership walker)

**What:** find every `if X not in <set/enum>: raise` and `if X in <bad_set>: raise` pattern within validator methods.

**Method:**
- Walk classified validator methods (output of Primitives 1 + 5).
- Tree-sitter query for `not_in_operator` / `in_operator` adjacent to raise statement.
- Resolve the set membership constant (often a module-level allowlist).

**Catches:** the 9 generate-only kwargs in transformers GenerationConfig.validate; equivalent allowlist gates across engines.

**Cost:** sub-second.

### Primitive 5: Per-engine validator-naming convention walker

**What:** discover ALL validator methods, not just the ones in the hand-curated landmarks list.

**Method:**
- Per-engine conventions:
  - vllm: `_verify_*` methods
  - transformers: `validate*` methods, `from_pretrained` pre-flight, `save_pretrained` strict-gate
  - tensorrt: `__post_init__`, `_validate_*`, Pydantic `@field_validator` / `@model_validator`
- For each method, parse body for raise + warning + normalization patterns.

**Catches:** validator methods on the LONG-TAIL classes Primitive 1 surfaces.

**Cost:** sub-second.

### Primitive 6: Decorator-discovered validator walker

**What:** specifically for Pydantic-style decorated validators (tensorrt heavy use).

**Method:**
- Tree-sitter query for `@field_validator(...)`, `@model_validator(...)`, `@validator(...)`, `@root_validator(...)`.
- For each decorated method, extract:
  - Field(s) being validated (from decorator args)
  - Body raise / return patterns
  - Mode (before / after / wrap)

**Catches:** tensorrt's per-Pydantic-class validators.

**Cost:** sub-second.

### Primitive 7: Aggregator __post_init__ deep walker

**What:** specifically for the TOP-LEVEL Config classes whose `__post_init__` orchestrates cross-class normalisation.

**Method:**
- Per-engine entry: vllm `VllmConfig`, transformers `GenerationConfig`, tensorrt `LlmArgs`.
- Walk `__post_init__` body for:
  - Calls to sibling `_verify_*` methods
  - Cross-field assignments (silent normalisation)
  - Conditional raises that reference multiple fields

**Catches:** the cross-field invariants baseline missed at the aggregator level.

**Cost:** sub-second.

## What the LLM is STILL needed for (the residual)

After applying primitives 1-7, the gap to ground truth shrinks to roughly these categories:

| Residual category | Why det can't do it | LLM role |
|---|---|---|
| Dynamic registries (transformers `ALL_ATTENTION_FUNCTIONS`, HF kernel-hub) | Runtime-introspected | LLM reads runtime + docs |
| C++ pybind classes (tensorrt `ModelConfig`, `WorldConfig`, `ExecutorConfig`) | Need `.so` import to inspect | LLM cross-references C++ source |
| Semantic resolution (`self.foo.bar` referent disambiguation) | AST sees syntax not types | LLM resolves with type knowledge |
| Free-text predicate interpretation | Predicate doesn't match common shapes | LLM interprets validator-body narrative |
| Cross-engine semantic equivalence | Naming conventions differ | LLM normalises across engines |

Estimated post-improved-det coverage: ~70-80% of GT (up from 26-30%). LLM extension closes the remaining 20-30%.

## Implications for Wave 2 protocol

- The Tier A pure-det strategies should be measured AGAINST the improved primitive set, not against the existing hand-walker baseline. Otherwise we'd be comparing "current det" with "current det + new det primitives" without isolating the new-primitive contribution.
- The cost-frontier becomes: improved-det floor (cheap, 70-80% coverage) + LLM-extend tail (more expensive, closes residual). Pure-det workflow likely viable for many engines IF the residual is small.
- Wave 1's H4 (LLM-patches-walker) finding may improve: with smaller residual + cleaner failure modes, LLM patch suggestions land more accurately.
- The "self-updating" dimension also gets cleaner: improved det primitives are engine-agnostic AND don't depend on hand-curated landmark lists, so they survive upstream renames automatically. The bumped-version brittleness profile should be MUCH better than current baseline.

## Implementation outline

A new strategy module: `scripts/strategies/wave2/a_improved_det.py` implementing the 7 primitives, dispatched via the existing wave2_runner. Estimated ~600-1000 LoC (much of it shared tree-sitter query infrastructure).

For Wave 2 measurement: run W2-a-improved-det on (transformers + vllm + tensorrt) x (active + v+1) = 6 cells. Compare per-task recall + precision against GT and against baseline. The delta is the "what does the better det tool buy us" answer.

## Out of scope for this proposal

- Universal cross-engine class enumeration (depending on import behaviour, might be too dynamic).
- Plugin-style extension (where engines load 3rd-party code at runtime).
- Full type inference (we use AST-syntactic patterns, not type-resolution).

These remain LLM territory or future-Wave-3.

## Cross-references

- `findings/wave2_treesitter_probe.md` — the initial tree-sitter probe whose findings prefigured this proposal.
- `findings/ground_truth/<engine>/v<v>/delta.md` — per-engine delta artefacts whose patterns drove the primitive design.
- `WAVE2_PRIMITIVES.md` — places `improved-det` as a new substrate level on Axis 1.
- `WAVE2_PROTOCOL.md` — to be updated to include W2-a-improved-det in the substrate measurement set.
- `DECISIONS_LOG.md` — 2026-06-05 entry capturing the rationale + scope.
