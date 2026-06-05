# Wave 2 protocol deviations

Pre-registered-protocol deviations recorded as they are made. Append-only.

## 2026-06-05 - Ground-truth scoring harness (gt_adapter / gt_scoring)

The Opus-established ground truth (GT) does not match the shape the locked
scorer (`scripts/trial_scoring.py`) expects. Rather than touch the locked
scorer, a canonicalising adapter (`scripts/gt_adapter.py`) translates GT into
the canonical envelope, and `scripts/gt_scoring.py` scores cells against it.
Three matching-method decisions are deviations from a naive "feed GT straight
to the scorer" reading and are recorded here.

### 1. Namespace normaliser (GT native_type / native_field -> baseline namespace)

GT encodes the owning config differently per engine. The synthesised
`match.fields` key is `<namespace>.<leaf_field>`, with namespace derived to
match the convention the baseline walker emits (learned by surveying
`engine_versions/<e>/<v>/outputs/invariants.proposed.yaml` match.fields keys).

| engine | GT source of namespace | rule | example |
|---|---|---|---|
| transformers | `native_field` is already dotted (`native_type` absent) | namespace = native_field up to last dot; leaf = last segment | `transformers.sampling.early_stopping` -> ns `transformers.sampling`, leaf `early_stopping` |
| tensorrt | `native_field` is `ClassName.field` (`native_type` absent) | namespace = `tensorrt` (flat); leaf = last segment | `BaseLlmArgs.tokenizer_mode` -> ns `tensorrt`, leaf `tokenizer_mode` |
| vllm | `native_type` carries the class; `native_field` is the leaf | namespace from per-class table below; leaf = last segment of native_field | `vllm.SamplingParams` + `n` -> ns `vllm.sampling`, leaf `n` |

vllm per-class namespace table:

| native_type | namespace |
|---|---|
| `vllm.SamplingParams`, `vllm.sampling_params.SamplingParams` | `vllm.sampling` |
| `vllm.sampling_params.GuidedDecodingParams` | `vllm.sampling` |
| `vllm.config.LoRAConfig` | `vllm.engine.lora` |
| `vllm.config.PromptAdapterConfig` | `vllm.engine.prompt_adapter` |
| `vllm.config.TokenizerPoolConfig` | `vllm.engine.tokenizer` |
| any other `vllm.config.*Config` / fallback | `vllm.engine` |

KNOWN RESIDUAL DRIFT: the GT author sometimes nests a namespace deeper than
the baseline (e.g. GT `transformers.engine.quantization_config.bnb` vs
baseline `transformers` for `llm_int8_has_fp16_weight`). For transformers we
honour the GT-embedded namespace verbatim (it IS the GT's own convention),
which means such fields will NOT match a flatter baseline on the strict axis.
This is the primary reason the TOLERANT (namespace-dropped) numbers are the
defensible headline. It is recorded, not "fixed", because rewriting the GT
author's namespace to guess the baseline's would be a second, unaudited
normalisation.

### 2. Predicate-kind mapping (GT vocab -> scorer PREDICATE_KINDS)

The scorer's `_predicate_kind_for` classifies an invariant by the SHAPE of
the synthesised match-field value into a closed set
`{exact, not_in, not_equal, gt, lt, ge, le, type_is_not, present, unknown}`.
GT's predicate vocabulary is large and descriptive. We emit a match-field
value whose shape lands the GT kind in the intended bucket:

| GT predicate_kind | scorer bucket | emitted value shape |
|---|---|---|
| `type_check`, `type_is`, `type_is_not` | `type_is_not` | `{"type_is_not": v}` |
| `not_in`, `not_in_range_inclusive`, `not_in_range_half_open`, `strenum_in`, `literal_in`, `allowlist_constant`, `enum` | `not_in` | `{"not_in": v}` |
| `lt`, `lt_or_eq`, `lt_either`, `cross_field_lt` | `lt` | `{"<": v}` |
| `gt`, `cross_field_gt`, `assert_positive`, `gt_fraction_of_host_memory` | `gt` | `{">": v}` |
| `le` | `le` | `{"<=": v}` |
| `ge` | `ge` | `{">=": v}` |
| `eq`, `identity`, `decode_dispatch`, `model_property_check` | `exact` | scalar `v` |
| `is_none`, `is_not_none`, `required`, `file_exists`, `presence_conflict`, `mutual_exclusion`, `mutually_exclusive`, `presence`, `any_falsy_in_list`, `range`, `numeric_range`, `in_open_range`(+inclusive), `cross_field`(+`_combo`), `cross_config_combo`, `cross_subconfig_combo`, `cross_runtime_combo`, `env_var_combo`, `platform_combo` | `present` | `{"present": v}` |
| anything unrecognised | `present` (conservative, non-crashing; still a stable intersecting identity rather than the `unknown` fallback) | `{"present": v}` |

RATIONALE for collapsing cross-field / range / combo families to `present`:
these are multi-field or interval predicates the scorer's single-field flat
shape cannot represent precisely. `present` yields a stable, intersecting
identity (whereas leaving them to fall into `unknown` would push them into
the `("", id, "unknown", "")` fallback bucket and intersect nothing). This
inflates the apparent agreement on predicate KIND for those families - which
is exactly why STRICT predicate matching is reported separately and the
tolerant axis collapses to coarse buckets (below).

### 3. Tolerant-match definition (the research headline)

For each cell we report TWO recall/precision pairs:

- STRICT: the unmodified `trial_scoring` set-intersection on the full identity
  tuple (`(namespace, native_field, predicate_kind, secondary_field)` for
  invariants; `(namespace, name)` for schema). A LOWER BOUND - it penalises
  every namespace or predicate-bucket convention difference as a miss.
- TOLERANT (headline): convention-insensitive re-match.
  - Invariants match on `(leaf_native_field, coarse_predicate_bucket)` ONLY -
    namespace dropped; predicate collapsed to coarse buckets
    `type | membership | numeric | exact | presence`
    (`gt_adapter.coarse_predicate_bucket`; the scorer's fine kinds are mapped
    into the same coarse space via `gt_scoring._SCORER_PK_TO_COARSE`).
  - Schema matches on `(field_name)` only - namespace dropped (a field name is
    unique enough within one engine's catalogue that namespace drift, not
    genuine cross-namespace collision, dominates).

A large STRICT<->TOLERANT gap is itself a finding: it quantifies how much of
the apparent miss is convention noise vs genuine absence.

SANITY CHECK (canonical GT vs the engine's own baseline catalogue, which is a
subset of GT, so recall-of-baseline-vs-GT is low and precision-of-baseline-
vs-GT is high-ish):

| engine/version | strict schema r/p | strict inv r/p | tolerant schema r/p | tolerant inv r/p |
|---|---|---|---|---|
| transformers/v4_57_3 | 0.354/0.920 | 0.050/0.154 | 0.451/0.991 | 0.149/0.607 |
| vllm/v0_7_3 | 0.494/0.978 | 0.275/0.846 | 0.615/0.978 | 0.316/0.960 |
| tensorrt/v0_21_0 | 0.344/1.000 | 0.131/0.258 | 0.386/1.000 | 0.190/0.400 |

All three engines give non-zero invariant recall with zero `unknown`-fallback
collapses, confirming the match synthesis is sound. (v5_6_2 / v0_19_1 / v1_2_1
baselines are not on disk, so the sanity table covers the active version per
engine.)
