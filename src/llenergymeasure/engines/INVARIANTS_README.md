# Engine invariants corpus

This directory contains the SSOT for per-engine configuration invariants - the
data that tells users "this combination will error", "this field will be
silently ignored", or "this will trigger a library warning". Invariants live as
**data**, not code; a single generic Pydantic validator
(landing in phase 50.2c) consumes the corpus and emits error / warn / dormant
annotations per invariant.

Design doc: [`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`](../../.product/designs/config-deduplication-dormancy/runtime-config-validation.md)
(primary).

## Layout

Each engine is a sub-package containing its validation invariants and discovered schema:

- `{engine}/invariants.proposed.yaml` - post-mine corpus from `scripts/engine_producers/{engine}_miner.py`
- `{engine}/invariants.validated.yaml` - post-validation envelope from `scripts/validate_invariants.py`
- `{engine}/schema.discovered.json` - discovered parameters from `scripts/refresh_discovered_schemas.sh`
- `{engine}/_staging/` - gitignored scratch directory for intermediate artefacts

The `.proposed.yaml` is the post-mine corpus - declared expectations only. The
`.validated.yaml` is the post-validation envelope capturing observed outcomes plus any
divergences from the proposed expectations. The runtime
`EngineInvariantsLoader` reads the `.validated.yaml` when present and overlays
its observations onto the proposed YAML, falling back to the proposed YAML
alone for local development without a validation run.

## File envelope

```yaml
schema_version: "1.0.0"
engine: transformers
engine_version: "4.56.0"
mined_at: "2026-04-23T00:00:00Z"
invariants:
  - id: ...
  - id: ...
```

- `schema_version` - semver. Loader supports major 1; mismatches raise
  `UnsupportedSchemaVersionError` (see
  `src/llenergymeasure/config/engine_invariants/loader.py`).
- `engine_version` - the library version the corpus was seeded against.
  Informational; the validation-CI pipeline will revalidate against each
  Dockerfile-pinned version.
- `mined_at` - ISO-8601 UTC timestamp of the miner run. Byte-reproducibility
  via `LLENERGY_MINER_FROZEN_AT=<timestamp>` env variable.

## Invariant schema

Every entry under `invariants:` must populate the following fields.

```yaml
- id: transformers_greedy_strips_temperature
  engine: transformers
  library: transformers
  invariant_under_test: >
    GenerationConfig.validate() records dormant `temperature` when
    do_sample=False and `temperature` is set to a non-default value
  severity: dormant                      # dormant | warn | error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 598
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.temperature: {present: true, not_equal: 1.0}
  kwargs_positive:
    do_sample: false
    temperature: 0.9
  kwargs_negative:
    do_sample: true
    temperature: 0.9
  expected_outcome:
    outcome: dormant_announced           # pass | warn | error | dormant_silent | dormant_announced
    emission_channel: logger_warning_once  # warnings_warn | logger_warning | logger_warning_once | minor_issues_dict | runtime_exception | none
    #   ^ what users actually see. Reserve `minor_issues_dict` for cases
    #     where a caller receives a dict; HF's internal minor_issues dict
    #     is emitted via logger.warning_once, so the log channel is the
    #     user-observable one.
    normalised_fields: []                # populated for dormant_silent outcomes
  message_template: |
    `do_sample=False` is set, so `temperature` ({declared_value}) has no effect.
    Remove it or set do_sample=True.
  references:
    - "transformers.GenerationConfig.validate() (line ~598)"
  added_by: static_miner                 # static_miner | dynamic_miner | manual_seed | runtime_warning | observed_collision
  added_at: "2026-04-23"
```

### Match predicate operators

`match.fields` maps dotted field paths (e.g.
`transformers.sampling.temperature`) to a predicate. Shapes:

| Shape | Meaning |
|---|---|
| Bare value (`0.9`, `false`, `"greedy"`) | Equality. |
| `{">": 1}`, `{">=": 1}`, `{"<": 1}`, `{"<=": 1}` | Comparison. None-safe. |
| `{"==": x}`, `{"equals": x}` | Equality (long form). |
| `{"!=": x}`, `{"not_equal": x}` | Inequality, None-safe. |
| `{"in": [a, b]}`, `{"not_in": [a, b]}` | Membership. |
| `{"present": true}`, `{"absent": true}` | None-presence check. |
| `{"type_is": "X"}`, `{"type_is_not": "X"}` | Concrete `type(value).__name__` match. Spec accepts a single name or a list of names (any-of). |
| `{"divisible_by": n}`, `{"not_divisible_by": n}` | Integer divisibility. Both operands must be non-bool ints; zero divisor never fires. |
| Multi-key dict | All predicates AND-combined. |

Multi-key example - "field is set AND isn't default":

```yaml
transformers.sampling.temperature: {present: true, not_equal: 1.0}
```

`match.fields` is also AND-combined **across field paths** - every entry
under `match.fields` must satisfy its predicate for the invariant to fire.
Use this for cross-field preconditions (e.g. "fires when
`num_beam_groups > 1` AND `diversity_penalty <= 0`").

`type_is_not` accepts a list to express an allowlist negation - the
predicate fires when the field's concrete type name is **not** in any of
the listed names. Useful for "must be an instance of one of these
classes" checks:

```yaml
transformers.sampling.watermarking_config:
  present: true
  type_is_not: [WatermarkingConfig, SynthIDTextWatermarkingConfig]
```

Every predicate in `match.fields` must hold for `invariant.try_match(config)` to
return an `InvariantMatch`.

### Cross-field references (`@field_path`)

Any operator's right-hand side may carry a `@field_path` reference,
substituted from the same config before evaluation. Two resolution modes:

| Form | Resolves against | Example |
|---|---|---|
| Bare `@name` | Sibling of the predicate's field path | `'>': '@num_beams'` resolves `num_beams` next to the predicate's field |
| Dotted `@a.b.c` | Config root | `'>': '@transformers.sampling.num_beams'` |

Example - fires when `num_return_sequences > num_beams`:

```yaml
match:
  fields:
    transformers.sampling.num_return_sequences:
      '>': '@num_beams'
```

Example - fires when `num_beams` is not a multiple of `num_beam_groups`,
guarded by `num_beam_groups > 1`:

```yaml
match:
  fields:
    transformers.sampling.num_beam_groups:
      '>': 1
    transformers.sampling.num_beams:
      not_divisible_by: '@num_beam_groups'
```

References resolve to `None` when the target is missing; comparison
operators treat `None` as "predicate does not fire", so a partially-set
config never trips a cross-field invariant by accident. References are also
walked recursively through list / tuple specs, so e.g.
`{in: ['@x', '@y']}` resolves both items before evaluation.

The `divisible_by` / `not_divisible_by` operators reject non-`int`
operands (including `bool`, which would otherwise pass via Python's
`bool < int`) and zero divisors, so a malformed config silently fails the
predicate rather than raising.

### ID convention

`{engine}_{invariant_summary_snake}` - unique within engine. Miner-authored IDs
encode the pattern (`greedy_strips_X`, `single_beam_strips_X`,
`bnb_X_type`); manual seeds use a descriptive snake-case tail.

### Corpus invariants

These are enforced via `tests/unit/config/engine_invariants/test_corpus_invariants.py`
and extended by the validation-CI gate (`scripts/validate_invariants.py`).

1. Every invariant has a unique `id` within the engine.
2. `match.fields` is non-empty.
3. Both `kwargs_positive` and `kwargs_negative` are populated.
4. `severity` ↔ `expected_outcome.outcome` consistency:
   - `error` ↔ `error`
   - `warn` ↔ `warn`
   - `dormant` ↔ `dormant_silent | dormant_announced`
5. `dormant_silent` outcomes imply non-empty `expected_outcome.normalised_fields`.

## Adding an invariant

### Via miner (preferred)

1. Rerun the miner for the engine:
   `python -m scripts.engine_producers.{engine}_miner --out src/llenergymeasure/engines/{engine}/invariants.proposed.yaml`
   (optionally with `LLENERGY_MINER_FROZEN_AT=<iso-utc>` for reproducibility).
2. Inspect the diff against the previous corpus file. Review the predicate
   shape and verify it matches the library source before merging.
3. Open a draft PR. The validation-CI gate verifies each invariant fires on
   `kwargs_positive` and doesn't fire on `kwargs_negative`.

### Via manual seed

Some invariants live outside miner scope (e.g., vLLM's greedy-strip in
`EngineArgs.create_engine_config()` rather than `__post_init__`). Author the
YAML entry directly:

- Pick a descriptive `id`.
- Set `added_by: manual_seed` and today's `added_at`.
- Cite the source location in `references`.
- Populate both `kwargs_positive` and `kwargs_negative`; the validation-CI step
  verifies both.

### Via feedback loop

Two automated discovery channels populate invariants from study runs: runtime
warning capture (`runtime_warning`) and observed-collision detection
(`observed_collision`). Both open draft PRs with `added_by` set accordingly.

## PR review checklist

When reviewing a corpus PR:

- [ ] Invariants pass (`pytest tests/unit/config/engine_invariants/test_corpus_invariants.py`).
- [ ] Each new invariant's `message_template` reads correctly when substituted.
- [ ] `kwargs_positive` genuinely triggers the invariant in the target library.
- [ ] `kwargs_negative` genuinely does NOT trigger it.
- [ ] `miner_source.path` is a stable relative path (site-packages-rooted).

## Non-goals

- This directory does not drive sweep generation - that's
  `configs/example-study-full.yaml` and related. Corpus YAMLs are data, not
  executable configs.
- The corpus is not consumed at runtime yet (phase 50.2c wires the generic
  `@model_validator`). Today the loader parses the corpus so tests and validation
  tooling have a stable entry point.
