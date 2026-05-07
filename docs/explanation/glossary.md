# Terminology Reference

This document is the canonical glossary for the LLenergyMeasure codebase. When a concept or component has a standard name, this page defines it. Consistent naming prevents the "vendor/rules" drift that prompted PR #570 and this file.

---

## Engine-invariants pipeline

The subsystem that mines, validates, and ships per-engine configuration constraint data.

| Term | Definition | Do NOT use |
|---|---|---|
| **Invariant** | A single constraint entry in the corpus. Has `id`, `severity`, `match.fields`, `kwargs_positive`, `kwargs_negative`, `expected_outcome`. | "rule", "rule entry" |
| **Invariants corpus** / **Engine invariants** | The full set of invariants for one engine, loaded via `EngineInvariantsLoader`. | "rules corpus", "vendored rules" |
| **Proposed corpus** | The post-mine YAML (`invariants.proposed.yaml`). Declared expectations only; no CI observation. | "candidate corpus", "mined corpus" |
| **Validated corpus** | The post-validation YAML (`invariants.validated.yaml`). CI-observed outcomes overlaid onto the proposed corpus. | "vendored corpus", "vendored rules", "vendored YAML" |
| **Validation gate** | `scripts/validate_invariants.py` — replays each invariant's `kwargs_positive`/`kwargs_negative` against the live library. Divergences fail CI. | "vendor gate", "vendor CI gate", "vendor-CI", "vendor-replay" |
| **Validation-CI** | The CI step that runs `validate_invariants.py` inside the engine's Docker container. | "vendor-CI", "vendor CI" |
| **Validated at** | YAML envelope field: ISO-8601 timestamp of the validation run. | "vendored_at" |
| **Validation commit** | YAML envelope field: git SHA that produced the validated envelope. | "vendor_commit" |
| **`EngineInvariantsLoader`** | Python class that loads and caches the invariants corpus. | `VendoredRulesLoader`, `RulesLoader` |
| **`load_invariants(engine)`** | Method on `EngineInvariantsLoader`. | `load_rules` |
| **`_apply_invariants`** | Model validator in `config/models.py` that dispatches error/warn/dormant per invariant. | `_apply_vendored_rules` |
| **`InvariantCandidate`** | Miner output type — one candidate invariant before validation. | `RuleCandidate` |
| **`_InferredInvariant`** | Internal miner type for dynamically-inferred invariants. | `_InferredRule` |
| **`_ProjectedInvariant`** | Fixpoint-test type: minimal invariant projection for dormancy tests. | `_ProjectedRule` |
| **`miner_pinned_range`** | YAML field on the proposed corpus envelope. SSOT-sourced PEP 440 specifier for the library version the miner was tested against. | `walker_pinned_range` |
| **`LLENERGY_VALIDATION_FROZEN_AT`** | Env var: pins the `validated_at` timestamp for reproducibility. | `LLENERGY_VENDOR_FROZEN_AT` |

---

## Miner pipeline

The subsystem that produces the proposed corpus.

| Term | Definition | Do NOT use |
|---|---|---|
| **Miner** | The script or module that extracts invariant candidates from engine library source. | "walker" (when used as a bare concept for the miner module) |
| **AST walker** | The static analysis component inside a miner that walks abstract syntax trees. **Preserved as a technical term** — do not rename to "AST miner". | — |
| **Static miner** | Miner variant that uses AST walking of validator methods. | "static walker" |
| **Dynamic miner** | Miner variant that probes constructors at runtime. | "dynamic walker" |
| **Lift module** | Type-system adapter (`_pydantic_lift.py`, `_msgspec_lift.py`, `_dataclass_lift.py`) that extracts constraints from type metadata. | — |
| **`miner_source`** | YAML field: `{path, method, line_at_scan}` pointing to the library source that produced the invariant. | `walker_source` |
| **`build_corpus.py`** | Orchestration script: runs miners, merges staging files, calls `validate_invariants.py`. | — |

---

## Discovered schemas

The subsystem that introspects engine parameter APIs and writes JSON schema files.

| Term | Definition | Do NOT use |
|---|---|---|
| **Discovered schema** | `schema.discovered.json` — introspected parameter metadata for one engine version. | "vendored schema", "vendored JSON" |
| **Discovery pipeline** | `scripts/engine_introspectors/` + `engine-pipeline.yml`. | "schema vendoring" |
| **`SchemaLoader`** | Python class that loads and caches discovered schemas. | — |

---

## Severity and outcome vocabulary

| Term | Used in | Meaning |
|---|---|---|
| `error` | `severity`, `outcome` | Config combination raises an exception at construction time |
| `warn` | `severity`, `outcome` | Config combination emits a library warning |
| `dormant` | `severity` | Field is set but silently ignored or normalised by the library |
| `dormant_silent` | `outcome` | Dormant: no user-visible signal; field normalised silently |
| `dormant_announced` | `outcome` | Dormant: library logs a warning about the ignored field |

---

## Python class / function name map

Quick reference for consistent naming across `src/`, `scripts/`, and `tests/`.

| Concept | Canonical name | Old name (do not use) |
|---|---|---|
| Corpus container dataclass | `EngineInvariants` | `VendoredRules` |
| Single entry dataclass | `Invariant` | `Rule` |
| Corpus loader class | `EngineInvariantsLoader` | `VendoredRulesLoader` |
| Loader cache reset | `_reset_rules_loader_cache()` | — (name unchanged) |
| Corpus field on container | `.invariants` | `.rules` |
| Miner output type | `InvariantCandidate` | `RuleCandidate` |
| Dynamic miner internal type | `_InferredInvariant` | `_InferredRule` |
| Fixpoint test projection | `_ProjectedInvariant` | `_ProjectedRule` |
| Load corpus by engine | `load_invariants(engine)` | `load_rules(engine)` |
| Dormant-invariant loader | `load_dormant_invariants(corpus)` | `load_dormant_rules(corpus)` |
| Config model validator | `_apply_invariants` | `_apply_vendored_rules` |
| Loader accessor | `_get_invariants_loader()` | `_get_rules_loader()` |
| API public function | `load_engine_invariants()` | `load_rules_corpus()` |
| Corpus field on study config | `engine_invariants` | `rules_corpus` |
| Divergence field | `invariant_id` | `rule_id` |
| diff kind: new invariant | `"added_invariant"` | `"added_rule"` |
| diff kind: dropped invariant | `"removed_invariant"` | `"removed_rule"` |
| YAML top-level key | `invariants:` | `rules:` |
| Validated YAML envelope field | `validated_at:` | `vendored_at:` |
| Validated YAML envelope field | `validation_commit:` | `vendor_commit:` |
| Corpus envelope key | `miner_pinned_range` | `walker_pinned_range` |
| CLI arg | `--validation-commit` | `--vendor-commit` |
| Env var | `LLENERGY_VALIDATION_FROZEN_AT` | `LLENERGY_VENDOR_FROZEN_AT` |
| SSOT artefact key | `validated_invariants` | `vendored_rules` |
| SSOT artefact key | `engine_invariants` | `rules_corpus` |
| YAML file name | `invariants.validated.yaml` | `invariants.vendored.yaml` |

---

## Scope guards (do NOT rename)

These identifiers look similar but are NOT part of the engine-invariants subsystem and must not be touched:

- `get_validation_rules()` in `src/llenergymeasure/config/introspection.py` — returns doc-generation combos, unrelated to the corpus.
- `AST walker` — preserved as a technical term for the static analysis component.
- `vendored_rules_version` — backward-compatible JSON field on `EquivalenceGroups`; old study artefacts carry this field.

---

## See also

- [validation-invariant-corpus.md](/contributing/validation-invariant-corpus) — corpus YAML format specification
- [miner-pipeline.md](/contributing/miner-pipeline) — pipeline architecture
- [extending-miners.md](/contributing/extending-miners) — adding a new engine
