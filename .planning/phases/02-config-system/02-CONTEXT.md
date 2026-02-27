# Phase 2: Config System - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Researchers can express any single-experiment configuration as a YAML file or Python object,
validate it, and load it — with clear errors on bad input and a user config file for persistent
defaults. This phase covers `ExperimentConfig` schema, YAML loading, user config merging, and
the introspection module. `StudyConfig` / sweep parsing is out of scope (later phase).

</domain>

<decisions>
## Implementation Decisions

### Error presentation

- **Format**: ConfigError includes field name + file path + did-you-mean suggestion
  (Levenshtein-based — e.g. `modell` → `model`)
- **Collect all errors**: Collect ALL validation errors before raising — researchers fix
  everything in one edit cycle, not one error at a time
- **Cross-validator messages**: Rule + fields + short explanation of why invalid
  (e.g. "pytorch: config block requires backend: pytorch, got backend: vllm")
- **Cross-validators in Phase 2**: 3 static structural rules only
  1. Backend section (`pytorch:`, `vllm:`, `tensorrt:`) must match the `backend` field
  2. `precision: fp16/bf16` disallowed when `backend: cpu` (static rule — GPU detection is Phase 4)
  3. `passthrough_kwargs` keys must not collide with top-level `ExperimentConfig` fields
- **Sampling params NOT validated**: temperature + top_p/top_k interactions are NOT caught at
  config layer — backends handle them. However: Phase 6 result schema MUST record effective
  params actually used AND any ignored params in `measurement_warnings`
- **Fatal**: Config errors are always fatal (exit 2). No "continue with warnings" mode

### Config merge & priority

- **Priority order**: CLI flags > experiment YAML > user config defaults
  (explicit beats specific; specific beats general — standard layered config)
- **User config scope**: Researcher workflow preferences only — e.g. default backend,
  default output_dir, default n_cycles. NOT model-specific fields or experiment parameters.
  User config must not silently affect experiment results.
- **Missing user config**: Silently apply all defaults, no error
- **Invalid user config**: Fatal ConfigError — same treatment as a bad experiment YAML
- **Forward-compatible design**: Config loader must accept a `cli_overrides` dict even though
  CLI is Phase 7 — merge semantics defined here so Phase 7 can pass overrides cleanly

### Introspection module

- **Scope split**: Phase 2 = update for field renames (model_name→model, fp_precision→precision
  etc.) + add `backend_support` to per-field metadata. Phase 4.1 = full param completeness audit.
- **Architecture unchanged**: Introspection is independent of composition model — it directly
  reflects `PyTorchConfig`, `VLLMConfig`, `TensorRTConfig` as standalone classes and does NOT
  traverse through `ExperimentConfig`. No traversal logic change needed.
- **Per-field metadata format**: Keep existing fields (type, default, description, constraints,
  test_values) + ADD `backend_support: list[str]` to flag which backends expose each param.
  This enables auto-generated backend capability matrix.
- **Consumers**: Test suite (zero-maintenance test generation), doc generator (backend capability
  tables), future `llem init` display. Not a public library API.
- **Doc generation deferred**: `docs/generated/config-reference.md` is Phase 8 scope.
  Phase 2 only ensures introspection returns correct data.

### YAML format

- **Version field**: Optional `version: "2.0"` field — loader handles with or without.
  Present enables migration detection; absent loads with current schema. Not required.
- **Config reuse**: Drop v1.x `_extends: base.yaml` custom inheritance. Support native YAML
  anchors (`&`/`*`) instead — PyYAML handles automatically, no custom loader logic needed.
- **Discoverability**: Docs-only for v2.0 (Phase 8 generates config-reference.md from introspection).
  No template generator command in Phase 2.

### Claude's Discretion

- Exact Levenshtein threshold for did-you-mean suggestions
- ConfigError message formatting (indentation, ANSI vs plain text — follow existing exceptions.py style)
- Merge implementation detail (deep merge vs shallow for nested backend sections)
- YAML loading library choice (PyYAML vs ruamel.yaml — ruamel preserves comments if needed)

</decisions>

<specifics>
## Specific Ideas

- Error messages should include the file path so the researcher knows which YAML triggered the error
  (important when user config + experiment YAML both exist)
- `passthrough_kwargs` key collision detection: e.g. user writes `passthrough_kwargs: {model: gpt2}`
  when `model` is a top-level field — this should raise ConfigError with a clear message

</specifics>

<deferred>
## Deferred Ideas

- **Sampling parameter cross-validators** (temperature + top_p/do_sample interactions): let backends
  handle. Phase 6 result schema must record effective params + ignored params in measurement_warnings.
- **StudyConfig / sweep YAML parsing**: out of Phase 2 scope — later phase
- **Doc generator** (`generate_config_docs.py` → `docs/generated/config-reference.md`): Phase 8
- **Config template generator** (`llem run --generate-template`): post-v2.0 if researchers ask

## CLI Decision Change (update .product/ before Phase 7)

- **`llem config` → `llem init`**: Rename the diagnostic command. Simple, no flags.
  Shows: GPU, installed backends, user config path — what a new researcher needs to start.
  Drops `--verbose` flag. The command name (`init`) better communicates intent.
- **Impact**: Update `.product/designs/cli-commands.md` and `.product/decisions/cli-ux.md`
  before Phase 7 planning. The Phase 2 introspection module is unaffected (internal consumer
  name doesn't matter).

</deferred>

---

*Phase: 02-config-system*
*Context gathered: 2026-02-26*
