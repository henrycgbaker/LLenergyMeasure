# N-X13: Config Loader — YAML Resolution Pipeline

**Module**: `src/llenergymeasure/config/loader.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0 (with updates to field names and removal of deprecated fields)
**Planning Gap**: `designs/architecture.md` lists `config/loader.py` as `load_experiment_config(), load_study_config()` without describing the inheritance resolution, deep-merge semantics, or the 4-layer provenance pipeline. The `_extends` inheritance mechanism is not mentioned in any planning document.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/config/loader.py`
**Key classes/functions**:

- `deep_merge(base, overlay)` (line 22) — recursive dict merge; overlay takes precedence; nested dicts are merged (not replaced); leaf values are deep-copied. Returns new dict (non-destructive).

- `load_config_dict(path)` (line 41) — reads YAML or JSON from path; returns raw dict; raises `ConfigurationError` on parse errors or unsupported format. Supports `.yaml`, `.yml`, `.json`.

- `resolve_inheritance(config_dict, config_path, seen)` (line 72) — resolves `_extends: <relative_path>` in config dict:
  1. Adds current path to `seen` set (cycle detection)
  2. Pops `_extends` from config dict
  3. Loads and recursively resolves the base config
  4. `deep_merge(base_resolved, config_dict)` — child overrides parent
  - Raises `ConfigurationError` on circular inheritance

- `load_config(path)` (line 113) — simple two-step: `load_config_dict(path)` + `resolve_inheritance()` + `ExperimentConfig(**resolved)`. No provenance tracking.

- `validate_config(config)` (line 149) — returns `list[ConfigWarning]`; checks schema version, max_output_tokens, decoder conflict warnings (preset + individual params, do_sample+temp=0), traffic QPS bounds, parallelism constraints; delegates to backend's `validate_config()` for backend-specific validation. Severity levels: `"error"`, `"warning"`, `"info"`.

- `has_blocking_warnings(warnings)` (line 326) — `any(w.severity == "error" for w in warnings)` — single-line helper

- `get_pydantic_defaults()` (line 331) — creates a minimal `ExperimentConfig` with placeholder values to extract all Pydantic defaults; returns flattened dict via `flatten_dict()`

- `load_config_with_provenance(path, preset_name, preset_dict, cli_overrides)` (line 351) — full 4-layer resolution:
  1. **Layer 1 — Pydantic defaults**: `get_pydantic_defaults()` → all marked as `ParameterSource.PYDANTIC_DEFAULT`
  2. **Layer 2 — Preset**: from `EXPERIMENT_PRESETS[preset_name]` → changed params marked as `ParameterSource.PRESET`
  3. **Layer 3 — Config file**: YAML + inheritance resolution → changed params marked as `ParameterSource.CONFIG_FILE`
  4. **Layer 4 — CLI overrides**: highest precedence → params marked as `ParameterSource.CLI`
  - Returns `ResolvedConfig(config, provenance, preset_chain, config_file_path)`

The provenance tracking uses `compare_dicts()` to detect which parameters changed at each layer — only changed params get a non-default `ParameterProvenance` entry. This is what drives the `display_non_default_summary()` display in `summaries.py`.

## Why It Matters

The `_extends` inheritance mechanism allows researchers to maintain a `base.yaml` shared across experiment files, with per-experiment files overriding only the parameters that differ. Without this, study configurations that share 90% of their parameters must duplicate them across every config file. The 4-layer provenance tracking in `load_config_with_provenance()` is the foundation of the reproducibility story: it answers "where did this parameter value come from?" for every field in the config, which is stored in results and shown to users in the parameter provenance display. This is a meaningful differentiator from tools that just load YAML without provenance.

## Planning Gap Details

- `designs/architecture.md` lists `config/loader.py` as containing `load_experiment_config()` and `load_study_config()` — the function names differ from the current `load_config()` and `load_config_with_provenance()`. This naming change needs to be resolved.
- No planning doc mentions `_extends` inheritance, `deep_merge`, or `resolve_inheritance`
- No planning doc describes the 4-layer provenance resolution or the `ParameterSource` enum
- `validate_config()` is a load-bearing function that runs backend-specific validation — but it references field names from the current schema (e.g., `config.fp_precision`, `config.num_input_prompts`). These need to be updated when fields are renamed.

## Recommendation for Phase 5

Carry the full pipeline forward. The `_extends` inheritance, `deep_merge`, `load_config_with_provenance`, and `validate_config` are all high-value implementations worth preserving.

Required updates for v2.0:
1. Rename `load_config()` → `load_experiment_config()` per `designs/architecture.md`, or keep as-is and add an alias
2. Add `load_study_config()` for `StudyConfig`/`study.yaml` loading
3. Update field references in `validate_config()` for renamed fields: `fp_precision`→`precision`, `num_input_prompts`→`n`
4. Remove validation rules for removed fields: `schedule`, `streaming`, `streaming_warmup_requests`, `io`, `query_rate` (per session 3 removals)
5. The `_extends` mechanism should work for `study.yaml` as well — ensure `resolve_inheritance` is called when loading study configs

The `deep_merge` function is particularly important: it must handle the backend-specific config sections (`pytorch: {...}`, `vllm: {...}`) correctly, merging nested dicts rather than replacing them. This is already implemented correctly (lines 34–37).
