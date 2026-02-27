# P-16: Parameter Naming Alias Registry

**Module**: `src/llenergymeasure/config/naming.py`
**Risk Level**: MEDIUM
**Decision**: Pending — needs human decision on backward compat vs clean break
**Planning Gap**: `designs/experiment-config.md` declares a clean break (no shims), but the alias registry exists and is actively maintained. Whether to delete it or preserve it for a deprecation-warning path is unresolved.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/config/naming.py`
**Key classes/functions**:
- `PARAMETER_ALIASES` (line 30) — dict mapping 18 canonical parameter paths to CLI flags, legacy YAML names, deprecation status, and descriptions
- `get_canonical_name(alias: str) -> str` (line 214) — resolves any alias (CLI flag or legacy YAML key) to its canonical parameter path
- `get_cli_flag_for_param(canonical: str) -> str | None` (line 245) — reverse lookup: canonical → primary CLI flag
- `is_deprecated_cli_flag(flag: str) -> bool` (line 261) — checks whether a specific CLI flag is marked deprecated
- `get_all_deprecated_cli_flags() -> dict[str, dict[str, str]]` (line 281) — returns all deprecated flags with migration instructions

The module maintains a structured registry of 18 parameter entries covering batching (`batching.batch_size`, `batching.strategy`, `batching.max_tokens_per_batch`), decoder sampling (`decoder.temperature`, `decoder.top_p`, `decoder.top_k`, `decoder.do_sample`, `decoder.repetition_penalty`, `decoder.preset`), precision/quantisation (`fp_precision`, `quantization.load_in_4bit`, `quantization.load_in_8bit`), parallelism (`parallelism.strategy`, `parallelism.degree`, `num_processes`, `gpus`), token/input parameters (`max_input_tokens`, `max_output_tokens`, `min_output_tokens`, `num_input_prompts`), backend and streaming, and reproducibility (`random_seed`). Each entry records CLI flags (e.g. `--batch-size`, `-b`), legacy YAML field names (e.g. `batching_options.batch_size`), and a `deprecated_cli: True` flag where applicable. The deprecated CLI flags are: `--batch-size`, `--temperature`, `--precision`, `--quantization`, `--num-processes`, `--gpu-list`.

Notably, `fp_precision` appears as a canonical name in the registry (line 96–102) with `yaml_legacy: ["precision"]` — this is the reverse of the v2.0 planned rename (where `fp_precision` becomes `precision`). This inconsistency signals the registry was not updated to match planning session decisions.

## Why It Matters

The alias registry is the mechanism by which old YAML configs (written for v1.x) and old CLI invocations could continue to work with helpful deprecation warnings rather than silent failures or cryptic errors. Without it, every v1 user hits a wall at v2.0. With it, the tool can emit: *"`--batch-size` is deprecated — set `pytorch.batch_size` in your config YAML"*. The `get_all_deprecated_cli_flags()` function is purpose-built for generating this kind of warning output.

However, `designs/experiment-config.md` explicitly states "No aliases, no shims. v2.0 is a clean break." If that decision holds, this entire module should be deleted — but the decision must be made consciously, not by default.

## Planning Gap Details

`designs/experiment-config.md` (Renamed Fields section, bottom of doc) states:

> No aliases, no shims. v2.0 is a clean break.

This directly contradicts the existence of the alias registry. The planning doc does not mention `naming.py` at all. The registry also contains parameters that were removed in v2.0 planning (`streaming`, `streaming_warmup_requests` appear in the registry but were confirmed removed in session 3 decisions). Additionally, `num_input_prompts` is registered (line 167) but was renamed to `n` per planning decisions — the registry does not yet reflect this rename. The `fp_precision` / `precision` rename is also mis-mapped in the registry.

No planning doc describes what happens to `naming.py` in Phase 5. The `MEMORY.md` planning file makes no mention of it.

## Recommendation for Phase 5

**Decision required from project owner before Phase 5 implementation begins.**

Option A — Clean break (aligns with current planning): Delete `naming.py` entirely. `llem experiment --batch-size 8` fails with a standard Typer unknown-flag error. Users must update their configs. Document the rename table in the migration guide.

Option B — Deprecation-warning path (safer for early adopters): Keep the registry concept but rewrite it to reflect the v2.0 canonical names accurately. Wire `is_deprecated_cli_flag()` into the CLI to emit `DeprecationWarning` with migration text before executing. This adds ~1 day of implementation for significantly better user experience.

If Option B is chosen, the registry needs corrections before Phase 5:
- `fp_precision` canonical entry should be updated: the new canonical is `precision`, not `fp_precision`
- `num_input_prompts` → `n` rename must be reflected
- `streaming` and `streaming_warmup_requests` entries should be removed (fields cut from v2.0)
- `yaml_legacy` entries for all renamed fields should be populated (currently empty for `n`, `precision`, `model`)
