# N-X04: Display Architecture (cli/display/ Subpackage)

**Module**: `src/llenergymeasure/cli/display/` (console.py, tables.py, summaries.py, results.py)
**Risk Level**: HIGH
**Decision**: Keep — v2.0 (the patterns and logic are sound; will need updates for renamed config fields)
**Planning Gap**: `designs/observability.md` specifies Rich as the display technology and shows example output, but does not document the `display/` subpackage structure, the shared console instance pattern, or the verbosity-aware field-filtering logic in `summaries.py`. A Phase 5 rebuild that ignores this subpackage risks duplicating or fragmenting this logic.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/cli/display/` (4 files, estimated ~950 lines total)

### `console.py` (~34 lines)
- `console = Console(no_color=os.environ.get("NO_COLOR") == "1")` (line 14) — shared singleton; respects `NO_COLOR` env var for accessibility and test environments
- `format_duration()` (line 17) — formats seconds to "5.2s", "3.1m", "1.5h" — used throughout result display

### `tables.py` (~92 lines)
- `format_field(name, value, is_default, nested)` (line 16) — returns `(formatted_name, formatted_value)` tuple; dim+grey for defaults, green for non-defaults at top level, cyan for nested non-defaults
- `add_section_header(table, name)` (line 41) — bold row with no value column
- `print_value(name, value, is_default, indent, show_defaults)` (line 46) — prints directly to console with verbosity-aware gating (`show_defaults` param)
- `format_dict_field(name, value, default, nested)` (line 68) — same as `format_field` but takes default value for direct comparison (used when rendering from result dict rather than live config object)

### `summaries.py` (~793 lines — the largest display file)
- `display_config_summary(config, overrides, preset_name)` (line 30) — verbose/normal/quiet-aware config rendering; reads `LLM_ENERGY_VERBOSITY` env var directly; in QUIET mode returns immediately; in NORMAL mode shows only non-default values; in VERBOSE mode shows all fields
- Renders sections: core, tokens, streaming, batching (backend-specific), parallelism (backend-specific), traffic, decoder (+ backend-specific extensions), quantization (backend-specific), schedule, prompts, CLI overrides
- `display_incomplete_experiment()` (line 513) — shows state info for interrupted experiments
- `show_effective_config()` (line 534) — renders full config from result dict (post-experiment display); ~205 lines
- `display_non_default_summary()` (line 751) — renders compact summary from `ResolvedConfig.get_non_default_parameters()`, grouped by section and source (preset/config/CLI)

### `results.py` (~348 lines)
- `show_raw_result()` (line 21) — renders per-process result with MIG warning, duration, tokens, throughput, energy, FLOPs
- `show_aggregated_result()` (line 59) — renders full aggregated result including extended efficiency metrics and latency stats
- `_show_extended_metrics()` (line 134) — renders `ExtendedEfficiencyMetrics` table with N/A for null fields; handles both dict (from JSON) and object (live) representations
- `_show_latency_stats()` (line 234) — renders TTFT and ITL statistics inline; handles both dict and `LatencyStatistics` object
- `show_parameter_provenance()` (line 295) — renders parameter source table grouped by preset/config/CLI
- `_format_metric(value, unit, fmt)` (line 117) — returns `"[dim]N/A[/dim]"` for None values

## Why It Matters

The display subpackage encodes the visual contract between the tool and its users. It represents significant design work: the verbosity-aware field filtering in `summaries.py` (distinguishing default vs non-default values, backend-specific sections), the dual-representation handling in `_show_extended_metrics()` and `_show_latency_stats()` (accepting both live objects and deserialized dicts from JSON), and the `NO_COLOR` accessibility pattern. Rebuilding from scratch would risk losing any of these behaviours. The shared `console` singleton is particularly important — it ensures all output goes through a single Rich console instance, which is required for correct Rich `Live` integration.

## Planning Gap Details

`designs/observability.md` describes:
- Two console instances (`stderr_console` for progress, `stdout_console` for final results) — the current code uses a single `console` that is not split by stream
- Rich `Progress` + `Live` for study-level display — not yet implemented; the current display layer will need to adapt

The planned v2.0 display layer in `designs/observability.md` (line 131–137) is:
```python
stderr_console = Console(file=sys.stderr)
stdout_console = Console(file=sys.stdout)
```

The current code uses a single `console` with default stderr. This needs addressing: the final summary (`show_aggregated_result`) should go to stdout; progress should go to stderr. The split is architecturally important for pipeline use (`llem run > result.json`).

## Recommendation for Phase 5

Consolidate the `display/` subpackage into `cli/display.py` as specified in `designs/architecture.md` (line 68: `display.py — Rich progress display, summary formatting`). The logic from all four files should be preserved — it is the content that matters, not the file split.

Key actions:
1. Split the single `console` into `stderr_console` and `stdout_console` per the observability design
2. Update `summaries.py` config field names to match confirmed renames: `model_name`→`model`, `fp_precision`→`precision`, `num_input_prompts`→`n`
3. Remove `schedule` section rendering (schedule removed from ExperimentConfig per session 3)
4. Remove `streaming` / `streaming_warmup_requests` rendering (removed per session 3)
5. Add `query_rate` section decision: was `query_rate` also removed or just `traffic_simulation`? (check session 3 removal list — `query_rate` is referenced in `summaries.py` line 251)
6. The `_format_metric()` N/A pattern is valuable — preserve it
7. The `display_non_default_summary()` provenance display is a key differentiator — preserve it
