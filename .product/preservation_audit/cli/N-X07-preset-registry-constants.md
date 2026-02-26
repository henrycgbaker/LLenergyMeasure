# N-X07: Preset Registry and Constants

**Module**: `src/llenergymeasure/constants.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0 (presets confirmed in planning; constants are referenced throughout the codebase)
**Planning Gap**: The 10 built-in presets are not documented in any planning doc. The planning confirms presets exist as a concept, but no design document lists the preset names, their config fields, or the `_meta` SSOT pattern.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/constants.py`
**Key constants and functions**:
- `DEFAULT_RESULTS_DIR = Path(os.environ.get("LLM_ENERGY_RESULTS_DIR", "results"))` (line 9) — env-var-overridable
- `RAW_RESULTS_SUBDIR = "raw"` (line 10), `AGGREGATED_RESULTS_SUBDIR = "aggregated"` (line 11)
- `DEFAULT_WARMUP_RUNS = 3` (line 14), `DEFAULT_ACCELERATE_PORT = 29500` (line 16)
- `DEFAULT_STREAMING_WARMUP_REQUESTS = 5` (line 24)
- `SCHEMA_VERSION = "3.0.0"` (line 27)
- `DEFAULT_STATE_DIR = Path(os.environ.get("LLM_ENERGY_STATE_DIR", ".state"))` (line 31)
- `COMPLETION_MARKER_PREFIX = ".completed_"` (line 32)
- `GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 2` (line 35)
- `DEFAULT_BARRIER_TIMEOUT_SEC = 600` (line 36), `DEFAULT_FLOPS_TIMEOUT_SEC = 30` (line 37), `DEFAULT_GPU_INFO_TIMEOUT_SEC = 10` (line 38)
- `PRESETS: dict[str, dict[str, Any]]` (line 48) — 10 built-in presets (see below)
- `get_preset_metadata(preset_name)` (line 208) — returns `_meta` dict or `None`
- `get_preset_config(preset_name)` (line 223) — returns config dict excluding `_meta`
- `DEPRECATED_CLI_FLAGS: dict[str, dict[str, str]]` (line 252) — 7 deprecated flags with migration guidance
- `is_cli_flag_deprecated(flag)` (line 292), `get_deprecation_info(flag)` (line 303)

**The 10 built-in presets**:
| Name | Backend | Key Config | Use Case |
|------|---------|------------|----------|
| `quick-test` | agnostic | max_input=64, max_output=32, batch=1, greedy | CI, sanity checks |
| `benchmark` | agnostic | max_input=2048, max_output=512, batch=1, fp16, greedy | Reproducible benchmarks |
| `throughput` | agnostic | max_input=512, max_output=256, batch=8, dynamic, fp16, greedy | Max tokens/second |
| `vllm-throughput` | vllm | max_input=2048, max_output=512, max_num_seqs=512, prefix_caching, chunked_prefill | Production serving |
| `vllm-speculative` | vllm | max_input=2048, n-gram speculation (5 tokens, ngram_max=4) | Lower latency |
| `vllm-memory-efficient` | vllm | max_input=4096, kv_cache_dtype=fp8, prefix_caching, gpu_memory_utilization=0.95 | Large context |
| `vllm-low-latency` | vllm | max_input=512, max_num_seqs=32, enforce_eager=True | Interactive/TTFT |
| `pytorch-optimized` | pytorch | flash_attention_2, torch_compile=reduce-overhead, fp16 | Best PyTorch perf |
| `pytorch-speculative` | pytorch | sdpa, assisted_generation 5 tokens | Lower latency |
| `pytorch-compatible` | pytorch | eager attn, torch_compile=False, fp16 | Older GPUs, debug |

The `_meta` pattern provides `description` and `use_case` fields as SSOT for CLI display (`lem list presets`) — the preset registry is both the configuration data and its own documentation.

## Why It Matters

Presets are the zero-config UX mechanism. Without them, `llem run --model X` would need to ask for every parameter, or silently use undocumented defaults. The 10 presets represent significant domain expertise: the vLLM presets in particular encode production-optimisation knowledge (prefix caching, chunked prefill, enforce_eager for TTFT). The timeout constants (`GRACEFUL_SHUTDOWN_TIMEOUT_SEC`, `DEFAULT_BARRIER_TIMEOUT_SEC`) are load-bearing — removing or changing them affects experiment reliability. `SCHEMA_VERSION` guards result compatibility.

## Planning Gap Details

- `decisions/cli-ux.md` confirms presets exist and `quick-test`, `benchmark` are expected in the zero-config flow — but does not list the 10 presets or their fields
- `designs/architecture.md` does not mention `constants.py` or the preset registry
- No planning doc describes the `_meta` SSOT pattern or the `DEPRECATED_CLI_FLAGS` structure

## Recommendation for Phase 5

Carry `constants.py` forward essentially unchanged. Several updates are needed:

1. **Field names in presets**: presets reference `num_processes`, `fp_precision`, `batching.batch_size` — update to match the confirmed renamed fields (`precision` for `fp_precision`; verify the backend-section field paths are correct for the composition architecture)
2. **`SCHEMA_VERSION`**: increment to match v2.0 result schema version
3. **`DEPRECATED_CLI_FLAGS`**: these flag the v1.x CLI params; review which flags survive in v2.0 and remove entries for flags that no longer exist
4. **Preset names**: `quick-test`, `benchmark`, `throughput` are backend-agnostic. Consider documenting that the default preset (when `--preset` is not specified) is `quick-test` for zero-config runs — this should be in `designs/cli-commands.md`
5. `EXPERIMENT_PRESETS = PRESETS` alias (line 239) — this backward-compat alias can be removed in v2.0 (it is a clean break)
