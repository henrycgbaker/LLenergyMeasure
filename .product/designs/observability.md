# Observability Design (Progress & Output)

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/live-observability.md](../decisions/live-observability.md)
**Status**: Confirmed

---

## Design Principle

**Maximally explicit.** Show everything that is happening — in standard mode, all experiment-level
events; in verbose mode, all subprocess-level events too. Researchers running overnight studies need
to know what the tool is doing at a glance. Opacity is a bug.

**Output routing** (Unix convention):
- Progress display → `stderr`
- Result JSON / final summary → `stdout`
- Allows: `llem run > result.json 2>/dev/null`

---

## Verbosity Levels

| Flag | Behaviour |
|---|---|
| (default) | Experiment-level progress + inline results + final summary |
| `--quiet` | No live display; final summary always printed |
| `--verbose` | Default + subprocess-level events (model loading, warmup CV, per-prompt metrics) |

`--quiet` suppresses progress only. The final summary is a scientific record — it always prints.

---

## `llem run` Output

### Standard (default)

```
Experiment: llama-3.1-8b / pytorch / bf16 / batch=8
  Pre-flight     ✓  GPU, backend, model, energy backend
  Warmup         ████████░░░░░░░░  8/20 prompts  CV=0.087 → target <0.050
  Measuring      ██████████████░░  70/100 prompts

Result saved: results/llama-3.1-8b_pytorch_2026-02-19T14-30.json

  Energy         312.4 J  (3.12 J/request)
  Throughput     847 tok/s
  Latency        TTFT 142ms  ITL 28ms
  Duration       4m 32s
```

### `--verbose` (additional lines)

```
  [Loading model]  meta-llama/Llama-3.1-8B  bf16  →  14.2 GB VRAM
  [Energy backend]  NVML direct (Zeus not installed)
  [Warmup]  prompt 8: latency=148ms  CV=0.087 (running avg of last 5)
  [Measuring]  prompt 70: latency=141ms  tokens=87
```

### `--quiet`

```
Result saved: results/llama-3.1-8b_pytorch_2026-02-19T14-30.json

  Energy         312.4 J  (3.12 J/request)
  Throughput     847 tok/s
  Latency        TTFT 142ms  ITL 28ms
  Duration       4m 32s
```

---

## `llem run study.yaml` Output (multi-experiment study)

### Standard (default)

```
Study: batch-size-sweep-2026-02                  cycle 1/3   00:45:32   ETA ~2:15:00
  [================================================] 3/12 experiments

  ✓ [1/12]  pytorch / batch=1 / bf16    →   87.3 J   1,243 tok/s   (2m 14s)
  ✓ [2/12]  pytorch / batch=4 / bf16    →  142.1 J     847 tok/s   (3m 02s)
  ▶ [3/12]  pytorch / batch=8 / bf16    →  measuring...  (00:01:23 elapsed)
  · [4/12]  pytorch / batch=16 / bf16   →  (queued — thermal gap 60s)
  · [5/12]  pytorch / batch=32 / bf16   →  (queued)
  ...
```

Symbols: `✓` completed, `▶` running, `·` queued, `✗` failed.

Inline results appear as each experiment finishes — live signal during long studies.

### `--verbose` (subprocess events as child lines)

```
  ▶ [3/12]  pytorch / batch=8 / bf16    →  measuring...
      Loading model: meta-llama/Llama-3.1-8B  (14.2 GB, bf16)
      Energy backend: NVML direct
      Warmup: 12/20 prompts  CV=0.062 → target <0.050
      Measuring: 44/100 prompts
```

Pattern: Docker BuildKit per-layer progress nested under overall build progress.

### `--quiet`

```
Study complete: 12/12 ran, 0 failed, 0 skipped
Results: results/batch-size-sweep-2026-02/
```

---

## Thermal Gap Display

During `config_gap_seconds` and `cycle_gap_seconds` pauses, show a countdown — researchers
need to know the tool is alive and waiting intentionally:

```
  · [4/12]  pytorch / batch=16 / bf16   →  waiting thermal gap  (55s remaining)
```

Not showing a countdown was considered a UX failure — silence during a 5-minute gap between
cycles would make users think the tool crashed.

---

## Implementation: Plain Output (~200 LOC)

> **Superseded (2026-02-26):** The Rich library stack below was the original design. Preservation
> audit N-X04 (2026-02-26) decided to simplify drastically: 950 LOC Rich-based display → ~200 LOC
> plain output. 0/5 peers use Rich for results display. vLLM bench pattern (key:value + ASCII
> separators) is the new target. tqdm for progress bars. No Rich dependency required.
> See [../preservation_audit/INDEX.md](../preservation_audit/INDEX.md) decision #14.

| Layer | Library | Notes |
|---|---|---|
| Study-level progress | tqdm | Simple progress bar |
| Experiment status | `print()` to stderr | key:value lines + ASCII separators |
| Verbose subprocess logs | `print()` to stderr | Indented, prefixed |
| Final summary | `print()` to stdout | Plain text table |

---

## Event Queue Pattern (Study)

> **Superseded (2026-02-26):** Rich-based event queue below replaced by plain print() output.
> The event queue pattern itself remains — subprocess events still flow via `multiprocessing.Queue`.
> The display consumer just uses `print()` + tqdm instead of Rich Progress/Console.

See [experiment-isolation.md](experiment-isolation.md) for the full Queue + consumer thread pattern.

---

## Environment Variable Support

| Variable | Effect |
|---|---|
| `NO_COLOR` | Disables all colour (Rich respects natively) |
| `TERM=dumb` | Falls back to ASCII-only progress (Rich detects) |
| `LLM_ENERGY_JSON_OUTPUT=true` | Suppresses all human-readable output; emits machine-readable JSON on stdout only (pipeline use) |

---

## `llem config` Output

Two scenarios:

**First use — no user config file:**

```
Environment
  GPU        NVIDIA A100-SXM4-80GB · CUDA 12.4 · Driver 535.86        ✓
  Docker     available (v27.2.0)                                       ✓

Backends
  pytorch    installed (torch 2.5.1 · transformers 4.47.0)            ✓
  vllm       not installed → Docker recommended (docker pull ghcr.io/llenergymeasure/vllm:latest)
  tensorrt   not installed → Docker recommended (docker pull ghcr.io/llenergymeasure/tensorrt:latest)

Energy
  nvml       available (base energy measurement)                       ✓
  zeus       not installed (more accurate) → pip install llenergymeasure[zeus]

User config: ~/.config/llenergymeasure/config.yaml — not found
  Running with all defaults. To customise, create the file with:

    runners:
      pytorch: local

    output:
      results_dir: ./results
```

**Configured state:**

```
llem config   # env snapshot + user config display

Environment
  GPU        NVIDIA A100-SXM4-80GB (1 available, CUDA 12.4)
  Docker     available (v27.2.0)

Backends
  pytorch    transformers 4.47.0 / torch 2.4.0   installed
  vllm       not installed → Docker recommended (docker pull ghcr.io/llenergymeasure/vllm:latest)
  tensorrt   not installed → Docker recommended (docker pull ghcr.io/llenergymeasure/tensorrt:latest)

Energy
  zeus       v0.13.1 (NVML counter mode)          installed
  codecarbon not installed → pip install llenergymeasure[codecarbon]

User Config   ~/.config/llenergymeasure/config.yaml
  output.results_dir       ./results
  measurement.n_cycles     3
  (no runner overrides — all backends use local)

  → To configure Docker runners: add runners.pytorch: local
```
> **Resolved (2026-02-25):** Per NEEDS_ADDRESSING item 25, show Docker recommendations for
> vLLM/TRT (not pip install hints). vLLM and TRT require Docker for multi-backend studies.
> For single-backend local use, users can install via pip extras — but the `llem config`
> output guides toward Docker as the primary multi-backend path.

---

## Pre-flight Output (Implicit, Not User-Invoked)

Runs automatically before every `llem run` invocation. Hard block on failure.

```
Study pre-flight: 12 experiments across pytorch, vllm, tensorrt

  Backends
  ✓ pytorch    transformers 4.47.0 / torch 2.4.0
  ✗ vllm       NOT installed → Docker recommended: docker pull ghcr.io/llenergymeasure/vllm:latest
  ✗ tensorrt   NOT installed → Docker recommended: docker pull ghcr.io/llenergymeasure/tensorrt:latest

  Energy
  ✓ zeus       v0.13.1 (NVML counter mode — Volta+)
  ✗ codecarbon NOT installed (required by 4 experiments)
               → pip install llenergymeasure[codecarbon]

  Models
  ✗ meta-llama/Llama-3-70B   gated — no HF_TOKEN → export HF_TOKEN=<your_token>

4 issues found. Resolve all before study can run.
```

Validation order: hardware → backends → energy → models → datasets (dependency order).
All failures reported at once — not one at a time.

---

## Deferred

- Live power draw display (requires Zeus streaming — later v2.0 milestone)
- Per-backend container log streaming for Docker studies (v2.0 Docker milestone)
- TUI (full terminal UI) — Rich Live approach is sufficient

---

## Related

- [../decisions/live-observability.md](../decisions/live-observability.md): Decision rationale
- [experiment-isolation.md](experiment-isolation.md): Queue + consumer thread pattern
- [cli-commands.md](cli-commands.md): `--quiet`, `--verbose` flags per command
