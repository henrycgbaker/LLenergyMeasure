# Research Synthesis

Synthesised from 9 raw research documents (5,377 lines) in
`.planning/phases/04.5-strategic-reset/research/`. These files are the summary layer —
see the raw docs for full detail and code examples.

**Date**: 2026-02-17

## Files

| File | Contents |
|------|----------|
| [competitive-landscape.md](competitive-landscape.md) | Optimum-Benchmark, Zeus/ML.ENERGY, AIPerf, lm-eval comparison; our positioning; deployment vs model distinction |
| [industry-patterns.md](industry-patterns.md) | Validated industry norms: pip extras, no default backend, lib→CLI→web, local-first results, opt-in upload |
| [energy-measurement-tools.md](energy-measurement-tools.md) | Zeus vs CodeCarbon accuracy comparison, NVML measurement, our existing EnergyBackend Protocol |
| [web-platform-patterns.md](web-platform-patterns.md) | ML.ENERGY static JSON leaderboard, ClearML-Agent outbound worker model, FastAPI+React stack |

## Key Finding

LLenergyMeasure occupies a unique position: the only tool combining energy measurement +
LLM streaming latency (TTFT/ITL) + FLOPs estimation + multi-backend comparison. The
closest competitor (Optimum-Benchmark) is broader but shallower on energy and LLM-specific
metrics.
