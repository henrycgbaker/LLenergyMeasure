# Competitive Landscape

**Source**: `phases/04.5-strategic-reset/research/09-broader-landscape.md` (970 lines, 38+ tools scanned)

## Positioning Matrix

|                      | Shallow energy | Deep energy |
|----------------------|----------------|-------------|
| **LLM-specific**     | *(gap)*        | **LLenergyMeasure** (us) |
| **General ML**       | Optimum-Benchmark, AIPerf | Zeus/ML.ENERGY |

We occupy the upper-right quadrant with no direct competitor.

## Tool Comparison

| Tool | What they do | How we differ |
|------|-------------|---------------|
| **Optimum-Benchmark** (HF, 329 stars) | Multi-backend perf benchmarking, 8+ backends, CodeCarbon energy | Broader backend coverage; we have deeper energy (Zeus target), LLM streaming latency (TTFT/ITL), FLOPs, thermal throttling, SSOT config |
| **Zeus / ML.ENERGY** | Precise NVML energy + "which model is efficient?" leaderboard | They compare models; we compare deployment configs — complementary. Zeus is an integration target (energy backend), not a competitor |
| **AIPerf** (NVIDIA) | Serving-endpoint benchmarking, DCGM GPU telemetry, SLO-based goodput | They benchmark serving endpoints; we benchmark direct model loading + deployment parameter sweeps |
| **lm-eval-harness** | Quality evaluation across 28+ backends, powers HF Open LLM Leaderboard | Quality only — no energy/throughput. Integration target (v3.0), not competitor |
| **LLMPerf** | LLM serving throughput (archived) | Archived; we cover the same ground and more |
| **GenAI-Perf** (NVIDIA) | Triton endpoint benchmarking | Endpoint-only, deprecated in favour of AIPerf |

## The Core Differentiation

**ML.ENERGY asks**: "Which model is more efficient?"
**We ask**: "Which deployment configuration is more efficient for a given model?"

ML.ENERGY's own data validates our thesis — a single parameter change (batch size) produces
**7.5x energy difference** on the same model. This is what no other tool demonstrates
systematically.

## Unique Value Combination

No single tool combines all of:
- Direct GPU energy measurement (NVML-level accuracy)
- LLM streaming latency (TTFT, ITL)
- FLOPs estimation
- Multi-backend comparison (PyTorch / vLLM / TensorRT-LLM)
- SSOT config introspection for parameter sweeps

## Two High-Value Integration Opportunities

1. **Zeus as energy backend** (P0): More accurate measurement, our Protocol already supports it
2. **lm-eval integration** (P1, v3.0): Quality-alongside-efficiency — unique differentiator. "INT4 quantisation gives 2.3x energy improvement at cost of 1.2% MMLU accuracy."
