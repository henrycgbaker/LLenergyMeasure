# Campaign Mode

Campaigns enable multi-config comparison experiments with statistical robustness.
Run multiple experiment configurations across multiple cycles to fairly compare
different backends, optimisations, or hyperparameters.

## Quick Start

```bash
# Compare two configs with 3 cycles
llm-energy-measure campaign configs/pytorch.yaml configs/vllm.yaml \
  --campaign-name "backend-comparison" \
  --cycles 3 \
  --dataset alpaca \
  -n 100

# Use a campaign YAML file
llm-energy-measure campaign configs/examples/campaign_example.yaml

# Dry-run to preview execution plan
llm-energy-measure campaign configs/*.yaml --campaign-name "test" --dry-run
```

## Why Use Campaigns?

| Feature | Single Experiment | Campaign |
|---------|-------------------|----------|
| Statistical robustness | One measurement | Multiple cycles with statistics |
| Ordering bias | N/A | Eliminated with shuffled structure |
| Thermal consistency | Variable | Controlled gaps between experiments |
| Comparison fairness | Sequential runs | Interleaved execution |
| Warmup | Per-experiment | Per-config, tracked separately |

## Configuration

### Campaign YAML Format

```yaml
# Required
campaign_name: "pytorch-vs-vllm-comparison"

# Prompt source (overrides individual configs)
dataset: alpaca        # HuggingFace dataset or alias
num_samples: 100       # Number of prompts

# Experiment configs to compare
configs:
  - configs/pytorch_base.yaml
  - configs/vllm_optimised.yaml

# Execution parameters
execution:
  cycles: 5                      # Number of complete cycles (1-20)
  structure: shuffled            # interleaved | shuffled | grouped
  warmup_prompts: 5              # Min warmup prompts per config
  warmup_timeout_seconds: 30     # Max warmup time
  config_gap_seconds: 60         # Gap between configs
  cycle_gap_seconds: 300         # Gap between cycles

# Optional: scheduled execution
schedule:
  at: "03:00"                    # Time of day (HH:MM)
  days: ["mon", "wed", "fri"]    # Or: "weekdays", "weekends"
```

### CLI Options

All YAML fields have corresponding CLI flags:

```bash
llm-energy-measure campaign CONFIG_PATHS... \
  --campaign-name NAME          # Required when using multiple configs
  --dataset DATASET             # Dataset override
  --sample-size N               # Sample size override
  --cycles N                    # Number of cycles (default: 3)
  --structure TYPE              # interleaved|shuffled|grouped
  --warmup-prompts N            # Min warmup prompts
  --warmup-timeout SECONDS      # Max warmup time
  --config-gap SECONDS          # Gap between configs
  --cycle-gap SECONDS           # Gap between cycles
  --seed N                      # Random seed for shuffled mode
  --results-dir PATH            # Output directory
  --dry-run                     # Preview without running
  --yes                         # Skip confirmation
```

**CLI flags override YAML values.**

## Execution Structures

### Interleaved (Default)

Configs run in fixed order within each cycle. Fair comparison with predictable ordering.

```
Cycle 1: A → B → C
Cycle 2: A → B → C
Cycle 3: A → B → C
```

### Shuffled

Configs run in random order within each cycle. Eliminates ordering bias.

```
Cycle 1: B → A → C
Cycle 2: C → B → A
Cycle 3: A → C → B
```

Use `--seed N` for reproducible random ordering.

### Grouped

All cycles of one config complete before the next. Useful for thermal baseline establishment.

```
Config A: cycle 1, cycle 2, cycle 3
Config B: cycle 1, cycle 2, cycle 3
Config C: cycle 1, cycle 2, cycle 3
```

## Warmup Strategy

Warmup runs before each config's measurement phase to prime:
- Model weights in GPU memory
- KV cache patterns
- CUDA kernels and memory allocation

**Dual-criteria approach**: Warmup stops when EITHER condition is met:
- Minimum prompts completed (`warmup_prompts`, default: 5)
- Timeout reached (`warmup_timeout_seconds`, default: 30s)

This balances thorough warmup with time efficiency.

## Thermal Management

GPU temperature affects performance and energy measurements. Campaigns provide controlled gaps:

| Gap Type | Default | Purpose |
|----------|---------|---------|
| `config_gap_seconds` | 60s | Allow GPU to cool between configs |
| `cycle_gap_seconds` | 300s | Full thermal reset between cycles |

Adjust based on your GPU's thermal characteristics. Higher-TDP GPUs may need longer gaps.

## Results

Campaign results are stored alongside regular experiment results:

```
results/
├── raw/
│   ├── 0042/              # Individual experiment results
│   │   └── process_0.json
│   └── 0043/
│       └── process_0.json
└── aggregated/
    ├── 0042.json
    └── 0043.json
```

Each experiment result includes campaign metadata:
- `campaign_name`: The campaign this experiment belongs to
- `campaign_id`: 8-character hash of campaign name
- `cycle_id`: Which cycle (0-indexed)

## Examples

### Backend Comparison

Compare PyTorch vs vLLM on the same model:

```bash
llm-energy-measure campaign \
  configs/examples/pytorch_example.yaml \
  configs/examples/vllm_example.yaml \
  --campaign-name "llama-backend-comparison" \
  --cycles 5 \
  --structure shuffled \
  --dataset alpaca \
  -n 200
```

### Overnight Benchmark Suite

Run comprehensive comparison overnight:

```yaml
# campaigns/overnight.yaml
campaign_name: "comprehensive-benchmark"
dataset: ai-energy-score
num_samples: 500

configs:
  - configs/pytorch_fp16.yaml
  - configs/pytorch_bf16.yaml
  - configs/vllm_default.yaml
  - configs/vllm_flash_attn.yaml

execution:
  cycles: 10
  structure: shuffled
  warmup_prompts: 10
  warmup_timeout_seconds: 60
  config_gap_seconds: 120
  cycle_gap_seconds: 600

schedule:
  at: "22:00"
```

### Quick Validation

Quick comparison with minimal resources:

```bash
llm-energy-measure campaign configs/*.yaml \
  --campaign-name "quick-test" \
  --cycles 2 \
  --warmup-prompts 1 \
  --config-gap 10 \
  --cycle-gap 30 \
  -n 20
```

## Troubleshooting

### Experiment Failures

If an experiment fails mid-campaign:
- The campaign continues with remaining experiments
- Failed experiments are logged in the summary
- Campaign exits with code 1 if any experiments failed

### Memory Issues

For large models or limited GPU memory:
- Use longer `config_gap_seconds` to allow memory cleanup
- Consider `grouped` structure for better memory locality
- Monitor GPU memory between experiments

### Thermal Throttling

If seeing performance degradation:
- Increase `cycle_gap_seconds` (try 600s)
- Use `shuffled` structure to distribute load
- Monitor GPU temperature during runs

## See Also

- [CLI Reference](cli.md) - Full command documentation
- [Backends Guide](backends.md) - Backend-specific configuration
- [Configuration](../src/llm_energy_measure/config/README.md) - Config schema details
