# Adapter Support (LoRA / QLoRA)

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-19
**Research:** [../research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)

---

## Context

Researchers frequently measure fine-tuned models via parameter-efficient adapters (LoRA, QLoRA) rather than fully fine-tuned weights. Supporting adapters in LLenergyMeasure allows measurement of:

- The inference overhead of running an adapter (unmerged) vs the base model
- The deployed cost of a fine-tuned model (merged weights)
- Efficiency comparisons across adapter configurations (rank, target modules)

The initial position was to defer adapter support to v2.1. This was reversed after peer tool research confirmed that LoRA support is standard across the ecosystem (lm-eval, optimum-benchmark, vLLM all support it) and that the implementation is simpler than initially assumed — adapter FLOPs are negligible, so the FLOPs formula does not change.

> **Superseded 2026-02-19:** Initial recommendation was "defer to v2.1". Reversed after peer research confirmed lightweight implementation path and broad peer tool support. See Confirmed Decisions table below.

---

## Considered Options

### A1: When to add LoRA support

| Option | Pros | Cons |
|--------|------|------|
| **v2.0 — optional `lora:` block in ExperimentConfig (chosen)** | Peer tools (lm-eval, optimum-benchmark, vLLM) all support it; low implementation cost; LoRA FLOPs formula unchanged | Adds schema complexity to ExperimentConfig from day one |
| v2.1 deferral | Simpler v2.0 schema | Misses a common researcher use case; peer tools treat it as standard, not an extension |

**Rejected: v2.1 deferral (2026-02-19).** Peer tool evidence makes clear that LoRA support is expected at launch, not as a future addition.

---

### A2: Adapter source — Hub ID vs local path

| Option | Pros | Cons |
|--------|------|------|
| **Both `adapter_id` (HF Hub) and `adapter_path` (local) — both supported (chosen)** | Mirrors lm-eval `peft=` flexibility; covers offline/air-gapped research environments | Two code paths to maintain |
| Hub ID only | Single code path | Blocks offline / private adapter workflows |
| Local path only | Works offline | Blocks Hub-hosted public adapters |

---

### A3: Default merge behaviour

| Option | Pros | Cons |
|--------|------|------|
| **`merge_weights: false` (unmerged, default — chosen)** | Measures the adapter's actual inference overhead (~10–12% throughput penalty per Fireworks AI 2024); reflects real-world adapter deployment cost | Slightly slower than merged; memory footprint includes separate adapter weights |
| `merge_weights: true` (merged default) | Zero overhead; simpler memory footprint | Hides the adapter cost; user must explicitly opt into measuring what they deployed |

The two modes serve distinct research questions:
- `merge_weights: false`: measuring *adapter overhead* — how much does running this adapter cost?
- `merge_weights: true`: measuring *deployed cost* of a fine-tuned model (production deployment with single adapter)

| Mode | Throughput | Memory | Notes |
|------|-----------|--------|-------|
| **Unmerged** (default) | ~10–12% penalty | Base + adapter weights separate | Measures adapter inference cost |
| **Merged** (`merge_weights: true`) | Zero overhead | Single fused tensor | Models production deployment (single adapter) |

Source: Fireworks AI (2024); multiple independent confirmations. Research confidence: MEDIUM.

---

### A4: FLOPs accounting for adapters

| Option | Pros | Cons |
|--------|------|------|
| **Use base model N_params unchanged — adapter FLOPs negligible (chosen)** | For rank-16 LoRA on 7B model, adapter FLOPs ≈ 0.06% of total; well within formula uncertainty | Technically an approximation |
| Separate FLOPs formula for adapter matrices | Technically correct | Adds complexity for an effect smaller than the formula's own approximation error |

For a 7B parameter model with a rank-16 LoRA adapter:
- Base model FLOPs: `2 × 7,000,000,000 × T` per token batch
- Adapter FLOPs: `2 × (2 × 16 × d_model × n_layers)` per token — roughly `2 × 4,200,000 × T`
- Adapter fraction: ~0.06% of total FLOPs

Treating adapter FLOPs as negligible is within the measurement uncertainty of the `2×N×T` formula itself. See [decisions/flops-estimation.md](flops-estimation.md) for full FLOPs formula.

---

### A5: Backend compatibility

Backend support is constrained by the backends' own capabilities — this is not an LLenergyMeasure design choice:

| Backend | LoRA support | Notes |
|---------|-------------|-------|
| **PyTorch** | Via PEFT `PeftModel.from_pretrained()` | Works with merged or unmerged; standard PEFT API, same pattern as lm-eval |
| **vLLM** | First-class via `LoRARequest` | `enable_lora=True` required in VLLMConfig; adapter loaded per-request; partial TP sharding supported |
| **TensorRT-LLM** | Not supported — validation error if `lora:` specified | TRT-LLM requires pre-merged weights before engine compilation; users must merge before TRT engine build |

---

## Decision

We will add an **optional `lora:` block to `ExperimentConfig` in v2.0**, supporting both HF Hub adapter IDs and local paths, defaulting to unmerged weights.

- `adapter_id` (HF Hub) and `adapter_path` (local) — both supported
- `merge_weights: false` is the default (measures adapter overhead)
- LoRA FLOPs use the same formula as the base model — adapter contribution is negligible
- Adapter presence recorded in result metadata: `lora_adapter_id`, `lora_merged`
- TensorRT-LLM raises a validation error if `lora:` is specified — pre-merge is required
- `config_hash` (now `measurement_config_hash`) includes `lora.adapter_id`/`lora.adapter_path` and `lora.merge_weights` — different adapters produce different hashes

**Rationale:** Peer tool evidence (lm-eval, optimum-benchmark, vLLM) shows LoRA is standard, not an extension. The implementation cost is low because the FLOPs formula is unchanged. Defaulting to unmerged makes the measurement intent explicit.

---

## Consequences

**Positive:**
- Researchers can measure adapter overhead directly — a common and important use case
- Results are reproducible: adapter identity and merge state are in the config hash
- Consistent with peer tool behaviour (lm-eval `peft=`, vLLM `LoRARequest`)

**Negative / Trade-offs:**
- TensorRT-LLM users cannot use `lora:` without pre-merging — this is a TRT constraint, not a design choice
- Two adapter source code paths (`adapter_id` vs `adapter_path`) to maintain

**Neutral / Follow-up decisions triggered:**
- `lora:` block schema: see [designs/experiment-config.md](../designs/experiment-config.md) § "LoRA / Adapter Support"
- `lora_adapter_id`, `lora_merged` result fields: see [designs/result-schema.md](../designs/result-schema.md)
- FLOPs formula unchanged: see [decisions/flops-estimation.md](flops-estimation.md)

---

## Peer References

| Tool | LoRA support | Notes |
|------|-------------|-------|
| **lm-eval** | `peft=<hub-id-or-path>` in model_args | Unmerged by default; PEFT optional extra |
| **optimum-benchmark** | PEFT extra; delegates to backend | No dedicated LoRA scenario |
| **vLLM** | First-class `LoRARequest` API | Multi-LoRA; runtime loading; partial TP sharding |
| **TensorRT-LLM** | No native support | Pre-merge required |
| **AIEnergyScore** | Base models only | Not relevant to their benchmark design |

Full peer analysis: [research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)

---

## Related

- [decisions/flops-estimation.md](flops-estimation.md): FLOPs formula unchanged for LoRA
- [designs/experiment-config.md](../designs/experiment-config.md): ExperimentConfig `lora:` block schema
- [designs/result-schema.md](../designs/result-schema.md): `lora_adapter_id`, `lora_merged` result fields
- [research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)
- NEEDS_ADDRESSING.md item 22
