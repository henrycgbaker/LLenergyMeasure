# mining-substrate-trial

Empirical trial investigating the substrate question for engine-config invariant mining: given a target engine + a frozen reference version, what extraction substrate (pure-static, pure-LLM, hybrid combinations) best balances recall, precision, brittleness across version bumps, and operational cost?

The trial spans four phases:

- **Phase 1** - version-lock + per-engine reference matrices (47 active + bumped cells across transformers / vLLM / TensorRT-LLM).
- **Phase 2** - LLM infrastructure + locked prompts.
- **Phase 3a** - pure-strategy bake-off (47 cells).
- **Phase 3b** - hybrid pattern catalogue (H2/H3/H4/H6/H7/H9 + E6/E9).
- **Phase 3c** (pending) - Claude-Opus / Claude-Sonnet agentic patterns once `ANTHROPIC_API_KEY` arrives.
- **Phase 4** - synthesis + recommendation.

The primary deliverable is [`findings/empirical_trial_outcome.md`](findings/empirical_trial_outcome.md), which records the chosen substrate and defended trade-offs against the validated-union ground truth.

## Status

ACTIVE through the Phase 3c addendum (claude-opus / claude-sonnet patterns); ARCHIVED thereafter. The directory carries `DECISIONS_LOG.md` as the running narrative; expect a research-paper-style IA restructure (problem-statement / methodology / results / decision-space / recommendation) once the addendum lands.

## Quick links

- [`findings/empirical_trial_outcome.md`](findings/empirical_trial_outcome.md) - Phase 4 synthesis.
- [`findings/trial_epistemic_framing.md`](findings/trial_epistemic_framing.md) - the framing this synthesis answers to.
- [`findings/phase4_0_validated_union_summary.md`](findings/phase4_0_validated_union_summary.md) - validated-union ground truth (the corrected reference used for re-scoring).
- [`findings/trial_matrix_vu.md`](findings/trial_matrix_vu.md) - validated-union per-cell results matrix.
- [`findings/trial_matrix.md`](findings/trial_matrix.md) - original (a)-as-reference matrix (retained for delta comparison).
- [`DECISIONS_LOG.md`](DECISIONS_LOG.md) - the full trial narrative (3500+ lines; chronological).

## Reproducibility

Trial scripts live under `scripts/`. Because the directory name contains hyphens, the scripts are not importable as a Python package via the `-m` flag. Invoke via file path:

```bash
uv run python research/mining-substrate-trial/scripts/trial_aggregate.py
```

Each script prepends the trial scripts directory to `sys.path` so sibling imports (`from trial_scoring import ...`, `from strategies.llm_b_oss import ...`) resolve at module load time. The project root is also on `sys.path` so trial code can import production helpers (`scripts.validate_invariants`, `engine_versions.<engine>`).

### Tests

Run the trial smoke tests:

```bash
uv run python -m pytest research/mining-substrate-trial/scripts/test_trial_scoring.py -v
uv run python -m pytest research/mining-substrate-trial/scripts/strategies/test_agentic_tool_harness.py -v
```

### Container Ollama

The (b)/(c)/(d-ab)/h*/e* strategies dispatch to a local Ollama container on port 11435. See the Phase 1 + Phase 2 entries in `DECISIONS_LOG.md` for the launch incantation and image tag.

### Per-engine validator containers

Validation (Phase 4.0 validated-union builder) dispatches to per-engine production containers:

- transformers: `llenergymeasure:transformers-4.57.3`
- vLLM: `vllm/vllm-openai:v0.7.3`
- TensorRT-LLM: `nvcr.io/nvidia/tensorrt-llm/release` (1.2.1)

The dispatch logic is in `scripts/trial_scoring.py` (`runtime_validate_invariants`, `build_validated_union`).
