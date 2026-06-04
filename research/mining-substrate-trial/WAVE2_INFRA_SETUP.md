# Wave 2 infrastructure setup

What needs to be in place before Wave 2 cells can run. Each entry is keyed by which strategy needs it.

---

## 1. Trial worktree + branch

```bash
cd ~/workspace/llenergymeasure-trial      # the worktree, already created
git rev-parse --abbrev-ref HEAD            # should print: trial/mining-substrate-bakeoff
```

If the worktree is missing:

```bash
git -C ~/workspace/llenergymeasure worktree add ~/workspace/llenergymeasure-trial trial/mining-substrate-bakeoff
```

---

## 2. Python deps for Tier A strategies

The project's `.venv` already has `pydantic`, `msgspec`, `pyyaml`. Tier A strategies need additional packages installed into the same venv:

```bash
cd ~/workspace/llenergymeasure-trial
uv add --dev tree-sitter==0.23.0 tree-sitter-python==0.23.6
uv add --dev hypothesis      # for W2-a-fuzz pilot
```

Confirm:

```bash
uv run python -c "import tree_sitter, tree_sitter_python, hypothesis; print('OK')"
```

---

## 3. Engine venvs (Tier A strategies that import the engine)

`a_pydantic_native` and `a_runtime_trace` both need the engine importable at runtime. The Wave 1 trial venv layout is `/tmp/trial_<engine>_<version_slug>_venv/`. Build via:

```bash
uv run python research/mining-substrate-trial/scripts/venv_setup.py \\
    --engine vllm --version 0.7.3 --mode source-only
```

For runtime-trace specifically, you need a FULL install (not source-only) so the engine actually imports + executes. Use the corresponding container instead:

```bash
docker pull vllm/vllm-openai:v0.7.3
```

And dispatch the cell inside the container (the runner does not do this automatically; the user wires it up per WAVE2_PROTOCOL section 5 discipline F).

---

## 4. Ollama for LLM strategies

Tier B + C strategies (W2f small-LLM sweep, W2C big-LLM ceiling, W2-h11 / W2-h15) need Ollama serving on port 11435.

```bash
docker run -d --runtime=nvidia --gpus all \\
    -p 11435:11434 \\
    -v ollama:/root/.ollama \\
    --name trial-ollama \\
    ollama/ollama:latest

# Pull Tier A small models first (smallest -> largest)
docker exec trial-ollama ollama pull qwen2.5-coder:7b-instruct-fp16
docker exec trial-ollama ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M
docker exec trial-ollama ollama pull llama3.1:8b-instruct-fp16
docker exec trial-ollama ollama pull phi4:14b-instruct-fp16

# Update findings/wave2_model_digests.toml with each model's digest:
docker exec trial-ollama ollama show qwen2.5-coder:7b-instruct-fp16 --modelfile | head -1
# -> FROM <ref>@sha256:<digest>; copy the digest into the toml file.
```

Tier C big-LLM benchmark models (pull only when ready to run W2C; each is 60-140 GB):

```bash
docker exec trial-ollama ollama pull llama3.3:70b-instruct-fp16
docker exec trial-ollama ollama pull qwen2.5-coder:32b-instruct-fp16
docker exec trial-ollama ollama pull deepseek-coder-v2:236b-instruct-q4_K_M
docker exec trial-ollama ollama pull mixtral:8x22b-instruct-q4_K_M
```

---

## 5. Per-engine validator containers (for h15_closed_loop runtime gate)

The closed-loop hybrid calls `trial_scoring.runtime_validate_invariants_dispatch`, which routes per engine. Each engine needs a container the dispatcher can `docker run` against.

```bash
# transformers: built locally
docker images | grep llenergymeasure:transformers || \\
    docker build -t llenergymeasure:transformers-4.57.3 -f docker/transformers.Dockerfile .

# vllm
docker pull vllm/vllm-openai:v0.7.3

# tensorrt
docker pull nvcr.io/nvidia/tensorrt-llm/release:0.21.0
```

Pin the container digests into `findings/wave2_model_digests.toml` under `[engine_container_pins.*]`.

---

## 6. Smoke-test order (verify each piece works)

Run from cheapest -> most-expensive. Each command should succeed before moving to the next.

```bash
cd ~/workspace/llenergymeasure-trial

# 6a. List wave2 registry (no infra needed)
uv run python research/mining-substrate-trial/scripts/wave2_runner.py --list

# 6b. Tier A: pydantic-native against vllm v0.7.3 (needs vllm in venv)
uv run python research/mining-substrate-trial/scripts/wave2_runner.py \\
    --strategy w2-a-pydantic-native --engine vllm --version v0.7.3

# 6c. Tier A: runtime-trace against vllm v0.7.3 (needs vllm in venv)
uv run python research/mining-substrate-trial/scripts/wave2_runner.py \\
    --strategy w2-a-runtime-trace --engine vllm --version v0.7.3

# 6d. Tier B: h15 closed-loop on transformers (needs Ollama + transformers container)
uv run python research/mining-substrate-trial/scripts/wave2_runner.py \\
    --strategy w2-h15-closed-loop --engine transformers --version v4.57.3
```

Each emits to `research/mining-substrate-trial/findings/trial_scores/wave2/`. Status field on the result JSON: `scored`, `deferred` (NotImplementedError raised at LLM dispatch line; expected for LLM-needing strategies before Ollama is up), `reference_missing`, or `crashed`.

---

## 7. What is intentionally NOT in scope for Wave 2 infra

Per WAVE2_PROTOCOL section 9, the following infrastructure is NOT required:

- Anthropic API key / Claude SDK (Wave 3 deferred).
- Bootstrap CI / seed-variance infra (statistical inference deferred).
- Differential-testing harness (Layer B validation deferred).
- llama.cpp / consumer-GPU runtimes.

---

## 8. Known infrastructure quirks

Surfaced during Wave 2 scaffolding; flagged for future runners.

**Q1: `runtime_validate_invariants_dispatch` silently swallows infra-missing errors.**
If the transformers container is unbuilt or the trial venv lacks `transformers`, the dispatcher emits one stub `RuntimeValidation(error=...)` per invariant instead of raising. H15's closed-loop will then ask the LLM to "fix" entries whose only problem is missing validation infrastructure. Classify any cell whose error count == invariant count as `cell-failure-mode=runtime_gate_unavailable` and discard from cost-frontier aggregation.

**Q2: Wave 1 trial venvs at `/tmp/trial_*_venv/` are ephemeral.**
The `/tmp` venvs created by Wave 1's `venv_setup.py` may have been cleaned up. Rebuild on first cell hit.

**Q3: Model digest pinning is by hand.**
There is no auto-pin step. After `ollama pull`, manually copy the digest from `ollama show <tag> --modelfile` into `findings/wave2_model_digests.toml`. The wave2_runner does not currently verify digests; that's a follow-up.

---

*Last updated: 2026-06-04 (Wave 2 strategy scaffolding round).*
