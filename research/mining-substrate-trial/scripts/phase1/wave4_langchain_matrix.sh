#!/usr/bin/env bash
# Phase 1, Wave 4 - langchain-cell matrix: {multistage, hybrid} x {mid-32B, large-70B}.
# Per user: try the langchain cells across mid + large OSS. vllm cell. Container ollama.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/tmp/round0b_venv/bin/python
OUT=findings/study/phase1_wave4/results
mkdir -p "$OUT"
export WAVE_OLLAMA=http://localhost:11435

run () {  # strategy model
  local strat=$1 model=$2 tag
  tag=$(echo "$model" | tr '/:.' '___')
  echo "=== LANGCHAIN $strat  $model  vllm ($(date -u +%H:%M)Z) ==="
  if [ "$strat" = multistage ]; then
    "$PY" scripts/phase1/wave4_multistage.py --engine vllm --vslug v0_19_1 --version 0.19.1 \
      --model "$model" 2>&1 | grep -E "PHASE1_W4MS_RESULT" | tail -1
    cp -f "/tmp/phase1_w4ms_vllm_v0_19_1_${tag}.json" "$OUT/w4ms_${tag}_vllm_v0_19_1.json" 2>/dev/null
    cp -f "/tmp/phase1_w4ms_vllm_v0_19_1_${tag}_corpus.yaml" "$OUT/w4ms_${tag}_vllm_v0_19_1_corpus.yaml" 2>/dev/null
  else
    "$PY" scripts/phase1/wave4_hybrid_chain.py --engine vllm --vslug v0_19_1 --version 0.19.1 \
      --model "$model" 2>&1 | grep -E "PHASE1_W4HC_RESULT" | tail -1
    cp -f "/tmp/phase1_w4hc_vllm_v0_19_1_${tag}.json" "$OUT/w4hc_${tag}_vllm_v0_19_1.json" 2>/dev/null
    cp -f "/tmp/phase1_w4hc_vllm_v0_19_1_${tag}_corpus.yaml" "$OUT/w4hc_${tag}_vllm_v0_19_1_corpus.yaml" 2>/dev/null
  fi
}

# mid (32B code) first - faster - then large (70B); hybrid before multistage (more promising)
run hybrid qwen2.5-coder:32b
run multistage qwen2.5-coder:32b
run hybrid llama3.1:70b
run multistage llama3.1:70b
echo "LANGCHAIN_MATRIX_DONE"
