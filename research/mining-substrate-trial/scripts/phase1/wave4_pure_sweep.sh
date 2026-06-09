#!/usr/bin/env bash
# Phase 1, Wave 4a - pure-LLM (prompt) sweep driver.
# Runs the locked pure-LLM+kwargs prompt (NO floor) across the given models, both
# cells (vllm 0.19.1, tensorrt 1.2.1), scoring recall vs GT. Models served by the
# containerized ollama on :11435 (WAVE_OLLAMA). Archives tier-tagged results.
# Usage: wave4_pure_sweep.sh <model> [<model> ...]
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> research/mining-substrate-trial
PY=/tmp/round0b_venv/bin/python
PROMPT=findings/study/phase1_wave4/pure_llm_kwargs_prompt.md
OUT=findings/study/phase1_wave4/results
mkdir -p "$OUT"
export WAVE_OLLAMA=http://localhost:11435

run_cell () {
  local model=$1 engine=$2 vslug=$3 version=$4
  local tag; tag=$(echo "$model" | tr '/:.' '___')
  echo "=== 4a PURE $model  $engine $vslug  ($(date -u +%H:%M:%S)Z) ==="
  "$PY" scripts/phase1/wave4_pure.py --engine "$engine" --vslug "$vslug" \
    --version "$version" --model "$model" --prompt-file "$PROMPT"
  cp -f "/tmp/phase1_w4_${engine}_${vslug}_${tag}.json" "$OUT/w4a_${tag}_${engine}_${vslug}.json" 2>/dev/null \
    && echo "archived $OUT/w4a_${tag}_${engine}_${vslug}.json"
  cp -f "/tmp/phase1_w4_${engine}_${vslug}_raw.txt" "$OUT/w4a_${tag}_${engine}_${vslug}_raw.txt" 2>/dev/null
  cp -f "/tmp/phase1_w4_${engine}_${vslug}_llmcorpus.yaml" "$OUT/w4a_${tag}_${engine}_${vslug}_corpus.yaml" 2>/dev/null
}

for model in "$@"; do
  run_cell "$model" vllm v0_19_1 0.19.1
  run_cell "$model" tensorrt v1_2_1 1.2.1
done
echo "WAVE4A_DONE"
