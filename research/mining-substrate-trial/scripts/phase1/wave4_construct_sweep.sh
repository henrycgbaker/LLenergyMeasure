#!/usr/bin/env bash
# Phase 1, Wave 4 - construction-grounding sweep driver.
# Runs the construct-grounded prompt (AST signature context, NO floor) across the
# given models, both cells, scoring recall vs GT. Container ollama :11435.
# Usage: wave4_construct_sweep.sh <model> [<model> ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/tmp/round0b_venv/bin/python
PROMPT=findings/study/phase1_wave4/construct_grounded_prompt.md
OUT=findings/study/phase1_wave4/results
mkdir -p "$OUT"
export WAVE_OLLAMA=http://localhost:11435

run_cell () {
  local model=$1 engine=$2 vslug=$3 version=$4
  local tag; tag=$(echo "$model" | tr '/:.' '___')
  echo "=== w4c CONSTRUCT $model  $engine $vslug  ($(date -u +%H:%M:%S)Z) ==="
  "$PY" scripts/phase1/wave4_construct.py --engine "$engine" --vslug "$vslug" \
    --version "$version" --model "$model" --prompt-file "$PROMPT"
  cp -f "/tmp/phase1_w4c_${engine}_${vslug}_${tag}.json" "$OUT/w4c_${tag}_${engine}_${vslug}.json" 2>/dev/null \
    && echo "archived $OUT/w4c_${tag}_${engine}_${vslug}.json"
  cp -f "/tmp/phase1_w4c_${engine}_${vslug}_${tag}_corpus.yaml" "$OUT/w4c_${tag}_${engine}_${vslug}_corpus.yaml" 2>/dev/null
  cp -f "/tmp/phase1_w4c_${engine}_${vslug}_raw.txt" "$OUT/w4c_${tag}_${engine}_${vslug}_raw.txt" 2>/dev/null
}

for model in "$@"; do
  run_cell "$model" vllm v0_19_1 0.19.1
  run_cell "$model" tensorrt v1_2_1 1.2.1
done
echo "WAVE4C_DONE"
