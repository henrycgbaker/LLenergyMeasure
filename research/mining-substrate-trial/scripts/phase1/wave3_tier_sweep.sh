#!/usr/bin/env bash
# Phase 1, Wave 3 Arm A - tier-sweep driver.
# Runs the LOCKED wave-2 kwargs prompt (sha256 7cd74960...) at a given OSS tier
# across the two wave-1/2 cells (vllm 0.19.1, tensorrt 1.2.1), scoring through the
# real production gate. Archives tier-tagged results so successive tiers do not
# clobber the ephemeral /tmp output. Per PHASE1_WAVE3_PREREG.md.
#
# Usage: wave3_tier_sweep.sh <ollama-model>
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> research/mining-substrate-trial
MODEL="${1:?usage: wave3_tier_sweep.sh <ollama-model>}"
PY=/tmp/round0b_venv/bin/python
PROMPT=findings/study/phase1_wave2/wg_extend_kwargs_prompt.md
OUT=findings/study/phase1_wave3/results
mkdir -p "$OUT"
TAG=$(echo "$MODEL" | tr '/:.' '___')

run_cell () {
  local engine=$1 vslug=$2 version=$3
  echo "=== TIER $MODEL  CELL $engine $vslug  ($(date -u +%H:%M:%S)Z) ==="
  "$PY" scripts/phase1/wave1.py --engine "$engine" --vslug "$vslug" \
    --version "$version" --rung oss --model "$MODEL" --prompt-file "$PROMPT"
  cp -f "/tmp/phase1_w1_${engine}_${vslug}_oss.json" \
        "$OUT/w3_${TAG}_${engine}_${vslug}.json" 2>/dev/null \
    && echo "archived -> $OUT/w3_${TAG}_${engine}_${vslug}.json"
  cp -f "/tmp/phase1_w1_${engine}_${vslug}_oss_raw.txt" \
        "$OUT/w3_${TAG}_${engine}_${vslug}_raw.txt" 2>/dev/null
}

run_cell vllm v0_19_1 0.19.1
run_cell tensorrt v1_2_1 1.2.1
echo "WAVE3_TIER_DONE $MODEL"
