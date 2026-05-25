#!/usr/bin/env bash
# Renovate-bump cascade test, parameterised per engine.
#
# Usage:
#   bash research/mining-substrate-trial/scratch/cascade_test.sh transformers
#   bash research/mining-substrate-trial/scratch/cascade_test.sh vllm
#   bash research/mining-substrate-trial/scratch/cascade_test.sh tensorrt
#   bash research/mining-substrate-trial/scratch/cascade_test.sh all
#
# Exercises the full chain (see research/mining-substrate-trial/DECISIONS_LOG.md task #6 entry
# for the step-by-step contract). Auto-derives synthetic version from
# the current pin; falls back to v999.999.999 if current can't be read.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

ENGINE_ARG="${1:-all}"

if [[ "${ENGINE_ARG}" == "all" ]]; then
  ENGINES=("transformers" "vllm" "tensorrt")
else
  ENGINES=("${ENGINE_ARG}")
fi

# Sample synthetic field name per engine - must be plausibly absent from
# the real schema so we can verify it travels into config.py.
declare -A SAMPLE_FIELD
SAMPLE_FIELD[transformers]="new_in_synthetic_pin"
SAMPLE_FIELD[vllm]="new_in_synthetic_pin"
SAMPLE_FIELD[tensorrt]="new_in_synthetic_pin"

# Bump versions far enough above current to live in their own bucket.
declare -A BUMP_VER
BUMP_VER[transformers]="4.99.99"
BUMP_VER[vllm]="0.99.99"
BUMP_VER[tensorrt]="9.99.99"

FAILED=()
for ENGINE in "${ENGINES[@]}"; do
  printf "\n========================================\n"
  printf "Cascade test: engine=%s\n" "${ENGINE}"
  printf "========================================\n"

  CURRENT_TOML="engine_versions/${ENGINE}/current.toml"
  NEW_VER="${BUMP_VER[${ENGINE}]}"
  NEW_SAFE="v${NEW_VER//./_}"
  NEW_DIR="engine_versions/${ENGINE}/${NEW_SAFE}/outputs"
  FIELD="${SAMPLE_FIELD[${ENGINE}]}"

  CURRENT_TOML_BACKUP="$(mktemp --suffix=.toml.bak)"
  cp "${CURRENT_TOML}" "${CURRENT_TOML_BACKUP}"

  cleanup() {
    if [[ -f "${CURRENT_TOML_BACKUP}" ]]; then
      cp "${CURRENT_TOML_BACKUP}" "${CURRENT_TOML}"
      rm -f "${CURRENT_TOML_BACKUP}"
    fi
    if [[ -d "engine_versions/${ENGINE}/${NEW_SAFE}" ]]; then
      rm -rf "engine_versions/${ENGINE}/${NEW_SAFE}"
    fi
    uv run python scripts/engine_producers/regen_engine_corpus.py --write \
      --engine "${ENGINE}" >/dev/null 2>&1 || true
    uv run python scripts/engine_producers/regen_engine_configs.py --write \
      --engine "${ENGINE}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  # Step counter local to this engine's run.
  STEP=0
  say()  { STEP=$((STEP + 1)); printf "\n[%s/step%d] %s\n" "${ENGINE}" "${STEP}" "$*"; }
  ok()   { printf "  ok: %s\n" "$*"; }
  fail_step() {
    printf "  FAIL: %s\n" "$*" >&2
    cleanup
    trap - EXIT
    FAILED+=("${ENGINE}")
    continue 2 2>/dev/null || return 0
  }

  ENGINE_FAILED=false

  # 1. baseline
  say "Baseline: --check green at current pin."
  if ! uv run python scripts/engine_producers/regen_engine_corpus.py --check --engine "${ENGINE}" >/dev/null 2>&1; then
    fail_step "baseline regen_corpus --check unexpectedly red"
    ENGINE_FAILED=true
  fi
  if ! uv run python scripts/engine_producers/regen_engine_configs.py --check --engine "${ENGINE}" >/dev/null 2>&1; then
    fail_step "baseline regen_configs --check unexpectedly red"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  ok "both --check green"

  # 2. bump current.toml
  say "Synthetic Renovate bump: ${NEW_VER}"
  sed -i "s/^current_version = .*$/current_version = \"${NEW_VER}\"/" "${CURRENT_TOML}"
  ok "current.toml -> ${NEW_VER}"

  # 3. F#5 tolerance: both regen --check skip cleanly
  say "F#5 tolerance: --check skips (outputs/ absent)."
  if ! uv run python scripts/engine_producers/regen_engine_corpus.py --check --engine "${ENGINE}" 2>&1 | grep -q "outputs directory not present"; then
    fail_step "regen_corpus did not log the skip"
    ENGINE_FAILED=true
  fi
  if ! uv run python scripts/engine_producers/regen_engine_configs.py --check --engine "${ENGINE}" 2>&1 | grep -q "outputs directory not present"; then
    fail_step "regen_configs did not log the skip"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  ok "both regen scripts skipped + exited 0"

  # 4. dispatcher fallback
  say "Dispatcher fallback: load_producer must resolve to a prior vendored version."
  DISPATCH_OUT="$(uv run python -c "
import sys
sys.path.insert(0, '.')
from engine_versions._dispatcher import load_producer
mod = load_producer(engine='${ENGINE}', version='${NEW_VER}', producer='schema_introspector')
print('LOADED:', mod.__name__)
" 2>&1)" || { fail_step "dispatcher raised: ${DISPATCH_OUT}"; ENGINE_FAILED=true; }
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  if ! echo "${DISPATCH_OUT}" | grep -q "LOADED:"; then
    fail_step "dispatcher did not surface LOADED line (got: ${DISPATCH_OUT})"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  if ! echo "${DISPATCH_OUT}" | grep -q "falling back to"; then
    fail_step "dispatcher did not log fallback to stderr"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  ok "dispatcher fell back + logged stderr line"

  # 5. simulate cells writeback
  say "Simulate cells writeback: create synthetic ${NEW_SAFE}/outputs/."
  mkdir -p "${NEW_DIR}"
  cat > "${NEW_DIR}/schema.discovered.json" <<JSON
{
  "schema_version": "2.0.0",
  "engine": "${ENGINE}",
  "engine_version": "${NEW_VER}",
  "engine_commit_sha": null,
  "image_ref": "llenergymeasure:${ENGINE}-${NEW_VER}",
  "base_image_ref": null,
  "discovered_at": "2026-05-24T11:00:00+02:00",
  "discovery_limitations": [],
  "engine_params": {},
  "sampling_params": {
    "temperature": {"type": "number", "default": 1.0},
    "${FIELD}": {"type": "boolean", "default": true}
  }
}
JSON
  cat > "${NEW_DIR}/curated.yaml" <<YAML
schema_version: 1.0.0
engine: ${ENGINE}
engine_version: ${NEW_VER}
exposed_fields:
  engine_params: []
  sampling_params:
    - temperature
    - ${FIELD}
YAML
  cat > "${NEW_DIR}/invariants.proposed.yaml" <<YAML
schema_version: 1.0.0
engine: ${ENGINE}
engine_version: ${NEW_VER}
mined_at: '2026-05-24T11:00:00+02:00'
invariants: []
YAML
  cat > "${NEW_DIR}/invariants.validated.yaml" <<YAML
schema_version: 1.0.0
engine: ${ENGINE}
engine_version: ${NEW_VER}
image_ref: llenergymeasure:${ENGINE}-${NEW_VER}
base_image_ref: llenergymeasure:${ENGINE}-${NEW_VER}
validated_at: '2026-05-24T11:00:00+02:00'
validation_commit: null
cases: []
YAML
  ok "wrote 4 synthetic files"

  # 6. --check now reports drift
  say "regen_corpus --check: drift expected (exit 1)."
  set +e
  uv run python scripts/engine_producers/regen_engine_corpus.py --check --engine "${ENGINE}" >/dev/null 2>&1
  RC=$?
  set -e
  if [[ ${RC} -ne 1 ]]; then
    fail_step "expected exit 1, got ${RC}"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  ok "drift detected (exit 1)"

  # 7. --write mirrors + regenerates config.py
  say "regen --write: archive -> shadow, codegen config.py."
  uv run python scripts/engine_producers/regen_engine_corpus.py --write --engine "${ENGINE}" >/dev/null 2>&1 \
    || { fail_step "regen_corpus --write failed"; ENGINE_FAILED=true; }
  uv run python scripts/engine_producers/regen_engine_configs.py --write --engine "${ENGINE}" >/dev/null 2>&1 \
    || { fail_step "regen_configs --write failed"; ENGINE_FAILED=true; }
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  if ! diff -q "${NEW_DIR}/schema.discovered.json" "src/llenergymeasure/engines/${ENGINE}/schema.discovered.json" >/dev/null; then
    fail_step "shadow schema not byte-matching archive"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  if ! grep -q "${FIELD}" "src/llenergymeasure/engines/${ENGINE}/config.py"; then
    fail_step "config.py missing synthetic ${FIELD}"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  # round-trip the new field
  uv run python -c "
import sys
sys.path.insert(0, 'src')
for mod in list(sys.modules):
    if 'llenergymeasure.engines.${ENGINE}' in mod:
        del sys.modules[mod]
from llenergymeasure.engines.${ENGINE}.config import Config
cfg = Config(sampling_params={'temperature': 0.5, '${FIELD}': False})
assert cfg.sampling_params is not None
assert getattr(cfg.sampling_params, '${FIELD}') is False
print('round-trip OK')
" 2>&1 | grep -q "round-trip OK" \
    || { fail_step "round-trip failed for ${FIELD}"; ENGINE_FAILED=true; }
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  ok "shadow mirrors archive; config.py round-trips with ${FIELD}"

  # 8. steady state green
  say "Post-write steady state: --check both green."
  if ! uv run python scripts/engine_producers/regen_engine_corpus.py --check --engine "${ENGINE}" >/dev/null 2>&1; then
    fail_step "post-write regen_corpus --check red"
    ENGINE_FAILED=true
  fi
  if ! uv run python scripts/engine_producers/regen_engine_configs.py --check --engine "${ENGINE}" >/dev/null 2>&1; then
    fail_step "post-write regen_configs --check red"
    ENGINE_FAILED=true
  fi
  ${ENGINE_FAILED} && cleanup && trap - EXIT && FAILED+=("${ENGINE}") && continue
  ok "both --check green"

  # Cleanup for this engine.
  cleanup
  trap - EXIT
  printf "\n=== %s: ALL %d STEPS PASSED ===\n" "${ENGINE}" "${STEP}"
done

printf "\n========================================\n"
if [[ ${#FAILED[@]} -eq 0 ]]; then
  printf "Cascade test: all engines PASSED.\n"
  exit 0
else
  printf "Cascade test: FAILED for: %s\n" "${FAILED[*]}"
  exit 1
fi
