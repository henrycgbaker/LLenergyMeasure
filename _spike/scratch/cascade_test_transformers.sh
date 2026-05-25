#!/usr/bin/env bash
# Transformers renovate-bump cascade test.
#
# Exercises the full chain that fires when Renovate bumps
# engine_versions/transformers/current.toml to a new upstream version:
#
#   1. current.toml bumped to a not-yet-vendored version (4.99.99).
#   2. regen_engine_corpus.py --check tolerates the gap (F#5 -> stderr skip,
#      exit 0). Renovate-pre-vendor window is legitimate.
#   3. regen_engine_configs.py --check tolerates the gap too (same skip).
#   4. Dispatcher falls back to the highest vendored archive at or below
#      4.99.99 (here: v4_57_3). The fallback is observable via stderr.
#   5. Simulate cells writeback: create engine_versions/transformers/v4_99_99
#      /outputs/ with synthetic schema + curated.yaml + invariants files.
#   6. regen_engine_corpus.py --check now reports drift (archive has fresh
#      content; shadow is stale). --write mirrors archive -> shadow.
#   7. regen_engine_configs.py --check reports drift (synthetic schema
#      produces a different config.py). --write regenerates config.py.
#   8. Re-imports of the loader + the generated config.py round-trip cleanly.
#   9. Cleanup: revert current.toml; remove the synthetic v4_99_99 tree;
#      regen --write restores the live state.
#
# Exit 0 = cascade verified end-to-end. Exit 1 = a step failed; the line
# preceding the FAIL is the diagnostic.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CURRENT_TOML="engine_versions/transformers/current.toml"
SYNTHETIC_VER="4.99.99"
SYNTHETIC_SAFE="v4_99_99"
SYNTHETIC_DIR="engine_versions/transformers/${SYNTHETIC_SAFE}/outputs"
CURRENT_TOML_BACKUP=""

# ASCII-safe step counter + outcome reporter.
STEP=0
say() {
  STEP=$((STEP + 1))
  printf "\n[STEP %d] %s\n" "$STEP" "$*"
}
fail() {
  printf "FAIL: %s\n" "$*" >&2
  exit 1
}
pass() {
  printf "  ok: %s\n" "$*"
}

cleanup() {
  printf "\n=== CLEANUP ===\n"
  if [[ -n "${CURRENT_TOML_BACKUP}" && -f "${CURRENT_TOML_BACKUP}" ]]; then
    cp "${CURRENT_TOML_BACKUP}" "${CURRENT_TOML}"
    rm -f "${CURRENT_TOML_BACKUP}"
    printf "  restored: %s\n" "${CURRENT_TOML}"
  fi
  if [[ -d "engine_versions/transformers/${SYNTHETIC_SAFE}" ]]; then
    rm -rf "engine_versions/transformers/${SYNTHETIC_SAFE}"
    printf "  removed:  engine_versions/transformers/%s\n" "${SYNTHETIC_SAFE}"
  fi
  # Re-mirror to restore src/ shadow to live-archive parity.
  uv run python scripts/engine_producers/regen_engine_corpus.py --write \
    --engine transformers >/dev/null 2>&1 || true
  uv run python scripts/engine_producers/regen_engine_configs.py --write \
    --engine transformers >/dev/null 2>&1 || true
  printf "  re-synced shadow + config.py from live archive\n"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 0: baseline assertion
# ---------------------------------------------------------------------------

say "Baseline: regen scripts green at current pin (4.57.3)."
uv run python scripts/engine_producers/regen_engine_corpus.py --check \
  --engine transformers >/dev/null 2>&1 || fail "baseline regen_corpus --check unexpectedly red"
uv run python scripts/engine_producers/regen_engine_configs.py --check \
  --engine transformers >/dev/null 2>&1 || fail "baseline regen_configs --check unexpectedly red"
pass "both --check green"

# ---------------------------------------------------------------------------
# Step 1: synthetic Renovate bump
# ---------------------------------------------------------------------------

say "Synthetic Renovate bump: current.toml -> ${SYNTHETIC_VER}"
CURRENT_TOML_BACKUP="$(mktemp --suffix=.toml.bak)"
cp "${CURRENT_TOML}" "${CURRENT_TOML_BACKUP}"
sed -i "s/^current_version = .*$/current_version = \"${SYNTHETIC_VER}\"/" "${CURRENT_TOML}"
NEW_VER=$(grep '^current_version' "${CURRENT_TOML}")
[[ "${NEW_VER}" == "current_version = \"${SYNTHETIC_VER}\"" ]] \
  || fail "sed did not bump current_version (got: ${NEW_VER})"
pass "current.toml now pins ${SYNTHETIC_VER}"

# ---------------------------------------------------------------------------
# Step 2: regen --check is tolerant of missing outputs/ (F#5)
# ---------------------------------------------------------------------------

say "F#5 tolerance: regen --check should SKIP (not raise) when outputs/ is absent."
CORPUS_OUT="$(uv run python scripts/engine_producers/regen_engine_corpus.py \
  --check --engine transformers 2>&1)"
CORPUS_RC=$?
echo "${CORPUS_OUT}" | grep -q "outputs directory not present" \
  || fail "regen_corpus did not log the expected skip message"
[[ ${CORPUS_RC} -eq 0 ]] || fail "regen_corpus --check exited non-zero (${CORPUS_RC})"
pass "regen_corpus skipped + exited 0"

CONFIGS_OUT="$(uv run python scripts/engine_producers/regen_engine_configs.py \
  --check --engine transformers 2>&1)"
CONFIGS_RC=$?
echo "${CONFIGS_OUT}" | grep -q "outputs directory not present" \
  || fail "regen_configs did not log the expected skip message"
[[ ${CONFIGS_RC} -eq 0 ]] || fail "regen_configs --check exited non-zero (${CONFIGS_RC})"
pass "regen_configs skipped + exited 0"

# ---------------------------------------------------------------------------
# Step 3: dispatcher fallback resolves
# ---------------------------------------------------------------------------

say "Dispatcher fallback: load_producer should resolve to v4_57_3 (no exact match for ${SYNTHETIC_VER})."
DISPATCH_OUT="$(uv run python -c "
import sys
sys.path.insert(0, '.')
from engine_versions._dispatcher import load_producer
mod = load_producer(engine='transformers', version='${SYNTHETIC_VER}', producer='schema_introspector')
print('LOADED:', mod.__name__)
" 2>&1)"
echo "${DISPATCH_OUT}" | grep -q "LOADED: engine_versions.transformers.v4_57_3.producers.schema_introspector" \
  || fail "fallback did not resolve to v4_57_3 (got: ${DISPATCH_OUT})"
echo "${DISPATCH_OUT}" | grep -q "falling back to" \
  || fail "fallback was not logged to stderr"
pass "dispatcher fell back to v4_57_3 + emitted the expected stderr log line"

# ---------------------------------------------------------------------------
# Step 4: simulate cells writeback (synthetic v4_99_99 outputs)
# ---------------------------------------------------------------------------

say "Simulate cells writeback: create synthetic v4_99_99/outputs/."
mkdir -p "${SYNTHETIC_DIR}"

# Synthetic schema.discovered.json: minimal canonical envelope + one
# field that doesn't exist in v4_57_3's schema (so config.py changes).
cat > "${SYNTHETIC_DIR}/schema.discovered.json" <<'JSON'
{
  "schema_version": "2.0.0",
  "engine": "transformers",
  "engine_version": "4.99.99",
  "engine_commit_sha": null,
  "image_ref": "llenergymeasure:transformers-4.99.99",
  "base_image_ref": null,
  "discovered_at": "2026-05-24T11:00:00+02:00",
  "discovery_limitations": [],
  "engine_params": {},
  "sampling_params": {
    "temperature": {"type": "number", "default": 1.0},
    "top_k": {"type": "integer", "default": 50},
    "new_in_4_99": {"type": "boolean", "default": true}
  }
}
JSON

cat > "${SYNTHETIC_DIR}/curated.yaml" <<'YAML'
schema_version: 1.0.0
engine: transformers
engine_version: 4.99.99
exposed_fields:
  engine_params: []
  sampling_params:
    - temperature
    - top_k
    - new_in_4_99
YAML

# Minimal invariants (envelope shape only).
cat > "${SYNTHETIC_DIR}/invariants.proposed.yaml" <<'YAML'
schema_version: 1.0.0
engine: transformers
engine_version: 4.99.99
mined_at: '2026-05-24T11:00:00+02:00'
invariants: []
YAML

cat > "${SYNTHETIC_DIR}/invariants.validated.yaml" <<'YAML'
schema_version: 1.0.0
engine: transformers
engine_version: 4.99.99
image_ref: llenergymeasure:transformers-4.99.99
base_image_ref: llenergymeasure:transformers-4.99.99
validated_at: '2026-05-24T11:00:00+02:00'
validation_commit: null
cases: []
YAML

pass "wrote 4 synthetic files in ${SYNTHETIC_DIR}"

# ---------------------------------------------------------------------------
# Step 5: regen --check now reports drift (archive has fresh content)
# ---------------------------------------------------------------------------

say "regen --check after synthetic mining: should report drift (exit 1)."
set +e
uv run python scripts/engine_producers/regen_engine_corpus.py --check \
  --engine transformers >/dev/null 2>&1
CORPUS_RC_DRIFT=$?
set -e
[[ ${CORPUS_RC_DRIFT} -eq 1 ]] \
  || fail "regen_corpus --check should have exited 1 with drift (got ${CORPUS_RC_DRIFT})"
pass "regen_corpus --check exited 1 (drift detected)"

# ---------------------------------------------------------------------------
# Step 6: regen --write mirrors archive -> shadow
# ---------------------------------------------------------------------------

say "regen --write: archive -> shadow + regen config.py."
uv run python scripts/engine_producers/regen_engine_corpus.py --write \
  --engine transformers >/dev/null 2>&1 \
  || fail "regen_corpus --write failed"
uv run python scripts/engine_producers/regen_engine_configs.py --write \
  --engine transformers >/dev/null 2>&1 \
  || fail "regen_configs --write failed"
pass "both --write succeeded"

# Verify shadow reflects synthetic content.
diff -q "${SYNTHETIC_DIR}/schema.discovered.json" \
  src/llenergymeasure/engines/transformers/schema.discovered.json >/dev/null \
  || fail "shadow schema does not match synthetic archive"
pass "src/.../schema.discovered.json matches archive byte-for-byte"

# Verify config.py contains the new field.
grep -q "new_in_4_99" src/llenergymeasure/engines/transformers/config.py \
  || fail "generated config.py is missing the synthetic new_in_4_99 field"
pass "config.py contains synthetic new_in_4_99 field"

# Verify config.py imports + round-trips.
uv run python -c "
import sys, importlib
sys.path.insert(0, 'src')
# Force re-import in case prior import cached the old shape.
for mod in list(sys.modules):
    if 'llenergymeasure.engines.transformers' in mod:
        del sys.modules[mod]
from llenergymeasure.engines.transformers.config import Config
cfg = Config(sampling_params={'temperature': 0.5, 'new_in_4_99': False})
assert cfg.sampling_params is not None
assert cfg.sampling_params.temperature == 0.5
assert cfg.sampling_params.new_in_4_99 is False
print('round-trip OK')
" 2>&1 | grep -q "round-trip OK" \
  || fail "regenerated config.py did not round-trip with new_in_4_99 field"
pass "config.py round-trips with the synthetic field"

# ---------------------------------------------------------------------------
# Step 7: --check now green again (steady state at the bumped pin)
# ---------------------------------------------------------------------------

say "Post-write steady state: both --check should be green."
uv run python scripts/engine_producers/regen_engine_corpus.py --check \
  --engine transformers >/dev/null 2>&1 \
  || fail "post-write regen_corpus --check unexpectedly red"
uv run python scripts/engine_producers/regen_engine_configs.py --check \
  --engine transformers >/dev/null 2>&1 \
  || fail "post-write regen_configs --check unexpectedly red"
pass "both --check green"

# ---------------------------------------------------------------------------
# Done. Cleanup runs via trap.
# ---------------------------------------------------------------------------

printf "\n=== ALL %d STEPS PASSED ===\n" "${STEP}"
