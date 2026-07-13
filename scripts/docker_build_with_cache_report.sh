#!/usr/bin/env bash
# Wraps `docker compose build <engine>` with BuildKit progress=plain so the
# cache-import status is visible in real time, then emits a single-line ✓/⚠
# summary post-build that says whether the GHCR cache was actually used.
#
# Silent fallback to a cold build is BuildKit's documented behaviour when
# cache_from cannot resolve. This wrapper exists so contributors notice when
# a "warm" rebuild is actually cold instead of tolerating 20-minute builds.
#
# Usage: scripts/docker_build_with_cache_report.sh <engine>
#        e.g. scripts/docker_build_with_cache_report.sh transformers

set -euo pipefail

engine="${1:?usage: $0 <engine>}"
log="/tmp/llem-build-${engine}.log"

# progress=plain exposes per-step output including "importing cache manifest"
# and per-stage "CACHED" markers; auto/tty hides them behind a collapsed UI.
export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"

# Resolve the transformers engine pin (engine_versions/transformers/current.yaml)
# so docker-compose.yml's cache_from can target the seed's per-version mode=max
# buildcache ref - the only ref that carries the FA3 builder-stage layers - and
# so the build installs the pinned transformers version. A caller-provided
# TRANSFORMERS_VERSION wins; on resolution failure the var stays empty and
# compose's fallbacks kick in (BuildKit skips the unresolvable cache source
# silently - a cold build, not an error).
if [[ -z "${TRANSFORMERS_VERSION:-}" ]]; then
    TRANSFORMERS_VERSION=$(python3 -c "import yaml; print(yaml.safe_load(open('engine_versions/transformers/current.yaml'))['library']['current_version'])" 2>/dev/null || true)
fi
export TRANSFORMERS_VERSION

start=$(date +%s)
# tee preserves the on-screen build log; PIPESTATUS captures compose's exit.
docker compose build "${engine}" 2>&1 | tee "${log}"
rc=${PIPESTATUS[0]}
elapsed=$(( $(date +%s) - start ))

mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))
human=$(printf "%dm %02ds" "${mins}" "${secs}")

# grep -c exits 1 on zero matches; || echo 0 keeps the value numeric for the
# arithmetic tests below (|| true would leave an empty string under set -e).
imported=$(grep -c "importing cache manifest from ghcr.io" "${log}" 2>/dev/null || echo 0)
cached_steps=$(grep -cE "^#[0-9]+ CACHED$" "${log}" 2>/dev/null || echo 0)

echo
if [[ "${rc}" -ne 0 ]]; then
    echo "✗ ${engine} build FAILED after ${human} (exit ${rc}) - see ${log}"
elif [[ "${imported}" -gt 0 && "${cached_steps}" -gt 0 ]]; then
    echo "✓ ${engine} build: ${human} - GHCR cache imported, ${cached_steps} layers reused"
elif [[ "${imported}" -gt 0 ]]; then
    echo "✓ ${engine} build: ${human} - GHCR cache reachable but no layers reused (source changed)"
else
    echo "⚠ ${engine} build: ${human} - no GHCR cache imported (cold build)"
    echo "  see docs/troubleshooting.md → 'Docker rebuild is slow' for diagnosis"
fi

exit "${rc}"
