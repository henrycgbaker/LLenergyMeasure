#!/usr/bin/env bash
# Verify pyproject.toml force-include paths agree with each engine's
# library.current_version in engine_versions/<engine>/current.yaml.
#
# When Renovate bumps current.yaml, the matching pyproject.toml line under
# [tool.hatch.build.targets.wheel.force-include] must move in lockstep so the
# wheel bundles the per-version outputs/ directory the runtime loaders
# expect. Drift here means the wheel ships data mined for the wrong engine
# version - silent at install time, broken at experiment time.
#
# Run: scripts/ci/check_engine_bundle_pin.sh
# Exit 0: all engines agree. Exit 1: at least one engine drifted.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYPROJECT="${REPO_ROOT}/pyproject.toml"

fail=0
for engine in vllm tensorrt transformers; do
    ssot="${REPO_ROOT}/engine_versions/${engine}/current.yaml"
    if [[ ! -f "${ssot}" ]]; then
        echo "::error::SSOT missing: ${ssot}" >&2
        fail=1
        continue
    fi
    version=$(yq '.library.current_version' "${ssot}")
    if [[ -z "${version}" || "${version}" == "null" ]]; then
        echo "::error::No library.current_version in ${ssot}" >&2
        fail=1
        continue
    fi
    safe="v${version//[.-]/_}"
    expected_path="engine_versions/${engine}/${safe}/outputs/"
    if ! grep -q "${expected_path}" "${PYPROJECT}"; then
        echo "::error::pyproject.toml force-include for ${engine} does not reference ${safe} (current.yaml says ${version}). Update [tool.hatch.build.targets.wheel.force-include] entries to match, or revert current.yaml. Renovate's customManagers regex should bump both in lockstep." >&2
        fail=1
    fi
done

if [[ "${fail}" -eq 0 ]]; then
    echo "engine bundle pin OK: pyproject.toml force-include agrees with current.yaml for all engines." >&2
fi
exit "${fail}"
