#!/usr/bin/env bash
# Run the probe kernel (scripts/probe_candidates.py) inside the engine's Docker
# image, so construction/identity probes execute against the real engine.
#
# Usage: ./scripts/probe_candidates.sh {transformers|vllm|tensorrt} [KERNEL ARGS...]
#   e.g. ./scripts/probe_candidates.sh vllm \
#            --candidates src/llenergymeasure/engines/vllm/rules.yaml \
#            --out /tmp/verdicts.yaml
#
# Everything after the engine is forwarded verbatim to the kernel, resolved
# against /repo (the mounted repo root, which is the container working dir).
#
# Image selection matches the CI engine cells: the pinned version is read from
# engine_versions/<engine>/current.yaml (the Renovate-writable SSOT), then
#   vllm         -> pristine vllm/vllm-openai:v<version>
#   tensorrt     -> pristine nvcr.io/nvidia/tensorrt-llm/release:<version>
#   transformers -> llenergymeasure:transformers-<version> (built locally if absent)
# tensorrt binds CUDA at import, so it keeps the NVIDIA entrypoint and needs a
# GPU; vllm/transformers need no GPU to construct config objects. The tensorrt
# GPU selection defaults to devices 1-3 (device 0 stays free on the reference
# host) and is overridable via LLEM_PROBE_GPUS for hosts with a different layout
# (e.g. LLEM_PROBE_GPUS='device=0' on a single-GPU host).
set -euo pipefail

# Docker --gpus device selector for the tensorrt probe; override for other hosts.
LLEM_PROBE_GPUS="${LLEM_PROBE_GPUS:-device=1,2,3}"

usage() {
    cat <<'EOF'
Usage: ./scripts/probe_candidates.sh {transformers|vllm|tensorrt} [KERNEL ARGS...]

Pulls or builds the engine image, then runs scripts/probe_candidates.py inside
it. Kernel args (--candidates, --out, --date) are forwarded verbatim.
EOF
}

if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage >&2
    exit 1
fi

ENGINE="$1"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pinned engine version from the per-engine SSOT (same locus the CI cells read).
_pinned_version() {
    yq '.library.current_version' "$REPO_ROOT/engine_versions/$1/current.yaml"
}

# docker pull only when the image is not already cached: gate the remote call
# on a local inspect (project convention for docker pull guards).
_ensure_pulled() {
    if ! docker image inspect "$1" >/dev/null 2>&1; then
        echo "[$ENGINE] Pulling $1..." >&2
        docker pull "$1"
    fi
}

VER="$(_pinned_version "$ENGINE")"
if [[ -z "$VER" || "$VER" == "null" ]]; then
    echo "Could not resolve $ENGINE version from engine_versions/$ENGINE/current.yaml" >&2
    exit 1
fi

case "$ENGINE" in
    transformers)
        IMAGE="llenergymeasure:transformers-${VER}"
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            echo "[$ENGINE] Image $IMAGE not found; building from docker/Dockerfile.transformers..." >&2
            docker build --build-arg "TRANSFORMERS_VERSION=${VER}" \
                -f "$REPO_ROOT/docker/Dockerfile.transformers" -t "$IMAGE" "$REPO_ROOT"
        fi
        ;;
    vllm)
        IMAGE="vllm/vllm-openai:v${VER}"
        _ensure_pulled "$IMAGE"
        ;;
    tensorrt)
        IMAGE="nvcr.io/nvidia/tensorrt-llm/release:${VER}"
        _ensure_pulled "$IMAGE"
        ;;
    *)
        echo "Unknown engine: $ENGINE" >&2
        usage >&2
        exit 1
        ;;
esac

# --user maps the host uid, which has no /etc/passwd entry in the image; torch
# calls getpass.getuser() at import and raises "uid not found" without USER set.
# HOME=/tmp keeps any torch/HF cache writes on a writable ephemeral path.
DOCKER_ARGS=(
    --rm --user "$(id -u):$(id -g)"
    -e HOME=/tmp -e USER=llem-probe
    -v "$REPO_ROOT:/repo" -w /repo
)

echo "[$ENGINE] Running probe_candidates.py inside $IMAGE..." >&2
if [[ "$ENGINE" == "tensorrt" ]]; then
    # Keep the NVIDIA entrypoint (it sets up the CUDA env before exec); GPU
    # selection from LLEM_PROBE_GPUS (default device=1,2,3, device 0 free).
    docker run "${DOCKER_ARGS[@]}" --gpus "\"${LLEM_PROBE_GPUS}\"" "$IMAGE" \
        python3 scripts/probe_candidates.py --engine "$ENGINE" "$@"
else
    docker run "${DOCKER_ARGS[@]}" --entrypoint python3 "$IMAGE" \
        scripts/probe_candidates.py --engine "$ENGINE" "$@"
fi
