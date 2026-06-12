# shellcheck shell=bash
# Populate the docker-run scaffold arrays for an engine cell step.
#
# SOURCE this (do not execute) from a workflow `run:` block - it sets three
# bash arrays in the caller's shell so the caller can assemble its own
# `docker run`:
#
#   DOCKER_FLAGS   - runtime flags (self-hosted GPU mounts + tensorrt source mount)
#   ENTRYPOINT_FLAG - the `--entrypoint <x>` pair
#   CMD_PREFIX     - the leading command token(s) inside the container
#
# Usage:
#   ENGINE=transformers VER=5.7.0 RUNNER=self-hosted \
#     source scripts/ci/run_in_engine_container.sh python3
#   docker run --rm "${DOCKER_FLAGS[@]}" ... "${ENTRYPOINT_FLAG[@]}" "$IMAGE" \
#     "${CMD_PREFIX[@]}" -c "..."
#
# The entrypoint mode (first arg) is `python3` (probe step) or `bash` (mine +
# re-gate steps). For tensorrt the NVIDIA entrypoint wrapper is substituted and
# the mode becomes the CMD_PREFIX token, matching the image's launcher contract.
#
# Reads from env: ENGINE, VER, RUNNER. Requires `set -u`-safe callers.

_entry_mode="${1:?entrypoint mode required (python3|bash)}"

DOCKER_FLAGS=()
if [[ "${RUNNER:-}" == "self-hosted" ]]; then
  DOCKER_FLAGS+=(
    --gpus all
    --user "$(id -u):$(id -g)"
    -v /tmp/llem-passwd-synth:/etc/passwd:ro
    -v /tmp/llem-group-synth:/etc/group:ro
    -e LD_LIBRARY_PATH
  )
fi

ENTRYPOINT_FLAG=(--entrypoint "${_entry_mode}")
CMD_PREFIX=()
if [[ "${ENGINE:-}" == "tensorrt" ]]; then
  DOCKER_FLAGS+=( -v "/tmp/trt-llm-${VER}:/tmp/trt-llm-${VER}:ro" )
  ENTRYPOINT_FLAG=(--entrypoint /opt/nvidia/nvidia_entrypoint.sh)
  CMD_PREFIX=("${_entry_mode}")
fi
