#!/bin/bash
# Entrypoint for llem container dispatch (vLLM, TensorRT-LLM, transformers).
#
# Runs INSIDE the container at dispatch time. Bind-mounted at /llem-entry.sh
# from the host repo by docker_runner.py.
#
# Responsibilities:
#   1. Compare /llem-src/pyproject.toml's [project.dependencies] against
#      what's importable in the container; pip-install only the missing
#      ones to a host-mounted persistent cache (/llem-runtime-deps/pyN.M),
#      keyed by container Python minor.
#   2. Set PYTHONPATH so /llem-src (bind-mounted framework source) and the
#      runtime-deps cache are discoverable.
#   3. Exec the framework's in-container entrypoint module. For TensorRT-LLM,
#      route through the upstream nvidia_entrypoint.sh which sets up
#      LD_LIBRARY_PATH for libnvinfer (closes #608). For TP > 1, wrap in
#      mpirun (LLEM_MPI_NP set by docker_runner).
#
# The pyproject-vs-container diff is computed at every dispatch. Cached
# installs in /llem-runtime-deps shadow the comparison naturally: once a
# missing dep is primed, it's importable via the cache, so the probe
# stops flagging it. No stamp file or explicit invalidation needed -
# pyproject is the single source of truth.

set -euo pipefail

DEPS_CACHE_ROOT="/llem-runtime-deps"
PY_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
DEPS_TARGET="${DEPS_CACHE_ROOT}/py${PY_MINOR}"
mkdir -p "$DEPS_TARGET"

# PYTHONPATH must be set BEFORE the dep probe so cached installs are visible
# to importlib.metadata. /llem-src is bind-mounted framework source; the
# cache dir hosts primed deps.
export PYTHONPATH="/llem-src:${DEPS_TARGET}:${PYTHONPATH:-}"

# Fast-path stamp: a sweep typically dispatches the same image hundreds of
# times with an unchanged pyproject.toml. Running the importlib.metadata
# probe every dispatch is ~200ms of pure overhead that the steady-state
# path doesn't need. The stamp file records the pyproject hash that was
# last verified against this cache; matching hash means "deps probe was
# done, nothing changed, skip it." Mismatch (or absent stamp) triggers a
# full probe and updates the stamp.
STAMP="${DEPS_TARGET}/.llem_pyproject_hash"
PYPROJECT_HASH=$(sha256sum /llem-pyproject.toml | head -c 16)

if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$PYPROJECT_HASH" ]; then
    # Diff pyproject's [project.dependencies] against installed dists
    # (importlib.metadata sees both site-packages and our cache mount,
    # since we already added the cache to PYTHONPATH above).
    MISSING=$(python3 - <<'PYEOF'
import importlib.metadata
import sys

try:
    import tomllib
except ImportError:
    # Python < 3.11 - fail loudly; current upstream images (vllm 0.7.3,
    # TRT-LLM 0.21, transformers first-party) all ship Python >= 3.11.
    print("FATAL: tomllib unavailable; container Python < 3.11", file=sys.stderr)
    sys.exit(2)

with open("/llem-pyproject.toml", "rb") as f:
    data = tomllib.load(f)

# Strip version bounds to get the bare distribution name for the
# installation probe. Keep the full spec for pip install.
def bare_name(spec: str) -> str:
    name = spec
    for sep in (">=", "<=", "==", "!=", "~=", "<", ">", "["):
        if sep in name:
            name = name.split(sep)[0]
    return name.strip()

missing = []
for spec in data.get("project", {}).get("dependencies", []):
    name = bare_name(spec)
    if not name:
        continue
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        missing.append(spec)

print(" ".join(missing))
PYEOF
    )

    if [ -n "${MISSING}" ]; then
        echo "[llem-entry] Priming missing runtime deps for py${PY_MINOR}: ${MISSING}" >&2
        # shellcheck disable=SC2086
        pip install --no-deps --no-cache-dir --only-binary=:all: \
            --target "$DEPS_TARGET" ${MISSING}
        # Container runs as root (default for upstream engine images) but
        # the deps cache is host-bind-mounted; without chown the freshly-
        # primed files would be root-owned on the host and require sudo
        # to clean up. docker_runner.py passes LLEM_HOST_UID/LLEM_HOST_GID
        # from the dispatch.
        if [ -n "${LLEM_HOST_UID:-}" ] && [ -n "${LLEM_HOST_GID:-}" ]; then
            chown -R "${LLEM_HOST_UID}:${LLEM_HOST_GID}" "${DEPS_TARGET}" || true
        fi
    fi

    # Stamp even when MISSING was empty - records that this pyproject hash
    # was verified against the current cache so subsequent dispatches can
    # short-circuit the probe.
    echo "$PYPROJECT_HASH" > "$STAMP"
    if [ -n "${LLEM_HOST_UID:-}" ] && [ -n "${LLEM_HOST_GID:-}" ]; then
        chown "${LLEM_HOST_UID}:${LLEM_HOST_GID}" "$STAMP" 2>/dev/null || true
    fi
fi

# Build the final launch command. For TP > 1, mpirun wraps python3 (matches
# the prior docker_runner.py behaviour where mpirun was the entrypoint).
if [ -n "${LLEM_MPI_NP:-}" ]; then
    LAUNCH=(mpirun -n "${LLEM_MPI_NP}" --allow-run-as-root python3)
else
    LAUNCH=(python3)
fi

# Framework module to exec. Defaults to the experiment container entrypoint;
# the baseline dispatch overrides it with LLEM_ENTRY_MODULE so it reuses this
# same package-mount + dep-prime bootstrap instead of duplicating it.
ENTRY_MODULE="${LLEM_ENTRY_MODULE:-llenergymeasure.entrypoints.container}"

# Engine-conditional final exec. TensorRT-LLM upstream image needs
# nvidia_entrypoint.sh to set up LD_LIBRARY_PATH for libnvinfer.so
# (closes #608); other engines exec python3 directly.
if [ "${LLEM_ENGINE:-}" = "tensorrt" ]; then
    exec /opt/nvidia/nvidia_entrypoint.sh "${LAUNCH[@]}" -m "${ENTRY_MODULE}" "$@"
else
    exec "${LAUNCH[@]}" -m "${ENTRY_MODULE}" "$@"
fi
