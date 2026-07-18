#!/bin/bash
# Entrypoint for llem container dispatch (vLLM, TensorRT-LLM, transformers).
#
# Runs INSIDE the container at dispatch time. Bind-mounted at /llem-entry.sh
# from a host tempdir by docker_runner.py, which materialises this script (and
# the requirements list below) from the installed llenergymeasure package data
# so dispatch works from both a source checkout and a pip install.
#
# Responsibilities:
#   1. Compare the runtime dependency list at /llem-requirements.txt against
#      what's importable in the container; pip-install only the missing
#      ones to a host-mounted persistent cache (/llem-runtime-deps/pyN.M),
#      keyed by container Python minor.
#   2. Set PYTHONPATH so /llem-src (bind-mounted framework source) and the
#      runtime-deps cache are discoverable.
#   3. Exec the framework's in-container entrypoint module. For TensorRT-LLM,
#      route through the upstream nvidia_entrypoint.sh which sets up
#      LD_LIBRARY_PATH for libnvinfer (closes #608). Always a single python3
#      process - multi-GPU tensorrt is NOT wrapped in mpirun (the TRT-LLM LLM
#      API self-manages tensor parallelism by spawning its own workers).
#
# The requirements-vs-container diff is computed at every dispatch. Cached
# installs in /llem-runtime-deps shadow the comparison naturally: once a
# missing dep is primed, it's importable via the cache, so the probe stops
# flagging it. The requirements list is derived on the host from the installed
# llenergymeasure distribution metadata, so a re-install (or an edited-then-
# reinstalled checkout) is the invalidation event: its content - and therefore
# its hash below - changes, and the probe re-runs.

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
# times with an unchanged requirements list. Running the import-verification
# probe every dispatch is pure overhead that the steady-state path doesn't
# need. The stamp file records the requirements hash that was last verified
# against this cache; matching hash means "deps probe was done, nothing
# changed, skip it." Mismatch (or absent stamp) triggers a full probe and
# updates the stamp. The stamp is written only after a fully-verified prime,
# so a hit is a sound skip.
#
# The stamp is keyed by ENGINE as well as python minor: different engines run
# different images whose site-packages differ, so "verified, nothing missing"
# is an image-level fact, not a python-minor-level one. With a shared stamp,
# whichever engine dispatches first would suppress the probe for the others
# (observed live 2026-07-14: the TRT-LLM NGC image, which ships every llem
# dep, stamped py3.12 verified; the vllm image, which lacks pyarrow, then
# skipped the probe and crashed at the parquet write).
STAMP="${DEPS_TARGET}/.llem_requirements_hash_${LLEM_ENGINE:-default}"
REQUIREMENTS_HASH=$(sha256sum /llem-requirements.txt | head -c 16)

if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$REQUIREMENTS_HASH" ]; then
    # Diff the runtime requirements list against what actually imports in
    # this interpreter. A dist-info presence check is not sufficient: metadata
    # can be present while the package fails to import - a compiled extension
    # built for the wrong ABI, or an otherwise broken install - so each present
    # requirement is verified by importing its resolved module(s), and primed
    # if the import raises. Absent metadata short-circuits as missing with no
    # import attempt. importlib.metadata sees both the image site-packages and
    # our cache mount, since we already added the cache to PYTHONPATH above.
    MISSING=$(python3 - <<'PYEOF'
import functools
import importlib
import importlib.metadata
import sys
from pathlib import Path

# The requirements list is bind-mounted here inside the container. An optional
# argv override (defaulting to the mount path) lets the probe run against a
# synthetic requirements file without changing container behaviour.
REQUIREMENTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "/llem-requirements.txt"

# Distributions whose top-level import name differs from the distribution name.
# Keys are canonical (lowercase, dashes) distribution names; values are the
# module to import. Authoritative: consulted before top_level.txt so noisy or
# absent top-level metadata cannot mis-resolve these known cases.
IMPORT_NAME_OVERRIDES = {
    "nvidia-ml-py": "pynvml",
    "pyyaml": "yaml",
    "python-dotenv": "dotenv",
}


def bare_name(spec: str) -> str:
    # Strip version bounds / extras to get the bare distribution name for the
    # metadata lookup. The full spec is kept for pip install.
    name = spec
    for sep in (">=", "<=", "==", "!=", "~=", "<", ">", "["):
        if sep in name:
            name = name.split(sep)[0]
    return name.strip()


def canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-")


@functools.cache
def reverse_packages() -> dict[str, list[str]]:
    # Invert importlib.metadata.packages_distributions() (import name -> dist
    # names) into dist name -> import names, so a distribution's top-level
    # module(s) can be recovered even when its top_level.txt is absent.
    mapping: dict[str, list[str]] = {}
    try:
        items = importlib.metadata.packages_distributions().items()
    except Exception:
        return mapping
    for import_name, dists in items:
        for dist_name in dists:
            mapping.setdefault(canonical(dist_name), []).append(import_name)
    return mapping


def import_names(dist_name, dist) -> list[str]:
    # Resolve the top-level import name(s) for an installed distribution.
    key = canonical(dist_name)
    if key in IMPORT_NAME_OVERRIDES:
        return [IMPORT_NAME_OVERRIDES[key]]
    try:
        top_level = dist.read_text("top_level.txt")
    except Exception:
        top_level = None
    if top_level:
        names = [ln.strip() for ln in top_level.splitlines() if ln.strip()]
        if names:
            return names
    resolved = reverse_packages().get(key)
    if resolved:
        return resolved
    # Fall back to the normalised distribution name (dashes to underscores).
    return [dist_name.replace("-", "_")]


missing = []
for line in Path(REQUIREMENTS_FILE).read_text().splitlines():
    spec = line.strip()
    if not spec or spec.startswith("#"):
        continue
    name = bare_name(spec)
    if not name:
        continue
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        # Absent metadata: definitely missing, no import attempt needed.
        missing.append(spec)
        continue
    try:
        for import_name in import_names(name, dist):
            importlib.import_module(import_name)
    except Exception:
        # Present metadata but the module does not import (wrong-ABI extension,
        # broken install): prime it so the container's pip reinstalls an
        # ABI-correct copy into the deps cache.
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

    # Stamp even when MISSING was empty - records that this requirements hash
    # was verified against the current cache so subsequent dispatches can
    # short-circuit the probe.
    echo "$REQUIREMENTS_HASH" > "$STAMP"
    if [ -n "${LLEM_HOST_UID:-}" ] && [ -n "${LLEM_HOST_GID:-}" ]; then
        chown "${LLEM_HOST_UID}:${LLEM_HOST_GID}" "$STAMP" 2>/dev/null || true
    fi
fi

# Build the final launch command. Always a single python3 process, for every
# engine including multi-GPU tensorrt. TensorRT-LLM's LLM API (both the trt and
# pytorch backends at 1.2.x) self-manages tensor parallelism: setting
# tensor_parallel_size makes the LLM class spawn its own worker processes
# internally (MPI/RPC orchestrator). Wrapping the container in mpirun -n N
# instead ran the WHOLE entrypoint on every rank, so each rank redundantly
# built the engine and constructed a full LLM - corrupting the on-disk build
# cache (a rank race in the upstream write_guard) and OOMing on the executor.
LAUNCH=(python3)

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
