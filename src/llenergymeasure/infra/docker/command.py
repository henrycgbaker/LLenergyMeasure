"""Pure ``docker run`` command builders for container dispatch.

Everything here is a pure builder: it assembles the argv (env forwarding,
mounts, GPU/shm flags, image ref, and the package-dispatch bootstrap) without
running docker, so it is unit-testable without a daemon. The one side effect is
materialising the dispatch assets (entrypoint script + requirements list) to a
host tempdir for bind-mounting, and ensuring the host deps cache exists - both
process-cached and idempotent.

This is the single home for ``docker run`` argv construction, so the container
shapes cannot drift apart unnoticed: :func:`build_docker_cmd` builds the
run-to-completion experiment dispatch and :func:`build_server_container_argv`
builds the long-lived engine-server launch. Consumed by
``DockerRunner._build_docker_cmd`` (the experiment dispatch), by
``study.baseline_container`` (the baseline dispatch) - which share
:func:`append_package_dispatch` and :func:`append_nccl_env` so the two setups
cannot drift - and by each engine's server adapter, which passes the argv built
here to the serving layer's launcher. :func:`docker_label_args` is shared by all
three, so their emission of the study's container ownership labels cannot drift
either.
"""

from __future__ import annotations

import atexit
import functools
import importlib.metadata
import importlib.resources
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final

import platformdirs

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    ENV_BASELINE_SPEC_PATH,
    ENV_CONFIG_PATH,
    ENV_DEPS_CACHE_DIR,
    ENV_ENGINE,
    ENV_ENTRY_MODULE,
    ENV_HOST_GID,
    ENV_HOST_UID,
    ENV_OUTPUT_DIR,
    ENV_SAVE_TIMESERIES,
    Engine,
)
from llenergymeasure.utils.env_config import (
    ENV_TRT_BUILD_CACHE_PATH,
    HF_CACHE_CONTAINER_PATH,
    docker_gpus_arg,
    docker_hf_cache_dir,
    docker_shm_size,
    hf_cache_mount_args,
    trt_build_cache_host_dir,
)
from llenergymeasure.utils.exceptions import DockerPreFlightError

__all__ = [
    "append_nccl_env",
    "append_package_dispatch",
    "build_docker_cmd",
    "build_server_container_argv",
    "docker_label_args",
]

# Reserved exchange env keys the docker command builders set deliberately
# (via -e or --env-file). The blanket "forward every host LLEM_* var" loop
# must skip these: docker applies -e last-wins over --env-file, so a stray
# host LLEM_CONFIG_PATH / LLEM_OUTPUT_DIR / ... would silently clobber the
# intended dispatch value. Built from the actual ENV_* constants so it can't
# drift from what the builders set.
_RESERVED_EXCHANGE_ENV: frozenset[str] = frozenset(
    {
        ENV_CONFIG_PATH,
        ENV_OUTPUT_DIR,
        ENV_SAVE_TIMESERIES,
        ENV_ENGINE,
        ENV_ENTRY_MODULE,
        ENV_HOST_UID,
        ENV_HOST_GID,
        ENV_BASELINE_SPEC_PATH,
    }
)

# Container-side mount target for the TRT-LLM engine build cache. The host
# ~/.cache/trt-llm is bind-mounted here; TRT-LLM's own default cache root
# (/tmp/.cache/tensorrt_llm/llmapi/) is NOT this path and lives on the
# ephemeral container filesystem, so without pinning the cache to the mount
# it would silently die with each container. We default
# LLEM_TRT_BUILD_CACHE_PATH to this mount so the cache works out of the box;
# a host-set value still wins (forwarded later, docker -e is last-wins).
_TRT_BUILD_CACHE_CONTAINER_PATH: Final = "/root/.cache/trt-llm"

# Container-side mount target for the HuggingFace cache. Single-sourced from
# env_config (HF_CACHE_CONTAINER_PATH) so the offline dispatch and the
# server-container launch, both built in this module, cannot drift; the host
# source is configurable via LLEM_DOCKER_HF_CACHE (see docker_hf_cache_dir),
# the in-container target and HF_HOME are fixed.
_HF_CACHE_CONTAINER_PATH: Final = HF_CACHE_CONTAINER_PATH

# The in-container entrypoint script is shipped as package data (rather than
# resolved from a repo-root scripts/ dir) so container dispatch works from an
# installed wheel, not only a source checkout. It is read via
# importlib.resources and materialised to a host tempdir before bind-mounting
# (see _materialise_dispatch_assets); docker bind-mounts need a real host path.
_ENTRY_SCRIPT_PACKAGE: Final = "llenergymeasure.infra"
_ENTRY_SCRIPT_RESOURCE: Final = "_container/container_entrypoint.sh"
# Distribution name whose metadata declares the runtime deps the container
# primes. Read from importlib.metadata so it resolves identically from a
# checkout and a site-packages install (the old path read the repo-root
# pyproject.toml, which does not exist for a pip-installed user).
_DISPATCH_DIST_NAME: Final = "llenergymeasure"
# Temp-dir prefix for the materialised dispatch assets (entrypoint script +
# requirements list). One dir per process, cleaned up at interpreter exit.
_TEMP_PREFIX_DISPATCH: Final = "llem-dispatch-"


@functools.cache
def _resolve_package_dir() -> Path:
    """Return the ``llenergymeasure`` package directory itself.

    Used to bind-mount the host package source into upstream engine images
    that don't ship llenergymeasure (vllm, tensorrt) and into our own
    transformers image. Resolved from ``__file__`` rather than via
    ``import llenergymeasure`` to keep the infra layer free of upper-layer
    imports (import-linter contract).

    Layout::

        <site-packages-or-src>/
            llenergymeasure/       <-- returned (the package dir)
                infra/
                    docker/
                        command.py   <-- __file__

    Three ``.parent`` hops walk command.py -> docker/ -> infra/ -> llenergymeasure/.
    Encapsulation here localises path knowledge so a future relayout only needs
    to touch this helper. Cached because ``__file__`` is fixed for the life of
    the process.

    INVARIANT - why the package dir and not its parent: the mount this feeds
    (``/llem-src/llenergymeasure``) must expose the ``llenergymeasure`` package
    and NOTHING else - never the package's on-disk siblings. For a pip/wheel
    install the parent IS the venv's ``site-packages``. Mounting the parent (the
    historical ``parent.parent.parent`` walk) put every host third-party package
    at ``/llem-src``, and ``container_entrypoint.sh`` prepends ``/llem-src`` to
    ``PYTHONPATH``, so ``PYTHONPATH`` entries always precede the container's own
    site-packages on ``sys.path`` - every host copy then shadowed the image's
    native one. Observed: a py3.12 host ``pydantic_core`` C extension shadowing
    the image's py3.11 build (transformers), and a fresh ``huggingface-hub``
    shadowing the pinned engine stack (tensorrt). Mounting the package dir alone
    makes ``/llem-src`` contain only ``llenergymeasure``, so nothing can shadow a
    container-native dependency. A source checkout is safe either way (its parent
    is ``src/``, holding only the package); resolving the package dir makes the
    two install shapes one uniform, always-safe path.
    """
    return Path(__file__).resolve().parent.parent.parent


@functools.cache
def _runtime_requirements() -> tuple[str, ...]:
    """Return llenergymeasure's always-on runtime dependency specs.

    Read from the installed distribution metadata (``importlib.metadata``)
    rather than a repo-root ``pyproject.toml``, so it resolves identically from
    a source checkout and a site-packages install. Optional-extra requirements
    (those carrying an ``extra == ...`` environment marker) are excluded: the
    container primes only the always-on runtime deps, exactly as the old
    ``[project.dependencies]`` diff did. Sorted so the materialised file is
    deterministic (the container hashes it for the deps-probe fast-path stamp).
    Non-extra environment markers (none exist on current core deps) are
    discarded rather than forwarded to pip; revisit if a marker-carrying core
    dependency is ever added.
    """
    core: list[str] = []
    for spec in importlib.metadata.requires(_DISPATCH_DIST_NAME) or []:
        requirement, _, marker = spec.partition(";")
        if "extra" in marker:
            continue
        requirement = requirement.strip()
        if requirement:
            core.append(requirement)
    return tuple(sorted(core))


@functools.cache
def _materialise_dispatch_assets() -> tuple[Path, Path]:
    """Materialise the dispatch assets to a host tempdir; return their paths.

    Returns ``(entry_script, requirements_file)`` as real on-disk paths suitable
    for a docker bind-mount. Both are sourced from the installed package (the
    script via ``importlib.resources``, the requirements via
    ``importlib.metadata``) so they resolve from an installed wheel, not only a
    source checkout - the defect this fixes was resolving them relative to
    ``__file__``, which landed outside the package for a site-packages install
    and let docker auto-create empty bind-mount dirs that broke the entrypoint
    exec.

    Materialised once per process (cached): the content is process-invariant and
    the mounts are read-only, so a sweep dispatching the same assets hundreds of
    times pays the extraction cost once. The tempdir is removed at interpreter
    exit. Raises :class:`DockerPreFlightError` (before any ``docker run``) if an
    asset cannot be produced, rather than letting docker silently mount an empty
    directory.
    """
    try:
        script_bytes = (
            importlib.resources.files(_ENTRY_SCRIPT_PACKAGE)
            .joinpath(_ENTRY_SCRIPT_RESOURCE)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise DockerPreFlightError(
            "Cannot read the container entrypoint script from the installed "
            "llenergymeasure package "
            f"({_ENTRY_SCRIPT_RESOURCE}). The package data is "
            "incomplete: reinstall llenergymeasure (e.g. "
            "'pip install --force-reinstall llenergymeasure'), or in a source "
            "checkout confirm the file exists under src/llenergymeasure/infra/."
        ) from exc

    requirements = _runtime_requirements()
    if not requirements:
        raise DockerPreFlightError(
            "Cannot derive llenergymeasure's runtime dependencies from the "
            "installed distribution metadata. Reinstall llenergymeasure so its "
            "metadata is present (e.g. 'pip install llenergymeasure')."
        )

    asset_dir = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX_DISPATCH))
    # Late-bind shutil.rmtree so a test patch of it active at registration time
    # cannot be captured into the process atexit chain; the real function is
    # resolved at shutdown, after any patch has been reverted.
    atexit.register(lambda: shutil.rmtree(asset_dir, ignore_errors=True))

    entry_script = asset_dir / "container_entrypoint.sh"
    entry_script.write_bytes(script_bytes)
    entry_script.chmod(0o755)

    requirements_file = asset_dir / "requirements.txt"
    requirements_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    return entry_script, requirements_file


@functools.cache
def _ensure_deps_cache_dir() -> Path:
    """Resolve the host-side runtime-deps cache directory, creating it if absent.

    The in-container entrypoint script primes any missing runtime deps here on
    first dispatch and short-circuits subsequent dispatches via a
    requirements-hash stamp. The cache is keyed by container
    Python minor (script writes to ``$DEPS_CACHE_ROOT/py{N}.{M}/``), so a
    single host directory serves multiple engine images even when their
    Python minors differ.

    Uses ``platformdirs`` for XDG-conformant path resolution; users can
    override via ``ENV_DEPS_CACHE_DIR`` if they want to share the cache
    across machines (e.g. on shared cluster storage). Cached because the
    result is invariant within a process and the mkdir is the only
    side-effect.
    """
    override = os.environ.get(ENV_DEPS_CACHE_DIR)
    if override:
        cache_dir = Path(override).expanduser().resolve()
    else:
        cache_dir = Path(platformdirs.user_cache_dir("llem")) / "deps"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def append_package_dispatch(
    cmd: list[str],
    *,
    engine: str,
    entry_module: str | None = None,
) -> None:
    """Append the bind-mounts + env + entrypoint that make the package importable.

    Upstream engine images (vllm, tensorrt) and even our transformers image do
    NOT have ``llenergymeasure`` installed; the framework runs from the host
    package source bind-mounted at ``/llem-src/llenergymeasure``. This appends
    the four mounts the in-container entrypoint script needs (package dir, the
    runtime requirements list, the script itself, and the host deps cache), the
    env it reads, and points ``--entrypoint`` at the script. The script prepends
    ``/llem-src`` to ``PYTHONPATH`` and primes any missing runtime deps, then
    exec's the module named by ``LLEM_ENTRY_MODULE``.

    INVARIANT - ``/llem-src`` must expose the ``llenergymeasure`` package and
    NOTHING else. The package dir is mounted at the nested target
    ``/llem-src/llenergymeasure`` (not the package's PARENT at ``/llem-src``)
    precisely so that no on-disk sibling of the package leaks in. For a
    pip/wheel install the parent is the venv's ``site-packages``; because the
    script prepends ``/llem-src`` to ``PYTHONPATH`` and ``PYTHONPATH`` precedes
    the container's own site-packages on ``sys.path``, mounting the parent let
    every host third-party package shadow the image's native copy (e.g. a host
    ``pydantic_core`` C extension built for a different Python minor, or a fresh
    ``huggingface-hub`` over a pinned engine stack). See ``_resolve_package_dir``.

    The entrypoint script and the requirements list are shipped as package data
    and materialised to a host tempdir (see ``_materialise_dispatch_assets``) so
    dispatch works from an installed wheel, not only a source checkout. The
    package dir at ``/llem-src/llenergymeasure`` resolves from ``__file__``
    (see ``_resolve_package_dir``): one uniform path for editable and wheel
    installs, no editable-detection, no llem dist-info in-container (the editable
    flow has always run without it, proving container-side code never needs it).

    Shared by the experiment dispatch (``DockerRunner._build_docker_cmd``) and
    the baseline dispatch (``study.baseline_container.build_baseline_docker_cmd``)
    so the two package-import setups cannot drift.

    Args:
        cmd: Docker command list to mutate in place (mounts/env appended before
            the image name, which the caller adds afterwards).
        engine: Engine value; sets ``LLEM_ENGINE`` so the script routes tensorrt
            through ``nvidia_entrypoint.sh`` for the libnvinfer ``LD_LIBRARY_PATH``.
        entry_module: Override for the module the script exec's. ``None`` leaves
            the script default (``llenergymeasure.entrypoints.container``).

    Raises:
        DockerPreFlightError: A dispatch asset (entrypoint script or requirements
            list) could not be materialised from the installed package. Raised
            before ``docker run`` so a missing source never becomes a silent
            docker-auto-created empty-dir mount.
    """
    pkg_dir = _resolve_package_dir()
    entry_script, requirements_file = _materialise_dispatch_assets()
    deps_cache = _ensure_deps_cache_dir()
    cmd.extend(
        [
            "-v",
            f"{pkg_dir}:/llem-src/llenergymeasure:ro",
            "-v",
            f"{requirements_file}:/llem-requirements.txt:ro",
            "-v",
            f"{entry_script}:/llem-entry.sh:ro",
            "-v",
            f"{deps_cache}:/llem-runtime-deps",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"{ENV_ENGINE}={engine}",
            "-e",
            f"{ENV_HOST_UID}={os.getuid()}",
            "-e",
            f"{ENV_HOST_GID}={os.getgid()}",
        ]
    )
    if entry_module is not None:
        cmd.extend(["-e", f"{ENV_ENTRY_MODULE}={entry_module}"])
    cmd.extend(["--entrypoint", "/llem-entry.sh"])


def append_nccl_env(cmd: list[str]) -> None:
    """Forward host ``NCCL_*`` env vars into the container.

    NCCL tuning/workaround settings must reach the engine process, which runs
    inside the container rather than on the host. The canonical case is
    ``NCCL_P2P_DISABLE=1`` on PCIe multi-GPU hosts whose topology lacks
    functional GPU peer-to-peer (P2P): without it, every tensor-parallel run
    hangs at the first NCCL collective.

    Uses explicit ``-e KEY=VALUE`` (matching the ``LLEM_*`` forwarding idiom in
    ``build_docker_cmd``) and iterates keys in sorted order so the built command
    is deterministic (tests assert on argv). Shared by the experiment dispatch
    (``DockerRunner._build_docker_cmd``) and the baseline dispatch
    (``study.baseline_container.build_baseline_docker_cmd``) so the two
    env-forwarding setups cannot drift.

    Args:
        cmd: Docker command list to mutate in place (env appended before the
            image name, which the caller adds afterwards).
    """
    for env_key in sorted(k for k in os.environ if k.startswith("NCCL_")):
        env_val = os.environ[env_key]
        if env_val:
            cmd.extend(["-e", f"{env_key}={env_val}"])


def docker_label_args(labels: dict[str, str] | None) -> list[str]:
    """Return the ``--label KEY=VALUE`` argv fragment for container ownership labels.

    Every container a study launches (experiment, engine server, baseline) wears
    the same ownership labels, so the study-scoped cleanup and the orphan reaper
    can see all of them. This is the one place that renders them as argv, shared
    by all three dispatch paths so their label emission cannot drift.

    ``docker run`` only accepts flags BEFORE the image reference - anything after
    it is passed to the container - so callers must splice this fragment in while
    still building the flag section, ahead of appending the image.

    Args:
        labels: Label key/values, or None/empty when the container is unlabelled.

    Returns:
        A flat argv fragment, empty when there are no labels. Insertion order is
        preserved so the built command is deterministic (tests assert on argv).
    """
    argv: list[str] = []
    for key, value in (labels or {}).items():
        argv += ["--label", f"{key}={value}"]
    return argv


def build_server_container_argv(
    *,
    image: str,
    container_name: str | None,
    gpu_indices: list[int] | None,
    serve_args: list[str],
    shm_size: str | None = None,
    labels: dict[str, str] | None = None,
) -> list[str]:
    """Build the ``docker run`` argv for a long-lived engine server container.

    Detached (``-d``) so the launch returns immediately and readiness is polled.
    Deliberately NO ``--rm``: a container that crashes during startup must
    SURVIVE so ``docker logs`` can recover the startup diagnostic (the
    failure-artefact hand-off the server handle's log reader promises). ``--rm``
    destroys the exited container within ~1s of the crash and the logs with it, so
    the diagnostic would be permanently lost. Leak-freeness is instead the
    explicit responsibility of the serving layer that runs this argv: its
    shutdown (``docker stop`` then ``docker rm -f``), its crashed-startup
    fast-detection, and its failed-launch cleanup all force-remove.

    ``--network host`` is UNCONDITIONAL (the peer convention: genai-perf, vllm
    benchmark_serving and MLPerf vendor repros
    co-locate client + server on one host; docker bridge overhead is real and
    directional). Because the container shares the host network namespace, the
    port is NOT published with ``-p`` - the server binds the host port directly,
    which the port passed inside ``serve_args`` (e.g. ``--port 8000``) selects.

    ``serve_args`` are the engine command appended after the image; for vLLM the
    upstream ``vllm/vllm-openai`` image's ``ENTRYPOINT`` is ``["vllm", "serve"]``,
    so the adapter passes ``[<model>, "--port", <port>]`` and the entrypoint
    supplies ``vllm serve``. TRT-LLM's NGC image is NOT entrypoint-baked with
    ``trtllm-serve``, so its adapter passes the full ``["trtllm-serve", <model>,
    "--port", <port>]`` command (the NGC entrypoint sets up the CUDA libs and
    execs it).

    The host HuggingFace cache is bind-mounted at ``/root/.cache/huggingface``
    with ``HF_HOME`` pointed at it (the SAME LLEM_DOCKER_HF_CACHE-driven mount the
    offline docker dispatch uses, via :func:`hf_cache_mount_args`), so a launched
    server reuses already-downloaded weights instead of re-downloading the full
    model on every run.

    ``labels`` are the study's container ownership labels (the same ones the
    offline docker dispatch sets), rendered by :func:`docker_label_args`, so a
    server container is attributable to its study and reachable by the
    study-scoped cleanup and the orphan reaper. A server container that outlives
    its launching process - the exact case shutdown cannot cover - is otherwise
    invisible to them.
    """
    argv = [
        "docker",
        "run",
        "-d",
        "--network",
        "host",
        "--gpus",
        docker_gpus_arg(gpu_indices),
    ]
    if container_name:
        argv += ["--name", container_name]
    argv += docker_label_args(labels)
    argv += ["--shm-size", shm_size or docker_shm_size()]
    argv += hf_cache_mount_args()
    argv.append(image)
    argv += list(serve_args)
    return argv


def _mount_if_absent(
    cmd: list[str],
    host: str | Path,
    container: str,
    extra_mounts: list[tuple[str, str]],
    *,
    extra_env: str | None = None,
) -> None:
    """Append ``-v host:container`` unless the user already mounts ``container``.

    Optionally appends ``-e <extra_env>`` after the mount (e.g. HF_HOME).
    """
    if any(cp == container for _, cp in extra_mounts):
        return
    cmd.extend(["-v", f"{host}:{container}"])
    if extra_env is not None:
        cmd.extend(["-e", extra_env])


def build_docker_cmd(
    *,
    image: str,
    config: Any,
    config_hash: str,
    exchange_dir: str,
    env_path: Path | None,
    extra_mounts: list[tuple[str, str]],
    container_name: str | None,
    labels: dict[str, str],
    gpu_indices: list[int] | None,
) -> list[str]:
    """Build the ``docker run`` command list.

    All three engines (transformers, vllm, tensorrt) follow the same
    dispatch shape: the image carries only the engine substrate, the host
    package source is bind-mounted at ``/llem-src/llenergymeasure``, the
    runtime requirements list is bind-mounted as a single-file mount at
    ``/llem-requirements.txt``, the in-container entrypoint script is
    bind-mounted at ``/llem-entry.sh``, and a host-side deps cache is
    bind-mounted at ``/llem-runtime-deps``. ``--entrypoint`` always points
    at ``/llem-entry.sh``; that script
    (a) diffs the requirements list against installed dists and
    pip-installs any missing ones to the cache (fast-path skips this when a
    requirements-hash stamp matches), (b) sets
    ``PYTHONPATH`` to include the cache and ``/llem-src``, and (c) exec's
    the framework entrypoint module - routing through
    ``/opt/nvidia/nvidia_entrypoint.sh`` when ``LLEM_ENGINE=tensorrt``
    (sets up LD_LIBRARY_PATH for libnvinfer).

    Multi-GPU tensorrt runs are NOT wrapped in mpirun: TensorRT-LLM's LLM
    API self-manages tensor parallelism (setting ``tensor_parallel_size``
    makes the LLM class spawn its own worker processes via its MPI/RPC
    orchestrator), so a single python3 process is correct.

    Args:
        image:        Docker image to run.
        config:       ExperimentConfig for the current experiment. Used to
                      detect TRT-LLM engine and read ``tensorrt.tensor_parallel_size``.
        config_hash:  Hash prefix for config/result file names.
        exchange_dir: Host path of the temporary exchange directory.
        env_path:     Path to a temp env-file (written by ``_env_file``), or None.
                      When set, ``--env-file <path>`` is added to the command.
                      Secrets (e.g. HF_TOKEN) are never passed as ``-e KEY=VALUE``
                      arguments to avoid exposure in ``/proc/<pid>/cmdline``.
        extra_mounts: User-supplied ``(host, container)`` mount pairs.
        container_name: Optional ``--name`` for lifecycle management.
        labels:       Optional ``--label`` key/values for the reaper.
        gpu_indices:  Optional host GPU indices to scope the container to.

    Returns:
        List of strings suitable for ``subprocess.run``.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        docker_gpus_arg(gpu_indices),
        "-v",
        f"{exchange_dir}:{CONTAINER_EXCHANGE_DIR}",
        "-e",
        f"{ENV_CONFIG_PATH}={CONTAINER_EXCHANGE_DIR}/{config_hash}_config.json",
        "--shm-size",
        docker_shm_size(),
    ]

    # Propagate secrets via --env-file (never as -e KEY=VALUE CLI args)
    if env_path is not None:
        cmd.extend(["--env-file", str(env_path)])

    # TRT-LLM engine cache: persist compiled engines across ephemeral
    # containers. Also default LLEM_TRT_BUILD_CACHE_PATH to the mount target
    # via extra_env so the cache lands on the mount out of the box (TRT-LLM's
    # own default root is unmounted); a host-set value overrides this because
    # the blanket LLEM_* forwarding loop below re-emits it last (docker -e is
    # last-wins).
    if config.engine == Engine.TENSORRT:
        _mount_if_absent(
            cmd,
            str(trt_build_cache_host_dir()),
            _TRT_BUILD_CACHE_CONTAINER_PATH,
            extra_mounts,
            extra_env=f"{ENV_TRT_BUILD_CACHE_PATH}={_TRT_BUILD_CACHE_CONTAINER_PATH}",
        )

    # Auto-mount the host HuggingFace cache so model weights persist across
    # ephemeral containers; otherwise each run re-downloads the full model.
    # The host source is configurable via LLEM_DOCKER_HF_CACHE.
    _mount_if_absent(
        cmd,
        docker_hf_cache_dir(),
        _HF_CACHE_CONTAINER_PATH,
        extra_mounts,
        extra_env=f"HF_HOME={_HF_CACHE_CONTAINER_PATH}",
    )

    # Auto-mount the host flashinfer JIT cache so TRT-LLM warm runs reuse
    # already-compiled per-arch attention kernels (cold compile is minutes).
    if config.engine == Engine.TENSORRT:
        _mount_if_absent(
            cmd, Path.home() / ".cache" / "flashinfer", "/root/.cache/flashinfer", extra_mounts
        )

    # Forward LLEM_* env vars into the container so framework defaults set
    # on the host (e.g. LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP) reach the
    # experiment process, which actually runs inside the container. Reserved
    # exchange keys are excluded: we set those deliberately above (or via
    # the package-dispatch helper), and docker's -e is last-wins over the
    # --env-file, so a stray host copy would silently clobber the intended
    # value.
    for env_key, env_val in os.environ.items():
        if env_key.startswith("LLEM_") and env_val and env_key not in _RESERVED_EXCHANGE_ENV:
            cmd.extend(["-e", f"{env_key}={env_val}"])

    # Forward host NCCL_* env vars so multi-GPU tuning/workaround settings
    # (e.g. NCCL_P2P_DISABLE=1 on PCIe hosts without functional GPU P2P)
    # reach the engine process, which runs inside the container.
    append_nccl_env(cmd)

    # Extra volume mounts (engine cache, model cache, etc.)
    for host_path, container_path in extra_mounts:
        cmd.extend(["-v", f"{host_path}:{container_path}"])

    # All engines: bind-mount the host package source + bootstrap (so the
    # package is importable in images that don't ship it) and point
    # --entrypoint at the script. Shared with the baseline dispatch via
    # append_package_dispatch so the two cannot drift. The experiment path
    # uses the script's default entry module
    # (``llenergymeasure.entrypoints.container``), so entry_module stays None.
    # ``Engine`` is a (str, Enum) so ``f"{config.engine}"`` resolves to the
    # raw value via its ``__str__`` override.
    append_package_dispatch(cmd, engine=f"{config.engine}")

    # Container name and labels for lifecycle management (cleanup, reaper).
    if container_name:
        cmd.extend(["--name", container_name])
    cmd += docker_label_args(labels)

    cmd.append(image)

    # No post-image args - the entrypoint script invokes the framework
    # module itself; config is passed via env vars (LLEM_CONFIG_PATH etc.).
    return cmd
