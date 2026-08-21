"""Pure ``docker run`` command builders for container dispatch.

Everything here is a pure builder: it assembles the argv (env forwarding,
mounts, GPU/shm flags, image ref, and the package-dispatch bootstrap) without
running docker, so it is unit-testable without a daemon. The one side effect is
materialising the dispatch assets (entrypoint script + requirements list) to a
host tempdir for bind-mounting, and ensuring the host deps cache exists - both
process-cached and idempotent.

This is the single home for ``docker run`` argv construction. One core,
:func:`build_container_argv`, assembles every container this framework launches;
the three shapes are parameterisations of it:

- :func:`build_docker_cmd` - the offline experiment dispatch. Runs to completion,
  forwards the host LLEM_* environment, and carries the full mount set.
- :func:`build_baseline_container_argv` - the idle-baseline measurement. Runs to
  completion, shares the package-dispatch bootstrap with the experiment shape,
  and is anonymous.
- :func:`build_server_container_argv` - the long-lived engine server. Detached,
  on the host network, and deliberately not auto-removed.

All three forward the host ``NCCL_*`` environment (:func:`append_nccl_env`).
That is not a divergence and never was: the server shape used to omit it, which
read as a deliberate asymmetry but was an omission. A tensor-parallel server run
needs the same host NCCL workarounds a multi-GPU offline run needs - on a PCIe
host without functional GPU peer-to-peer, a server started without
``NCCL_P2P_DISABLE=1`` hangs at its first NCCL collective exactly as an offline
run does. Note what does NOT hold that parity: the forwarding is three separate
``append_nccl_env`` calls, one per shape, not one line inside the shared core -
which is precisely how a shape came to be missing it. What guards it is a test,
the cross-shape assertion in ``tests/unit/docker/test_container_argv_shapes.py``
that every shape forwards a host ``NCCL_*`` var, so a new shape or a moved call
site has to keep the parity deliberately.

Their remaining divergences are real and each is stated where it is chosen, but
nothing they have in common is written more than once, so the shapes cannot drift
apart on the removal policy, the GPU selector, the ownership labels, or the rule
that every flag precedes the image. Consumed by
``DockerRunner._build_docker_cmd``, ``study.baseline_container``, and each
engine's server adapter, which passes the argv it builds here to the serving
layer's launcher.

Scope: these are the shapes llenergymeasure runs a WORKLOAD in. The two one-shot
diagnostic ``docker run`` probes elsewhere in ``infra`` (the engine-version probe
in ``version_handshake``, the GPU-visibility probe in ``docker_preflight``) stay
separate on purpose - neither carries a measurement, and neither can go through
this core without changing what it does: the version probe requests NO GPU at all
(so it still answers on a host without the NVIDIA container runtime) and the
preflight probe runs a fixed CUDA base image rather than an engine image.
"""

from __future__ import annotations

import atexit
import functools
import importlib.metadata
import importlib.resources
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

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

# The three container shapes, and the one filename the study side has to agree
# with. Everything else here - the shared core, the removal-policy constants, the
# flag-fragment helpers - is construction material with no consumer outside this
# module, and it stays unexported on purpose: an advertised core is an invitation
# to assemble a fourth shape somewhere else, which is the exact drift this module
# exists to prevent. A future need to define shapes elsewhere can export the core
# deliberately, as a decision rather than as a leftover.
__all__ = [
    "BASELINE_SPEC_FILENAME",
    "build_baseline_container_argv",
    "build_docker_cmd",
    "build_server_container_argv",
]

#: The container runs to completion and docker removes it on exit (``--rm``).
LIFETIME_RUN_TO_COMPLETION: Final = "run_to_completion"
#: The container is launched detached (``-d``) and deliberately NOT ``--rm``, so
#: a crash-on-startup survives for its logs to be read. Removal is then the
#: launcher's explicit responsibility.
LIFETIME_DETACHED: Final = "detached"
#: Which of the two removal policies a container shape wants.
ContainerLifetime = Literal["run_to_completion", "detached"]

#: Filename of the baseline spec inside the exchange dir. The host writes it and
#: the container reads it back from the mounted exchange dir, so both sides name
#: it from here.
BASELINE_SPEC_FILENAME: Final = "baseline_spec.json"

#: Entry module the baseline shape points the in-container entrypoint script at,
#: instead of the experiment module the script defaults to.
_BASELINE_ENTRY_MODULE: Final = "llenergymeasure.entrypoints.baseline_measure"

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

    Shared by the experiment dispatch (:func:`build_docker_cmd`) and the baseline
    dispatch (:func:`build_baseline_container_argv`)
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
    is deterministic (tests assert on argv). An exported-but-empty ``NCCL_*`` var
    is skipped rather than forwarded as an empty value.

    Called by all three container shapes - the experiment dispatch
    (:func:`build_docker_cmd`), the baseline dispatch
    (:func:`build_baseline_container_argv`) and the engine server
    (:func:`build_server_container_argv`). The forwarding rule is written once
    here, but that each shape actually calls it is enforced by the cross-shape
    test rather than by construction: a shape that forgets the call still builds
    a valid argv, which is how the server shape went without it. A
    tensor-parallel server run reaches the same first-collective hang as a
    tensor-parallel offline run, so no shape may skip this.

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


def build_container_argv(
    *,
    image: str,
    gpu_indices: list[int] | None,
    lifetime: ContainerLifetime,
    host_network: bool = False,
    flags: Sequence[str] = (),
    container_name: str | None = None,
    labels: dict[str, str] | None = None,
    identity_before_flags: bool = False,
    command: Sequence[str] = (),
) -> list[str]:
    """Assemble one ``docker run`` argv. The single core the three shapes share.

    Every container llenergymeasure launches is built here, so the invariants
    none of the shapes may break are enforced in one place rather than repeated
    three times: the argv opens with ``docker run``, the removal policy is a
    named decision rather than a literal that can be silently "fixed", the GPU
    selector always goes through :func:`docker_gpus_arg`, identity (name and
    ownership labels) is always rendered by :func:`docker_label_args` and always
    lands before the image, and the image itself comes after every flag with the
    container command after it. ``docker run`` treats everything following the
    image reference as the container's own command, so that last ordering rule is
    not cosmetic.

    What differs per shape is passed in, and every divergence is deliberate:

    - ``lifetime`` picks the removal policy. ``"run_to_completion"`` emits
      ``--rm``: the caller blocks until the container exits and docker reaps it.
      ``"detached"`` emits ``-d`` and NO ``--rm``, because a detached server that
      crashes during startup must SURVIVE for ``docker logs`` to recover the
      diagnostic; ``--rm`` would destroy the exited container within about a
      second and the logs with it. That makes removal the launcher's explicit
      job, not docker's.
    - ``host_network`` emits ``--network host``. Only the engine-server shape
      wants it (client and server co-located on one host, the convention
      genai-perf, vllm benchmark_serving and MLPerf vendor repros follow, because
      docker bridge overhead is real and directional). Sharing the host network
      namespace is also why that shape publishes no port with ``-p``: the server
      binds the host port directly.
    - ``flags`` are the shape's own flags: mounts, environment, resource limits,
      and the package-dispatch bootstrap where the shape needs one. Passed
      already assembled because their internal order is part of each shape's
      contract.
    - ``identity_before_flags`` places ``--name`` and the ownership labels ahead
      of ``flags`` instead of after them. Docker parses every pre-image flag
      order-independently, so this changes nothing about the container; it exists
      only so each shape's emitted argv keeps the exact flag order its own tests
      and reviewers read, and no shape has to hand-roll its identity to get it.
    - ``command`` is appended after the image, for shapes that pass a command
      rather than relying on the image entrypoint.
    """
    argv = ["docker", "run"]
    argv.append("-d" if lifetime == LIFETIME_DETACHED else "--rm")
    if host_network:
        argv += ["--network", "host"]
    argv += ["--gpus", docker_gpus_arg(gpu_indices)]
    identity: list[str] = []
    if container_name:
        identity += ["--name", container_name]
    identity += docker_label_args(labels)
    argv += identity + list(flags) if identity_before_flags else list(flags) + identity
    argv.append(image)
    argv += list(command)
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

    The detached shape: ``-d`` so the launch returns immediately and readiness is
    polled separately, no ``--rm`` so a crash-on-startup leaves its logs
    recoverable, and ``--network host`` so a co-located client reaches the server
    over loopback. The port is selected inside ``serve_args`` (e.g.
    ``--port 8000``) rather than published with ``-p``, because the container
    shares the host network namespace. See :func:`build_container_argv` for why
    each of those is the right call for this shape.

    This shape carries NO package-dispatch bootstrap, unlike the other two. The
    process it starts is the engine's own upstream server binary (``vllm serve``,
    ``trtllm-serve``), never llenergymeasure code, so the framework has nothing to
    make importable inside the container: no source bind-mount, no requirements
    priming, no ``--entrypoint`` override. The measurement code stays on the host
    and talks to the server over HTTP.

    It DOES forward the host ``NCCL_*`` environment (:func:`append_nccl_env`),
    exactly as the other two shapes do. The engine's tensor-parallel workers run
    inside this container, so a host that needs ``NCCL_P2P_DISABLE=1`` for
    multi-GPU (PCIe topology without functional GPU peer-to-peer) needs it here
    too: without it the server hangs at its first NCCL collective and never
    becomes ready.

    ``serve_args`` are the engine command appended after the image. For vLLM the
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
    model on every run. This shape accepts no user-supplied mounts, so it takes
    the unconditional form rather than the offline shape's
    do-not-clobber-the-user variant, and it is the ONLY mount it needs.

    ``labels`` are the study's container ownership labels (the same ones the
    offline docker dispatch sets), so a server container is attributable to its
    study and reachable by the study-scoped cleanup and the orphan reaper. A
    server container that outlives its launching process - the exact case
    shutdown cannot cover - is otherwise invisible to them.
    """
    flags = ["--shm-size", shm_size or docker_shm_size(), *hf_cache_mount_args()]
    # Forward host NCCL_* env vars so multi-GPU tuning/workaround settings
    # (e.g. NCCL_P2P_DISABLE=1 on PCIe hosts without functional GPU P2P) reach
    # the engine's server process, matching the offline and baseline paths. A
    # tensor-parallel server hangs at the first NCCL collective without them on
    # exactly the hosts that need the workaround.
    append_nccl_env(flags)
    return build_container_argv(
        image=image,
        gpu_indices=gpu_indices,
        lifetime=LIFETIME_DETACHED,
        host_network=True,
        flags=flags,
        container_name=container_name,
        labels=labels,
        identity_before_flags=True,
        command=serve_args,
    )


def build_baseline_container_argv(
    *,
    image: str,
    exchange_dir: str,
    gpu_indices: list[int],
    engine: str,
    config_gpu_indices: list[int] | None = None,
    labels: dict[str, str] | None = None,
) -> list[str]:
    """Build the ``docker run`` argv for a short-lived baseline-only container.

    A host-measured idle baseline underestimates the container's idle GPU power,
    because the host has no CUDA context and no torch memory pool seeded. This
    shape measures the baseline inside a container whose CUDA state matches the
    experiment container's, which removes that bias.

    Run-to-completion like the experiment shape (``--rm``, the caller blocks),
    and it routes through :func:`append_package_dispatch` for the same reason:
    upstream engine images do not ship the ``llenergymeasure`` package, so
    without the bind-mounted source the baseline entry module would fail with
    ``ModuleNotFoundError``. It differs from the experiment shape only in what it
    needs: the baseline entry module instead of the experiment one, no
    ``--shm-size`` (it allocates no shared-memory dataloader workers), no
    LLEM_* forwarding, and no user-supplied mounts.

    Two distinct GPU params: ``gpu_indices`` are the LOGICAL in-container
    monitoring indices (``CUDA_VISIBLE_DEVICES``); ``config_gpu_indices`` are the
    study's HOST ``--gpus`` selector (``study_execution.gpu_indices``, env>config
    via ``docker_gpus_arg``; see ``utils.env_config.ENV_DOCKER_GPUS``). Threading
    the latter scopes the baseline container to the same physical devices as the
    experiment container, so a config-pinned study does not baseline the wrong GPU.

    This shape is anonymous - it takes ``labels`` but no ``--name``. A baseline
    container is short-lived, but it holds the GPU while it samples, so the
    ownership labels are what make it attributable to its study and reachable by
    the study-scoped cleanup and the orphan reaper if the launching process dies
    mid-measurement.

    Kept separate from the dispatch that runs it so tests can assert on the argv
    without mocking subprocess internals.
    """
    cuda_visible = ",".join(str(i) for i in gpu_indices) if gpu_indices else ""
    spec_container_path = f"{CONTAINER_EXCHANGE_DIR}/{BASELINE_SPEC_FILENAME}"
    flags = [
        "-v",
        f"{exchange_dir}:{CONTAINER_EXCHANGE_DIR}",
        "-e",
        f"{ENV_BASELINE_SPEC_PATH}={spec_container_path}",
        "-e",
        f"CUDA_VISIBLE_DEVICES={cuda_visible}",
    ]
    # Forward host NCCL_* env vars so multi-GPU tuning/workaround settings
    # (e.g. NCCL_P2P_DISABLE=1 on PCIe hosts without functional GPU P2P) reach
    # the baseline process inside the container, matching the experiment path.
    append_nccl_env(flags)
    # Mount the package + bootstrap and point --entrypoint at /llem-entry.sh,
    # which makes the package importable and exec's the baseline entry module.
    append_package_dispatch(flags, engine=engine, entry_module=_BASELINE_ENTRY_MODULE)
    return build_container_argv(
        image=image,
        gpu_indices=config_gpu_indices,
        lifetime=LIFETIME_RUN_TO_COMPLETION,
        flags=flags,
        labels=labels,
    )


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
    flags = [
        "-v",
        f"{exchange_dir}:{CONTAINER_EXCHANGE_DIR}",
        "-e",
        f"{ENV_CONFIG_PATH}={CONTAINER_EXCHANGE_DIR}/{config_hash}_config.json",
        "--shm-size",
        docker_shm_size(),
    ]

    # Propagate secrets via --env-file (never as -e KEY=VALUE CLI args)
    if env_path is not None:
        flags.extend(["--env-file", str(env_path)])

    # TRT-LLM engine cache: persist compiled engines across ephemeral
    # containers. Also default LLEM_TRT_BUILD_CACHE_PATH to the mount target
    # via extra_env so the cache lands on the mount out of the box (TRT-LLM's
    # own default root is unmounted); a host-set value overrides this because
    # the blanket LLEM_* forwarding loop below re-emits it last (docker -e is
    # last-wins).
    if config.engine == Engine.TENSORRT:
        _mount_if_absent(
            flags,
            str(trt_build_cache_host_dir()),
            _TRT_BUILD_CACHE_CONTAINER_PATH,
            extra_mounts,
            extra_env=f"{ENV_TRT_BUILD_CACHE_PATH}={_TRT_BUILD_CACHE_CONTAINER_PATH}",
        )

    # Auto-mount the host HuggingFace cache so model weights persist across
    # ephemeral containers; otherwise each run re-downloads the full model.
    # The host source is configurable via LLEM_DOCKER_HF_CACHE.
    _mount_if_absent(
        flags,
        docker_hf_cache_dir(),
        _HF_CACHE_CONTAINER_PATH,
        extra_mounts,
        extra_env=f"HF_HOME={_HF_CACHE_CONTAINER_PATH}",
    )

    # Auto-mount the host flashinfer JIT cache so TRT-LLM warm runs reuse
    # already-compiled per-arch attention kernels (cold compile is minutes).
    if config.engine == Engine.TENSORRT:
        _mount_if_absent(
            flags, Path.home() / ".cache" / "flashinfer", "/root/.cache/flashinfer", extra_mounts
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
            flags.extend(["-e", f"{env_key}={env_val}"])

    # Forward host NCCL_* env vars so multi-GPU tuning/workaround settings
    # (e.g. NCCL_P2P_DISABLE=1 on PCIe hosts without functional GPU P2P)
    # reach the engine process, which runs inside the container.
    append_nccl_env(flags)

    # Extra volume mounts (engine cache, model cache, etc.)
    for host_path, container_path in extra_mounts:
        flags.extend(["-v", f"{host_path}:{container_path}"])

    # All engines: bind-mount the host package source + bootstrap (so the
    # package is importable in images that don't ship it) and point
    # --entrypoint at the script. Shared with the baseline dispatch via
    # append_package_dispatch so the two cannot drift. The experiment path
    # uses the script's default entry module
    # (``llenergymeasure.entrypoints.container``), so entry_module stays None.
    # ``Engine`` is a (str, Enum) so ``f"{config.engine}"`` resolves to the
    # raw value via its ``__str__`` override.
    append_package_dispatch(flags, engine=f"{config.engine}")

    # No post-image command - the entrypoint script invokes the framework module
    # itself; config is passed via env vars (LLEM_CONFIG_PATH etc.). The name and
    # ownership labels for lifecycle management (cleanup, reaper) are emitted by
    # the core, after these flags and before the image.
    return build_container_argv(
        image=image,
        gpu_indices=gpu_indices,
        lifetime=LIFETIME_RUN_TO_COMPLETION,
        flags=flags,
        container_name=container_name,
        labels=labels,
    )
