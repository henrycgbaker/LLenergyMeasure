"""Env-var helpers for opinionated runtime defaults.

This module is the canonical location for ``LLEM_*`` env-var constants and
the thin passthrough helpers that read them. Helpers are pure - they return
``os.environ.get(...)`` (or a parsed form) and return ``None`` / ``False``
when unset, deferring to each inference library's own default.

Opinionated defaults (e.g. ``LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto``) live in the
repo-root ``.env.example`` - the single reviewable source of truth for what
llem overrides out-of-the-box. Users copy ``.env.example`` to ``.env`` (auto-
loaded by the CLI) and edit to taste. Because helpers have no baked-in
defaults, removing a line from ``.env`` always restores the library default.

Layer: ``utils/`` (Layer 0). Cannot import ``config/``. This module is
consumed by engine plugins in Layer 2.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})


def parse_bool_env(var: str) -> bool:
    """Parse an env var as a boolean (``1``/``true``/``yes``/``on`` -> True)."""
    return os.environ.get(var, "").strip().lower() in _TRUTHY


ENV_TRANSFORMERS_DEFAULT_DEVICE_MAP: Final = "LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP"
"""Override for the HuggingFace ``device_map`` argument at model load.

Unset / empty → ``None`` (caller omits the kwarg; HuggingFace's own default
applies, which is CPU-only). Any non-empty value is forwarded as-is
(e.g. ``auto``, ``balanced``, ``sequential``). The opinionated default
``auto`` is shipped via ``.env.example`` - not baked into this helper.
"""


def default_device_map() -> str | None:
    """Return the configured default ``device_map`` or ``None`` if unset.

    Pure passthrough: no opinionated default is baked in. The repo-root
    ``.env.example`` ships ``LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto`` so that the
    out-of-the-box experience on multi-GPU hosts is ``device_map="auto"``;
    deleting the line from a user's ``.env`` reverts to HuggingFace's own
    default (``None`` = CPU-only).

    Callers typically apply typed-config precedence first (e.g. the
    ``transformers.device_map`` field in YAML), then fall back to this
    helper. If this returns ``None``, callers should omit the kwarg.
    """
    return os.environ.get(ENV_TRANSFORMERS_DEFAULT_DEVICE_MAP) or None


ENV_DOCKER_GPUS: Final = "LLEM_DOCKER_GPUS"
"""Docker ``--gpus`` request for llem-launched containers (experiment + baseline).

Passed verbatim as the value of ``docker run --gpus``. Unset / empty means
``all`` (every visible GPU - the historical behaviour). On a shared multi-GPU
host, set a device selector (e.g. ``device=2`` or ``device=2,3``) to pin
llem's containers to free GPUs.

Index space and why docker-level scoping (the canonical rationale referenced
elsewhere): GPU indices - both here and in ``study_execution.gpu_indices`` - are
HOST device indices as the NVIDIA driver / NVML enumerate them (the ordering
``nvidia-smi`` shows and ``docker run --gpus device=N`` selects). Restricting
visibility at the DOCKER level (rather than via ``CUDA_VISIBLE_DEVICES`` inside
the container) keeps measurement indices consistent: inside the container the
only visible GPU(s) re-enumerate from 0 for BOTH CUDA and NVML, so compute,
energy sampling, and thermal monitoring all address the same physical device
without any index translation.

Precedence over the config ``study_execution.gpu_indices`` field: this env var
WINS (env > config, matching llem's env>config convention). When both are set
the config indices are ignored and :func:`warn_on_gpu_selector_conflict` logs a
one-line warning. Because the env fully overrides config, the two selectors
never compose - there is no "config indices index into the env-restricted set"
case to disambiguate.
"""


def _gpu_indices_to_selector(gpu_indices: list[int]) -> str:
    """Format host GPU indices as a docker device selector.

    ``[2]`` -> ``"device=2"``; ``[2, 3]`` -> ``"device=2,3"``. Indices are host
    device indices - see :data:`ENV_DOCKER_GPUS` for the index space.
    """
    return "device=" + ",".join(str(i) for i in gpu_indices)


def docker_gpus(config_gpu_indices: list[int] | None = None) -> str:
    """Return the raw ``docker run --gpus`` selector for llem-launched containers.

    Single precedence point for GPU scoping (env > config > default):

    - ``LLEM_DOCKER_GPUS`` set / non-empty -> its value verbatim (config indices
      ignored; the env WINS per llem's env>config convention).
    - else ``config_gpu_indices`` non-empty -> ``device=<comma-joined host ids>``.
    - else -> ``all`` (every visible GPU - the historical default).

    ``config_gpu_indices`` are host device indices; see :data:`ENV_DOCKER_GPUS`
    for the index space and why scoping happens at the docker level.
    """
    raw = os.environ.get(ENV_DOCKER_GPUS, "").strip()
    if raw:
        return raw
    if config_gpu_indices:
        return _gpu_indices_to_selector(config_gpu_indices)
    return "all"


def docker_gpus_arg(config_gpu_indices: list[int] | None = None) -> str:
    """Return the ``--gpus`` value formatted for use as a docker-run argument.

    A multi-device ``device=<a>,<b>`` selector MUST be wrapped in literal double
    quotes: docker parses the ``--gpus`` value as CSV, so an unquoted
    ``device=1,3`` is split at the comma into a device id (``device=1``) and a
    trailing GPU *count* (``3``), which docker rejects with "cannot set both
    Count and DeviceIDs on device request". Quoting keeps the whole list a
    single device-ids field. ``all``, count forms, and a single-device
    ``device=N`` (no comma) need no quoting and are returned verbatim.

    ``config_gpu_indices`` follows the same env>config precedence as
    :func:`docker_gpus` (which this delegates to). :func:`docker_gpus` stays the
    raw selector - :func:`pinned_gpu_lock_ids` parses that form for lock naming
    and must not see the quotes.
    """
    raw = docker_gpus(config_gpu_indices)
    if raw.startswith("device=") and "," in raw:
        return f'"{raw}"'
    return raw


def warn_on_gpu_selector_conflict(config_gpu_indices: list[int] | None) -> None:
    """Log one warning when both ``LLEM_DOCKER_GPUS`` and config indices are set.

    Precedence resolution (env wins) is silent inside :func:`docker_gpus`; this
    is the loud surface for the conflict so the researcher is told their config
    ``gpu_indices`` are being ignored. Call once per study/experiment dispatch,
    not once per ``docker run`` build, to avoid per-experiment log spam.
    """
    raw = os.environ.get(ENV_DOCKER_GPUS, "").strip()
    if raw and config_gpu_indices:
        logger.warning(
            "Both %s=%r (env) and study_execution.gpu_indices=%s (config) are set. "
            "Env wins (env>config): containers are scoped to %r and the config "
            "gpu_indices are ignored. Unset one to silence this warning.",
            ENV_DOCKER_GPUS,
            raw,
            config_gpu_indices,
            raw,
        )


_UNSAFE_LOCK_ID_CHARS: Final = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_lock_id(raw: str) -> str:
    """Make a docker ``--gpus`` device token safe as a lock-file name component.

    Replaces any character outside ``[A-Za-z0-9._-]`` with ``_`` so the token
    cannot contain a path separator (no directory escape) and is a valid
    filename. Real device selectors - integer indices and ``GPU-<uuid>`` /
    ``MIG-<uuid>`` strings - are already within this set, so this is a no-op for
    them; it only hardens against pathological ``LLEM_DOCKER_GPUS`` values.
    """
    return _UNSAFE_LOCK_ID_CHARS.sub("_", raw)


def pinned_gpu_lock_ids(config_gpu_indices: list[int] | None = None) -> list[str] | None:
    """Return per-physical-device lock identifiers for the effective GPU selector.

    llem's per-GPU advisory locks (``study/gpu_locks.py``) must be named by the
    PHYSICAL device a study occupies so that two studies pinned to different
    physical GPUs never share a lock. Under ``docker run --gpus device=N`` the
    container sees its granted GPU as LOGICAL index ``0``, so the in-container
    index (what ``device/gpu_info._resolve_gpu_indices`` returns) is the wrong
    key for a host-side lock. This resolves the effective selector (via
    :func:`docker_gpus`, so the same env>config precedence applies:
    ``LLEM_DOCKER_GPUS`` wins, else ``config_gpu_indices``) and parses it back
    into the physical identity:

    - ``device=2``          -> ``["2"]``
    - ``device=2,3``        -> ``["2", "3"]``
    - ``device=GPU-<uuid>`` -> ``["GPU-<uuid>"]`` - a UUID cannot be mapped to a
      small integer index without an ``nvidia-smi`` lookup, but the UUID string
      is itself globally unique and stable, so it serves directly as a
      collision-correct per-device lock id (same UUID -> same lock, distinct
      UUIDs -> distinct locks).
    - ``all`` / unset       -> ``None`` - every visible GPU is granted, so
      logical == physical and the caller falls back to the in-container logical
      indices (unchanged historical behaviour).
    - any other shape (e.g. ``count=2``, a bare count, malformed) -> ``None`` -
      the physical identity is unknowable, so fall back to logical indices.

    Threading ``config_gpu_indices`` here keeps lock naming correct when a study
    pins physical GPUs via config rather than the env var: without it, two
    config-pinned studies on different physical GPUs would both fall back to the
    logical index ``0`` and collide on one lock.

    This is a LOCK-NAMING concern only. Measurement-side index resolution is
    deliberately untouched: NVML / CUDA indices inside the container enumerate
    from ``0`` under pinning, and ``_resolve_gpu_indices`` still returns those
    logical indices to address the energy samplers.
    """
    raw = docker_gpus(config_gpu_indices)
    prefix = "device="
    if not raw.startswith(prefix):
        # "all", unset (-> "all"), count forms, or anything unrecognised.
        return None
    tokens = [tok.strip() for tok in raw[len(prefix) :].split(",")]
    ids = [_sanitize_lock_id(tok) for tok in tokens if tok]
    return ids or None


def docker_gpus_cache_token(config_gpu_indices: list[int] | None = None) -> str | None:
    """Return a filename-safe token for the effective ``--gpus`` selector.

    ``None`` when the selector resolves to ``all`` (unrestricted); otherwise a
    sanitised form of the effective selector (env>config, via :func:`docker_gpus`),
    e.g. ``device=2,3`` -> ``device_2_3``. Used to qualify the per-target baseline
    cache key so a study resumed with a different GPU pin does not reuse a baseline
    measured on a different physical device (idle GPU power is per-device). The
    ``all`` -> ``None`` case keeps the unqualified cache-key shape unchanged for the
    default (unpinned) path.
    """
    selector = docker_gpus(config_gpu_indices)
    if selector == "all":
        return None
    return _sanitize_lock_id(selector)


_DEFAULT_DOCKER_SHM_SIZE: Final = "8g"

ENV_DOCKER_SHM_SIZE: Final = "LLEM_DOCKER_SHM_SIZE"
"""Docker ``--shm-size`` for llem-launched experiment containers.

Passed verbatim as the value of ``docker run --shm-size``. Unset / empty means
``8g`` - large inference engines (notably vLLM) need far more than docker's
64 MB ``/dev/shm`` default or they fail at startup. Raise it (e.g. ``16g``) for
very large tensor-parallel runs, or lower it on memory-constrained hosts. Any
docker-accepted size string (``512m``, ``8g``, ...) is forwarded as-is.
"""


def docker_shm_size() -> str:
    """Return the ``docker run --shm-size`` value for llem experiment containers.

    Pure passthrough with the historical fallback: unset / empty -> ``8g``.
    """
    return os.environ.get(ENV_DOCKER_SHM_SIZE, "").strip() or _DEFAULT_DOCKER_SHM_SIZE


ENV_TRT_BUILD_CACHE_ENABLED: Final = "LLEM_TRT_BUILD_CACHE_ENABLED"
"""Toggle for TRT-LLM on-disk engine build cache.

Unset / empty / any falsy value (``0``, ``false``, ``no``, ``off``) → False,
matching TRT-LLM's own default. Truthy values (``1``, ``true``, ``yes``,
``on``, case-insensitive) → True. The opinionated default ``1`` is shipped
via ``.env.example`` - not baked into this helper.
"""


ENV_TRT_BUILD_CACHE_PATH: Final = "LLEM_TRT_BUILD_CACHE_PATH"
"""Cache directory for the TRT-LLM engine build cache.

If set and non-empty, the engine plugin wraps it into TRT-LLM's
``BuildCacheConfig.cache_root``. Unset / empty leaves TRT-LLM's internal
default cache root in place (``/tmp/.cache/tensorrt_llm/llmapi/``).

Under docker dispatch the runner defaults this to the bind-mounted cache
directory (``/root/.cache/trt-llm``) so the cache persists across ephemeral
containers out of the box; TRT-LLM's own default root lives on the container
filesystem and would not survive. A host-set value still overrides the
docker default.
"""


def trt_build_cache_enabled() -> bool:
    """Return whether TRT-LLM on-disk engine build cache should be enabled.

    Pure passthrough: no opinionated default is baked in. The repo-root
    ``.env.example`` ships ``LLEM_TRT_BUILD_CACHE_ENABLED=1`` so the
    out-of-the-box experience preserves the cache (engine compilation takes
    minutes); deleting the line reverts to TRT-LLM's disabled default.
    """
    return parse_bool_env(ENV_TRT_BUILD_CACHE_ENABLED)


def trt_build_cache_path() -> Path | None:
    """Return the user-supplied TRT-LLM build cache root, if any.

    Returns a ``Path`` when set to a non-empty value; otherwise ``None`` so
    TRT-LLM uses its internal default root (``/tmp/.cache/tensorrt_llm/llmapi/``).
    Under docker dispatch the runner sets this to the mounted cache directory
    so the default is the persistent mount, not the ephemeral container path.
    """
    raw = os.environ.get(ENV_TRT_BUILD_CACHE_PATH)
    return Path(raw) if raw else None


def trt_build_cache_host_dir() -> Path:
    """Return the host directory bind-mounted as the TRT-LLM build cache.

    This is the out-of-the-box location the docker runner mounts into the
    container (``~/.cache/trt-llm``) and the directory ``llem doctor`` reports.
    A single source of truth so the mount source and the doctor view cannot
    drift. A user who overrides ``LLEM_TRT_BUILD_CACHE_PATH`` to a custom
    container path is responsible for their own host mount.
    """
    return Path.home() / ".cache" / "trt-llm"
