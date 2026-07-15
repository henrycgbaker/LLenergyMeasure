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

import os
import re
from pathlib import Path
from typing import Final

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

Restricting visibility at the DOCKER level keeps measurement indices
consistent: inside the container the only visible GPU(s) enumerate from 0 for
BOTH CUDA and NVML, so compute, energy sampling, and thermal monitoring all
address the same physical device without any index translation.
"""


def docker_gpus() -> str:
    """Return the ``docker run --gpus`` value for llem-launched containers.

    Pure passthrough with the historical fallback: unset / empty -> ``all``.
    """
    raw = os.environ.get(ENV_DOCKER_GPUS, "").strip()
    return raw or "all"


def docker_gpus_arg() -> str:
    """Return the ``--gpus`` value formatted for use as a docker-run argument.

    A multi-device ``device=<a>,<b>`` selector MUST be wrapped in literal double
    quotes: docker parses the ``--gpus`` value as CSV, so an unquoted
    ``device=1,3`` is split at the comma into a device id (``device=1``) and a
    trailing GPU *count* (``3``), which docker rejects with "cannot set both
    Count and DeviceIDs on device request". Quoting keeps the whole list a
    single device-ids field. ``all``, count forms, and a single-device
    ``device=N`` (no comma) need no quoting and are returned verbatim.

    ``docker_gpus`` stays the raw selector - :func:`pinned_gpu_lock_ids` parses
    that form for lock naming and must not see the quotes.
    """
    raw = docker_gpus()
    if raw.startswith("device=") and "," in raw:
        return f'"{raw}"'
    return raw


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


def pinned_gpu_lock_ids() -> list[str] | None:
    """Return per-physical-device lock identifiers parsed from ``LLEM_DOCKER_GPUS``.

    llem's per-GPU advisory locks (``study/gpu_locks.py``) must be named by the
    PHYSICAL device a study occupies so that two studies pinned to different
    physical GPUs never share a lock. Under ``docker run --gpus device=N`` the
    container sees its granted GPU as LOGICAL index ``0``, so the in-container
    index (what ``device/gpu_info._resolve_gpu_indices`` returns) is the wrong
    key for a host-side lock. This parses the docker selector back into the
    physical identity:

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

    This is a LOCK-NAMING concern only. Measurement-side index resolution is
    deliberately untouched: NVML / CUDA indices inside the container enumerate
    from ``0`` under pinning, and ``_resolve_gpu_indices`` still returns those
    logical indices to address the energy samplers.
    """
    raw = docker_gpus()
    prefix = "device="
    if not raw.startswith(prefix):
        # "all", unset (-> "all"), count forms, or anything unrecognised.
        return None
    tokens = [tok.strip() for tok in raw[len(prefix) :].split(",")]
    ids = [_sanitize_lock_id(tok) for tok in tokens if tok]
    return ids or None


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
