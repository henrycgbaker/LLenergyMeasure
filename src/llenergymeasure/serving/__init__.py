"""Serving layer: engine-server lifecycle vocabulary, placement, launch and readiness.

Owns everything that is identical for every OpenAI-compatible inference server,
so no engine adapter re-implements it and no consumer above reaches into the
container plumbing to get it. Split by concern so a caller takes only what it
needs:

- :mod:`llenergymeasure.serving.types` - the vocabulary. Placement, handle, probe
  shape and the four lifecycle errors, with no mechanism attached.
- :mod:`llenergymeasure.serving.lifecycle` - the mechanism. Port allocation,
  container/process launch, the readiness wait, and leak-free shutdown.
- :mod:`llenergymeasure.serving.transport` - the wire. One request out, one
  streamed completion back, for the load issuer above to schedule.

Per-engine knowledge (the serve command, the probe body) stays in the engine
adapters, which compose these primitives through the ``ServerCapable`` protocol
extension. This package init re-exports the public surface.
"""

from __future__ import annotations

from llenergymeasure.serving.lifecycle import (
    DEFAULT_HEALTH_PATH,
    allocate_free_port,
    await_ready,
    build_server_container_argv,
    default_server_log_path,
    launch_container_server,
    launch_process_server,
    server_container_name,
    shutdown,
)
from llenergymeasure.serving.transport import (
    PARTIAL_COMPLETION_ATTR,
    CompletionResult,
    HttpxTransport,
    RequestShape,
    require_httpx,
)
from llenergymeasure.serving.types import (
    ProbeRequest,
    ServerHandle,
    ServerLaunchError,
    ServerLifecycleError,
    ServerPlacement,
    ServerReadinessError,
    ServerTopologyError,
)

__all__ = [
    "DEFAULT_HEALTH_PATH",
    "PARTIAL_COMPLETION_ATTR",
    "CompletionResult",
    "HttpxTransport",
    "ProbeRequest",
    "RequestShape",
    "ServerHandle",
    "ServerLaunchError",
    "ServerLifecycleError",
    "ServerPlacement",
    "ServerReadinessError",
    "ServerTopologyError",
    "allocate_free_port",
    "await_ready",
    "build_server_container_argv",
    "default_server_log_path",
    "launch_container_server",
    "launch_process_server",
    "require_httpx",
    "server_container_name",
    "shutdown",
]
