"""Docker dispatch internals, decomposed by concern.

The public surface is the :class:`~llenergymeasure.infra.docker_runner.DockerRunner`
facade (kept at ``llenergymeasure.infra.docker_runner`` so consumers import it
unchanged). This package holds the concerns that facade composes:

- :mod:`command` - the single home for ``docker run`` argv: one shared core plus the
  three container shapes (offline experiment dispatch, idle baseline, engine server)
  as parameterisations of it. Pure builders, unit-testable without docker.
- :mod:`lifecycle` - container process execution: the guarded image pull (one image
  or several concurrently), launch (detach-capable), block-until-exit wait, and the
  stdout-silence watchdog isolated inside the wait path.
- :mod:`exchange` - exchange-dir lifecycle, result read, and the artefact rescue sweep.
- :mod:`diagnostics` - container failure classification (error payload, log tail).
- :mod:`ownership` - who owns a container and the machinery that reclaims it:
  naming, ownership labels, the atexit net, the SIGTERM bridge, the orphan reaper.
  Mechanics only; whether to install the net is the caller's decision.
"""

from __future__ import annotations
