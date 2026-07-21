"""Container-side dispatch assets (bind-mounted into the engine container).

Holds the entrypoint shell script and the runtime-dependency import probe
(:mod:`probe_imports`) the entrypoint invokes. These run INSIDE the dispatch
container against the bind-mounted package source; the probe is a real module
(lint/mypy covered, unit-tested) rather than an inline shell heredoc.
"""

from __future__ import annotations
