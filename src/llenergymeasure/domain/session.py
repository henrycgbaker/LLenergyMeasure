"""Session-facts block shared by result.json and system.json.

A :class:`SessionBlock` is the per-window record of the measurement SESSION a
window belongs to. In server mode one session is one server lifetime (launch ->
warm up -> N windows -> drain); every window bundle of that session carries the
SAME block (identity by a stamped ``session_id`` field, never a directory or a
separate index artefact). Offline degenerates to the one-window case: a
fresh session id, ``window_count=1``, and all raw phase quantities null (the
offline pre-window phases are not instrumented yet).

The block holds RAW quantities only. Amortised / derived values (energy per
window after spreading the launch cost, etc.) are deferred to a later release and
deliberately absent here so the schema stays stable. Any phase whose energy could not be
measured stamps ``None`` (never ``0.0``), so a null reads as "unmeasured", not
"zero joules".
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionBlock(BaseModel):
    """Session facts stamped identically into every window bundle of a session.

    Home in BOTH result.json (via :attr:`ExperimentResult.session`) and
    system.json (via :attr:`EnvironmentSnapshot.session`), mirroring the
    runner-provenance dual-serialisation. The persistence layer treats it as an
    opaque data block - it never learns that sessions exist.
    """

    model_config = {"extra": "ignore"}

    session_id: str = Field(
        ...,
        description="uuid4 hex identifying one session realisation. The parent linkage "
        "between a session's sibling window bundles: siblings share this value, which is "
        "how a session's windows are rediscovered (a field, never a directory or DB).",
    )
    window_count: int = Field(
        ...,
        description="Total measured windows in the session. 1 for an offline experiment "
        "(the degenerate one-window session).",
    )
    level_count: int | None = Field(
        default=None,
        description="Number of rate levels the session drove (server mode). None for offline.",
    )
    launch_duration_s: float | None = Field(
        default=None,
        description="Wall-clock seconds from server launch to readiness (model load rides "
        "inside this one phase). None outside server mode.",
    )
    launch_energy_j: float | None = Field(
        default=None,
        description="GPU energy (J) measured over the launch-to-ready phase, or None when "
        "unmeasured (never 0.0).",
    )
    warmup_total_duration_s: float | None = Field(
        default=None,
        description="Sum of the per-level warmup durations across the session. None outside "
        "server mode.",
    )
    warmup_total_energy_j: float | None = Field(
        default=None,
        description="Sum of the per-level warmup GPU energies (J) across the session, or None "
        "when unmeasured (never 0.0).",
    )
    drain_duration_s: float | None = Field(
        default=None,
        description="Wall-clock seconds of the session drain/teardown phase. None when the "
        "session did not close cleanly (e.g. interrupted).",
    )
    drain_energy_j: float | None = Field(
        default=None,
        description="GPU energy (J) measured over the session drain phase, or None when "
        "unmeasured or the session did not close cleanly (never 0.0).",
    )
