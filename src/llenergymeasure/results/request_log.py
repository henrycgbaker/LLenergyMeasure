"""Per-request log Parquet writer for server-mode measurement (requests.parquet).

One row per issued request per window: the client-observed request lifecycle
(issue / dispatch / first-token / completion timestamps and the per-token
receipt times), the client-side canonical output-token count (O8: llem's own
count of the streamed deltas is the J/token denominator), the engine's
self-reported usage as auxiliary provenance, the stream's finish reason (so a
length-truncation is distinguishable from a natural stop), and the D7
boundary-attribution flags (measurement-window vs ramp, drain-tail). SM12
derives TTFT / ITL percentiles, goodput, and SLO attainment from these rows
WITHOUT re-sampling.

Mirrors :mod:`llenergymeasure.harness.timeseries`: a locked columnar schema
written with pyarrow (a core dependency), with file-level identity metadata so
the artefact stays attributable if separated from its bundle. It lives in the
results layer (not harness) so :class:`~llenergymeasure.results.bundle.BundleWriter`
can own the write without a results -> harness import; the row shape
(:class:`RequestLogRow`) is a plain dataclass of scalars plus one list column, so
the layer boundary stays clean (the study layer builds the rows from harness
request records and passes them down).

Timestamps are the traffic issuer's ``time.monotonic`` basis - relative, not
wall-clock - so only their DIFFERENCES are meaningful; SM12 derives every latency
as a difference. The schema is locked: do not rename or retype a column without
folding the change into the bundle-format break (there is no per-artefact schema
version; Parquet stays self-describing under the single bundle_version).

CLIPPING SEMANTICS (read before deriving a denominator). Rows are ISSUE-partitioned
(a row lands in the window that owns its issue time) and per-row token
counts/times are receipt-UNCLIPPED: ``output_token_times`` is the request's whole
client receipt series, so it can include receipts outside the window's measured
span (a straddler's drain-tail tokens, or a ramp row's tokens). The AUTHORITATIVE
span-clipped denominator is ``result.json`` ``output_tokens`` (the window manager
clips receipts to the measured span). An alternative attribution is re-derivable
offline by clipping each row's ``output_token_times`` to the window's
``span_start`` / ``span_end`` (persisted as this file's key-value metadata); the
per-row ``client_output_tokens`` is NOT that clipped count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Status vocabulary for a request row. ``ok`` = the transport call returned;
#: ``error`` = it raised (record.error set); ``timeout`` = it never completed
#: (cancelled at the drain timeout under a stalled transport).
REQUEST_STATUS_OK = "ok"
REQUEST_STATUS_ERROR = "error"
REQUEST_STATUS_TIMEOUT = "timeout"


@dataclass
class RequestLogRow:
    """One issued request's row in a window's requests.parquet.

    Timestamps share the issuer's ``time.monotonic`` basis (relative seconds).
    ``output_token_times`` is the client-side canonical token receipt series (one
    entry per streamed content delta); ``client_output_tokens`` is its length -
    the count that feeds the J/token denominator (O8). ``server_prompt_tokens`` /
    ``server_completion_tokens`` are the engine's self-reported usage, auxiliary
    only (None when the engine reported none). ``finish_reason`` is the stream's
    terminal reason (e.g. ``"stop"`` vs ``"length"``), None for an error/timeout
    row and when the engine reported none; SM12 needs it to tell a natural stop
    from a length-truncation for goodput / SLO attainment. ``in_measurement_window`` /
    ``is_ramp`` / ``completed_in_drain`` are the D7 boundary attribution: a
    request issued in the level's prospective ramp (is_ramp) never counts toward
    the window's steady-state metrics; a request issued in-span but completing
    after span_end (completed_in_drain) keeps its full latency yet only its
    in-span tokens count (the two-policy separation).
    """

    request_index: int
    issued_at: float
    dispatched_at: float | None
    first_token_at: float | None
    completed_at: float | None
    ttft_ms: float | None
    e2e_latency_ms: float | None
    client_output_tokens: int
    server_prompt_tokens: int | None
    server_completion_tokens: int | None
    status: str
    finish_reason: str | None
    level_index: int
    window_index: int
    in_measurement_window: bool
    is_ramp: bool
    completed_in_drain: bool
    output_token_times: list[float] = field(default_factory=list)


def _requests_schema() -> Any:
    """Return the locked Parquet schema for the per-request log.

    Locked: do not change column names or types without folding it into the
    bundle-format break (Parquet is unversioned under the single bundle_version).
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("request_index", pa.int64()),
            pa.field("issued_at", pa.float64()),
            pa.field("dispatched_at", pa.float64()),
            pa.field("first_token_at", pa.float64()),
            pa.field("completed_at", pa.float64()),
            pa.field("ttft_ms", pa.float64()),
            pa.field("e2e_latency_ms", pa.float64()),
            pa.field("client_output_tokens", pa.int64()),
            pa.field("server_prompt_tokens", pa.int64()),
            pa.field("server_completion_tokens", pa.int64()),
            pa.field("status", pa.string()),
            pa.field("finish_reason", pa.string()),
            pa.field("level_index", pa.int32()),
            pa.field("window_index", pa.int32()),
            pa.field("in_measurement_window", pa.bool_()),
            pa.field("is_ramp", pa.bool_()),
            pa.field("completed_in_drain", pa.bool_()),
            pa.field("output_token_times", pa.list_(pa.float64())),
        ]
    )


def write_requests_parquet(
    rows: list[RequestLogRow],
    output_path: Path,
    *,
    experiment_id: str | None = None,
    declared_config_hash: str | None = None,
    span_start: float | None = None,
    span_end: float | None = None,
) -> Path:
    """Write a window's request rows to a Parquet file (schema locked).

    An empty ``rows`` still writes a schema-only Parquet, so a window that issued
    no requests lands a truthful (empty) log rather than a missing artefact. When
    ``experiment_id`` / ``declared_config_hash`` are given they are stored as
    Parquet file-level key-value metadata (not columns), mirroring the timeseries
    writer, so the log stays attributable if separated from its bundle dir.

    ``span_start`` / ``span_end`` are the window's measured monotonic span bounds,
    also stored as file-level metadata (the span is file-scoped, not row-scoped).
    They make the per-row (issue-partitioned, receipt-UNCLIPPED) token series
    re-clippable to the measured window offline, so an alternative attribution can
    be re-derived without re-running (M1); the authoritative span-clipped
    denominator remains ``result.json`` ``output_tokens``.

    Args:
        rows: The window's request rows, in issue order.
        output_path: Destination path for requests.parquet.
        experiment_id: Unique experiment identifier, stored as file metadata.
        declared_config_hash: Config hash for orphan attribution, file metadata.
        span_start: Window measured-span start (monotonic s), file metadata.
        span_end: Window measured-span end (monotonic s), file metadata.

    Returns:
        The output_path after writing.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = _requests_schema()
    identity = {
        key: str(value)
        for key, value in (
            ("experiment_id", experiment_id),
            ("declared_config_hash", declared_config_hash),
            ("span_start", span_start),
            ("span_end", span_end),
        )
        if value is not None
    }
    if identity:
        schema = schema.with_metadata(identity)

    columns: dict[str, list[Any]] = {field.name: [] for field in schema}
    for row in rows:
        columns["request_index"].append(row.request_index)
        columns["issued_at"].append(row.issued_at)
        columns["dispatched_at"].append(row.dispatched_at)
        columns["first_token_at"].append(row.first_token_at)
        columns["completed_at"].append(row.completed_at)
        columns["ttft_ms"].append(row.ttft_ms)
        columns["e2e_latency_ms"].append(row.e2e_latency_ms)
        columns["client_output_tokens"].append(row.client_output_tokens)
        columns["server_prompt_tokens"].append(row.server_prompt_tokens)
        columns["server_completion_tokens"].append(row.server_completion_tokens)
        columns["status"].append(row.status)
        columns["finish_reason"].append(row.finish_reason)
        columns["level_index"].append(row.level_index)
        columns["window_index"].append(row.window_index)
        columns["in_measurement_window"].append(row.in_measurement_window)
        columns["is_ramp"].append(row.is_ramp)
        columns["completed_in_drain"].append(row.completed_in_drain)
        columns["output_token_times"].append(list(row.output_token_times))

    table = pa.table(
        {f.name: pa.array(columns[f.name], type=f.type) for f in schema},
        schema=schema,
    )
    pq.write_table(table, output_path)
    return output_path


def read_requests_parquet(path: Path) -> list[dict[str, Any]]:
    """Read requests.parquet back into a list of per-row dicts (schema round-trip).

    The read-side counterpart used by tests and any consumer that wants the raw
    rows without a dataframe dependency.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path))
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows
