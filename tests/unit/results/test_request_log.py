"""Unit tests for the server-mode per-request log Parquet writer (SM11).

Host-only, no GPU: exercises the locked requests.parquet schema and its
write/read round-trip, the empty (schema-only) log, and the file-level identity
metadata. The row-building / attribution logic is covered in
tests/unit/study/test_server_session.py (it consumes harness request records).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from llenergymeasure.results.request_log import (
    REQUEST_STATUS_OK,
    RequestLogRow,
    read_requests_parquet,
    write_requests_parquet,
)

_SCHEMA_COLUMNS = {
    "request_index",
    "issued_at",
    "dispatched_at",
    "first_token_at",
    "completed_at",
    "ttft_ms",
    "e2e_latency_ms",
    "client_output_tokens",
    "server_prompt_tokens",
    "server_completion_tokens",
    "status",
    "finish_reason",
    "level_index",
    "window_index",
    "in_measurement_window",
    "is_ramp",
    "completed_in_drain",
    "output_token_times",
}


def _row(**overrides: object) -> RequestLogRow:
    defaults: dict = {
        "request_index": 0,
        "issued_at": 100.0,
        "dispatched_at": 100.01,
        "first_token_at": 100.1,
        "completed_at": 100.5,
        "ttft_ms": 100.0,
        "e2e_latency_ms": 500.0,
        "client_output_tokens": 3,
        "server_prompt_tokens": 7,
        "server_completion_tokens": 3,
        "status": REQUEST_STATUS_OK,
        "finish_reason": "stop",
        "level_index": 0,
        "window_index": 0,
        "in_measurement_window": True,
        "is_ramp": False,
        "completed_in_drain": False,
        "output_token_times": [100.1, 100.3, 100.5],
    }
    defaults.update(overrides)
    return RequestLogRow(**defaults)  # type: ignore[arg-type]


def test_empty_log_writes_schema_only_parquet(tmp_path: Path) -> None:
    """An empty rows list still lands a Parquet carrying the full locked schema."""
    path = write_requests_parquet([], tmp_path / "requests.parquet")
    assert path.exists()
    table = pq.read_table(path)
    assert set(table.column_names) == _SCHEMA_COLUMNS
    assert table.num_rows == 0


def test_round_trip_preserves_every_column(tmp_path: Path) -> None:
    """write + read back returns each row's fields, including the token-time list."""
    rows = [
        _row(request_index=0, output_token_times=[100.1, 100.3], finish_reason="stop"),
        _row(
            request_index=1,
            is_ramp=True,
            in_measurement_window=False,
            output_token_times=[],
            finish_reason="length",
        ),
    ]
    write_requests_parquet(rows, tmp_path / "requests.parquet")
    read_back = read_requests_parquet(tmp_path / "requests.parquet")

    assert [r["request_index"] for r in read_back] == [0, 1]
    assert read_back[0]["output_token_times"] == [100.1, 100.3]
    assert read_back[1]["output_token_times"] == []
    assert read_back[1]["is_ramp"] is True
    assert read_back[1]["in_measurement_window"] is False
    assert read_back[0]["status"] == REQUEST_STATUS_OK
    # finish_reason distinguishes a natural stop from a length-truncation (SM12).
    assert read_back[0]["finish_reason"] == "stop"
    assert read_back[1]["finish_reason"] == "length"


def test_null_auxiliary_and_timing_fields_round_trip(tmp_path: Path) -> None:
    """Nullable columns (server usage, timings) survive as None, never a false 0."""
    row = _row(
        first_token_at=None,
        ttft_ms=None,
        server_prompt_tokens=None,
        server_completion_tokens=None,
        finish_reason=None,
        client_output_tokens=0,
        output_token_times=[],
    )
    write_requests_parquet([row], tmp_path / "requests.parquet")
    read_back = read_requests_parquet(tmp_path / "requests.parquet")[0]
    assert read_back["first_token_at"] is None
    assert read_back["ttft_ms"] is None
    assert read_back["server_prompt_tokens"] is None
    assert read_back["server_completion_tokens"] is None
    assert read_back["finish_reason"] is None
    assert read_back["client_output_tokens"] == 0


def test_identity_and_span_metadata_stored_as_file_kv(tmp_path: Path) -> None:
    """Identity + the window span ride as Parquet file metadata, not columns (M1)."""
    write_requests_parquet(
        [_row()],
        tmp_path / "requests.parquet",
        experiment_id="server-abc-c1-L0-W0",
        declared_config_hash="abc123",
        span_start=1030.0,
        span_end=1040.0,
    )
    meta = pq.read_table(tmp_path / "requests.parquet").schema.metadata or {}
    assert meta.get(b"experiment_id") == b"server-abc-c1-L0-W0"
    assert meta.get(b"declared_config_hash") == b"abc123"
    # The span bounds make the receipt-unclipped rows re-clippable offline (M1).
    assert float(meta[b"span_start"]) == 1030.0
    assert float(meta[b"span_end"]) == 1040.0


def test_rows_are_receipt_unclipped_and_reclippable_to_span(tmp_path: Path) -> None:
    """Per-row token times are unclipped; clipping to the file span re-derives the count."""
    span_start, span_end = 10.0, 20.0
    # A straddler: 3 receipts in-span, 2 past span_end (drain tail).
    row = _row(
        request_index=0,
        output_token_times=[11.0, 15.0, 19.0, 22.0, 25.0],
        client_output_tokens=5,
    )
    write_requests_parquet(
        [row], tmp_path / "requests.parquet", span_start=span_start, span_end=span_end
    )
    read_back = read_requests_parquet(tmp_path / "requests.parquet")[0]
    meta = pq.read_table(tmp_path / "requests.parquet").schema.metadata
    lo, hi = float(meta[b"span_start"]), float(meta[b"span_end"])
    # Unclipped per-row count is the whole series...
    assert read_back["client_output_tokens"] == 5
    # ...but clipping to the persisted span re-derives the in-span (denominator) count.
    in_span = [t for t in read_back["output_token_times"] if lo <= t <= hi]
    assert len(in_span) == 3
