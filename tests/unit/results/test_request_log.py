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
        _row(request_index=0, output_token_times=[100.1, 100.3]),
        _row(request_index=1, is_ramp=True, in_measurement_window=False, output_token_times=[]),
    ]
    write_requests_parquet(rows, tmp_path / "requests.parquet")
    read_back = read_requests_parquet(tmp_path / "requests.parquet")

    assert [r["request_index"] for r in read_back] == [0, 1]
    assert read_back[0]["output_token_times"] == [100.1, 100.3]
    assert read_back[1]["output_token_times"] == []
    assert read_back[1]["is_ramp"] is True
    assert read_back[1]["in_measurement_window"] is False
    assert read_back[0]["status"] == REQUEST_STATUS_OK


def test_null_auxiliary_and_timing_fields_round_trip(tmp_path: Path) -> None:
    """Nullable columns (server usage, timings) survive as None, never a false 0."""
    row = _row(
        first_token_at=None,
        ttft_ms=None,
        server_prompt_tokens=None,
        server_completion_tokens=None,
        client_output_tokens=0,
        output_token_times=[],
    )
    write_requests_parquet([row], tmp_path / "requests.parquet")
    read_back = read_requests_parquet(tmp_path / "requests.parquet")[0]
    assert read_back["first_token_at"] is None
    assert read_back["ttft_ms"] is None
    assert read_back["server_prompt_tokens"] is None
    assert read_back["server_completion_tokens"] is None
    assert read_back["client_output_tokens"] == 0


def test_identity_metadata_is_stored_as_file_kv(tmp_path: Path) -> None:
    """experiment_id / declared_config_hash ride as Parquet file metadata, not columns."""
    write_requests_parquet(
        [_row()],
        tmp_path / "requests.parquet",
        experiment_id="server-abc-c1-L0-W0",
        declared_config_hash="abc123",
    )
    meta = pq.read_table(tmp_path / "requests.parquet").schema.metadata or {}
    assert meta.get(b"experiment_id") == b"server-abc-c1-L0-W0"
    assert meta.get(b"declared_config_hash") == b"abc123"
