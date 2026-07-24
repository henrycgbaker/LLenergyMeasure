"""Tests for the Parquet timeseries writer, focused on file-level identity metadata.

The writer tags the Parquet artefact with ``experiment_id`` and
``declared_config_hash`` as file-level key-value metadata (not columns) so the
sidecar stays attributable if separated from its result directory.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.timeseries import write_timeseries_parquet


def _sample(**overrides: object) -> PowerThermalSample:
    defaults = dict(
        timestamp=0.0,
        power_w=10.0,
        temperature_c=40.0,
        memory_used_mb=1024.0,
        memory_total_mb=40960.0,
        sm_utilisation=50.0,
    )
    defaults.update(overrides)
    return PowerThermalSample(**defaults)  # type: ignore[arg-type]


def test_identity_metadata_round_trips(tmp_path: Path) -> None:
    """experiment_id + declared_config_hash survive a write/read cycle as KV metadata."""
    path = tmp_path / "timeseries.parquet"
    write_timeseries_parquet(
        [_sample()],
        path,
        experiment_id="gpt2_20260715_120000",
        declared_config_hash="abcdef0123456789",
    )

    metadata = pq.read_table(path).schema.metadata
    assert metadata is not None
    assert metadata[b"experiment_id"] == b"gpt2_20260715_120000"
    assert metadata[b"declared_config_hash"] == b"abcdef0123456789"


def test_identity_metadata_on_empty_samples(tmp_path: Path) -> None:
    """Identity metadata is written even when there are no samples (empty parquet)."""
    path = tmp_path / "timeseries.parquet"
    write_timeseries_parquet(
        [],
        path,
        experiment_id="gpt2_20260715_120000",
        declared_config_hash="abcdef0123456789",
    )

    metadata = pq.read_table(path).schema.metadata
    assert metadata is not None
    assert metadata[b"experiment_id"] == b"gpt2_20260715_120000"
    assert metadata[b"declared_config_hash"] == b"abcdef0123456789"


def test_no_identity_metadata_when_omitted(tmp_path: Path) -> None:
    """Without identity args the writer carries no identity keys (unchanged default)."""
    path = tmp_path / "timeseries.parquet"
    write_timeseries_parquet([_sample()], path)

    metadata = pq.read_table(path).schema.metadata
    identity_keys = {b"experiment_id", b"declared_config_hash"}
    present = set(metadata.keys()) if metadata else set()
    assert not (identity_keys & present)
