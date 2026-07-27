"""Unit tests for domain/experiment.py - hashing, StudySummary, edge cases."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    StudySummary,
    compute_declared_config_hash,
)
from tests.conftest import (
    make_config,
    make_result,
)

# ---------------------------------------------------------------------------
# TestAggregationMetadata
# ---------------------------------------------------------------------------


class TestAggregationMetadata:
    """AggregationMetadata defaults."""

    def test_defaults(self):
        am = AggregationMetadata(num_processes=2)
        assert am.method == "sum_energy_avg_throughput"
        assert am.warnings == []
        assert am.temporal_overlap_verified is False
        assert am.gpu_attribution_verified is False

    def test_with_warnings(self):
        am = AggregationMetadata(num_processes=1, warnings=["something off"])
        assert len(am.warnings) == 1

    def test_verification_flags(self):
        am = AggregationMetadata(
            num_processes=4,
            temporal_overlap_verified=True,
            gpu_attribution_verified=True,
        )
        assert am.temporal_overlap_verified is True
        assert am.gpu_attribution_verified is True


# ---------------------------------------------------------------------------
# TestStudySummary
# ---------------------------------------------------------------------------


class TestStudySummary:
    """StudySummary optional fields and defaults."""

    def test_unique_configurations_default_none(self):
        ss = StudySummary(total_experiments=5)
        assert ss.unique_configurations is None

    def test_warnings_default_empty(self):
        ss = StudySummary(total_experiments=3)
        assert ss.warnings == []


# ---------------------------------------------------------------------------
# TestExperimentResultEdgeCases
# ---------------------------------------------------------------------------


class TestExperimentResultEdgeCases:
    """Properties and edge cases for ExperimentResult."""

    def test_tokens_per_joule_zero_energy(self):
        r = make_result(total_energy_j=0.0)
        assert r.tokens_per_joule == 0.0

    def test_tokens_per_joule_small_energy(self):
        r = make_result(total_energy_j=1e-10, total_tokens=1000)
        # Should produce a large number without overflow
        assert r.tokens_per_joule == pytest.approx(1000 / 1e-10)
        assert r.tokens_per_joule > 0

    def test_duration_sec_subsecond(self):
        start = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
        r = make_result(start_time=start, end_time=end)
        assert r.duration_sec == pytest.approx(0.5)

    def test_duration_sec_zero_when_same_time(self):
        t = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        r = make_result(start_time=t, end_time=t)
        assert r.duration_sec == 0.0

    def test_frozen_and_extra_forbid(self):
        r = make_result()
        with pytest.raises(ValidationError):
            r.total_tokens = 9999

    def test_llenergymeasure_version_default_none(self):
        r = make_result()
        assert r.llenergymeasure_version is None

    def test_llenergymeasure_version_set(self):
        r = make_result(llenergymeasure_version="0.9.0")
        assert r.llenergymeasure_version == "0.9.0"


# ---------------------------------------------------------------------------
# TestMeasurementConfigHash
# ---------------------------------------------------------------------------


class TestMeasurementConfigHash:
    """compute_declared_config_hash() determinism and shape."""

    def test_hash_length(self):
        config = make_config()
        h = compute_declared_config_hash(config)
        assert len(h) == 16

    def test_deterministic(self):
        config = make_config()
        h1 = compute_declared_config_hash(config)
        h2 = compute_declared_config_hash(config)
        assert h1 == h2

    def test_different_engines_different_hash(self):
        h1 = compute_declared_config_hash(make_config(engine="transformers"))
        h2 = compute_declared_config_hash(make_config(engine="vllm"))
        assert h1 != h2

    def test_hash_is_string(self):
        h = compute_declared_config_hash(make_config())
        assert isinstance(h, str)

    def test_latency_profiling_default_off(self):
        config = make_config()
        assert config.measurement.latency_profiling is False

    def test_latency_profiling_distinct_hash(self):
        """Toggling latency_profiling must change the declared config hash."""
        h_off = compute_declared_config_hash(make_config(latency_profiling=False))
        h_on = compute_declared_config_hash(make_config(latency_profiling=True))
        assert h_off != h_on

    def test_serving_mode_required_no_default(self):
        """serving_mode is required with no default: omitting it fails loudly."""
        from llenergymeasure.config.models import ExperimentConfig

        with pytest.raises((ValidationError, ValueError), match="serving_mode is required"):
            ExperimentConfig(task={"model": "gpt2"}, engine="transformers")  # type: ignore[call-arg]

    def test_serving_mode_distinct_hash(self):
        """serving_mode is an identity axis: offline and server hash differently.

        Both sides use a server-capable engine (vllm) so serving_mode is the axis
        under test; transformers server mode is gated off (E5 fast-follow).
        """
        h_offline = compute_declared_config_hash(make_config(engine="vllm", serving_mode="offline"))
        h_server = compute_declared_config_hash(
            make_config(
                engine="vllm",
                serving_mode="server",
                server={"traffic": {"rate": 10, "window_seconds": 60}},
            )
        )
        assert h_offline != h_server
