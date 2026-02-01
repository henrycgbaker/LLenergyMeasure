"""Unit tests for campaign manifest persistence and resume."""

from __future__ import annotations

from datetime import datetime

import pytest

from llenergymeasure.orchestration.manifest import (
    CampaignManifest,
    CampaignManifestEntry,
    ManifestManager,
)


def _make_entry(
    exp_id: str = "exp-001",
    status: str = "pending",
    **kwargs: object,
) -> CampaignManifestEntry:
    """Create a CampaignManifestEntry with sensible defaults."""
    defaults = {
        "exp_id": exp_id,
        "config_name": "test_config",
        "config_path": "configs/test.yaml",
        "config_hash": "abc123",
        "backend": "pytorch",
        "container": "pytorch",
        "cycle_index": 0,
        "status": status,
    }
    defaults.update(kwargs)
    return CampaignManifestEntry(**defaults)


def _make_manifest(
    entries: list[CampaignManifestEntry] | None = None,
    config_hash: str = "hash123",
) -> CampaignManifest:
    """Create a CampaignManifest with sensible defaults."""
    exps = entries or []
    now = datetime.now()
    return CampaignManifest(
        campaign_id="test-campaign",
        campaign_name="Test Campaign",
        created_at=now,
        updated_at=now,
        config_hash=config_hash,
        total_experiments=len(exps),
        experiments=exps,
    )


class TestCampaignManifestEntry:
    """Tests for CampaignManifestEntry model."""

    def test_manifest_entry_creation(self) -> None:
        """CampaignManifestEntry validates all required fields."""
        entry = _make_entry(
            exp_id="exp-abc",
            status="pending",
        )
        assert entry.exp_id == "exp-abc"
        assert entry.config_name == "test_config"
        assert entry.config_path == "configs/test.yaml"
        assert entry.config_hash == "abc123"
        assert entry.backend == "pytorch"
        assert entry.container == "pytorch"
        assert entry.cycle_index == 0
        assert entry.status == "pending"
        assert entry.result_path is None
        assert entry.started_at is None
        assert entry.completed_at is None
        assert entry.error is None
        assert entry.retry_count == 0


class TestCampaignManifest:
    """Tests for CampaignManifest model properties."""

    def test_manifest_status_tracking(self) -> None:
        """completed_count, pending_count, is_complete properties."""
        entries = [
            _make_entry("exp-1", status="completed"),
            _make_entry("exp-2", status="pending"),
            _make_entry("exp-3", status="failed"),
            _make_entry("exp-4", status="skipped"),
        ]
        manifest = _make_manifest(entries)

        assert manifest.completed_count == 1
        assert manifest.pending_count == 1
        assert manifest.failed_count == 1
        assert manifest.is_complete is False

    def test_manifest_is_complete_when_all_done(self) -> None:
        """is_complete is True when all experiments are completed or skipped."""
        entries = [
            _make_entry("exp-1", status="completed"),
            _make_entry("exp-2", status="completed"),
            _make_entry("exp-3", status="skipped"),
        ]
        manifest = _make_manifest(entries)
        assert manifest.is_complete is True

    def test_manifest_get_remaining(self) -> None:
        """get_remaining returns pending + failed entries."""
        entries = [
            _make_entry("exp-1", status="completed"),
            _make_entry("exp-2", status="pending"),
            _make_entry("exp-3", status="failed"),
            _make_entry("exp-4", status="skipped"),
        ]
        manifest = _make_manifest(entries)
        remaining = manifest.get_remaining()

        assert len(remaining) == 2
        ids = {e.exp_id for e in remaining}
        assert ids == {"exp-2", "exp-3"}

    def test_manifest_update_entry(self) -> None:
        """update_entry changes status on specific exp_id."""
        entries = [
            _make_entry("exp-1", status="pending"),
            _make_entry("exp-2", status="pending"),
        ]
        manifest = _make_manifest(entries)
        manifest.update_entry("exp-1", status="running")

        assert manifest.experiments[0].status == "running"
        assert manifest.experiments[1].status == "pending"

    def test_manifest_update_entry_not_found(self) -> None:
        """update_entry raises KeyError for unknown exp_id."""
        manifest = _make_manifest([_make_entry("exp-1")])
        with pytest.raises(KeyError, match="not found"):
            manifest.update_entry("nonexistent", status="running")

    def test_manifest_progress_fraction(self) -> None:
        """progress_fraction returns correct ratio."""
        entries = [
            _make_entry("exp-1", status="completed"),
            _make_entry("exp-2", status="completed"),
            _make_entry("exp-3", status="pending"),
            _make_entry("exp-4", status="pending"),
        ]
        manifest = _make_manifest(entries)
        assert manifest.progress_fraction == pytest.approx(0.5)


class TestManifestManager:
    """Tests for ManifestManager persistence operations."""

    def test_manifest_manager_save_load(self, tmp_path: object) -> None:
        """Round-trip persistence: save then load."""
        from pathlib import Path

        path = Path(str(tmp_path)) / "manifest.json"
        manager = ManifestManager(path)
        manifest = _make_manifest([_make_entry("exp-1", status="completed")])

        manager.save(manifest)
        loaded = manager.load()

        assert loaded is not None
        assert loaded.campaign_id == manifest.campaign_id
        assert loaded.campaign_name == manifest.campaign_name
        assert len(loaded.experiments) == 1
        assert loaded.experiments[0].exp_id == "exp-1"
        assert loaded.experiments[0].status == "completed"

    def test_manifest_manager_atomic_write(self, tmp_path: object) -> None:
        """Temp file renamed; no .tmp file left after save."""
        from pathlib import Path

        path = Path(str(tmp_path)) / "manifest.json"
        manager = ManifestManager(path)
        manifest = _make_manifest([_make_entry("exp-1")])

        manager.save(manifest)

        assert path.exists()
        tmp_file = path.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_manifest_manager_load_missing(self, tmp_path: object) -> None:
        """Returns None when file doesn't exist."""
        from pathlib import Path

        path = Path(str(tmp_path)) / "nonexistent.json"
        manager = ManifestManager(path)
        result = manager.load()
        assert result is None

    def test_manifest_manager_exists(self, tmp_path: object) -> None:
        """exists() reflects file presence."""
        from pathlib import Path

        path = Path(str(tmp_path)) / "manifest.json"
        manager = ManifestManager(path)

        assert not manager.exists()
        manager.save(_make_manifest())
        assert manager.exists()

    def test_manifest_config_hash_change_detection(self, tmp_path: object) -> None:
        """check_config_changed() detects when config hash differs."""
        from pathlib import Path

        path = Path(str(tmp_path)) / "manifest.json"
        manager = ManifestManager(path)
        manifest = _make_manifest(config_hash="original_hash")

        # Same hash - not changed
        assert not manager.check_config_changed(manifest, "original_hash")
        # Different hash - changed
        assert manager.check_config_changed(manifest, "new_hash")

    def test_manifest_manager_create_manifest(self, tmp_path: object) -> None:
        """Factory method creates manifest with proper timestamps."""
        from pathlib import Path

        path = Path(str(tmp_path)) / "manifest.json"
        manager = ManifestManager(path)
        entries = [_make_entry("exp-1"), _make_entry("exp-2")]

        manifest = manager.create_manifest(
            campaign_id="camp-1",
            campaign_name="Test",
            config_hash="hash456",
            experiments=entries,
        )

        assert manifest.campaign_id == "camp-1"
        assert manifest.total_experiments == 2
        assert manifest.created_at is not None
        assert manifest.updated_at is not None
