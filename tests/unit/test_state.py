"""Tests for experiment state management."""

import pytest

from llm_bench.state.experiment_state import ExperimentState, StateManager


class TestExperimentState:
    """Tests for ExperimentState model."""

    def test_create_minimal(self):
        state = ExperimentState(experiment_id="exp_001")
        assert state.experiment_id == "exp_001"
        assert state.cycle_id == 0
        assert state.completed_runs == {}
        assert state.failed_runs == {}

    def test_create_full(self):
        state = ExperimentState(
            experiment_id="exp_001",
            cycle_id=3,
            total_runs=10,
            completed_runs={"config_a": "/path/a.json"},
            failed_runs={"config_b": "OOM error"},
        )
        assert state.cycle_id == 3
        assert state.total_runs == 10
        assert state.completed_count == 1
        assert state.failed_count == 1

    def test_pending_count(self):
        state = ExperimentState(
            experiment_id="exp_001",
            total_runs=10,
            completed_runs={"a": "p", "b": "p", "c": "p"},
            failed_runs={"d": "err"},
        )
        assert state.pending_count == 6

    def test_is_complete(self):
        state = ExperimentState(
            experiment_id="exp_001",
            total_runs=3,
            completed_runs={"a": "p", "b": "p"},
            failed_runs={"c": "err"},
        )
        assert state.is_complete is True

    def test_is_not_complete(self):
        state = ExperimentState(
            experiment_id="exp_001",
            total_runs=5,
            completed_runs={"a": "p"},
        )
        assert state.is_complete is False

    def test_mark_completed(self):
        state = ExperimentState(experiment_id="exp_001")
        state.mark_completed("config_a", "/path/result.json")
        assert "config_a" in state.completed_runs
        assert state.completed_runs["config_a"] == "/path/result.json"

    def test_mark_completed_removes_from_failed(self):
        state = ExperimentState(
            experiment_id="exp_001",
            failed_runs={"config_a": "previous error"},
        )
        state.mark_completed("config_a", "/path/result.json")
        assert "config_a" in state.completed_runs
        assert "config_a" not in state.failed_runs

    def test_mark_failed(self):
        state = ExperimentState(experiment_id="exp_001")
        state.mark_failed("config_a", "CUDA OOM")
        assert "config_a" in state.failed_runs
        assert state.failed_runs["config_a"] == "CUDA OOM"

    def test_is_pending(self):
        state = ExperimentState(
            experiment_id="exp_001",
            completed_runs={"config_a": "p"},
            failed_runs={"config_b": "err"},
        )
        assert state.is_pending("config_c") is True
        assert state.is_pending("config_a") is False
        assert state.is_pending("config_b") is False


class TestStateManager:
    """Tests for StateManager."""

    @pytest.fixture
    def state_manager(self, tmp_path):
        return StateManager(state_dir=tmp_path / "state")

    def test_create_and_load(self, state_manager):
        state = state_manager.create("exp_001", total_runs=5, cycle_id=1)
        assert state.experiment_id == "exp_001"
        assert state.total_runs == 5

        loaded = state_manager.load("exp_001")
        assert loaded is not None
        assert loaded.experiment_id == "exp_001"

    def test_load_nonexistent(self, state_manager):
        result = state_manager.load("nonexistent")
        assert result is None

    def test_save_updates_timestamp(self, state_manager):
        state = state_manager.create("exp_001")
        old_time = state.last_updated

        state.mark_completed("config_a", "/path")
        state_manager.save(state)

        loaded = state_manager.load("exp_001")
        assert loaded is not None
        assert loaded.last_updated >= old_time

    def test_delete(self, state_manager):
        state_manager.create("exp_001")
        assert state_manager.load("exp_001") is not None

        deleted = state_manager.delete("exp_001")
        assert deleted is True
        assert state_manager.load("exp_001") is None

    def test_delete_nonexistent(self, state_manager):
        deleted = state_manager.delete("nonexistent")
        assert deleted is False

    def test_list_experiments(self, state_manager):
        state_manager.create("exp_001")
        state_manager.create("exp_002")
        state_manager.create("exp_003")

        experiments = state_manager.list_experiments()
        assert len(experiments) == 3
        assert "exp_001" in experiments
        assert "exp_002" in experiments
        assert "exp_003" in experiments

    def test_atomic_save(self, state_manager):
        """Test that save is atomic (no partial writes)."""
        state = state_manager.create("exp_001", total_runs=10)

        # Simulate multiple updates
        for i in range(5):
            state.mark_completed(f"config_{i}", f"/path/{i}.json")
            state_manager.save(state)

        loaded = state_manager.load("exp_001")
        assert loaded is not None
        assert loaded.completed_count == 5
