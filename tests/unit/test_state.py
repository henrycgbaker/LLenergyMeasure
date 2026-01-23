"""Tests for experiment state management."""

import pytest

from llenergymeasure.exceptions import InvalidStateTransitionError
from llenergymeasure.state.experiment_state import (
    EXPERIMENT_VALID_TRANSITIONS,
    ExperimentState,
    ExperimentStatus,
    StateManager,
)


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


class TestStateTransitions:
    """Tests for state machine transitions."""

    def test_valid_transition_initialised_to_running(self):
        state = ExperimentState(experiment_id="exp_001")
        assert state.status == ExperimentStatus.INITIALISED

        state.transition_to(ExperimentStatus.RUNNING)
        assert state.status == ExperimentStatus.RUNNING

    def test_valid_transition_running_to_completed(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.RUNNING)
        state.transition_to(ExperimentStatus.COMPLETED)
        assert state.status == ExperimentStatus.COMPLETED

    def test_valid_transition_running_to_failed_with_message(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.RUNNING)
        state.transition_to(
            ExperimentStatus.FAILED,
            error_message="CUDA OOM",
        )
        assert state.status == ExperimentStatus.FAILED
        assert state.error_message == "CUDA OOM"

    def test_valid_transition_running_to_interrupted(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.RUNNING)
        state.transition_to(
            ExperimentStatus.INTERRUPTED,
            error_message="Interrupted by user",
        )
        assert state.status == ExperimentStatus.INTERRUPTED
        assert state.error_message == "Interrupted by user"

    def test_valid_transition_completed_to_aggregated(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.COMPLETED)
        state.transition_to(ExperimentStatus.AGGREGATED)
        assert state.status == ExperimentStatus.AGGREGATED

    def test_valid_transition_failed_to_running(self):
        """Test retry from FAILED state."""
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.FAILED)
        state.transition_to(ExperimentStatus.RUNNING)
        assert state.status == ExperimentStatus.RUNNING

    def test_valid_transition_interrupted_to_running(self):
        """Test resume from INTERRUPTED state."""
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.INTERRUPTED)
        state.transition_to(ExperimentStatus.RUNNING)
        assert state.status == ExperimentStatus.RUNNING

    def test_invalid_transition_initialised_to_completed(self):
        state = ExperimentState(experiment_id="exp_001")
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            state.transition_to(ExperimentStatus.COMPLETED)

        assert exc_info.value.from_status == "initialised"
        assert exc_info.value.to_status == "completed"
        assert exc_info.value.entity == "experiment"

    def test_invalid_transition_aggregated_is_terminal(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.AGGREGATED)
        with pytest.raises(InvalidStateTransitionError):
            state.transition_to(ExperimentStatus.RUNNING)

    def test_invalid_transition_completed_to_running(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.COMPLETED)
        with pytest.raises(InvalidStateTransitionError):
            state.transition_to(ExperimentStatus.RUNNING)

    def test_transition_updates_timestamp(self):
        state = ExperimentState(experiment_id="exp_001")
        old_time = state.last_updated

        state.transition_to(ExperimentStatus.RUNNING)
        assert state.last_updated >= old_time

    def test_transition_without_validation(self):
        """Test that validate=False allows any transition."""
        state = ExperimentState(experiment_id="exp_001")
        # This would normally be invalid
        state.transition_to(ExperimentStatus.AGGREGATED, validate=False)
        assert state.status == ExperimentStatus.AGGREGATED

    def test_can_transition_to(self):
        state = ExperimentState(experiment_id="exp_001", status=ExperimentStatus.RUNNING)
        assert state.can_transition_to(ExperimentStatus.COMPLETED) is True
        assert state.can_transition_to(ExperimentStatus.FAILED) is True
        assert state.can_transition_to(ExperimentStatus.INTERRUPTED) is True
        assert state.can_transition_to(ExperimentStatus.AGGREGATED) is False
        assert state.can_transition_to(ExperimentStatus.INITIALISED) is False

    def test_valid_transitions_coverage(self):
        """Ensure all statuses have defined transitions."""
        for status in ExperimentStatus:
            assert status in EXPERIMENT_VALID_TRANSITIONS


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
