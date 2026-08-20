"""Tests for the container-ownership mechanics: naming, labels, cleanup, reaper."""

from __future__ import annotations

import os
import signal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.infra.docker.ownership import (
    cleanup_study_containers,
    generate_container_labels,
    generate_container_name,
    install_sigterm_bridge,
    reap_orphaned_containers,
    register_container_cleanup,
    require_study_id,
)
from llenergymeasure.utils.exceptions import StudyError

# ---------------------------------------------------------------------------
# require_study_id
# ---------------------------------------------------------------------------


class TestRequireStudyId:
    def test_returns_hash_unchanged(self) -> None:
        assert require_study_id("abcdef1234567890") == "abcdef1234567890"

    @pytest.mark.parametrize("missing", [None, "", "   ", "\t\n"])
    def test_missing_identity_raises(self, missing: str | None) -> None:
        with pytest.raises(StudyError) as exc_info:
            require_study_id(missing)

        message = str(exc_info.value)
        assert "study_design_hash" in message
        # The message must name the hazard, not just the missing field.
        assert "llem.study_id" in message

    def test_no_placeholder_identity_is_synthesised(self) -> None:
        """The old behaviour (fall back to a shared "unknown" id) is gone."""
        with pytest.raises(StudyError):
            require_study_id(None)


# ---------------------------------------------------------------------------
# generate_container_name
# ---------------------------------------------------------------------------


class TestGenerateContainerName:
    def test_standard_study_id(self) -> None:
        name = generate_container_name("abcdef1234567890", 1)
        assert name == "llem-abcdef12-0001"

    def test_zero_padded_index(self) -> None:
        name = generate_container_name("abcdef1234567890", 42)
        assert name == "llem-abcdef12-0042"

    def test_empty_study_id_is_refused(self) -> None:
        """No placeholder name: an unidentified study cannot own a container."""
        with pytest.raises(StudyError):
            generate_container_name("", 42)

    def test_short_study_id_used_as_is(self) -> None:
        name = generate_container_name("abc", 1)
        assert name == "llem-abc-0001"

    def test_large_index_zero_padded(self) -> None:
        name = generate_container_name("deadbeef12345678", 9999)
        assert name == "llem-deadbeef-9999"

    def test_index_zero(self) -> None:
        name = generate_container_name("abcdef1234567890", 0)
        assert name == "llem-abcdef12-0000"


# ---------------------------------------------------------------------------
# generate_container_labels
# ---------------------------------------------------------------------------


class TestGenerateContainerLabels:
    def test_returns_required_keys(self) -> None:
        labels = generate_container_labels("my-study-id")
        assert "llem.study_id" in labels
        assert "llem.parent_pid" in labels
        assert "llem.started_at" in labels

    def test_study_id_matches(self) -> None:
        labels = generate_container_labels("my-study-id")
        assert labels["llem.study_id"] == "my-study-id"

    def test_parent_pid_is_string_of_current_pid(self) -> None:
        labels = generate_container_labels("my-study-id")
        assert labels["llem.parent_pid"] == str(os.getpid())

    def test_started_at_is_iso8601(self) -> None:
        from datetime import datetime

        labels = generate_container_labels("my-study-id")
        # Should be parseable as a datetime with timezone
        dt = datetime.fromisoformat(labels["llem.started_at"])
        assert dt.tzinfo is not None

    def test_empty_study_id_is_refused(self) -> None:
        """An unlabelled-by-identity container would be un-scopeable; refuse it."""
        with pytest.raises(StudyError):
            generate_container_labels("")


# ---------------------------------------------------------------------------
# cleanup_study_containers
# ---------------------------------------------------------------------------


class TestCleanupStudyContainers:
    def test_calls_docker_ps_with_label_filter(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            cleanup_study_containers("test-study-id")

        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "docker" in cmd
        assert "ps" in cmd
        assert "label=llem.study_id=test-study-id" in " ".join(cmd)

    def test_stops_running_containers(self) -> None:
        ps_result = MagicMock(stdout="abc123\ndef456\n", returncode=0)
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result, stop_result]) as mock_run:
            cleanup_study_containers("test-study-id")

        # Should have called docker stop for each container ID
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 2
        stop_cmds = [" ".join(c[0][0]) for c in stop_calls]
        assert any("abc123" in cmd for cmd in stop_cmds)
        assert any("def456" in cmd for cmd in stop_cmds)

    def test_suppresses_all_exceptions(self) -> None:
        with patch("subprocess.run", side_effect=RuntimeError("Docker unavailable")):
            # Should not raise
            cleanup_study_containers("test-study-id")

    def test_skips_empty_container_ids(self) -> None:
        # docker ps output with blank lines
        ps_result = MagicMock(stdout="abc123\n\n  \n", returncode=0)
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run:
            cleanup_study_containers("test-study-id")

        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1

    def test_empty_study_id_issues_no_docker_command(self) -> None:
        """Never run an unscoped filter: it could reach containers we do not own.

        This path must stay silent (it is an atexit handler), so the refusal is
        a no-op plus a warning rather than an exception.
        """
        with patch("subprocess.run") as mock_run:
            cleanup_study_containers("")

        assert mock_run.call_count == 0

    def test_no_containers_running(self) -> None:
        ps_result = MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", return_value=ps_result) as mock_run:
            cleanup_study_containers("test-study-id")

        # Only the docker ps call, no docker stop calls
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# register_container_cleanup
# ---------------------------------------------------------------------------


class TestRegisterContainerCleanup:
    def test_registers_atexit_handler(self) -> None:
        with patch("atexit.register") as mock_register:
            register_container_cleanup("my-study")

        mock_register.assert_called_once_with(cleanup_study_containers, "my-study")

    def test_missing_identity_refuses_registration(self) -> None:
        with patch("atexit.register") as mock_register, pytest.raises(StudyError):
            register_container_cleanup("")

        assert mock_register.call_count == 0


# ---------------------------------------------------------------------------
# install_sigterm_bridge
# ---------------------------------------------------------------------------


class TestInstallSigtermBridge:
    def test_installs_handler_and_returns_original(self) -> None:
        original_handler = signal.getsignal(signal.SIGTERM)
        try:
            returned = install_sigterm_bridge()
            new_handler = signal.getsignal(signal.SIGTERM)

            assert returned is original_handler
            assert new_handler is not original_handler
            assert callable(new_handler)
        finally:
            # Restore to avoid affecting other tests
            signal.signal(signal.SIGTERM, original_handler)

    def test_installed_handler_calls_sys_exit(self) -> None:
        original_handler = signal.getsignal(signal.SIGTERM)
        try:
            install_sigterm_bridge()
            handler = signal.getsignal(signal.SIGTERM)

            with pytest.raises(SystemExit) as exc_info:
                handler(signal.SIGTERM, None)  # type: ignore[operator,misc]  # handler is int|Callable|None

            assert exc_info.value.code == 0
        finally:
            signal.signal(signal.SIGTERM, original_handler)

    def test_returns_none_on_value_error(self) -> None:
        with patch("signal.getsignal", side_effect=ValueError("not main thread")):
            result = install_sigterm_bridge()

        assert result is None


# ---------------------------------------------------------------------------
# reap_orphaned_containers
# ---------------------------------------------------------------------------


class TestReapOrphanedContainers:
    def _make_ps_result(self, lines: str) -> MagicMock:
        return MagicMock(stdout=lines, returncode=0)

    def test_stops_container_with_dead_parent_pid(self) -> None:
        ps_result = self._make_ps_result("abc123 99999999\n")
        stop_result = MagicMock(returncode=0)

        with (
            patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run,
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            count = reap_orphaned_containers()

        assert count == 1
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1
        assert "abc123" in " ".join(stop_calls[0][0][0])

    def test_skips_container_with_alive_parent_pid(self) -> None:
        ps_result = self._make_ps_result(f"abc123 {os.getpid()}\n")

        with (
            patch("subprocess.run", return_value=ps_result) as mock_run,
            patch("os.kill", return_value=None),
        ):
            count = reap_orphaned_containers()

        assert count == 0
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 0

    def test_skips_container_with_permission_error(self) -> None:
        ps_result = self._make_ps_result("abc123 1\n")

        with (
            patch("subprocess.run", return_value=ps_result) as mock_run,
            patch("os.kill", side_effect=PermissionError),
        ):
            count = reap_orphaned_containers()

        assert count == 0
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 0

    def test_skips_malformed_lines(self) -> None:
        # Lines missing pid field
        ps_result = self._make_ps_result("abc123\nxyz\n")

        with patch("subprocess.run", return_value=ps_result):
            count = reap_orphaned_containers()

        assert count == 0

    def test_handles_multiple_containers_mixed(self) -> None:
        """One orphan + one alive container."""
        lines = "dead111 11111111\nalive22 22222222\n"
        ps_result = self._make_ps_result(lines)
        stop_result = MagicMock(returncode=0)

        def kill_side_effect(pid: int, sig: int) -> None:
            if pid == 11111111:
                raise ProcessLookupError
            # 22222222 is "alive"

        with (
            patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run,
            patch("os.kill", side_effect=kill_side_effect),
        ):
            count = reap_orphaned_containers()

        assert count == 1
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1
        assert "dead111" in " ".join(stop_calls[0][0][0])

    def test_suppresses_all_exceptions(self) -> None:
        with patch("subprocess.run", side_effect=RuntimeError("Docker down")):
            # Should not raise
            count = reap_orphaned_containers()

        assert count == 0

    def test_invalid_pid_string_treated_as_dead(self) -> None:
        ps_result = self._make_ps_result("abc123 not-a-pid\n")
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result]):
            # ValueError from int("not-a-pid") should be treated as orphan
            count = reap_orphaned_containers()

        assert count == 1

    def test_empty_output_returns_zero(self) -> None:
        ps_result = self._make_ps_result("")

        with patch("subprocess.run", return_value=ps_result):
            count = reap_orphaned_containers()

        assert count == 0


# ---------------------------------------------------------------------------
# Concurrent-trial isolation
# ---------------------------------------------------------------------------


class _FakeDockerCLI:
    """An in-memory stand-in for the docker CLI the lifecycle helpers shell out to.

    Understands just enough to run the real cleanup and reaper code against a
    registry of labelled containers: ``docker ps`` with ``label=`` filters (bare
    key or key=value) in both the id-only and ``--format`` shapes, plus
    ``docker stop``. Anything else is a test bug, not a silent no-op.
    """

    def __init__(self) -> None:
        self.labels: dict[str, dict[str, str]] = {}
        self.running: set[str] = set()
        self.stopped: list[str] = []

    def add(self, container_id: str, *, study_id: str, parent_pid: int) -> None:
        self.labels[container_id] = {
            "llem.study_id": study_id,
            "llem.parent_pid": str(parent_pid),
        }
        self.running.add(container_id)

    def running_for(self, study_id: str) -> set[str]:
        return {c for c in self.running if self.labels[c]["llem.study_id"] == study_id}

    # subprocess.run replacement
    def run(self, cmd: list[str], **_kwargs: object) -> MagicMock:
        assert cmd[0] == "docker", f"unexpected command: {cmd}"
        if cmd[1] == "ps":
            return MagicMock(stdout=self._ps(cmd), stderr="", returncode=0)
        if cmd[1] == "stop":
            container_id = cmd[-1]
            self.running.discard(container_id)
            self.stopped.append(container_id)
            return MagicMock(stdout="", stderr="", returncode=0)
        raise AssertionError(f"unexpected docker subcommand: {cmd}")

    def _ps(self, cmd: list[str]) -> str:
        filters = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--filter"]
        selected = sorted(c for c in self.running if self._matches(c, filters))
        if "--format" in cmd:
            # The reaper's format string is '{{.ID}} {{.Label "llem.parent_pid"}}'.
            return "".join(f"{c} {self.labels[c]['llem.parent_pid']}\n" for c in selected)
        return "".join(f"{c}\n" for c in selected)

    def _matches(self, container_id: str, filters: list[str]) -> bool:
        labels = self.labels[container_id]
        for raw in filters:
            assert raw.startswith("label="), f"unsupported filter: {raw}"
            key, sep, value = raw[len("label=") :].partition("=")
            if key not in labels:
                return False
            if sep and labels[key] != value:
                return False
        return True


def _two_concurrent_trials() -> _FakeDockerCLI:
    """Two studies running side by side, each with its own containers.

    Trial A holds an experiment container and a server container; trial B holds
    an experiment container and a baseline container. Both processes are alive.
    """
    docker = _FakeDockerCLI()
    docker.add("a-experiment", study_id="study-aaaa", parent_pid=os.getpid())
    docker.add("a-server", study_id="study-aaaa", parent_pid=os.getpid())
    docker.add("b-experiment", study_id="study-bbbb", parent_pid=os.getpid())
    docker.add("b-baseline", study_id="study-bbbb", parent_pid=os.getpid())
    return docker


class TestConcurrentTrialIsolation:
    """One trial's cleanup must never touch another trial's containers.

    This is the hazard the mandatory study identity closes: while two studies
    could both fall back to a shared placeholder id, cleaning up after one of
    them stopped the other's containers mid-measurement.
    """

    def test_cleanup_stops_only_the_owning_studys_containers(self) -> None:
        docker = _two_concurrent_trials()

        with patch("subprocess.run", side_effect=docker.run):
            cleanup_study_containers("study-aaaa")

        assert docker.stopped == ["a-experiment", "a-server"]
        assert docker.running_for("study-bbbb") == {"b-experiment", "b-baseline"}

    def test_atexit_path_of_one_trial_leaves_the_other_untouched(self) -> None:
        """The registered handler carries its own study id, not a shared one."""
        docker = _two_concurrent_trials()
        registered: list[tuple[Any, tuple[Any, ...]]] = []

        with patch("atexit.register", side_effect=lambda fn, *a: registered.append((fn, a))):
            register_container_cleanup("study-aaaa")
            register_container_cleanup("study-bbbb")

        # Trial A's process exits first: only its handler fires.
        handler, args = registered[0]
        with patch("subprocess.run", side_effect=docker.run):
            handler(*args)

        assert docker.running_for("study-aaaa") == set()
        assert docker.running_for("study-bbbb") == {"b-experiment", "b-baseline"}

        # Trial B's own exit then cleans up exactly what is left.
        handler_b, args_b = registered[1]
        with patch("subprocess.run", side_effect=docker.run):
            handler_b(*args_b)

        assert docker.running == set()

    def test_reaper_spares_a_live_trials_containers(self) -> None:
        """The startup reaper is host-wide but PID-gated, so a live trial survives.

        A starting study reaps only containers whose launching process is gone;
        the concurrently running trial's containers stay up.
        """
        dead_pid = 4_242_424
        docker = _FakeDockerCLI()
        docker.add("orphan-experiment", study_id="study-orphan", parent_pid=dead_pid)
        docker.add("live-experiment", study_id="study-live", parent_pid=os.getpid())
        docker.add("live-server", study_id="study-live", parent_pid=os.getpid())

        def kill_side_effect(pid: int, _sig: int) -> None:
            if pid == dead_pid:
                raise ProcessLookupError

        with (
            patch("subprocess.run", side_effect=docker.run),
            patch("os.kill", side_effect=kill_side_effect),
        ):
            reaped = reap_orphaned_containers()

        assert reaped == 1
        assert docker.stopped == ["orphan-experiment"]
        assert docker.running_for("study-live") == {"live-experiment", "live-server"}
