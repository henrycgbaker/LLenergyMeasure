"""Tests for the container-ownership mechanics: naming, labels, cleanup, reaper."""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
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
    def test_calls_docker_ps_with_label_filter(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            cleanup_study_containers("test-study-id", tmp_path)

        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "docker" in cmd
        assert "ps" in cmd
        assert "label=llem.study_id=test-study-id" in " ".join(cmd)

    def test_stops_running_containers(self, tmp_path: Path) -> None:
        docker = _FakeDockerCLI()
        docker.add("abc123", study_id="test-study-id", parent_pid=os.getpid())
        docker.add("def456", study_id="test-study-id", parent_pid=os.getpid())

        with patch("subprocess.run", side_effect=docker.run):
            cleanup_study_containers("test-study-id", tmp_path)

        assert docker.stopped == ["abc123", "def456"]

    def test_suppresses_all_exceptions(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=RuntimeError("Docker unavailable")):
            # Should not raise
            cleanup_study_containers("test-study-id", tmp_path)

    def test_skips_empty_container_ids(self, tmp_path: Path) -> None:
        # docker ps output with blank lines
        ps_result = MagicMock(stdout="abc123\n\n  \n", returncode=0)
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run:
            cleanup_study_containers("test-study-id", tmp_path)

        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1

    def test_empty_study_id_issues_no_docker_command(self, tmp_path: Path) -> None:
        """Never run an unscoped filter: it could reach containers we do not own.

        This path must stay silent (it is an atexit handler), so the refusal is
        a no-op plus a warning rather than an exception.
        """
        with patch("subprocess.run") as mock_run:
            cleanup_study_containers("", tmp_path)

        assert mock_run.call_count == 0

    def test_no_containers_running(self, tmp_path: Path) -> None:
        ps_result = MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", return_value=ps_result) as mock_run:
            cleanup_study_containers("test-study-id", tmp_path)

        # Only the docker ps call, no docker stop calls
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# register_container_cleanup
# ---------------------------------------------------------------------------


class TestRegisterContainerCleanup:
    def test_registers_atexit_handler(self, tmp_path: Path) -> None:
        with patch("atexit.register") as mock_register:
            register_container_cleanup("my-study", tmp_path)

        mock_register.assert_called_once_with(cleanup_study_containers, "my-study", tmp_path)

    def test_missing_identity_refuses_registration(self, tmp_path: Path) -> None:
        with patch("atexit.register") as mock_register, pytest.raises(StudyError):
            register_container_cleanup("", tmp_path)

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
    """An in-memory stand-in for the docker CLI these helpers shell out to.

    Understands just enough to run the real cleanup and reaper code against a
    registry of labelled containers: ``docker ps`` with ``label=`` filters (bare
    key or key=value) in both the id-only and ``--format`` shapes, plus
    ``docker stop``, ``docker logs``, ``docker container inspect`` and
    ``docker rm``. Anything else is a test bug, not a silent no-op.

    Auto-removal is modelled, because it is what separates the two container
    shapes the cleanup meets: a container added with ``auto_remove=True`` (the
    ``--rm`` shapes) vanishes the moment it is stopped, exactly as the daemon
    would reap it, while one added without it survives as an exited container.
    """

    def __init__(self) -> None:
        self.labels: dict[str, dict[str, str]] = {}
        self.running: set[str] = set()
        self.exists: set[str] = set()
        self.auto_remove: set[str] = set()
        self.logs: dict[str, str] = {}
        self.stopped: list[str] = []
        self.removed: list[str] = []

    def add(
        self,
        container_id: str,
        *,
        study_id: str,
        parent_pid: int,
        auto_remove: bool = False,
        logs: str = "",
    ) -> None:
        self.labels[container_id] = {
            "llem.study_id": study_id,
            "llem.parent_pid": str(parent_pid),
        }
        self.running.add(container_id)
        self.exists.add(container_id)
        self.logs[container_id] = logs
        if auto_remove:
            self.auto_remove.add(container_id)

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
            if container_id in self.auto_remove:
                self.exists.discard(container_id)
            return MagicMock(stdout="", stderr="", returncode=0)
        if cmd[1] == "logs":
            container_id = cmd[-1]
            if container_id not in self.exists:
                return MagicMock(stdout="", stderr="No such container", returncode=1)
            return MagicMock(stdout=self.logs[container_id], stderr="", returncode=0)
        if cmd[1:3] == ["container", "inspect"]:
            present = cmd[-1] in self.exists
            return MagicMock(stdout="", stderr="", returncode=0 if present else 1)
        if cmd[1] == "rm":
            container_id = cmd[-1]
            if container_id not in self.exists:
                return MagicMock(stdout="", stderr="No such container", returncode=1)
            self.exists.discard(container_id)
            self.removed.append(container_id)
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

    def test_cleanup_stops_only_the_owning_studys_containers(self, tmp_path: Path) -> None:
        docker = _two_concurrent_trials()

        with patch("subprocess.run", side_effect=docker.run):
            cleanup_study_containers("study-aaaa", tmp_path)

        assert docker.stopped == ["a-experiment", "a-server"]
        assert docker.running_for("study-bbbb") == {"b-experiment", "b-baseline"}

    def test_atexit_path_of_one_trial_leaves_the_other_untouched(self, tmp_path: Path) -> None:
        """The registered handler carries its own study id, not a shared one."""
        docker = _two_concurrent_trials()
        registered: list[tuple[Any, tuple[Any, ...]]] = []

        with patch("atexit.register", side_effect=lambda fn, *a: registered.append((fn, a))):
            register_container_cleanup("study-aaaa", tmp_path / "a")
            register_container_cleanup("study-bbbb", tmp_path / "b")

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


# ---------------------------------------------------------------------------
# Reclaiming containers at exit: stop, keep the logs, then remove
# ---------------------------------------------------------------------------


class TestAtexitReclaim:
    """Stopping is not the end of the job for the container shape that is not --rm.

    The engine-server container is launched deliberately WITHOUT ``--rm`` so a
    crash-on-startup survives for its logs to be read. Stopping it therefore
    leaves an exited container on the host that no code will ever look at again.
    The atexit net removes it - but only once its log tail is safely on disk,
    because a stray container is untidy and fixable by hand whereas discarding
    the last record of why a study died is not.
    """

    def test_stopped_container_is_removed_once_its_log_tail_is_kept(self, tmp_path: Path) -> None:
        docker = _FakeDockerCLI()
        docker.add(
            "server-1",
            study_id="study-aaaa",
            parent_pid=os.getpid(),
            logs="engine ready\nfatal: CUDA out of memory\n",
        )

        with patch("subprocess.run", side_effect=docker.run):
            cleanup_study_containers("study-aaaa", tmp_path / "failed-runs")

        assert docker.stopped == ["server-1"]
        assert docker.removed == ["server-1"]
        persisted = list((tmp_path / "failed-runs").glob("*.log"))
        assert len(persisted) == 1
        assert "server-1" in persisted[0].name
        assert "CUDA out of memory" in persisted[0].read_text()

    def test_container_with_no_output_is_still_removed(self, tmp_path: Path) -> None:
        """An empty log is a persisted fact, not a persistence failure."""
        docker = _FakeDockerCLI()
        docker.add("server-1", study_id="study-aaaa", parent_pid=os.getpid(), logs="")

        with patch("subprocess.run", side_effect=docker.run):
            cleanup_study_containers("study-aaaa", tmp_path / "failed-runs")

        assert docker.removed == ["server-1"]
        persisted = (tmp_path / "failed-runs" / "abandoned-container-server-1.log").read_text()
        assert persisted.strip() != ""

    def test_unwritable_log_dir_leaves_the_container_in_place(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Crash evidence beats tidiness: no log written means no removal."""
        blocked = tmp_path / "blocked"
        blocked.write_text("this is a file, so it cannot become the log directory")

        docker = _FakeDockerCLI()
        docker.add("server-1", study_id="study-aaaa", parent_pid=os.getpid(), logs="boom\n")

        with (
            patch("subprocess.run", side_effect=docker.run),
            caplog.at_level(logging.WARNING),
        ):
            cleanup_study_containers("study-aaaa", blocked / "failed-runs")

        assert docker.stopped == ["server-1"]
        assert docker.removed == []
        assert "server-1" in docker.exists
        assert "server-1" in caplog.text

    def test_unreadable_logs_leave_an_existing_container_in_place(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A container that is still there but will not talk is kept, and named."""
        docker = _FakeDockerCLI()
        docker.add("server-1", study_id="study-aaaa", parent_pid=os.getpid())

        def run(cmd: list[str], **kwargs: object) -> MagicMock:
            if cmd[1] == "logs":
                raise OSError("docker daemon went away")
            return docker.run(cmd, **kwargs)

        with patch("subprocess.run", side_effect=run), caplog.at_level(logging.WARNING):
            cleanup_study_containers("study-aaaa", tmp_path / "failed-runs")

        assert docker.removed == []
        assert "server-1" in docker.exists
        assert "server-1" in caplog.text
        assert not (tmp_path / "failed-runs").exists()

    def test_rm_containers_are_unaffected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The --rm shapes are reaped by docker on stop; nothing else happens.

        No log file is written (there is no container left to read), no removal
        is attempted, and - crucially - no warning is emitted: this is the normal
        outcome for an experiment or baseline container, not a problem.
        """
        docker = _FakeDockerCLI()
        docker.add(
            "experiment-1",
            study_id="study-aaaa",
            parent_pid=os.getpid(),
            auto_remove=True,
            logs="never read\n",
        )

        with (
            patch("subprocess.run", side_effect=docker.run),
            caplog.at_level(logging.WARNING),
        ):
            cleanup_study_containers("study-aaaa", tmp_path / "failed-runs")

        assert docker.stopped == ["experiment-1"]
        assert docker.removed == []
        assert docker.exists == set()
        assert not (tmp_path / "failed-runs").exists()
        assert caplog.text == ""

    def test_mixed_shapes_each_get_their_own_treatment(self, tmp_path: Path) -> None:
        """One study, both shapes: the --rm one vanishes, the server one is kept then removed."""
        docker = _FakeDockerCLI()
        docker.add(
            "experiment-1",
            study_id="study-aaaa",
            parent_pid=os.getpid(),
            auto_remove=True,
        )
        docker.add("server-1", study_id="study-aaaa", parent_pid=os.getpid(), logs="tail\n")

        with patch("subprocess.run", side_effect=docker.run):
            cleanup_study_containers("study-aaaa", tmp_path / "failed-runs")

        assert sorted(docker.stopped) == ["experiment-1", "server-1"]
        assert docker.removed == ["server-1"]
        assert docker.exists == set()
        persisted = [p.name for p in (tmp_path / "failed-runs").glob("*.log")]
        assert persisted == ["abandoned-container-server-1.log"]
