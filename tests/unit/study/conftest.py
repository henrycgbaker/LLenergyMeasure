"""Shared fixtures and helpers for study/ tests."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock


def _make_mock_process(
    *,
    is_alive_after_join: bool = False,
    exitcode: int = 0,
    pid: int = 12345,
) -> MagicMock:
    """Factory for a mock Process with controllable lifecycle."""
    proc = MagicMock()
    proc.pid = pid
    proc.exitcode = exitcode

    def fake_join(timeout=None):
        # After join, is_alive reflects whether timed out
        pass

    proc.join.side_effect = fake_join
    proc.is_alive.return_value = is_alive_after_join
    return proc


def _make_mock_context(
    process: MagicMock,
    pipe_data: object = None,
    *,
    pipe_has_data: bool = True,
) -> MagicMock:
    """Build a mock multiprocessing context that returns a controlled process.

    pipe_data: the object that parent_conn.recv() will return.
    pipe_has_data: if False, parent_conn.poll() returns False (empty pipe).
    """
    ctx = MagicMock()
    ctx.Process.return_value = process

    parent_conn = MagicMock()
    child_conn = MagicMock()
    parent_conn.poll.return_value = pipe_has_data
    if pipe_data is not None:
        parent_conn.recv.return_value = pipe_data

    ctx.Pipe.return_value = (parent_conn, child_conn)

    # Queue: use a real in-process queue so the consumer thread works
    ctx.Queue.return_value = queue.SimpleQueue()

    return ctx
