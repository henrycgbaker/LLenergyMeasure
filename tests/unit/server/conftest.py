"""Shared fixtures for the server-lifecycle tests.

A ``launch_stub`` factory runs the tiny asyncio stub HTTP server
(``_stub_server.py``) through the real process-leg launcher, so the tests
exercise the actual ServerCapable mechanics. The fixture teardown shuts every
launched handle down, so a failing test can never leak a stub process.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from llenergymeasure.infra import server_lifecycle as sl
from llenergymeasure.infra.server_lifecycle import ServerHandle

STUB_SERVER = Path(__file__).parent / "_stub_server.py"


@pytest.fixture
def launch_stub(tmp_path: Path) -> Iterator[Callable[..., ServerHandle]]:
    launched: list[ServerHandle] = []

    def _launch(
        *,
        ignore_sigterm: bool = False,
        completions_ready_after: float = 0.0,
    ) -> ServerHandle:
        port = sl.allocate_free_port()
        argv = [sys.executable, str(STUB_SERVER), "--port", str(port)]
        if ignore_sigterm:
            argv.append("--ignore-sigterm")
        if completions_ready_after:
            argv += ["--completions-ready-after", str(completions_ready_after)]
        handle = sl.launch_process_server(
            argv,
            base_url=sl.base_url_for(port),
            engine="stub",
            log_path=tmp_path / f"stub-{port}.log",
        )
        launched.append(handle)
        return handle

    yield _launch

    for handle in launched:
        sl.shutdown(handle)
