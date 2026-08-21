"""A stand-in for the docker CLI, for tests of the guarded image pull.

``infra.docker.lifecycle`` reaches the daemon through one ``subprocess.run``, and
every pull it issues is guarded by a local ``docker image inspect``. Testing that
guard needs a fake that answers BOTH subcommands consistently: an image must read
as absent before its pull and as present afterwards, or the guard cannot be
distinguished from a fake that simply always says "absent".

Both the container-layer tests (``tests/unit/docker/test_image_ensure.py``) and
the study-layer tests that drive the same pull one level up
(``tests/unit/study/test_image_prep.py``) need exactly that, so the fake lives
here rather than once per file - two copies of a stateful fake are two chances
for one of them to stop resembling the daemon.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from subprocess import CompletedProcess

__all__ = ["fake_docker_pull_cli"]


def fake_docker_pull_cli(
    pull: Callable[[str], CompletedProcess[bytes]],
    *,
    present: Iterable[str] = (),
    inspect_stdout: bytes = b"[]",
) -> tuple[Callable[..., CompletedProcess[bytes]], list[list[str]]]:
    """Return a ``subprocess.run`` stand-in plus the argv log it records.

    Understands the two subcommands the pull path issues:

    - ``docker image inspect IMAGE`` - exit 0 with *inspect_stdout* once the
      image is present, exit 1 otherwise, so the guard answers truthfully both
      before and after a pull;
    - ``docker pull IMAGE`` - delegated to *pull*, which returns the result (or
      raises ``TimeoutExpired``). A successful pull marks the image present, so a
      following inspect reports it.

    Anything else is an assertion failure rather than a silent no-op: a test that
    reaches an unexpected docker subcommand is a test that is not measuring what
    it thinks it is.

    Args:
        pull: What ``docker pull`` does for a given image reference.
        present: Images the local cache already holds, so their pull is skipped.
        inspect_stdout: The inspect JSON handed back for a present image. Callers
            that assert on the metadata parsed out of it pass their own payload;
            the default is a well-formed but empty record.

    Returns:
        ``(run, calls)`` - the stand-in to install over the module's
        ``subprocess.run``, and the list of argvs it has been called with, in
        call order. The log is shared mutable state written under a lock, so it
        is safe to read after concurrent pulls have finished.
    """
    have = set(present)
    lock = threading.Lock()
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        with lock:
            calls.append(list(argv))
        if argv[:3] == ["docker", "image", "inspect"]:
            with lock:
                cached = argv[3] in have
            return CompletedProcess(
                argv, 0 if cached else 1, inspect_stdout if cached else b"", b""
            )
        assert argv[:2] == ["docker", "pull"], argv
        result = pull(argv[2])
        if result.returncode == 0:
            with lock:
                have.add(argv[2])
        return result

    return run, calls
