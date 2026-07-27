"""A tiny asyncio HTTP server for exercising the ServerCapable process leg.

Stands in for a real ``vllm serve`` process in host-only tests (no GPU, no
docker): it binds a port and answers the two endpoints the readiness protocol
drives - ``GET /health`` (liveness) and ``POST /v1/completions`` (the real
serving-path probe). Deliberately covers the awkward cases:

- ``--completions-ready-after S``: ``/health`` returns 200 immediately but
  ``/v1/completions`` returns 503 for the first ``S`` seconds, so a test can
  prove readiness is NOT satisfied by ``/health`` alone (R8) - either it waits
  for the real probe, or (with a large S and a short timeout) it times out.
- ``--ignore-sigterm``: the process ignores SIGTERM, forcing ``shutdown`` to
  escalate to SIGKILL (the kill-escalation path).

Stdlib only (asyncio + a hand-rolled minimal HTTP/1.1 responder), so it starts
fast and pulls in nothing engine-specific.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import time


def _route(
    method: str, path: str, *, ready_at: float, completions_ready_after: float
) -> tuple[int, str]:
    if path.startswith("/health"):
        return 200, '{"status":"ok"}'
    if path.startswith("/v1/completions"):
        if completions_ready_after and (time.monotonic() - ready_at) < completions_ready_after:
            return 503, '{"error":"model still loading"}'
        return 200, '{"choices":[{"text":"pong"}]}'
    return 404, '{"error":"not found"}'


async def _handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    ready_at: float,
    completions_ready_after: float,
) -> None:
    try:
        request_line = await reader.readline()
        if not request_line:
            return
        parts = request_line.decode("latin1").split(" ", 2)
        if len(parts) < 2:
            return
        method, path = parts[0], parts[1]
        content_length = 0
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin1").partition(":")
            if name.strip().lower() == "content-length":
                with contextlib.suppress(ValueError):
                    content_length = int(value.strip())
        if content_length:
            await reader.readexactly(content_length)
        status, body = _route(
            method, path, ready_at=ready_at, completions_ready_after=completions_ready_after
        )
        payload = body.encode("utf-8")
        head = (
            f"HTTP/1.1 {status} STUB\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("latin1")
        writer.write(head + payload)
        await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ignore-sigterm", action="store_true")
    parser.add_argument("--completions-ready-after", type=float, default=0.0)
    args = parser.parse_args()

    if args.ignore_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    ready_at = time.monotonic()
    server = await asyncio.start_server(
        lambda r, w: _handle(
            r, w, ready_at=ready_at, completions_ready_after=args.completions_ready_after
        ),
        "127.0.0.1",
        args.port,
    )
    # Announce readiness on stdout so read_logs() has something to return.
    print(f"stub server listening on 127.0.0.1:{args.port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())
    sys.exit(0)
