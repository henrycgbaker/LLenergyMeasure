"""vLLM-specific server command construction for the ServerCapable adapter.

Holds the engine knowledge the generic lifecycle mechanics
(:mod:`llenergymeasure.serving.lifecycle`) do not have: how vLLM is
invoked as an online server (``vllm serve <model> --port <port>``), and a
minimal OpenAI-completions readiness probe.

The upstream ``vllm/vllm-openai`` image's ``ENTRYPOINT`` is ``["vllm", "serve"]``
(verified from the Dockerfile), so the container leg passes only the serve
ARGUMENTS after the image and the entrypoint supplies ``vllm serve``. The
process leg runs the full command. Both share :func:`serve_args` so the two
legs cannot drift on model/port handling.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.serving.types import ProbeRequest

#: The upstream vllm/vllm-openai image ENTRYPOINT. The process leg prepends this;
#: the container leg relies on the image supplying it.
SERVE_ENTRYPOINT: tuple[str, ...] = ("vllm", "serve")

#: vLLM's OpenAI-compatible completions endpoint - the serving path a readiness
#: probe drives a real request through.
COMPLETIONS_PATH = "/v1/completions"


def serve_args(model: str, port: int, extra: list[str] | None = None) -> list[str]:
    """Return the ``vllm serve`` ARGUMENTS (after the ``vllm serve`` verb).

    ``<model> --port <port>`` plus any passthrough ``extra`` flags. Shared by
    both legs so the container entrypoint (``vllm serve``) and the process
    command produce the same logical invocation.
    """
    args = [model, "--port", str(port)]
    if extra:
        args += list(extra)
    return args


def process_argv(model: str, port: int, extra: list[str] | None = None) -> list[str]:
    """Return the full host-subprocess command: ``vllm serve <model> --port <port>``."""
    return [*SERVE_ENTRYPOINT, *serve_args(model, port, extra)]


def build_completions_probe(
    model: str, prompt: str = "ready?", max_tokens: int = 1
) -> ProbeRequest:
    """Return a minimal OpenAI-completions readiness probe for vLLM.

    A default request SHAPE for readiness when the caller supplies none; the
    server warmup protocol replaces it with a request drawn from the measured
    traffic distribution
    (warm the path you measure). Kept tiny (1 token, greedy) so the probe itself
    perturbs nothing measurable.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return ProbeRequest(path=COMPLETIONS_PATH, payload=payload)
