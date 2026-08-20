"""TensorRT-LLM-specific server command construction for the ServerCapable adapter.

Holds the engine knowledge the generic lifecycle mechanics
(:mod:`llenergymeasure.serving.lifecycle`) do not have: how TRT-LLM is
invoked as an online server (``trtllm-serve <model> --port <port>``), and a
minimal OpenAI-completions readiness probe.

Unlike vLLM (whose ``vllm/vllm-openai`` image bakes ``ENTRYPOINT ["vllm",
"serve"]``), the pinned NGC TRT-LLM image
(``nvcr.io/nvidia/tensorrt-llm/release``) is the documented ``trtllm-serve``
vehicle but does NOT bake it as the entrypoint - its entrypoint is the NVIDIA
setup script (``/opt/nvidia/nvidia_entrypoint.sh``), which sets up the CUDA libs
and execs whatever command it is given. So the container leg must invoke
``trtllm-serve`` EXPLICITLY as the command (it rides through the NGC entrypoint),
and the process leg runs the identical command on the host. Both share
:func:`serve_command` so the two legs cannot drift on the verb / model / port.

Endpoints verified against the TRT-LLM ``trtllm-serve`` docs (OpenAI-compatible
server): liveness ``/health`` (the module-default health path) and the real
serving path ``/v1/completions`` (also ``/v1/chat/completions``, ``/v1/models``,
``/metrics``, ``/version``).
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.serving.types import ProbeRequest

#: The TRT-LLM online-serving CLI verb. Invoked explicitly (the NGC image is not
#: entrypoint-baked with it - see the module docstring), so it leads the command
#: on BOTH the container and the process leg.
SERVE_COMMAND = "trtllm-serve"

#: TRT-LLM's OpenAI-compatible completions endpoint - the serving path a
#: readiness probe drives a real request through.
COMPLETIONS_PATH = "/v1/completions"


def serve_command(model: str, port: int, extra: list[str] | None = None) -> list[str]:
    """Return the full ``trtllm-serve`` command: ``trtllm-serve <model> --port <port>``.

    ``<model>`` is the positional argument and ``--port`` selects the bind port,
    plus any passthrough ``extra`` flags. Used verbatim by BOTH legs: the process
    leg runs it as the host subprocess argv, and the container leg passes it as
    the docker command after the image (the NGC entrypoint execs it), because the
    NGC image does not bake ``trtllm-serve`` as its entrypoint.
    """
    args = [SERVE_COMMAND, model, "--port", str(port)]
    if extra:
        args += list(extra)
    return args


def build_completions_probe(
    model: str, prompt: str = "ready?", max_tokens: int = 1
) -> ProbeRequest:
    """Return a minimal OpenAI-completions readiness probe for TRT-LLM.

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
