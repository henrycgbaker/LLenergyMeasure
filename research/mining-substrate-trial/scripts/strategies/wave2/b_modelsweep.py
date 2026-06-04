"""Wave 2a: model sweep on the existing (b) pipeline.

Same chunked-extraction pipeline as Wave 1's `llm_b_oss`; the served model
is a parameter. Pins by sha256 digest (see ``findings/wave2_model_digests.toml``).

Hypothesis (per WAVE2_PROTOCOL section 2a): the Wave 1 (b) recall ceiling
of ~50% is a property of llama3.1:70b at q4_K_M, not of the pipeline. If
a stronger model (unquantised, code-specialised, larger MoE) lifts the
ceiling, the substrate question shifts: pure-LLM extraction becomes
viable as Architecture III rather than only as Architecture II's
extension layer.

Models in scope (must fit on 4xA100-40G):
- llama-3.3-70b-instruct fp16
- qwen2.5-coder-32b-instruct fp16
- deepseek-coder-v2-instruct q4_K_M
- mixtral-8x22b-instruct q4_K_M
"""

from __future__ import annotations

import time
from pathlib import Path

from strategies.llm_b_oss import StrategyBOutputs, run_b_pipeline  # noqa: F401  (interface ref)
from strategies.wave2._base import CellOutput, standard_out_dir

SUPPORTED_MODELS = {
    "llama33-70b-fp16": {
        "ollama_tag": "llama3.3:70b-instruct-fp16",
        "hf_repo": "meta-llama/Llama-3.3-70B-Instruct",
        "num_ctx": 32768,
        "num_gpu_layers": -1,  # all on GPU
    },
    "qwen-coder-32b-fp16": {
        "ollama_tag": "qwen2.5-coder:32b-instruct-fp16",
        "hf_repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "num_ctx": 32768,
        "num_gpu_layers": -1,
    },
    "deepseek-coder-v2-q4": {
        "ollama_tag": "deepseek-coder-v2:236b-instruct-q4_K_M",
        "hf_repo": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "num_ctx": 32768,
        "num_gpu_layers": -1,
    },
    "mixtral-8x22b-q4": {
        "ollama_tag": "mixtral:8x22b-instruct-q4_K_M",
        "hf_repo": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "num_ctx": 32768,
        "num_gpu_layers": -1,
    },
}


def run_cell(
    *,
    engine: str,
    version: str,
    model: str,
    ollama_host: str = "http://localhost:11435",
    transcripts_dir: Path | None = None,
) -> CellOutput:
    """Execute one Wave 2a cell.

    Wires the standard (b) pipeline to a model-specific Ollama backend.
    Model digest pinning happens upstream (the runner verifies the
    served model's digest before calling run_cell).
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {list(SUPPORTED_MODELS)}")
    model_cfg = SUPPORTED_MODELS[model]
    strategy_id = f"w2-b-{model}"
    version_slug = version.replace(".", "_").lstrip("v")
    if not version_slug.startswith("v"):
        version_slug = f"v{version_slug}"

    out_dir = standard_out_dir(strategy_id, engine, version_slug)
    transcripts_dir = transcripts_dir or (out_dir / "llm_transcript")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # TODO: dispatch to run_b_pipeline with backend override per model_cfg.
    # The existing run_b_pipeline reads model + host from env or kwargs;
    # the wave2 runner sets OLLAMA_MODEL + OLLAMA_HOST before calling.
    raise NotImplementedError(
        "wave2 b_modelsweep dispatch: hook into llm_b_oss.run_b_pipeline. "
        "Scaffolded; not yet implemented. See WAVE2_PROTOCOL section 4 step 5."
    )
    # noqa: unreachable, kept for clarity
    wall_sec = {"total": time.perf_counter() - t0}
    return CellOutput(
        strategy_id=strategy_id,
        engine=engine,
        engine_version=version,
        schema_path=out_dir / "schema.json",
        invariants_path=out_dir / "invariants.proposed.yaml",
        observations=[f"model_config={model_cfg}"],
        wall_sec=wall_sec,
    )
