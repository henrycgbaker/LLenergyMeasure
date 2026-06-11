"""Trimmed tensorrt-1.2-style ``plugin/plugin.py`` excerpt for walker fixtures.

Reproduces the PluginConfig shape the study GT found entirely invisible to flat
entry-point introspection: ~20 ``Optional[Literal[...]]`` membership fields plus
a module-level ``Literal`` alias (``DefaultPluginDtype``) that several fields are
typed against. The alias resolution is what turns those aliased fields from
"membership unknown" into a probeable allowlist (the plugin-literal fold).

NOT importable tensorrt_llm source - a hand-trimmed structural excerpt. No
engine install required. The plugin/ directory layout exercises the
``plugin/*.py`` glob.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Module-level Literal alias: many plugin fields are typed Optional[<alias>].
DefaultPluginDtype = Literal["float16", "bfloat16", "float32"]

# A module-level lookup-map allowlist (keys are the accepted string values).
STR_DTYPE_TO_BINDING = {
    "float16": 0,
    "bfloat16": 1,
    "float32": 2,
    "int8": 3,
}


class PluginConfig(BaseModel):
    # Fields typed against the module-level alias (resolved -> 3 values each).
    gpt_attention_plugin: DefaultPluginDtype | None = "float16"
    gemm_plugin: DefaultPluginDtype | None = None
    gemm_swiglu_plugin: DefaultPluginDtype | None = None
    fp8_rowwise_gemm_plugin: DefaultPluginDtype | None = None
    nccl_plugin: DefaultPluginDtype | None = "float16"
    lora_plugin: DefaultPluginDtype | None = None
    moe_plugin: DefaultPluginDtype | None = "float16"
    mamba_conv1d_plugin: DefaultPluginDtype | None = "float16"
    low_latency_gemm_plugin: DefaultPluginDtype | None = None
    low_latency_gemm_swiglu_plugin: DefaultPluginDtype | None = None

    # Inline Literal membership fields.
    context_fmha: Literal["enabled", "disabled"] = "enabled"
    paged_kv_cache: Literal["enabled", "disabled"] = "enabled"
    remove_input_padding: Literal["enabled", "disabled"] = "enabled"
    reduce_fusion: Literal["enabled", "disabled"] = "disabled"
    user_buffer: Literal["enabled", "disabled"] = "disabled"
    tokens_per_block: Literal[32, 64, 128] = 64
    use_paged_context_fmha: Literal["enabled", "disabled"] = "disabled"
    multiple_profiles: Literal["enabled", "disabled"] = "disabled"
    paged_state: Literal["enabled", "disabled"] = "enabled"
    streamingllm: Literal["enabled", "disabled"] = "disabled"

    # A non-membership typed field: must NOT surface an enum.
    max_lora_rank: int = 64
    _internal: int = 0  # underscore field: must be ignored
