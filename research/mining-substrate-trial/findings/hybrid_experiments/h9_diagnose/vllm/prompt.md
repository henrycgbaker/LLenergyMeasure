You are a structural-gap analyst for an AST-based static-analysis
catalogue. The output (a) below is what a deterministic AST walker
emitted from the engine source. You are looking for STRUCTURAL gap
PATTERNS that explain why certain validations the source contains do
NOT appear in (a). Each gap pattern is a categorical reason - e.g. the
walker doesn't descend into orelse branches, or doesn't track local-
variable aliases.

For each gap pattern you can identify, emit ONE entry in the JSON
output. Cite a SPECIFIC example field name from this engine that
exhibits the gap. Be honest if you can't find an example - emit no
entries rather than fabricate.

Gap categories you should consider:
- branch-descent: walker handles if/raise but not if/elif/else, not
  else-raise, not nested branches.
- nested-config: walker doesn't recurse into Pydantic / dataclass
  fields typed as another config class.
- local-var-alias: walker compares self.X but source has x = self.X
  and compares the local x.
- normalisation-only: source uses normalisation pattern (no raise);
  walker only emits from if/raise patterns; these validators dormant
  for the walker.
- type-blindness: walker synthesises probe values without consulting
  the declared type; runtime fails for int-typed fields probed with
  string.
- defensive-import: walker fails at import time on bumped versions
  because some symbol was renamed in a dep.
- other: any structural gap that doesn't fit the above.

OUTPUT FORMAT: a single JSON object (no markdown, no commentary, no
code fences, just raw JSON). Use these EXACT keys:

{
  "gaps": [
    {
      "category": "branch-descent | nested-config | local-var-alias | normalisation-only | type-blindness | defensive-import | other",
      "severity": "blocks-correctness | reduces-recall | minor",
      "example_field": "<field name in vllm that exhibits the gap>",
      "structural_reason": "<1-2 sentences; use plain prose without colons or backticks>",
      "cross_engine_pattern": "yes-already-known | yes-new-here | engine-specific | unknown",
      "fix_estimate_loc": <integer>,
      "mergeable_into_spike_refactor": "yes | no | needs-broader-design"
    }
  ]
}

IMPORTANT: in structural_reason, avoid colons and backticks (the parser
is strict). Use plain English. If you must reference code, describe it
in words (e.g. "the if/raise pattern" not "`if X: raise`").

If you find no gaps, output: {"gaps": []}

=== ENGINE: vllm vv0_7_3 ===

=== (a) OUTPUT (26 entries) ===
- id: vllm_samplingparams_dormant_seed_eq_neg1
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source_method: __post_init__
  match.fields: {"vllm.sampling.seed": -1}
  message: 
- id: vllm_samplingparams_raises_best_of_lt_ref_n
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: __post_init__
  match.fields: {"vllm.sampling.best_of": {"present": true, "<": "@n"}}
  message: best_of must be greater than or equal to n, got n={n} and best_of={best_of}.
- id: vllm_samplingparams_raises_logprobs_lt_0
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.logprobs": {"present": true, "<": 0}}
  message: logprobs must be non-negative, got {self.logprobs}.
- id: vllm_samplingparams_raises_max_tokens_lt_1
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.max_tokens": {"present": true, "<": 1}}
  message: max_tokens must be at least 1, got {self.max_tokens}.
- id: vllm_samplingparams_raises_min_tokens_gt_ref_max_tokens
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.max_tokens": {"present": true}, "vllm.sampling.min_tokens": {">": "@max_tokens"}}
  message: min_tokens must be less than or equal to max_tokens={max_tokens}, got {min_tokens}.
- id: vllm_samplingparams_raises_min_tokens_lt_0
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.min_tokens": {"<": 0}}
  message: min_tokens must be greater than or equal to 0, got {min_tokens}.
- id: vllm_samplingparams_raises_n_lt_1
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.n": {"<": 1}}
  message: n must be at least 1, got {n}.
- id: vllm_samplingparams_raises_n_not_type_int
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.n": {"type_is_not": "int"}}
  message: n must be an int, but is of type {type(self.n)}
- id: vllm_samplingparams_raises_prompt_logprobs_lt_0
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.prompt_logprobs": {"present": true, "<": 0}}
  message: prompt_logprobs must be non-negative, got {self.prompt_logprobs}.
- id: vllm_samplingparams_raises_temperature_lt_0p0
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.temperature": {"<": 0.0}}
  message: temperature must be non-negative, got {temperature}.
- id: vllm_samplingparams_raises_top_k_not_type_int
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.top_k": {"type_is_not": "int"}}
  message: top_k must be an integer, got {type(self.top_k).__name__}
- id: vllm_samplingparams_raises_truncate_prompt_tokens_lt_1
  severity: error
  native_type: vllm.SamplingParams
  miner_source_method: _verify_args
  match.fields: {"vllm.sampling.truncate_prompt_tokens": {"present": true, "<": 1}}
  message: truncate_prompt_tokens must be >= 1, got {self.truncate_prompt_tokens}
- id: vllm_cacheconfig_raises_gpu_memory_utilization_gt_1p0
  severity: error
  native_type: vllm.config.CacheConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.gpu_memory_utilization": {">": 1.0}}
  message: GPU memory utilization must be less than 1.0. Got {self.gpu_memory_utilization}.
- id: vllm_loraconfig_dormant_max_cpu_loras_unset
  severity: dormant
  native_type: vllm.config.LoRAConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.lora.max_cpu_loras": {"absent": true}}
  message: 
- id: vllm_loraconfig_raises_max_cpu_loras_lt_ref_max_loras
  severity: error
  native_type: vllm.config.LoRAConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.lora.max_cpu_loras": {"present": true, "<": "@max_loras"}}
  message: max_cpu_loras ({self.max_cpu_loras}) must be >= max_loras ({self.max_loras})
- id: vllm_loraconfig_raises_max_loras_lt_1
  severity: error
  native_type: vllm.config.LoRAConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.lora.max_loras": {"<": 1}}
  message: max_loras ({self.max_loras}) must be >= 1.
- id: vllm_parallelconfig_raises_distributed_executor_backend_not_in_ray_mp_uni_external_launcher_none
  severity: error
  native_type: vllm.config.ParallelConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.distributed_executor_backend": {"not_in": ["ray", "mp", "uni", "external_launcher", null]}}
  message: Unrecognized distributed executor backend {self.distributed_executor_backend}. Supported values are 'ray', 'mp' 'uni', '
- id: vllm_promptadapterconfig_dormant_max_cpu_prompt_adapters_unset
  severity: dormant
  native_type: vllm.config.PromptAdapterConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.prompt_adapter.max_cpu_prompt_adapters": {"absent": true}}
  message: 
- id: vllm_promptadapterconfig_raises_max_prompt_adapter_token_eq_0
  severity: error
  native_type: vllm.config.PromptAdapterConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.prompt_adapter.max_prompt_adapter_token": 0}
  message: max_prompt_adapter_token must be set.
- id: vllm_promptadapterconfig_raises_max_prompt_adapters_lt_1
  severity: error
  native_type: vllm.config.PromptAdapterConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.prompt_adapter.max_prompt_adapters": {"<": 1}}
  message: max_prompt_adapters ({self.max_prompt_adapters}) must be >= 1.
- id: vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_model_len
  severity: error
  native_type: vllm.config.SchedulerConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.max_num_batched_tokens": {"<": "@max_model_len"}, "vllm.engine.enable_chunked_prefill": false}
  message: max_num_batched_tokens ({self.max_num_batched_tokens}) is smaller than max_model_len ({self.max_model_len}).
- id: vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs
  severity: error
  native_type: vllm.config.SchedulerConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.max_num_batched_tokens": {"<": "@max_num_seqs"}}
  message: max_num_batched_tokens ({self.max_num_batched_tokens}) must be greater than or equal to max_num_seqs ({self.max_num_seqs
- id: vllm_schedulerconfig_raises_max_num_partial_prefills_lt_1
  severity: error
  native_type: vllm.config.SchedulerConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.max_num_partial_prefills": {"<": 1}}
  message: max_num_partial_prefills ({max_num_partial_prefills}) must be greater than or equal to 1.
- id: vllm_schedulerconfig_raises_num_lookahead_slots_lt_0
  severity: error
  native_type: vllm.config.SchedulerConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.num_lookahead_slots": {"<": 0}}
  message: num_lookahead_slots ({num_lookahead_slots}) must be greater than or equal to 0.
- id: vllm_schedulerconfig_raises_num_scheduler_steps_lt_1
  severity: error
  native_type: vllm.config.SchedulerConfig
  miner_source_method: _verify_args
  match.fields: {"vllm.engine.num_scheduler_steps": {"<": 1}}
  message: num_scheduler_steps ({num_scheduler_steps}) must be greater than or equal to 1.
- id: vllm_tokenizerpoolconfig_raises_extra_config_not_type_dict
  severity: error
  native_type: vllm.config.TokenizerPoolConfig
  miner_source_method: __post_init__
  match.fields: {"vllm.engine.tokenizer.extra_config": {"type_is_not": "dict"}}
  message: extra_config must be a dictionary.

=== ENGINE SOURCE EXCERPTS ===
=== config.py ===
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import enum
import hashlib
import json
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Counter, Dict,
                    Final, List, Literal, Mapping, Optional, Protocol, Set,
                    Tuple, Type, Union)

import torch
from pydantic import BaseModel, Field, PrivateAttr
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (QUANTIZATION_METHODS,
                                                     get_quantization_config)
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import CpuArchEnum
from vllm.tracing import is_otel_available, otel_import_error_traceback
from vllm.transformers_utils.config import (
    ConfigFormat, get_config, get_hf_image_processor_config,
    get_hf_text_config, get_pooling_config,
    get_sentence_transformer_tokenizer_config, is_encoder_decoder,
    try_get_generation_config, uses_mrope)
from vllm.transformers_utils.s3_utils import S3Model
from vllm.transformers_utils.utils import is_s3
from vllm.utils import (GiB_bytes, LayerBlockType, cuda_device_count_stateless,
                        get_cpu_memory, random_uuid, resolve_obj_by_qualname)

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from vllm.executor.executor_base import ExecutorBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)
    from vllm.model_executor.model_loader.loader import BaseModelLoader
    from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
        BaseTokenizerGroup)
else:
    QuantizationConfig = None

logger = init_logger(__name__)

_POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
_MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

TaskOption = Literal["auto", "generate", "embedding", "embed", "classify",
                     "score", "reward", "transcription"]

_ResolvedTask = Literal["generate", "embed", "classify", "score", "reward",
                        "draft", "transcription"]

RunnerType = Literal["generate", "pooling", "draft", "transcription"]

_RUNNER_TASKS: Dict[RunnerType, List[_ResolvedTask]] = {
    "generate": ["generate"],
    "pooling": ["embed", "classify", "score", "reward"],
    "draft": ["draft"],
    "transcription": ["transcription"],
}

_TASK_RUNNER: Dict[_ResolvedTask, RunnerType] = {
    task: runner
    for runner, tasks in _RUNNER_TASKS.items()
    for task in tasks
}

HfOverrides = Union[Dict[str, Any], Callable[[PretrainedConfig],
                                             PretrainedConfig]]


class SupportsHash(Protocol):

    def compute_hash(self) -> str:
        ...


class ModelImpl(str, enum.Enum):
    AUTO = "auto"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics
            output when `served_model_name` is not specified.
        task: The task to use the model for. Each vLLM instance only supports
            one task, even if the same model can be used for multiple tasks.
            When the model only supports one task, "auto" can be used to select
            it; otherwise, you must specify explicitly which task to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, "slow" will always use the slow tokenizer,
            "mistral" will always use the tokenizer from `mistral_common`, and
            "custom" will use --tokenizer to select the preregistered tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images or
            videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        spec_target_max_model_len: Specify the the maximum length for spec
            decoding draft models.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
            If None, the user did not specify, so default to False.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        max_logprobs: Maximum number of log probabilities. Defaults to 20.
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model
            names provided, the first name will be used. If not specified,
            the model name will be the same as `model`.
        limit_mm_per_prompt: Maximum number of data items per modality
            per prompt. Only applicable for multimodal models.
        use_async_output_proc: Whether to use async output processor.
            Defaults to True.
        config_format: The config format which shall be loaded.
            Defaults to 'auto' which defaults to 'hf'.
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor.
        disable_mm_preprocessor_cache: If true, then disables caching of the
            multi-modal preprocessor/mapper. (not recommended)
        override_neuron_config: Initialize non default neuron config or
            override default neuron config that are specific to Neuron devices,
            this argument will be used to configure the neuron config that
            can not be gathered from the vllm arguments.
        override_pooler_config: Initialize non default pooling config or
            override default pooling config for the pooling model.
        logits_processor_pattern: Optional regex pattern specifying valid
            logits processor qualified names that can be passed with the
            `logits_processors` extra completion argument. Defaults to None,
            which allows no processors.
        generation_config: Configuration parameter file for generation.
        model_impl: Which implementation of the model to use:
            "auto" will try to use the vLLM implementation if it exists and
                fall back to the Transformers implementation if no vLLM
                implementation is available.
            "vllm" will use the vLLM model implementation.
            "transformers" will use the Transformers model implementation.
        override_generation_config: Override the generation config with the
            given config.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: List[Any] = []
        factors.append(self.model)
        factors.append(self.dtype)
        factors.append(self.quantization)
        factors.append(self.revision)
        factors.append(self.code_revision)
        factors.append(self.trust_remote_code)
        factors.append(self.rope_scaling)
        factors.append(self.rope_theta)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __init__(
        self,
        model: str,
        task: Union[TaskOption, Literal["draft"]],
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        dtype: Union[str, torch.dtype],
        seed: int,
        allowed_local_media_path: str = "",
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_theta: Optional[float] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        spec_target_max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 20,
        disable_sliding_window: bool = False,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
        limit_mm_per_prompt: Optional[Mapping[str, int]] = None,
        use_async_output_proc: bool = True,
        config_format: ConfigFormat = ConfigFormat.AUTO,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        disable_mm_preprocessor_cache: bool = False,
        override_neuron_config: Optional[Dict[str, Any]] = None,
        override_pooler_config: Optional["PoolerConfig"] = None,
        logits_processor_pattern: Optional[str] = None,
        generation_config: Optional[str] = None,
        enable_sleep_mode: bool = False,
        override_generation_config: Optional[Dict[str, Any]] = None,
        model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.allowed_local_media_path = allowed_local_media_path
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.model_impl = model_impl

        if hf_overrides is None:
            hf_overrides = {}

        if callable(hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = hf_overrides
        else:
            hf_overrides_kw = hf_overrides
            hf_overrides_fn = None

        if rope_scaling is not None:
            hf_override: Dict[str, Any] = {"rope_scaling": rope_scaling}
            hf_overrides_kw.update(hf_override)
            msg = ("`--rope-scaling` will be removed in a future release. "
                   f"'Please instead use `--hf-overrides '{hf_override!r}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if rope_theta is not None:
            hf_override = {"rope_theta": rope_theta}
            hf_overrides_kw.update(hf_override)
            msg = ("`--rope-theta` will be removed in a future release. "
                   f"'Please instead use `--hf-overrides '{hf_override!r}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)

        self.maybe_pull_model_tokenizer_for_s3(model, tokenizer)

        # The tokenizer version is consistent with the model version by default.
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init
        self.enable_sleep_mode = enable_sleep_mode

        from vllm.platforms import current_platform

        if self.enable_sleep_mode and not current_platform.is_cuda():
            raise ValueError("Sleep mode is only supported on CUDA devices.")

        hf_config = get_config(self.model, trust_remote_code, revision,
                               code_revision, config_format)

        if hf_overrides_kw:
            logger.info("Overriding HF config with %s", hf_overrides_kw)
            hf_config.update(hf_overrides_kw)
        if hf_overrides_fn:
            logger.info("Overriding HF config with %s", hf_overrides_fn)
            hf_config = hf_overrides_fn(hf_config)

        self.hf_config = hf_config

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, revision)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        self.use_async_output_proc = use_async_output_proc
        self.mm_processor_kwargs = mm_processor_kwargs
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache

        # Set enforce_eager to False if the value is unset.
        if self.enforce_eager is None:
            self.enforce_eager = False

        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
        has_interleaved_attention = (sliding_window is not None) and (
            isinstance(sliding_window, list) or
            (self.hf_text_config.model_type in ["gemma2", "cohere2"]))

        if (not self.disable_sliding_window and has_interleaved_attention):
            if (backend :=
                    envs.VLLM_ATTENTION_BACKEND) in ("XFORMERS", "FLASHINFER"):
                sliding_window_len_min = get_min_sliding_window(
                    self.hf_text_config.sliding_window)

                logger.warning_once(
                    f"{self.hf_text_config.model_type} has interleaved "
                    "attention, which is currently not supported by the "
                    f"{backend} backend. Disabling sliding window and capping "
                    "the max length to the sliding window size "
                    f"({sliding_window_len_min}).")
                self.disable_sliding_window = True
            else:
                # for a model with interleaved attention,
                # the scheduler and the model treat it as full attention
                # (i.e., not dropping any tokens outside the window).
                # only the attention layer itself is aware of the sliding
                # window, and use the window size to compute the attention.
                self.hf_text_config.interleaved_sliding_window = sliding_window
                delattr(self.hf_text_config, "sliding_window")
                sliding_window = None

        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=spec_target_max_model_len,
            encoder_config=self.encoder_config)
        self.served_model_name = get_served_model_name(model,
                                                       served_model_name)
        self.multimodal_config = self._init_multimodal_config(
            limit_mm_per_prompt)
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        self.is_attention_free = self._init_attention_free()
        self.is_hybrid = self._init_is_hybrid()
        self.has_inner_state = self._init_has_inner_state()

        if current_platform.is_neuron():
            self.override_neuron_config = override_neuron_config
        else:
            self.override_neuron_config = None

        supported_tasks, task = self._resolve_task(task, self.hf_config)
        self.supported_tasks = supported_tasks
        self.task: Final = task
        if self.task in ("draft", "generate"):
            self.truncation_side = "left"
        else:
            self.truncation_side = "right"

        self.pooler_config = self._init_pooler_config(override_pooler_config)
        self.logits_processor_pattern = logits_processor_pattern

        self.generation_config = generation_config
        self.override_generation_config = override_generation_config or {}

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    def maybe_pull_model_tokenizer_for_s3(self, model: str,
                                          tokenizer: str) -> None:
        """
        Pull the model config or tokenizer to a temporary
        directory in case of S3.

        Args:
            model: The model name or path.
            tokenizer: The tokenizer name or path.

        """
        if is_s3(model) or is_s3(tokenizer):
            if is_s3(model):
                s3_model = S3Model()
                s3_model.pull_files(
                    model, allow_pattern=["*.model", "*.py", "*.json"])
                self.model_weights = self.model
                self.model = s3_model.dir

            if is_s3(tokenizer):
                s3_tokenizer = S3Model()
                s3_tokenizer.pull_files(
                    model, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
                self.tokenizer = s3_tokenizer.dir

    def _init_multimodal_config(
        self, limit_mm_per_prompt: Optional[Mapping[str, int]]
    ) -> Optional["MultiModalConfig"]:
        architectures = getattr(self.hf_config, "architectures", [])
        if ModelRegistry.is_multimodal_model(architectures):
            return MultiModalConfig(limit_per_prompt=limit_mm_per_prompt or {})

        if limit_mm_per_prompt:
            raise ValueError("`limit_mm_per_prompt` is only supported for "
                             "multimodal models.")

        return None

    def _get_encoder_config(self):
        return get_sentence_transformer_tokenizer_config(
            self.model, self.revision)

    def _init_pooler_config(
        self,
        override_pooler_config: Optional["PoolerConfig"],
    ) -> Optional["PoolerConfig"]:

        if self.runner_type == "pooling":
            user_config = override_pooler_config or PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(user_config, k) is None:
                        setattr(user_config, k, v)

            return user_config

        return None

    def _init_attention_free(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_attention_free_model(architectures)

    def _init_is_hybrid(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_hybrid_model(architectures)

    def _init_has_inner_state(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.model_has_inner_state(architectures)

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow", "mistral", "custom"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto', 'slow', 'mistral' or 'custom'.")
        self.tokenizer_mode = tokenizer_mode

    def _get_preferred_task(
        self,
        architectures: List[str],
        supported_tasks: Set[_ResolvedTask],
    ) -> Optional[_ResolvedTask]:
        model_id = self.model
        if get_pooling_config(model_id, self.revision):
            return "embed"
        if ModelRegistry.is_cross_encoder_model(architectures):
            return "score"
        if ModelRegistry.is_transcription_model(architectures):
            return "transcription"

        suffix_to_preferred_task: List[Tuple[str, _ResolvedTask]] = [
            # Other models follow this pattern
            ("ForCausalLM", "generate"),
            ("ForConditionalGeneration", "generate"),
            ("ForSequenceClassification", "classify"),
            ("ChatModel", "generate"),
            ("LMHeadModel", "generate"),
            ("EmbeddingModel", "embed"),
            ("RewardModel", "reward"),
        ]
        _, arch = ModelRegistry.inspect_model_cls(architectures)

        for suffix, pref_task in suffix_to_preferred_task:
            if arch.endswith(suffix) and pref_task in supported_tasks:
                return pref_task

        return None

    def _resolve_task(
        self,
        task_option: Union[TaskOption, Literal["draft"]],
        hf_config: PretrainedConfig,
    ) -> Tuple[Set[_ResolvedTask], _ResolvedTask]:
        if task_option == "draft":
            return {"draft"}, "draft"

        architectures = getattr(hf_config, "architectures", [])

        runner_support: Dict[RunnerType, bool] = {
            # NOTE: Listed from highest to lowest priority,
            # in case the model supports multiple of them
            "transcription":
            ModelRegistry.is_transcription_model(architectures),
            "generate": ModelRegistry.is_text_generation_model(architectures),
            "pooling": ModelRegistry.is_pooling_model(architectures),
        }
        supported_runner_types_lst: List[RunnerType] = [
            runner_type
            for runner_type, is_supported in runner_support.items()
            if is_supported
        ]

        supported_tasks_lst: List[_ResolvedTask] = [
            task for runner_type in supported_runner_types_lst
            for task in _RUNNER_TASKS[runner_type]
        ]
        supported_tasks = set(supported_tasks_lst)

        if task_option == "auto":
            selected_task = next(iter(supported_tasks_lst))

            if len(supported_tasks_lst) > 1:
                preferred_task = self._get_preferred_task(
                    architectures, supported_tasks)
                if preferred_task is not None:
                    selected_task = preferred_task

                logger.info(
                    "This model supports multiple tasks: %s. "
                    "Defaulting to '%s'.", supported_tasks, selected_task)
        else:
            # Aliases
            if task_option == "embedding":
                preferred_task = self._get_preferred_task(
                    architectures, supported_tasks)
                if preferred_task != "embed":
                    msg = ("The 'embedding' task will be restricted to "
                           "embedding models in a future release. Please "
                           "pass `--task classify`, `--task score`, or "
                           "`--task reward` explicitly for other pooling "
                           "models.")
                    warnings.warn(msg, DeprecationWarning, stacklevel=2)

                task_option = preferred_task or "embed"

            if task_option not in supported_tasks:
                msg = (
                    f"This model does not support the '{task_option}' task. "
                    f"Supported tasks: {supported_tasks}")
                raise ValueError(msg)

            selected_task = task_option

        return supported_tasks, selected_task

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed_tensors",
            "compressed-tensors", "experts_int8", "quark"
        ]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Detect which checkpoint is it
            for name in QUANTIZATION_METHODS:
                method = get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)

    def _verify_cuda_graph(self) -> None:
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)

        MODEL_NOT_SUPPORT_CUDA_GRAPH = ['mllama']
        if (self.hf_config.model_type in MODEL_NOT_SUPPORT_CUDA_GRAPH
                and not self.enforce_eager):
            logger.warning(
                "CUDA graph is not supported for %s yet, fallback to the eager "
                "mode.", self.hf_config.model_type)
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.44.0) with 8-bit models does not
        yet support CUDA graph.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (getattr(self.hf_config,
                                           "quantization_config", None)
                                   is not None)
        is_8bit = (self.hf_config.quantization_config.get(
            "load_in_8bit", False) if has_quantization_config else False)
        if all([
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
        ]):
            logger.warning(
                "CUDA graph is not supported on BitAndBytes 8bit yet, "
                "fallback to the eager mode.")
            self.enforce_eager = True

    def verify_async_output_proc(self, parallel_config, speculative_config,
                                 device_config) -> None:
        if not self.use_async_output_proc:
            # Nothing to check
            return

        if parallel_config.pipeline_parallel_size > 1:
            logger.warning("Async output processing can not be enabled "
                           "with pipeline parallel")
            self.use_async_output_proc = False
            return

 
... [TRUNCATED 123870 chars]

=== engine/arg_utils.py ===
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional,
                    Tuple, Type, Union, cast, get_args)

import torch

import vllm.envs as envs
from vllm.config import (CacheConfig, CompilationConfig, ConfigFormat,
                         DecodingConfig, DeviceConfig, HfOverrides,
                         KVTransferConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ModelImpl, ObservabilityConfig,
                         ParallelConfig, PoolerConfig, PromptAdapterConfig,
                         SchedulerConfig, SpeculativeConfig, TaskOption,
                         TokenizerPoolConfig, VllmConfig)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.plugins import load_general_plugins
from vllm.transformers_utils.utils import check_gguf_file
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, StoreBoolean

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

logger = init_logger(__name__)

ALLOWED_DETAILED_TRACE_MODULES = ["model", "worker", "all"]

DEVICE_OPTIONS = [
    "auto",
    "cuda",
    "neuron",
    "cpu",
    "openvino",
    "tpu",
    "xpu",
    "hpu",
]


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def nullable_kvs(val: str) -> Optional[Mapping[str, int]]:
    """Parses a string containing comma separate key [str] to value [int]
    pairs into a dictionary.

    Args:
        val: String value to be parsed.

    Returns:
        Dictionary with parsed values.
    """
    if len(val) == 0:
        return None

    out_dict: Dict[str, int] = {}
    for item in val.split(","):
        kv_parts = [part.lower().strip() for part in item.split("=")]
        if len(kv_parts) != 2:
            raise argparse.ArgumentTypeError(
                "Each item should be in the form KEY=VALUE")
        key, value = kv_parts

        try:
            parsed_value = int(value)
        except ValueError as exc:
            msg = f"Failed to parse value of item {key}={value}"
            raise argparse.ArgumentTypeError(msg) from exc

        if key in out_dict and out_dict[key] != parsed_value:
            raise argparse.ArgumentTypeError(
                f"Conflicting values specified for key: {key}")
        out_dict[key] = parsed_value

    return out_dict


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = 'facebook/opt-125m'
    served_model_name: Optional[Union[str, List[str]]] = None
    tokenizer: Optional[str] = None
    task: TaskOption = "auto"
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    config_format: ConfigFormat = ConfigFormat.AUTO
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[str,
                                                 Type[ExecutorBase]]] = None
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = True
    swap_space: float = 4  # GiB
    cpu_offload_gb: float = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_partial_prefills: Optional[int] = 1
    max_long_partial_prefills: Optional[int] = 1
    long_prefill_token_threshold: Optional[int] = 0
    max_num_seqs: Optional[int] = None
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None
    hf_overrides: Optional[HfOverrides] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    tokenizer_pool_extra_config: Optional[Dict[str, Any]] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    disable_mm_preprocessor_cache: bool = False
    enable_lora: bool = False
    enable_lora_bias: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = 1
    max_prompt_adapter_token: int = 0
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: Optional[Union[str, torch.dtype]] = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    num_scheduler_steps: int = 1
    multi_step_stream_outputs: bool = True
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    guided_decoding_backend: str = 'xgrammar'
    logits_processor_pattern: Optional[str] = None
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_model_quantization: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_disable_mqa_scorer: Optional[bool] = False
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    qlora_adapter_name_or_path: Optional[str] = None
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    otlp_traces_endpoint: Optional[str] = None
    collect_detailed_traces: Optional[str] = None
    disable_async_output_proc: bool = False
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    scheduler_cls: Union[str, Type[object]] = "vllm.core.scheduler.Scheduler"

    override_neuron_config: Optional[Dict[str, Any]] = None
    override_pooler_config: Optional[PoolerConfig] = None
    compilation_config: Optional[CompilationConfig] = None
    worker_cls: str = "auto"

    kv_transfer_config: Optional[KVTransferConfig] = None

    generation_config: Optional[str] = None
    override_generation_config: Optional[Dict[str, Any]] = None
    enable_sleep_mode: bool = False
    model_impl: str = "auto"

    calculate_kv_scales: Optional[bool] = None

    additional_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model

        # Override the default value of enable_prefix_caching if it's not set
        # by user.
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = bool(envs.VLLM_USE_V1)

        # Override max_num_seqs if it's not set by user.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256 if not envs.VLLM_USE_V1 else 1024

        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, (int, dict)):
            self.compilation_config = CompilationConfig.from_cli(
                str(self.compilation_config))

        # Setup plugins
        from vllm.plugins import load_general_plugins
        load_general_plugins()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default=EngineArgs.model,
            help='Name or path of the huggingface model to use.')
        parser.add_argument(
            '--task',
            default=EngineArgs.task,
            choices=get_args(TaskOption),
            help='The task to use the model for. Each vLLM instance only '
            'supports one task, even if the same model can be used for '
            'multiple tasks. When the model only supports one task, ``"auto"`` '
            'can be used to select it; otherwise, you must specify explicitly '
            'which task to use.')
        parser.add_argument(
            '--tokenizer',
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help='Name or path of the huggingface tokenizer to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            '--skip-tokenizer-init',
            action='store_true',
            help='Skip initialization of tokenizer and detokenizer.')
        parser.add_argument(
            '--revision',
            type=nullable_str,
            default=None,
            help='The specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=nullable_str,
            default=None,
            help='The specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=nullable_str,
            default=None,
            help='Revision of the huggingface tokenizer to use. '
            'It can be a branch name, a tag name, or a commit id. '
            'If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-mode',
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=['auto', 'slow', 'mistral', 'custom'],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            'always use the slow tokenizer. \n* '
            '"mistral" will always use the `mistral_common` tokenizer. \n* '
            '"custom" will use --tokenizer to select the '
            'preregistered tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument(
            '--allowed-local-media-path',
            type=str,
            help="Allowing API requests to read local images or videos "
            "from directories specified by the server file system. "
            "This is a security risk. "
            "Should only be enabled in trusted environments.")
        parser.add_argument('--download-dir',
                            type=nullable_str,
                            default=EngineArgs.download_dir,
                            help='Directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface.')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=[f.value for f in LoadFormat],
            help='The format of the model weights to load.\n\n'
            '* "auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available.\n'
            '* "pt" will load the weights in the pytorch bin format.\n'
            '* "safetensors" will load the weights in the safetensors format.\n'
            '* "npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading.\n'
            '* "dummy" will initialize the weights with random values, '
            'which is mainly for profiling.\n'
            '* "tensorizer" will load the weights using tensorizer from '
            'CoreWeave. See the Tensorize vLLM Model script in the Examples '
            'section for more information.\n'
            '* "runai_streamer" will load the Safetensors weights using Run:ai'
            'Model Streamer \n'
            '* "bitsandbytes" will load the weights using bitsandbytes '
            'quantization.\n')
        parser.add_argument(
            '--config-format',
            default=EngineArgs.config_format,
            choices=[f.value for f in ConfigFormat],
            help='The format of the model config to load.\n\n'
            '* "auto" will try to load the config in hf format '
            'if available else it will try to load in mistral format ')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='Data type for model weights and activations.\n\n'
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            'BF16 precision for BF16 models.\n'
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.')
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
            'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='Model context length. If unspecified, will '
                            'be automatically derived from the model config.')
        parser.add_argument(
            '--guided-decoding-backend',
            type=str,
            default='xgrammar',
            choices=['outlines', 'lm-format-enforcer', 'xgrammar'],
            help='Which engine will be used for guided decoding'
            ' (JSON schema / regex etc) by default. Currently support '
            'https://github.com/outlines-dev/outlines, '
            'https://github.com/mlc-ai/xgrammar, and '
            'https://github.com/noamgat/lm-format-enforcer.'
            ' Can be overridden per request via guided_decoding_backend'
            ' parameter.')
        parser.add_argument(
            '--logits-processor-pattern',
            type=nullable_str,
            default=None,
            help='Optional regex pattern specifying valid logits processor '
            'qualified names that can be passed with the `logits_processors` '
            'extra completion argument. Defaults to None, which allows no '
            'processors.')
        parser.add_argument(
            '--model-impl',
            type=str,
            default=EngineArgs.model_impl,
            choices=[f.value for f in ModelImpl],
            help='Which implementation of the model to use.\n\n'
            '* "auto" will try to use the vLLM implementation if it exists '
            'and fall back to the Transformers implementation if no vLLM '
            'implementation is available.\n'
            '* "vllm" will use the vLLM model implementation.\n'
            '* "transformers" will use the Transformers model '
            'implementation.\n')
        # Parallel arguments
        parser.add_argument(
            '--distributed-executor-backend',
            choices=['ray', 'mp', 'uni', 'external_launcher'],
            default=EngineArgs.distributed_executor_backend,
            help='Backend to use for distributed model '
            'workers, either "ray" or "mp" (multiprocessing). If the product '
            'of pipeline_parallel_size and tensor_parallel_size is less than '
            'or equal to the number of GPUs available, "mp" will be used to '
            'keep processing on a single host. Otherwise, this will default '
            'to "ray" if Ray is installed and fail otherwise. Note that tpu '
            'only supports Ray for distributed inference.')

        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='Number of pipeline stages.')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='Number of tensor parallel replicas.')
        parser.add_argument(
            '--max-parallel-loading-workers',
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help='Load model sequentially in multiple batches, '
            'to avoid RAM OOM when using tensor '
            'parallel and large models.')
        parser.add_argument(
            '--ray-workers-use-nsight',
            action='store_true',
            help='If specified, use nsight to profile Ray workers.')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32, 64, 128],
                            help='Token block size for contiguous chunks of '
                            'tokens. This is ignored on neuron devices and '
                            'set to ``--max-model-len``. On CUDA devices, '
                            'only block sizes up to 32 are supported. '
                            'On HPU devices, block size defaults to 128.')

        parser.add_argument(
            "--enable-prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=EngineArgs.enable_prefix_caching,
            help="Enables automatic prefix caching. "
            "Use ``--no-enable-prefix-caching`` to disable explicitly.",
        )
        parser.add_argument('--disable-sliding-window',
                            action='store_true',
                            help='Disables sliding window, '
                            'capping to sliding window size.')
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            default=True,
                            help='[DEPRECATED] block manager v1 has been '
                            'removed and SelfAttnBlockSpaceManager (i.e. '
                            'block manager v2) is now the default. '
                            'Setting this flag to True or False'
                            ' has no effect on vLLM behavior.')
        parser.add_argument(
            '--num-lookahead-slots',
            type=int,
            default=EngineArgs.num_lookahead_slots,
            help='Experimental scheduling config necessary for '
            'speculative decoding. This will be replaced by '
            'speculative config in the future; it is present '
            'to enable correctness tests until then.')

        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='Random seed for operations.')
        parser.add_argument('--swap-space',
                            type=float,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU.')
        parser.add_argument(
            '--cpu-offload-gb',
            type=float,
            default=0,
            help='The space in GiB to offload to CPU, per GPU. '
            'Default is 0, which means no offloading. Intuitively, '
            'this argument can be seen as a virtual way to increase '
            'the GPU memory size. For example, if you have one 24 GB '
            'GPU and set this to 10, virtually you can think of it as '
            'a 34 GB GPU. Then you can load a 13B model with BF16 weight, '
            'which requires at least 26GB GPU memory. Note that this '
            'requires fast CPU-GPU interconnect, as part of the model is '
            'loaded from CPU memory to GPU memory on the fly in each '
            'model forward pass.')
        parser.add_argument(
            '--gpu-memory-utilization',
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help='The fraction of GPU memory to be used for the model '
            'executor, which can range from 0 to 1. For example, a value of '
            '0.5 would imply 50%% GPU memory utilization. If unspecified, '
            'will use the default value of 0.9. This is a per-instance '
            'limit, and only applies to the current vLLM instance.'
            'It does not matter if you have another vLLM instance running '
            'on the same GPU. For example, if you have two vLLM instances '
            'running on the same GPU, you can set the GPU memory utilization '
            'to 0.5 for each instance.')
        parser.add_argument(
            '--num-gpu-blocks-override',
            type=int,
            default=None,
            help='If specified, ignore GPU profiling result and use this number'
            ' of GPU blocks. Used for testing preemption.')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='Maximum number of batched tokens per '
                            'iteration.')
        parser.add_argument(
            "--max-num-partial-prefills",
            type=int,
            default=EngineArgs.max_num_partial_prefills,
            help="For chunked prefill, the max number of concurrent \
            partial prefills."
            "Defaults to 1",
        )
        parser.add_argument(
            "--max-long-partial-prefills",
            type=int,
            default=EngineArgs.max_long_partial_prefills,
            help="For chunked prefill, the maximum number of prompts longer "
            "than --long-prefill-token-threshold that will be prefilled "
            "concurrently. Setting this less than --max-num-partial-prefills "
            "will allow shorter prompts to jump the queue in front of longer "
            "prompts in some cases, improving latency. Defaults to 1.")
        parser.add_argument(
            "--long-prefill-token-threshold",
            type=float,
            default=EngineArgs.long_prefill_token_threshold,
            help="For chunked prefill, a request is considered long if the "
            "prompt is longer than this number of tokens. Defaults to 4%% of "
            "the model's context length.",
        )
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='Maximum number of sequences per iteration.')
        parser.add_argument(
            '--max-logprobs',
            type=int,
            default=EngineArgs.max_logprobs,
            help=('Max number of log probs to return logprobs is specified in'
                  ' SamplingParams.'))
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='Disable logging statistics.')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=nullable_str,
                            choices=[*QUANTIZATION_METHODS, None],
                            default=EngineArgs.quantization,
                            help='Method used to quantize the weights. If '
                            'None, we first check the `quantization_config` '
                            'attribute in the model config file. If that is '
                            'None, we assume the model weights are not '
                            'quantized and use `dtype` to determine the data '
                            'type of the weights.')
        parser.add_argument(
            '--rope-scaling',
            default=None,
            type=json.loads,
            help='RoPE scaling configuration in JSON format. '
            'For example, ``{"rope_type":"dynamic","factor":2.0}``')
        parser.add_argument('--rope-theta',
                            default=None,
                            type=float,
                            help='RoPE theta. Use with `rope_scaling`. In '
                            'some cases, changing the RoPE theta improves the '
                            'performance of the scaled model.')
        parser.add_argument('--hf-overrides',
                            type=json.loads,
                            default=EngineArgs.hf_overrides,
                            help='Extra arguments for the HuggingFace config. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary.')
        parser.add_argument('--enforce-eager',
                            action='store_true',
                            help='Always use eager-mode PyTorch. If False, '
                            'will use eager mode and CUDA graph in hybrid '
                            'for maximal performance and flexibility.')
        parser.add_argument('--max-seq-len-to-capture',
                            type=int,
                            default=EngineArgs.max_seq_len_to_capture,
                            help='Maximum sequence length covered by CUDA '
                            'graphs. When a sequence has context length '
                            'larger than this, we fall back to eager mode. '
                            'Additionally for encoder-decoder models, if the '
                            'sequence length of the encoder input is larger '
                            'than this, we fall back to the eager mode.')
        parser.add_argument('--disable-custom-all-reduce',
                            action='store_true',
                            default=EngineArgs.disable_custom_all_reduce,
                            help='See ParallelConfig.')
        parser.add_argument('--tokenizer-pool-size',
                            type=int,
                            default=EngineArgs.tokenizer_pool_size,
                            help='Size of tokenizer pool to use for '
                            'asynchronous tokenization. If 0, will '
                            'use synchronous tokenization.')
        parser.add_argument('--tokenizer-pool-type',
                            type=str,
                            default=EngineArgs.tokenizer_pool_type,
                            help='Type of tokenizer pool to use for '
                            'asynchronous tokenization. Ignored '
                            'if tokenizer_pool_size is 0.')
        parser.add_argument('--tokenizer-pool-extra-config',
                            type=nullable_str,
                            default=EngineArgs.tokenizer_pool_extra_config,
                            help='Extra config for tokenizer pool. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary. Ignored if '
                            'tokenizer_pool_size is 0.')

        # Multimodal related configs
        parser.add_argument(
            '--limit-mm-per-prompt',
            type=nullable_kvs,
            default=EngineArgs.limit_mm_per_prompt,
            # The default value is given in
            # MultiModalRegistry.init_mm_limits_per_prompt
            help=('For each multimodal plugin, limit how many '
                  'input instances to allow for each prompt. '
                  'Expects a comma-separated list of items, '
                  'e.g.: `image=16,video=2` allows a maximum of 16 '
                  'images and 2 videos per prompt. Defaults to 1 for '
                  'each modality.'))
        parser.add_argument(
            '--mm-processor-kwargs',
            default=None,
            type=json.loads,
            h
... [TRUNCATED 37203 chars]

=== sampling_params.py ===
# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for text generation."""
import copy
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Union

import msgspec
from pydantic import BaseModel
from typing_extensions import Annotated

from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# maybe make msgspec?
@dataclass
class GuidedDecodingParams:
    """One of these fields will be used to build a logit processor."""
    json: Optional[Union[str, Dict]] = None
    regex: Optional[str] = None
    choice: Optional[List[str]] = None
    grammar: Optional[str] = None
    json_object: Optional[bool] = None
    """These are other options that can be set"""
    backend: Optional[str] = None
    whitespace_pattern: Optional[str] = None

    @staticmethod
    def from_optional(
        json: Optional[Union[Dict, BaseModel, str]] = None,
        regex: Optional[str] = None,
        choice: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        json_object: Optional[bool] = None,
        backend: Optional[str] = None,
        whitespace_pattern: Optional[str] = None,
    ) -> Optional["GuidedDecodingParams"]:
        if all(arg is None
               for arg in (json, regex, choice, grammar, json_object)):
            return None
        # Extract json schemas from pydantic models
        if isinstance(json, (BaseModel, type(BaseModel))):
            json = json.model_json_schema()
        return GuidedDecodingParams(
            json=json,
            regex=regex,
            choice=choice,
            grammar=grammar,
            json_object=json_object,
            backend=backend,
            whitespace_pattern=whitespace_pattern,
        )

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([
            self.json is not None, self.regex is not None, self.choice
            is not None, self.grammar is not None, self.json_object is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")


class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOuputs
    FINAL_ONLY = 2


class SamplingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. By default,
            `best_of` is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        seed: Random seed to use for the generation.
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        bad_words: List of words that are not allowed to be generated.
            More precisely, only the last token of a corresponding
            token sequence is not allowed when the next generated token
            can complete the sequence.
        include_stop_str_in_output: Whether to include the stop strings in
            output text. Defaults to False.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens: Minimum number of tokens to generate per output sequence
            before EOS or stop_token_ids can be generated
        logprobs: Number of log probabilities to return per output token.
            When set to None, no probability is returned. If set to a non-None
            value, the result includes the log probabilities of the specified
            number of most likely tokens, as well as the chosen tokens.
            Note that the implementation follows the OpenAI API: The API will
            always return the log probability of the sampled token, so there
            may be up to `logprobs+1` elements in the response.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        detokenize: Whether to detokenize the output. Defaults to True.
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens in the output.  Defaults to True.
        logits_processors: List of functions that modify logits based on
            previously generated tokens, and optionally prompt tokens as
            a first argument.
        truncate_prompt_tokens: If set to an integer k, will use only the last k
            tokens from the prompt (i.e., left truncation). Defaults to None
            (i.e., no truncation).
        guided_decoding: If provided, the engine will construct a guided
            decoding logits processor from these parameters. Defaults to None.
        logit_bias: If provided, the engine will construct a logits processor
            that applies these logit biases. Defaults to None.
        allowed_token_ids: If provided, the engine will construct a logits
            processor which only retains scores for the given token ids.
            Defaults to None.
    """

    n: int = 1
    best_of: Optional[int] = None
    _real_n: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    bad_words: Optional[List[str]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    # Optional[List[LogitsProcessor]] type. We use Any here because
    # Optional[List[LogitsProcessor]] type is not supported by msgspec.
    logits_processors: Optional[Any] = None
    include_stop_str_in_output: bool = False
    truncate_prompt_tokens: Optional[Annotated[int, msgspec.Meta(ge=1)]] = None
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0
    _all_stop_token_ids: Set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    guided_decoding: Optional[GuidedDecodingParams] = None
    logit_bias: Optional[Dict[int, float]] = None
    allowed_token_ids: Optional[List[int]] = None

    @staticmethod
    def from_optional(
        n: Optional[int] = 1,
        best_of: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        repetition_penalty: Optional[float] = 1.0,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        bad_words: Optional[List[str]] = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: Optional[int] = 16,
        min_tokens: int = 0,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        logits_processors: Optional[List[LogitsProcessor]] = None,
        truncate_prompt_tokens: Optional[Annotated[int,
                                                   msgspec.Meta(ge=1)]] = None,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        guided_decoding: Optional[GuidedDecodingParams] = None,
        logit_bias: Optional[Union[Dict[int, float], Dict[str, float]]] = None,
        allowed_token_ids: Optional[List[int]] = None,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            logit_bias = {
                int(token): min(100.0, max(-100.0, bias))
                for token, bias in logit_bias.items()
            }

        return SamplingParams(
            n=1 if n is None else n,
            best_of=best_of,
            presence_penalty=0.0
            if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0
            if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            logits_processors=logits_processors,
            truncate_prompt_tokens=truncate_prompt_tokens,
            output_kind=output_kind,
            guided_decoding=guided_decoding,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
        )

    def __post_init__(self) -> None:
        # how we deal with `best_of``:
        # if `best_of`` is not set, we default to `n`;
        # if `best_of`` is set, we set `n`` to `best_of`,
        # and set `_real_n`` to the original `n`.
        # when we return the result, we will check
        # if we need to return `n` or `_real_n` results
        if self.best_of:
            if self.best_of < self.n:
                raise ValueError(
                    f"best_of must be greater than or equal to n, "
                    f"got n={self.n} and best_of={self.best_of}.")
            if not self._real_n:
                self._real_n = self.n
                self.n = self.best_of

        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature, _MAX_TEMP, _MAX_TEMP)
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None
        else:
            self.seed = self.seed

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        else:
            self.stop = list(self.stop)

        if self.stop_token_ids is None:
            self.stop_token_ids = []
        else:
            self.stop_token_ids = list(self.stop_token_ids)

        if self.bad_words is None:
            self.bad_words = []
        else:
            self.bad_words = list(self.bad_words)

        self.logprobs = 1 if self.logprobs is True else self.logprobs
        self.prompt_logprobs = (1 if self.prompt_logprobs is True else
                                self.prompt_logprobs)

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = -1
            self.min_p = 0.0
            self._verify_greedy_sampling()
        # eos_token_id is added to this by the engine
        self._all_stop_token_ids = set(self.stop_token_ids)

    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise ValueError(f"n must be an int, but is of "
                             f"type {type(self.n)}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2], got "
                             f"{self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if not isinstance(self.top_k, int):
            raise TypeError(
                f"top_k must be an integer, got {type(self.top_k).__name__}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1], got "
                             f"{self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, "
                             f"got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError(f"prompt_logprobs must be non-negative, got "
                             f"{self.prompt_logprobs}.")
        if (self.truncate_prompt_tokens is not None
                and self.truncate_prompt_tokens < 1):
            raise ValueError(f"truncate_prompt_tokens must be >= 1, "
                             f"got {self.truncate_prompt_tokens}")
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop.")
        if self.best_of != self._real_n and self.output_kind == (
                RequestOutputKind.DELTA):
            raise ValueError("best_of must equal n to use output_kind=DELTA")

    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise ValueError("n must be 1 when using greedy sampling, "
                             f"got {self.n}.")

    def update_from_generation_config(
            self,
            generation_config: Dict[str, Any],
            model_eos_token_id: Optional[int] = None) -> None:
        """Update if there are non-default values from generation_config"""

        if model_eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(model_eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if model_eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(model_eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def all_stop_token_ids(self) -> Set[int]:
        return self._all_stop_token_ids

    def clone(self) -> "SamplingParams":
        """Deep copy, but maybe not the LogitsProcessor objects.

        LogitsProcessor objects may contain an arbitrary, nontrivial amount of
        data that is expensive to copy. However, if not copied, the processor
        needs to support parallel decoding for multiple sequences
        See https://github.com/vllm-project/vllm/issues/3087
        """

        logit_processor_refs = None if self.logits_processors is None else {
            id(lp): lp.clone() if hasattr(lp, 'clone') else lp
            for lp in self.logits_processors
        }
        return copy.deepcopy(self, memo=logit_processor_refs)

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"truncate_prompt_tokens={self.truncate_prompt_tokens}, "
            f"guided_decoding={self.guided_decoding})")


class BeamSearchParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""
    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False



=== KNOWN GAPS FROM PRIOR REVIEW (post_trial_a_gap_closure.md) ===
G-vllm-1: EngineArgs.__post_init__ - ZERO raises, all normalisation
(if x is None: x = default). Walker only emits from `if X: raise`.
G-vllm-2: ModelConfig._verify_quantization / _verify_tokenizer_mode /
_verify_cuda_graph compare LOCAL variables (e.g. `quantization` not
`self.quantization`). Walker can't tie local-var-compare to a field
invariant without call-graph analysis.
G-vllm-3: CacheConfig._verify_cache_dtype - if/elif/else chain. Walker
only handles top-level if/raise pattern; doesn't descend into orelse
branches.


You may RE-CONFIRM known gaps (mark cross_engine_pattern as
yes-already-known) AND surface NEW ones (mark yes-new-here). Be
explicit which is which. Now emit the JSON. Start with `{`.
