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
      "example_field": "<field name in tensorrt that exhibits the gap>",
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

=== ENGINE: tensorrt vv0_21_0 ===

=== (a) OUTPUT (35 entries) ===
- id: tensorrt__autodeployllmargs_mla_backend_in_1_values
  severity: error
  native_type: tensorrt_llm._AutoDeployLlmArgs
  miner_source_method: <literal_field>
  match.fields: {"tensorrt.mla_backend": {"in": ["MultiHeadLatentAttention"]}}
  message: `mla_backend` must be one of ['MultiHeadLatentAttention']
- id: tensorrt__autodeployllmargs_model_factory_in_2_values
  severity: error
  native_type: tensorrt_llm._AutoDeployLlmArgs
  miner_source_method: <literal_field>
  match.fields: {"tensorrt.model_factory": {"in": ["AutoModelForCausalLM", "AutoModelForImageTextToText"]}}
  message: `model_factory` must be one of ['AutoModelForCausalLM', 'AutoModelForImageTextToText']
- id: tensorrt_autodeploy_free_mem_ratio_out_of_range
  severity: error
  native_type: tensorrt_llm._AutoDeployLlmArgs
  miner_source_method: validate_free_mem_ratio
  match.fields: {"tensorrt.free_mem_ratio": {"<": 0.0}}
  message: free_mem_ratio must be between 0.0 and 1.0, got {v}
- id: tensorrt_basellmargs_load_format_in_2_values
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: <literal_field>
  match.fields: {"tensorrt.load_format": {"in": ["auto", "dummy"]}}
  message: `load_format` must be one of ['auto', 'dummy']
- id: tensorrt_basellmargs_tokenizer_mode_in_2_values
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: <literal_field>
  match.fields: {"tensorrt.tokenizer_mode": {"in": ["auto", "slow"]}}
  message: `tokenizer_mode` must be one of ['auto', 'slow']
- id: tensorrt_batching_type_in_2_values
  severity: error
  native_type: tensorrt_llm.TrtLlmArgs
  miner_source_method: <strenum>
  match.fields: {"tensorrt.batching_type": {"in": ["STATIC", "INFLIGHT"]}}
  message: `batching_type` must be one of BatchingType members: ['STATIC', 'INFLIGHT']
- id: tensorrt_build_cache_max_records_ge_1
  severity: error
  native_type: tensorrt_llm.llmapi.build_cache.BuildCache
  miner_source_method: __init__
  match.fields: {"tensorrt.max_records": {"<": 1}}
  message: max_records should be greater than 0
- id: tensorrt_calibconfig_device_in_2_values
  severity: error
  native_type: tensorrt_llm.CalibConfig
  miner_source_method: <literal_field>
  match.fields: {"tensorrt.device": {"in": ["cuda", "cpu"]}}
  message: `device` must be one of ['cuda', 'cpu']
- id: tensorrt_quant_config_kv_cache_quant_algo_in_allowlist
  severity: error
  native_type: tensorrt_llm.QuantConfig
  miner_source_method: <KV_CACHE_QUANT_ALGO_LIST>
  match.fields: {"tensorrt.kv_cache_quant_algo": {"in": ["FP8", "INT8", "NVFP4", null]}}
  message: 
- id: tensorrt_quant_config_quant_algo_in_allowlist
  severity: error
  native_type: tensorrt_llm.QuantConfig
  miner_source_method: <QuantAlgo enum>
  match.fields: {"tensorrt.quant_algo": {"in": ["W8A16", "W4A16", "W4A16_AWQ", "W4A8_AWQ", "W8A16_GPTQ", "W4A16_GPTQ", "W8A8_SQ_PER_CHANNEL", "W8A8_SQ_PER_TENSOR_PLUGIN", "W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN", "W8A8
  message: 
- id: tensorrt_raises_cuda_graph_max_batch_size_lt_0_cuda_graph_max_batch_size
  severity: error
  native_type: tensorrt_llm.TorchLlmArgs
  miner_source_method: validate_cuda_graph_max_batch_size
  match.fields: {"tensorrt.cuda_graph_max_batch_size": {"<": 0}}
  message: cuda_graph_max_batch_size must be non-negative
- id: tensorrt_raises_cuda_graph_max_batch_size_ne_0_cuda_graph_config
  severity: error
  native_type: tensorrt_llm.TorchLlmArgs
  miner_source_method: validate_cuda_graph_config
  match.fields: {"tensorrt.cuda_graph_batch_sizes": {"present": true}, "tensorrt.cuda_graph_max_batch_size": {"!=": 0}}
  message: Please don't set both cuda_graph_batch_sizes and cuda_graph_max_batch_size.
cuda_graph_batch_sizes: {cuda_graph_batch_si
- id: tensorrt_raises_dtype_eq_bfloat16_dtype
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_dtype
  match.fields: {"tensorrt.dtype": "bfloat16"}
  message: Pre SM 80 GPUs do not support bfloat16
- id: tensorrt_raises_enable_build_cache_not_type_buildcacheconfig_enable_build_cache
  severity: error
  native_type: tensorrt_llm.TrtLlmArgs
  miner_source_method: validate_enable_build_cache
  match.fields: {"tensorrt.enable_build_cache": {"type_is_not": "BuildCacheConfig"}}
  message: Invalid build_cache_config: {enable_build_cache}
- id: tensorrt_raises_max_batch_size_set_True_build_config_with_runtime_params
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_build_config_with_runtime_params
  match.fields: {"tensorrt.max_batch_size": {"present": true}}
  message: max_batch_size [{max_batch_size}] is greater than build_config.max_batch_size [{self.build_config.max_batch_size}] in bu
- id: tensorrt_raises_max_ngram_size_le_0_positive_values
  severity: error
  native_type: tensorrt_llm.LookaheadDecodingConfig
  miner_source_method: validate_positive_values
  match.fields: {"tensorrt.max_ngram_size": {"<=": 0}}
  message: Value must be positive, got {v}
- id: tensorrt_raises_max_num_tokens_set_True_build_config_with_runtime_params
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_build_config_with_runtime_params
  match.fields: {"tensorrt.max_num_tokens": {"present": true}}
  message: max_num_tokens [{max_num_tokens}] is greater than build_config.max_num_tokens [{self.build_config.max_num_tokens}] in bu
- id: tensorrt_raises_max_verification_set_size_le_0_positive_values
  severity: error
  native_type: tensorrt_llm.LookaheadDecodingConfig
  miner_source_method: validate_positive_values
  match.fields: {"tensorrt.max_verification_set_size": {"<=": 0}}
  message: Value must be positive, got {v}
- id: tensorrt_raises_max_window_size_le_0_positive_values
  severity: error
  native_type: tensorrt_llm.LookaheadDecodingConfig
  miner_source_method: validate_positive_values
  match.fields: {"tensorrt.max_window_size": {"<=": 0}}
  message: Value must be positive, got {v}
- id: tensorrt_raises_model_not_type_model
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_model
  match.fields: {"tensorrt.model": {"type_is_not": ["str", "Path"]}}
  message: Invalid model: {v}
- id: tensorrt_raises_moe_load_balancer_type_str_moe_load_balancer
  severity: error
  native_type: tensorrt_llm.TorchLlmArgs
  miner_source_method: validate_moe_load_balancer
  match.fields: {"tensorrt.moe_load_balancer": {"type_is": "str"}}
  message: MoE load balancer config file not found: {moe_load_balancer}
- id: tensorrt_raises_speculative_config_set_True_speculative_config
  severity: error
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_speculative_config
  match.fields: {"tensorrt.speculative_config": {"present": true}}
  message: Speculative config type not recognized: {speculative_config}
- id: tensorrt_torch_llm_load_format_invalid
  severity: error
  native_type: tensorrt_llm.TorchLlmArgs
  miner_source_method: convert_load_format
  match.fields: {"tensorrt.load_format": {"not_in": ["AUTO", "DUMMY", "auto", "dummy"]}}
  message: Invalid LoadFormat: {v}
- id: tensorrt_warns_backend_in_lora_config_consistency
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_lora_config_consistency
  match.fields: {"tensorrt.enable_lora": {"present": true}, "tensorrt.lora_config": {"present": true}, "tensorrt.backend": {"in": ["pytorch", "_autodeploy"]}}
  message: enable_lora is ignored when lora_config is provided for {backend} backend.
- id: tensorrt_warns_build_config_set_True_model_format_misc
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_model_format_misc
  match.fields: {"tensorrt.backend": {"not_in": ["pytorch", "_autodeploy"]}, "tensorrt.build_config": {"present": true}}
  message: The build_config is ignored for model format of TLLM_ENGINE.
- id: tensorrt_warns_lora_config_set_True_lora_config_consistency
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_lora_config_consistency
  match.fields: {"tensorrt.lora_config": {"present": true}}
  message: max_loras is ignored when lora_config is provided.
- id: tensorrt_warns_max_batch_size_set_True_set_runtime_knobs_from_build_config
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: set_runtime_knobs_from_build_config
  match.fields: {"tensorrt.backend": "pytorch", "tensorrt.build_config": {"present": true}, "tensorrt.max_batch_size": {"present": true}}
  message: overriding {key} from build_config
- id: tensorrt_warns_max_beam_width_set_True_build_config_with_runtime_params
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_build_config_with_runtime_params
  match.fields: {"tensorrt.max_beam_width": {"present": true}}
  message: max_beam_width [{max_beam_width}] is overridden by build_config.max_beam_width [{self.build_config.max_beam_width}] in b
- id: tensorrt_warns_max_beam_width_set_True_set_runtime_knobs_from_build_config
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: set_runtime_knobs_from_build_config
  match.fields: {"tensorrt.backend": "pytorch", "tensorrt.build_config": {"present": true}, "tensorrt.max_beam_width": {"present": true}}
  message: overriding {key} from build_config
- id: tensorrt_warns_max_input_len_set_True_build_config_with_runtime_params
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_build_config_with_runtime_params
  match.fields: {"tensorrt.max_input_len": {"present": true}}
  message: max_input_len [{max_input_len}] is overridden by build_config.max_input_len [{self.build_config.max_input_len}] in build
- id: tensorrt_warns_max_input_len_set_True_set_runtime_knobs_from_build_config
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: set_runtime_knobs_from_build_config
  match.fields: {"tensorrt.backend": "pytorch", "tensorrt.build_config": {"present": true}, "tensorrt.max_input_len": {"present": true}}
  message: overriding {key} from build_config
- id: tensorrt_warns_max_lora_rank_set_True_lora_config_consistency
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_lora_config_consistency
  match.fields: {"tensorrt.lora_config": {"present": true}, "tensorrt.max_lora_rank": {"present": true}}
  message: max_lora_rank is ignored when lora_config is provided.
- id: tensorrt_warns_max_num_tokens_set_True_set_runtime_knobs_from_build_config
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: set_runtime_knobs_from_build_config
  match.fields: {"tensorrt.backend": "pytorch", "tensorrt.build_config": {"present": true}, "tensorrt.max_num_tokens": {"present": true}}
  message: overriding {key} from build_config
- id: tensorrt_warns_max_seq_len_set_True_build_config_with_runtime_params
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: validate_build_config_with_runtime_params
  match.fields: {"tensorrt.max_seq_len": {"present": true}}
  message: max_seq_len [{max_seq_len}] is overridden by build_config.max_seq_len [{self.build_config.max_seq_len}] in build_config
- id: tensorrt_warns_max_seq_len_set_True_set_runtime_knobs_from_build_config
  severity: warn
  native_type: tensorrt_llm.BaseLlmArgs
  miner_source_method: set_runtime_knobs_from_build_config
  match.fields: {"tensorrt.backend": "pytorch", "tensorrt.build_config": {"present": true}, "tensorrt.max_seq_len": {"present": true}}
  message: overriding {key} from build_config

=== ENGINE SOURCE EXCERPTS ===
=== llmapi/llm_args.py ===
import copy
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional,
                    Union)

import torch
import yaml
from pydantic import (BaseModel, Field, PrivateAttr, field_validator,
                      model_validator)
from strenum import StrEnum
from transformers import PreTrainedTokenizerBase

from tensorrt_llm.lora_manager import (LoraConfig,
                                       get_default_trtllm_modules_to_hf_modules)

from .._utils import mpi_rank
from ..auto_parallel import AutoParallelConfig, infer_cluster_config

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

# yapf: disable
# isort: off
from ..bindings.executor import (
                                 BatchingType as _BatchingType,
                                 CacheTransceiverConfig as _CacheTransceiverConfig,
                                 CapacitySchedulerPolicy as _CapacitySchedulerPolicy,
                                 ContextChunkingPolicy as _ContextChunkingPolicy,
                                 DecodingConfig,
                                 DecodingMode,
                                 DynamicBatchConfig as _DynamicBatchConfig,
                                 EagleConfig as _EagleConfig,
                                 ExecutorConfig as _ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig as _ExtendedRuntimePerfKnobConfig,
                                 KvCacheConfig as _KvCacheConfig,
                                 LookaheadDecodingConfig as _LookaheadDecodingConfig,
                                 PeftCacheConfig as _PeftCacheConfig,
                                 SchedulerConfig as _SchedulerConfig) # isort: skip
# isort: on

# yapf: enable
from ..builder import BuildConfig, EngineConfig
from ..logger import logger
from ..mapping import Mapping
from ..models.automodel import AutoConfig
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..sampling_params import BatchedLogitsProcessor
from .build_cache import BuildCacheConfig
from .tokenizer import TokenizerBase, tokenizer_factory
from .utils import (generate_api_docs_as_docstring, get_type_repr,
                    print_traceback_on_error)

# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import


@dataclass
class _ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    gpus_per_node: int = 8
    moe_cluster_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 1
    cp_config: dict = field(default_factory=dict)
    enable_attention_dp: bool = False
    auto_parallel: bool = False

    _world_size: int = field(default=1, init=False)
    _devices: Optional[List[int]] = field(default=None, init=False)

    @property
    def devices(self) -> List[int]:
        if self._devices is None:
            return list(range(self.world_size))
        return self._devices

    @devices.setter
    def devices(self, devices: List[int]):
        if len(devices) != self.world_size:
            raise ValueError(
                f"devices {devices} should have the same length as world_size {self.world_size}"
            )
        self._devices = devices

    @property
    def world_size(self) -> bool:

        if self.auto_parallel:
            if self.tp_size > 1 or self.pp_size > 1 or self.cp_size > 1:
                raise RuntimeError(
                    "manually TP and PP are not supported in auto parallel mode."
                )
            return self._world_size

        if self._world_size > 1:
            raise RuntimeError(
                "world_size > 1 is only supported in auto parallel mode.")
        return self.tp_size * self.pp_size * self.cp_size

    @property
    def world_size_per_node(self) -> int:
        world_size = self.world_size
        total_nodes = math.ceil(world_size / self.gpus_per_node)
        return world_size // total_nodes  #TODO is this right?

    @world_size.setter
    def world_size(self, world_size: int):
        if self.auto_parallel:
            self._world_size = world_size
        elif (not self.auto_parallel
              ) and world_size != self.tp_size * self.pp_size * self.cp_size:
            raise ValueError(
                f"world_size {world_size} should be equal to tp_size * pp_size {self.tp_size * self.pp_size * self.cp_size} "
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1

    def to_mapping(self) -> Mapping:
        return Mapping(world_size=self.world_size,
                       rank=mpi_rank(),
                       gpus_per_node=self.gpus_per_node,
                       tp_size=self.tp_size,
                       pp_size=self.pp_size,
                       cp_size=self.cp_size,
                       cp_config=self.cp_config,
                       enable_attention_dp=self.enable_attention_dp,
                       moe_cluster_size=self.moe_cluster_size,
                       moe_tp_size=self.moe_tp_size,
                       moe_ep_size=self.moe_ep_size,
                       auto_parallel=self.auto_parallel)


class CalibConfig(BaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")

    @classmethod
    def from_dict(cls, config: dict) -> 'CalibConfig':
        """Create a CalibConfig instance from a dict.

        Args:
            config (dict): The dict used to create CalibConfig.

        Returns:
            tensorrt_llm.llmapi.CalibConfig: The CalibConfig created from dict.
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Dump a CalibConfig instance to a dict.

        Returns:
            dict: The dict dumped from CalibConfig.
        """
        return self.model_dump()


class _ModelFormatKind(Enum):
    HF = 0
    TLLM_CKPT = 1
    TLLM_ENGINE = 2


class DecodingBaseConfig(BaseModel):
    max_draft_len: Optional[int] = None
    speculative_model: Optional[Union[str, Path]] = None

    @classmethod
    def from_dict(cls, data: dict):
        # dispatch to the correct decoding config
        decoding_type = data.get("decoding_type")
        config_classes = {
            "MTP": MTPDecodingConfig,
            "Medusa": MedusaDecodingConfig,
            "Eagle": EagleDecodingConfig,
            "Lookahead": LookaheadDecodingConfig,
            "NGram": NGramDecodingConfig,
            "DraftTarget": DraftTargetDecodingConfig,
        }

        config_class = config_classes.get(decoding_type)
        if config_class is None:
            raise ValueError(f"Invalid decoding type: {decoding_type}")

        return config_class(**data)

    def _check_fields(self):
        pass


class MedusaDecodingConfig(DecodingBaseConfig):
    medusa_choices: Optional[List[List[int]]] = None
    num_medusa_heads: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Medusa"


class EagleDecodingConfig(DecodingBaseConfig):
    eagle_choices: Optional[List[List[int]]] = None
    greedy_sampling: Optional[bool] = True
    posterior_threshold: Optional[float] = None
    use_dynamic_tree: Optional[bool] = False
    dynamic_tree_max_topK: Optional[int] = None
    num_eagle_layers: Optional[int] = None
    max_non_leaves_per_layer: Optional[int] = None
    pytorch_weights_path: Optional[str] = None
    eagle3_one_model: Optional[bool] = True

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Eagle"


class NGramDecodingConfig(DecodingBaseConfig):
    """
    Configuration for NGram drafter speculative decoding.

    Arguments:
        prompt_lookup_num_tokens: int
                The length maximum of draft tokens (can be understood as length maximum of output draft tokens).

        max_matching_ngram_size: int
            The length maximum of searching tokens (can be understood as length maximum of input tokens to search).

        is_keep_all: bool = True
            Whether to keep all candidate pattern-matches pairs, only one match is kept for each pattern if False.

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit, the newest one is provided if False.

        is_public_pool: bool = True
            Whether to use a common pool for all requests, or the pool is private for each request if False.
    """

    prompt_lookup_num_tokens: int = 2
    max_matching_ngram_size: int = 4
    is_keep_all: bool = True
    is_use_oldest: bool = True
    is_public_pool: bool = True

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "NGram"


class DraftTargetDecodingConfig(DecodingBaseConfig):
    pytorch_weights_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "DraftTarget"


class MTPDecodingConfig(DecodingBaseConfig):
    num_nextn_predict_layers: Optional[int] = 1
    use_relaxed_acceptance_for_thinking: Optional[bool] = False
    relaxed_topk: Optional[int] = 1
    relaxed_delta: Optional[float] = 0.

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "MTP"


class PybindMirror(ABC):
    ''' A class containing the utilities for mirroring Python classes to
    pybinding classes.
    '''

    @abstractmethod
    def _to_pybind(self):
        pass

    @staticmethod
    def maybe_to_pybind(ins):
        if isinstance(
                ins,
                PybindMirror) or type(ins).__class__ == PybindMirrorEnumMeta:
            return ins._to_pybind()
        return ins

    @staticmethod
    def mirror_pybind_fields(pybind_class):
        """
        Class decorator that ensures Python class fields mirror those of a C++ class.

        Args:
            pybind_class: The C++ class whose fields should be mirrored

        Returns:
            A decorator function that validates field mirroring
        """

        def decorator(cls):
            assert issubclass(cls, BaseModel)
            # Get all non-private fields from the C++ class
            cpp_fields = PybindMirror.get_pybind_variable_fields(pybind_class)
            python_fields = set(cls.model_fields.keys())

            # Check if all C++ fields exist in the Python class
            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )

            # Return the original class
            return cls

        return decorator

    @staticmethod
    def get_pybind_enum_fields(pybind_class):
        ''' Get all the enum fields from the pybind class. '''
        return [
            f for f in pybind_class.__members__.keys()
            if not f.startswith('_') and not callable(getattr(pybind_class, f))
        ]

    @staticmethod
    def mirror_pybind_enum(pybind_class):
        ''' Mirror the enum fields from the pybind class to the Python class. '''

        def decorator(cls):
            assert issubclass(cls, Enum)
            cpp_fields = PybindMirror.get_pybind_enum_fields(pybind_class)
            python_fields = set(cls.__members__.keys())

            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )
            return cls

        return decorator

    @staticmethod
    def get_pybind_variable_fields(config_cls):
        ''' Get all the variable fields from the pybind class. '''
        return [
            f for f in dir(config_cls)
            if not f.startswith('_') and not callable(getattr(config_cls, f))
        ]

    @staticmethod
    def pybind_equals(obj0, obj1):
        ''' Check if two pybind objects are equal. '''
        assert type(obj0) is type(obj1)
        for field in PybindMirror.get_pybind_variable_fields(type(obj0)):
            if getattr(obj0, field) != getattr(obj1, field):
                return False
        return True


class PybindMirrorMeta(type(PybindMirror)):
    pass


class PybindMirrorEnumMeta(EnumMeta, PybindMirrorMeta):
    """
    Combined metaclass for Enum and PybindMirror.  This is crucial.
    """


@PybindMirror.mirror_pybind_enum(_BatchingType)
class BatchingType(StrEnum, metaclass=PybindMirrorEnumMeta):
    STATIC = "STATIC"
    INFLIGHT = "INFLIGHT"

    def _to_pybind(self):
        return getattr(_BatchingType, self.value)


@PybindMirror.mirror_pybind_enum(_CapacitySchedulerPolicy)
class CapacitySchedulerPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)


@PybindMirror.mirror_pybind_enum(_ContextChunkingPolicy)
class ContextChunkingPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    ''' Context chunking policy. '''
    FIRST_COME_FIRST_SERVED = "FIRST_COME_FIRST_SERVED"
    EQUAL_PROGRESS = "EQUAL_PROGRESS"

    def _to_pybind(self):
        return getattr(_ContextChunkingPolicy, self.value)


@PybindMirror.mirror_pybind_fields(_DynamicBatchConfig)
class DynamicBatchConfig(BaseModel, PybindMirror):
    """Dynamic batch configuration.

    Controls how batch size and token limits are dynamically adjusted at runtime.
    """
    enable_batch_size_tuning: bool = Field(
        description="Controls if the batch size should be tuned dynamically")

    enable_max_num_tokens_tuning: bool = Field(
        description="Controls if the max num tokens should be tuned dynamically"
    )

    dynamic_batch_moving_average_window: int = Field(
        description=
        "The window size for moving average of input and output length which is used to calculate dynamic batch size and max num tokens"
    )

    def _to_pybind(self):
        return _DynamicBatchConfig(
            enable_batch_size_tuning=self.enable_batch_size_tuning,
            enable_max_num_tokens_tuning=self.enable_max_num_tokens_tuning,
            dynamic_batch_moving_average_window=self.
            dynamic_batch_moving_average_window)


@PybindMirror.mirror_pybind_fields(_SchedulerConfig)
class SchedulerConfig(BaseModel, PybindMirror):
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")

    context_chunking_policy: Optional[ContextChunkingPolicy] = Field(
        default=None, description="The context chunking policy to use")

    dynamic_batch_config: Optional[DynamicBatchConfig] = Field(
        default=None, description="The dynamic batch config to use")

    def _to_pybind(self):
        return _SchedulerConfig(
            capacity_scheduler_policy=self.capacity_scheduler_policy._to_pybind(
            ),
            context_chunking_policy=self.context_chunking_policy._to_pybind()
            if self.context_chunking_policy else None,
            dynamic_batch_config=self.dynamic_batch_config._to_pybind()
            if self.dynamic_batch_config else None)


@PybindMirror.mirror_pybind_fields(_PeftCacheConfig)
class PeftCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the PEFT cache.
    """
    num_host_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
    )
    num_device_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module sets of weights that can be stored in host cache"
    )
    optimal_adapter_size: int = Field(
        default=
        8,  # There are tests to keep the default value consistent with the pybind default value
        description="optimal adapter size used to set page width")
    max_adapter_size: int = Field(
        default=64,
        description="max supported adapter size. Used to compute minimum")
    num_put_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to put weights into host cache")
    num_ensure_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to copy weights from host to device")
    num_copy_streams: int = Field(
        default=1,
        description="number of streams used to copy weights from host to device"
    )
    max_pages_per_block_host: int = Field(
        default=24,
        description="Number of cache pages per allocation block (host)")
    max_pages_per_block_device: int = Field(
        default=8,
        description="Number of cache pages per allocation block (device)")
    device_cache_percent: Optional[float] = Field(
        default=None,
        description="percent of memory after engine load to use for cache")
    host_cache_size: Optional[int] = Field(
        default=None, description="size in bytes to use for host cache")
    lora_prefetch_dir: Optional[str] = Field(
        default=None,
        description=
        "folder to store the LoRA weights we hope to load during engine initialization"
    )

    def _to_pybind(self):
        return _PeftCacheConfig(
            num_host_module_layer=self.num_host_module_layer,
            num_device_module_layer=self.num_device_module_layer,
            optimal_adapter_size=self.optimal_adapter_size,
            max_adapter_size=self.max_adapter_size,
            num_put_workers=self.num_put_workers,
            num_ensure_workers=self.num_ensure_workers,
            num_copy_streams=self.num_copy_streams,
            max_pages_per_block_host=self.max_pages_per_block_host,
            max_pages_per_block_device=self.max_pages_per_block_device,
            device_cache_percent=self.device_cache_percent,
            host_cache_size=self.host_cache_size,
            lora_prefetch_dir=self.lora_prefetch_dir)


@PybindMirror.mirror_pybind_fields(_LookaheadDecodingConfig)
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    max_window_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: int = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @field_validator('max_window_size', 'max_ngram_size',
                     'max_verification_set_size')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._check_fields()

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def _to_pybind(self):
        return _LookaheadDecodingConfig(self.max_window_size,
                                        self.max_ngram_size,
                                        self.max_verification_set_size)

    decoding_type: ClassVar[str] = "Lookahead"


@PybindMirror.mirror_pybind_fields(_KvCacheConfig)
class KvCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the KV cache.
    """
    enable_block_reuse: bool = Field(
        default=True,
        description=
        "Controls if KV cache blocks can be reused for different requests.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=
        "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    max_attention_window: Optional[List[int]] = Field(
        default=None,
        description=
        "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    )
    sink_token_length: Optional[int] = Field(
        default=None,
        description=
        "Number of sink tokens (tokens to always keep in attention window).")
    free_gpu_memory_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    host_cache_size: Optional[int] = Field(
        default=None,
        description=
        "Size of the host cache in bytes. If both `max_tokens` and `host_cache_size` are specified, memory corresponding to the minimum will be used."
    )
    onboard_blocks: bool = Field(
        default=True, description="Controls if blocks are onboarded.")
    cross_kv_cache_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and cross attention will use p of KV Cache memory. Default is 50%. Should only be set when using encoder-decoder model."
    )
    secondary_offload_min_priority: Optional[int] = Field(
        default=None,
        description=
        "Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory."
    )
    event_buffer_max_size: int = Field(
        default=0,
        description=
        "Maximum size of the event buffer. If set to 0, the event buffer will not be used."
    )
    enable_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether blocks that are only partially matched can be reused.")
    copy_on_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether partially matched blocks that are in use can be reused after copying them."
    )

    def _to_pybind(self):
        return _KvCacheConfig(
            enable_block_reuse=self.enable_block_reuse,
            max_tokens=self.max_tokens,
            max_attention_window=self.max_attention_window,
            sink_token_length=self.sink_token_length,
            free_gpu_memory_fraction=self.free_gpu_memory_fraction,
            host_cache_size=self.host_cache_size,
            onboard_blocks=self.onboard_blocks,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
            secondary_offload_min_priority=self.secondary_offload_min_priority,
            event_buffer_max_size=self.event_buffer_max_size,
            enable_partial_reuse=self.enable_partial_reuse,
            copy_on_partial_reuse=self.copy_on_partial_reuse)


@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(BaseModel, PybindMirror):
    """
    Configuration for extended runtime performance knobs.
    """

    multi_block_mode: bool = Field(
        default=True, description="Whether to use multi-block mode.")

    enable_context_fmha_fp32_acc: bool = Field(
        default=False,
        description="Whether to enable context FMHA FP32 accumulation.")

    cuda_graph_mode: bool = Field(default=False,
                                  description="Whether to use CUDA graph mode.")

    cuda_graph_cache_size: int = Field(
        default=0,
        description=
        "Number of cuda graphs to be cached in the runtime. The larger the cache, the better the perf, but more GPU memory is consumed."
    )

    def _to_pybind(self):
        res = _ExtendedRuntimePerfKnobConfig(
            multi_block_mode=self.multi_block_mode,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc)
        res.cuda_graph_mode = self.cuda_graph_mode
        res.cuda_graph_cache_size = self.cuda_graph_cache_size
        return res


@PybindMirror.mirror_pybind_fields(_CacheTransceiverConfig)
class CacheTransceiverConfig(BaseModel, PybindMirror):
    """
    Configuration for the cache transceiver.
    """
    max_num_tokens: Optional[int] = Field(
        default=None,
        description="The max number of tokens the transfer buffer can fit.")

    def _to_pybind(self):
        return _CacheTransceiverConfig(max_num_tokens=self.max_num_tokens)


@dataclass
class _ModelWrapper:
    model: Union[str, Path]

    def __post_init__(self):
        if not self.model:
            raise ValueError("model should be provided.")
        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"

        model_dir = Path(self.model)

        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir

    @property
    def is_hub_model(self) -> bool:
        return not self.is_local_model

    @property
    def is_local_model(self) -> bool:
        return isinstance(self.model, Path)

    @property
    def model_dir(self) -> Path:
        assert self.is_local_model, f"model_dir is only available for local model, {self.model}."
        return self.model

    @model_dir.setter
    def model_dir(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        assert model_dir.exists() and model_dir.is_dir(
        ), f"model_dir is not a valid path, {model_dir}"
        self.model = model_dir

    @property
    def model_name(self) -> Union[str, Path]:
        return self.model if isinstance(self.model, str) else None


class BaseLlmArgs(BaseModel):
    """
    Base class for both TorchLlmArgs and TrtLlmArgs. It contains all the arguments that are common to both.
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    # Explicit arguments
    model: Union[str, Path] = Field(
        description=
        "The path to the model checkpoint or the model name from the Hugging Face Hub."
    )

    tokenizer: Optional[Union[
        str, Path, TokenizerBase, PreTrainedTokenizerBase]] = Field(
            description=
            "The path to the tokenizer checkpoint or the tokenizer name from the Hugging Face Hub.",
            default=None)

    tokenizer_mode: Literal['auto', 'slow'] = Field(
        default='auto',
        description="The mode to initialize the tokenizer.",
        json_schema_extra={"type": "Literal['auto', 'slow']"})

    skip_tokenizer_init: bool = Field(
        default=False,
        description="Whether to skip the tokenizer initialization.")

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust the remote code.")

    tensor_parallel_size: int = Field(default=1,
                                      description="The tensor parallel size.")

    dtype: str = Field(default="auto",
                       description="The data type to use for the model.")

    revision: Optional[str] = Field(
        default=None, description="The revision to use for the model.")

    tokenizer_revision: Optional[str] = Field(
        default=None, description="The revision to use for the tokenizer.")

    # Below are all remaining arguments

    pipeline_parallel_size: int = Field(
        default=1, description="The pipeline parallel size.")

    context_parallel_size: int = Field(default=1,
                                       description="The context parallel size.")

    gpus_per_node: Optional[int] = Field(
        default=None,
        description="The number of GPUs per node.",
        validate_default=True)

    moe_cluster_parallel_size: Optional[int] = Field(
        default=None,
        description="The cluster parallel size for MoE models's expert weights."
    )

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor pa
... [TRUNCATED 53752 chars]

=== llmapi/build_cache.py ===
import contextlib
import datetime
import enum
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import filelock

import tensorrt_llm
from tensorrt_llm import BuildConfig
from tensorrt_llm.llmapi.utils import enable_llm_debug, print_colored
from tensorrt_llm.logger import logger


def get_build_cache_config_from_env() -> tuple[bool, str]:
    """
    Get the build cache configuration from the environment variables
    """
    build_cache_enabled = os.environ.get('TLLM_LLMAPI_BUILD_CACHE') == '1'
    build_cache_root = os.environ.get(
        'TLLM_LLMAPI_BUILD_CACHE_ROOT',
        '/tmp/.cache/tensorrt_llm/llmapi/')  # nosec B108
    return build_cache_enabled, build_cache_root


class BuildCacheConfig:
    """
    Configuration for the build cache.

    Attributes:
        cache_root (str): The root directory for the build cache.
        max_records (int): The maximum number of records to store in the cache.
        max_cache_storage_gb (float): The maximum amount of storage (in GB) to use for the cache.

    Note:
        The build-cache assumes the weights of the model are not changed during the execution. If the weights are
        changed, you should remove the caches manually.
    """

    def __init__(self,
                 cache_root: Optional[Path] = None,
                 max_records: int = 10,
                 max_cache_storage_gb: float = 256):
        self._cache_root = cache_root
        self._max_records = max_records
        self._max_cache_storage_gb = max_cache_storage_gb

    @property
    def cache_root(self) -> Path:
        _build_cache_enabled, _build_cache_root = get_build_cache_config_from_env(
        )
        return self._cache_root or Path(_build_cache_root)

    @property
    def max_records(self) -> int:
        return self._max_records

    @property
    def max_cache_storage_gb(self) -> float:
        return self._max_cache_storage_gb


class BuildCache:
    """
    The BuildCache class is a class that manages the intermediate products from the build steps.

    NOTE: currently, only engine-building is supported
    TODO[chunweiy]: add support for other build steps, such as quantization, convert_checkpoint, etc.
    """
    # The version of the cache, will be used to determine if the cache is compatible
    CACHE_VERSION = 0

    def __init__(self, config: Optional[BuildCacheConfig] = None):

        _, default_cache_root = get_build_cache_config_from_env()
        config = config or BuildCacheConfig()

        self.cache_root = config.cache_root or Path(default_cache_root)
        self.max_records = config.max_records
        self.max_cache_storage_gb = config.max_cache_storage_gb

        if config.max_records < 1:
            raise ValueError("max_records should be greater than 0")

    def free_storage_in_gb(self) -> float:
        ''' Get the free storage capacity of the cache. '''
        # measure the root directory
        if self.cache_root.parent.exists():
            usage = shutil.disk_usage(self.cache_root.parent)
            return usage.free / 1024**3
        return 0

    def get_engine_building_cache_stage(self,
                                        build_config: BuildConfig,
                                        model_path: Optional[Path] = None,
                                        force_rebuild: bool = False,
                                        **kwargs) -> 'CachedStage':
        '''
        Get the build step for engine building.
        '''
        build_config_str = json.dumps(self.prune_build_config_for_cache_key(
            build_config.to_dict()),
                                      sort_keys=True)

        kwargs_str = json.dumps(kwargs, sort_keys=True)

        return CachedStage(parent=self,
                           kind=CacheRecord.Kind.Engine,
                           cache_root=self.cache_root,
                           force_rebuild=force_rebuild,
                           inputs=[build_config_str, model_path, kwargs_str])

    def prune_caches(self, has_incoming_record: bool = False):
        '''
        Clean up the cache records to make sure the cache size is within the limit

        Args:
            has_incoming_record (bool): If the cache has incoming record, the existing records will be further pruned to
            reserve space for the incoming record
        '''
        if not self.cache_root.exists():
            return
        self._clean_up_cache_dir()
        records = []
        for dir in self.cache_root.iterdir():
            records.append(self._load_cache_record(dir))
        records.sort(key=lambda x: x.time, reverse=True)
        max_records = self.max_records - 1 if has_incoming_record else self.max_records
        # prune the cache to meet max_records and max_cache_storage_gb limitation
        while len(records) > max_records or sum(
                r.storage_gb for r in records) > self.max_cache_storage_gb:
            record = records.pop()
            # remove the directory and its content
            shutil.rmtree(record.path)

    @staticmethod
    def prune_build_config_for_cache_key(build_config: dict) -> dict:
        # The BuildCache will be disabled once auto_pp is enabled, so 'auto_parallel_config' should be removed
        black_list = ['auto_parallel_config', 'dry_run']
        dic = build_config.copy()
        for key in black_list:
            if key in dic:
                dic.pop(key)
        return dic

    def load_cache_records(self) -> List["CacheRecord"]:
        '''
        Load all the cache records from the cache directory
        '''
        records = []
        if not self.cache_root.exists():
            return records

        for dir in self.cache_root.iterdir():
            records.append(self._load_cache_record(dir))
        return records

    def _load_cache_record(self, cache_dir) -> "CacheRecord":
        '''
        Get the cache record from the cache directory
        '''
        metadata = json.loads((cache_dir / 'metadata.json').read_text())
        storage_gb = sum(f.stat().st_size for f in cache_dir.glob('**/*')
                         if f.is_file()) / 1024**3
        return CacheRecord(kind=CacheRecord.Kind.__members__[metadata['kind']],
                           storage_gb=storage_gb,
                           path=cache_dir,
                           time=datetime.datetime.fromisoformat(
                               metadata['datetime']))

    def _clean_up_cache_dir(self):
        '''
        Clean up the files in the cache directory, remove anything that is not in the cache
        '''
        # get all the files and directies in the cache_root
        if not self.cache_root.exists():
            return
        for file_or_dir in self.cache_root.iterdir():
            if not self.is_cache_valid(file_or_dir):
                logger.info(f"Removing invalid cache directory {dir}")
                if file_or_dir.is_file():
                    file_or_dir.unlink()
                else:
                    shutil.rmtree(file_or_dir)

    def is_cache_valid(self, cache_dir: Path) -> bool:
        '''
        Check if the cache directory is valid
        '''
        if not cache_dir.exists():
            return False

        metadata_path = cache_dir / 'metadata.json'
        if not metadata_path.exists():
            return False

        metadata = json.loads(metadata_path.read_text())
        if metadata.get('version') != BuildCache.CACHE_VERSION:
            return False

        content = cache_dir / 'content'
        if not content.exists():
            return False

        return True


@dataclass
class CachedStage:
    '''
    CachedStage is a class that represents a stage in the build process, it helps to manage the intermediate product.

    The cache is organized as follows:

    this_cache_dir/     # name is like "engine-<hash>"
        metadata.json   # the metadata of the cache
        content/        # the actual product of the build step, such trt-llm engine directory
    '''
    # The parent should be kept alive by CachedStep instance
    parent: BuildCache
    cache_root: Path
    # The inputs will be used to determine if the step needs to be re-run, so all the variables should be put here
    inputs: List[Any]
    kind: "CacheRecord.Kind"
    # If force_rebuild is set to True, the cache will be ignored
    force_rebuild: bool = False

    def get_hash_key(self):
        lib_version = tensorrt_llm.__version__
        input_strs = [str(i) for i in self.inputs]
        return hashlib.md5(
            f"{lib_version}-{input_strs}".encode()).hexdigest()  # nosec B324

    def get_cache_path(self) -> Path:
        '''
        The path to the product of the build step, will be overwritten if the step is re-run
        '''
        return self.cache_root / f"{self.kind.value}-{self.get_hash_key()}"

    def get_engine_path(self) -> Path:
        return self.get_cache_path() / 'content'

    def get_cache_metadata(self) -> dict:
        res = {
            "version": BuildCache.CACHE_VERSION,
            "datetime": datetime.datetime.now().isoformat(),
            "kind": self.kind.name,
        }
        return res

    def is_cached(self) -> bool:
        '''
        Check if the product of the build step is in the cache
        '''
        if self.force_rebuild:
            return False
        try:
            if self.get_cache_path().exists():
                metadata = json.loads(
                    (self.get_cache_path() / 'metadata.json').read_text())
                if metadata["version"] == BuildCache.CACHE_VERSION:
                    return True
        except:
            pass

        return False

    @contextlib.contextmanager
    def write_guard(self):
        ''' Guard the cache writing process.

        The cache writing process should be atomic, so the filelock is used to protect the cache writing process. And
        the cache metadata will be written to the cache directory.

        Args:
            final_engien_dir: the final engine directory
        '''
        self.parent.prune_caches(has_incoming_record=True)

        target_dir = self.get_cache_path()
        if enable_llm_debug():
            print_colored(f"Writing cache to {target_dir}\n", "yellow")

        # To avoid the cache modification conflict, a dummy directory is used to write the cache, and then rename it to
        # the target directory
        dummy_target_dir = Path(f"{target_dir.parent}/{target_dir.name}.dummy")

        dummy_target_dir.mkdir(parents=True, exist_ok=True)
        # TODO[chunweiy]: deal with the cache modification conflict
        lock = filelock.FileLock(dummy_target_dir / '.filelock', timeout=10)

        with open(dummy_target_dir / 'metadata.json', 'w') as f:
            f.write(json.dumps(self.get_cache_metadata()))

        with lock:
            yield dummy_target_dir / 'content'

            # If engine building is successful, rename the dummy directory to the target directory
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(dummy_target_dir, target_dir)


@dataclass(unsafe_hash=True)
class CacheRecord:
    '''
    CacheRecord is a class that represents a record in the cache directory.
    '''

    class Kind(enum.Enum):
        Engine = 'engine'
        Checkpoint = 'checkpoint'

    kind: Kind
    storage_gb: float
    path: Path
    time: datetime.datetime



=== KNOWN GAPS FROM PRIOR REVIEW (post_trial_a_gap_closure.md) ===
G-trt-1: type-blind probe synthesis - `_value_satisfying("present",
True)` returns `"x"` even for int-typed fields. Probe construction
needs declared-type information.
G-trt-2: DeprecationWarning poisoning - NOT a walker gap; lives in
validation-emission capture. Out of scope for diagnosis here.
G-trt-3: nested-config dispatch - SchedulerConfig, QuantConfig,
KvCacheConfig are Pydantic-validator-bearing nested classes the walker
doesn't recurse into via class-level type references.


You may RE-CONFIRM known gaps (mark cross_engine_pattern as
yes-already-known) AND surface NEW ones (mark yes-new-here). Be
explicit which is which. Now emit the JSON. Start with `{`.
