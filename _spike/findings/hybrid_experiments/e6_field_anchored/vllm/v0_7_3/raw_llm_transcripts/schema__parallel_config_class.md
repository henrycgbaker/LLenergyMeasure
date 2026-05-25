# schema extraction transcript: parallel_config_class

- chunk_description: vllm.ParallelConfig dataclass (distributed engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 41.44
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
vllm library, version 0.7.3.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "vllm",
  "engine_version": "0.7.3",
  "chunk_name": "parallel_config_class",
  "chunk_fields": {
    "<field_name>": {
      "namespace": "<one of: engine_params, sampling_params, $defs.CompileConfig, ...>",
      "type": "<one of: string, integer, number, boolean, array, object, null>",
      "default": <value_or_null>,
      "description": "<brief one-liner>",
      "enum": [<values>],
      "anyOf": [{"type": "..."}, ...]
    }
  }
}

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params
(Other namespaces are extracted from other chunks. If you see fields
that belong to other namespaces, ignore them in this chunk.)

CRITICAL RULES:
1. Return ONLY the JSON document. NO markdown code fences (no ```).
   NO commentary, no preamble, no postamble. The first character of
   your response must be `{`.
2. Extract ONLY fields VISIBLE in the source below. Do not invent or
   hallucinate fields. If a field is referenced but its source is not
   shown, omit it.
3. Skip internal-plumbing fields: any name starting with `_`
   (e.g. `_commit_hash`, `_from_auto`) and these explicit names:
   `adapter_kwargs`, `model_kwargs`, `torch_dtype`.
4. For fields with NO clear type annotation, OMIT the "type" key. Do
   NOT guess. Only set "type" when the source explicitly annotates.
5. For Optional[X] / Union[X, None] / X | None: set "type" to X and
   "default" to null if applicable. If multiple non-null types use
   "anyOf": [{"type": "X"}, {"type": "Y"}, {"type": "null"}].
6. For typing.Union[A, B] without None: use "anyOf" not "type".
7. For defaults that are None, use null. For defaults that are
   complex objects, use null (the schema only carries simple defaults).
8. For Sphinx-documented kwargs (pulled by name from a `kwargs.pop(...)`
   call) the docstring usually documents the type and default - read
   it and emit accordingly. If the docstring is shown, USE IT.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference catalogue):

Example 1 (engine_params, simple bool):
  Source: `force_download: bool = False,`
  Emit: `"force_download": {"namespace": "engine_params", "type": "boolean", "default": false}`

Example 2 (engine_params, Optional Union with PathLike):
  Source: `cache_dir: Optional[Union[str, os.PathLike]] = None,`
  Emit: `"cache_dir": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null}`

Example 3 (engine_params, BitsAndBytesConfig field with default):
  Source: `bnb_4bit_compute_dtype=None,` (in BitsAndBytesConfig.__init__)
  Emit: `"bnb_4bit_compute_dtype": {"namespace": "engine_params", "default": null, "description": "BitsAndBytesConfig quantisation field"}`

Example 4 (sampling_params with enum from validate()):
  Source: GenerationConfig docstring mentions `cache_implementation (str, *optional*)` and validate() checks `not in ALL_CACHE_IMPLEMENTATIONS`.
  Emit: `"cache_implementation": {"namespace": "sampling_params", "type": "string"}`

Example 5 ($defs entry for CompileConfig):
  Source: CompileConfig dataclass with `fullgraph: bool = True`
  Emit: `"fullgraph": {"namespace": "$defs.CompileConfig", "type": "boolean", "default": true}`

Example 6 (sampling_params, unannotated default-None - common in GenerationConfig.__init__):
  Source: `temperature = kwargs.pop("temperature", None)` and docstring `temperature (float, *optional*, defaults to 1.0)`
  Emit: `"temperature": {"namespace": "sampling_params", "type": "number", "default": 1.0}`

CRITICAL: every field name MUST appear in the chunk_fields object.
Do NOT nest by namespace at the top level - the "namespace" key
inside each field's object is the namespace marker.

=== SOURCE: vllm.ParallelConfig (dataclass field list) ===
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1  # Number of pipeline parallel groups.
    tensor_parallel_size: int = 1  # Number of tensor parallel groups.

    # Maximum number of multiple batches
    # when load model sequentially. To avoid RAM OOM when using tensor
    # parallel and large models.
    max_parallel_loading_workers: Optional[int] = None

    # Disable the custom all-reduce kernel and fall back to NCCL.
    disable_custom_all_reduce: bool = False

    # Config for the tokenizer pool. If None, will use synchronous tokenization.
    tokenizer_pool_config: Optional[TokenizerPoolConfig] = None

    # Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.
    ray_workers_use_nsight: bool = False

    # ray distributed model workers placement group.
    placement_group: Optional["PlacementGroup"] = None

    # Backend to use for distributed model
    # workers, either "ray" or "mp" (multiprocessing). If the product
    # of pipeline_parallel_size and tensor_parallel_size is less than
    # or equal to the number of GPUs available, "mp" will be used to
    # keep processing on a single host. Otherwise, this will default
    # to "ray" if Ray is installed and fail otherwise. Note that tpu
    # and hpu only support Ray for distributed inference.
    distributed_executor_backend: Optional[Union[str,
                                                 Type["ExecutorBase"]]] = None

    # the full name of the worker class to use. If "auto", the worker class
    # will be determined based on the platform.
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"

    world_size: int = field(init=False)

    rank: int = 0

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: List[Any] = []
        factors.append(self.pipeline_parallel_size)
        factors.append(self.tensor_parallel_size)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        ray_only_devices = ["tpu"]
        from vllm.platforms import current_platform
        if (current_platform.device_type in ray_only_devices
                and self.world_size > 1):
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    f"{current_platform.device_type.upper()} backend only "
                    "supports Ray for distributed inference.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ra
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.7.3",
  "chunk_name": "parallel_config_class",
  "chunk_fields": {
    "pipeline_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Number of pipeline parallel groups."},
    "tensor_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Number of tensor parallel groups."},
    "max_parallel_loading_workers": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "Maximum number of multiple batches when load model sequentially."},
    "disable_custom_all_reduce": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Disable the custom all-reduce kernel and fall back to NCCL."},
    "tokenizer_pool_config": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null, "description": "Config for the tokenizer pool. If None, will use synchronous tokenization."},
    "ray_workers_use_nsight": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Whether to profile Ray workers with nsight."},
    "placement_group": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null, "description": "ray distributed model workers placement group."},
    "distributed_executor_backend": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "null"}], "default": null, "description": "Backend to use for distributed model workers."},
    "worker_cls": {"namespace": "engine_params", "type": "string", "default": "auto", "description": "the full name of the worker class to use. If \"auto\", the worker class will be determined based on the platform."},
    "sd_worker_cls": {"namespace": "engine_params", "type": "string", "default": "auto", "description": ""},
    "rank": {"namespace": "engine_params", "type": "integer", "default": 0, "description": ""}
  }
}
```
