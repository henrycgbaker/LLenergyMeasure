# schema extraction transcript: parallel_config_class

- chunk_description: vllm.ParallelConfig dataclass (distributed engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 111.41
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
vllm library, version 0.6.0.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "vllm",
  "engine_version": "0.6.0",
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
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Deprecated, use distributed_executor_backend instead.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
        tokenizer_pool_config: Config for the tokenizer pool.
            If None, will use synchronous tokenization.
        ray_workers_use_nsight: Whether to profile Ray workers with nsight, see
            https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.
        placement_group: ray distributed model workers placement group.
        distributed_executor_backend: Backend to use for distributed model
            workers, either "ray" or "mp" (multiprocessing). If either
            pipeline_parallel_size or tensor_parallel_size is greater than 1,
            will default to "ray" if Ray is installed or "mp" otherwise.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: Optional[bool] = None,
        max_parallel_loading_workers: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
        tokenizer_pool_config: Optional[TokenizerPoolConfig] = None,
        ray_workers_use_nsight: bool = False,
        placement_group: Optional["PlacementGroup"] = None,
        distributed_executor_backend: Optional[Union[
            str, Type["ExecutorBase"]]] = None,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.tokenizer_pool_config = tokenizer_pool_config
        self.ray_workers_use_nsight = ray_workers_use_nsight
        self.placement_group = placement_group
        self.world_size = pipeline_parallel_size * self.tensor_parallel_size

        if worker_use_ray:
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            elif not self.use_ray:
                raise ValueError(f"worker-use-ray can't be used with "
                                 f"distributed executor backend "
                                 f"'{self.distributed_executor_backend}'.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if cuda_device_count_stateless() < self.world_size:
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_curr
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.6.0",
  "chunk_name": "parallel_config_class",
  "chunk_fields": {
    "pipeline_parallel_size": {"namespace": "engine_params", "type": "integer"},
    "tensor_parallel_size": {"namespace": "engine_params", "type": "integer"},
    "worker_use_ray": {"namespace": "engine_params", "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null},
    "max_parallel_loading_workers": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "disable_custom_all_reduce": {"namespace": "engine_params", "type": "boolean", "default": false},
    "tokenizer_pool_config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "TokenizerPoolConfig"}, {"type": "null"}], "default": null},
    "ray_workers_use_nsight": {"namespace": "engine_params", "type": "boolean", "default": false},
    "placement_group": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "PlacementGroup"}, {"type": "null"}], "default": null},
    "distributed_executor_backend": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "ExecutorBase subclass"}, {"type": "null"}], "default": null}
  }
}
```
