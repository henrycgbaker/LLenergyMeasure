# schema extraction transcript: build_cache_config_class

- chunk_description: tensorrt_llm.BuildCacheConfig + BuildCache (classic class with __init__)
- expected_namespaces: ['engine_params', '$defs.BuildCacheConfig']
- attempts: 1
- elapsed_sec: 97.58
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
tensorrt library, version 0.19.0.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "tensorrt",
  "engine_version": "0.19.0",
  "chunk_name": "build_cache_config_class",
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

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params, $defs.BuildCacheConfig
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

=== CONTEXT ===
BuildCacheConfig is a CLASSIC class (not Pydantic) - the constructor takes cache_root, max_records, max_cache_storage_gb. BuildCache's __init__ does the validation (`if max_records < 1: raise ValueError`). Treat the class init signature for schema.

=== SOURCE: BuildCacheConfig ===
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



=== SOURCE: BuildCache (companion; only __init__ relevant for schema) ===
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
            has_incoming_record (bool): If the cache ha
...<truncated>...
```

### Response

```
{
  "engine": "tensorrt",
  "engine_version": "0.19.0",
  "chunk_name": "build_cache_config_class",
  "chunk_fields": {
    "cache_root": {
      "namespace": "$defs.BuildCacheConfig",
      "type": "object",
      "description": "The root directory for the build cache."
    },
    "max_records": {
      "namespace": "$defs.BuildCacheConfig",
      "type": "integer",
      "default": 10,
      "description": "The maximum number of records to store in the cache."
    },
    "max_cache_storage_gb": {
      "namespace": "$defs.BuildCacheConfig",
      "type": "number",
      "default": 256.0,
      "description": "The maximum amount of storage (in GB) to use for the cache."
    }
  }
}
```
