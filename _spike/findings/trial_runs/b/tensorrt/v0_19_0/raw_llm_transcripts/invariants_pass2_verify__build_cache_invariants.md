# invariants_pass2_verify extraction transcript: build_cache_invariants

- chunk_description: tensorrt_llm.BuildCache.__init__ (classic if X: raise pattern)
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 14.86
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v0.19.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: tensorrt_build_cache_max_records_lt_1
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_records:
        <: 1
  invariant_under_test: BuildCacheConfig validation flags max_records < 1


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
BuildCache.__init__ contains the only classic-style `if X: raise` invariant in the TRT codebase: `if config.max_records < 1: raise ValueError(...)`. Emit this as ONE invariant: namespace=tensorrt_llm, field=max_records (on BuildCacheConfig), predicate=lt 1, severity=error.

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



=== SOURCE: BuildCache (__init__ does the validation) ===
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
                dic



OUTPUT FORMAT: a YAML document with TWO sections:

confirmed:
- <id-of-pass1-invariant>  # one ID per line, no further detail needed.

flagged:
- id: <id-of-pass1-invariant>
  reason: <one-line: what looks wrong>
  fix: <one of: "drop", "correct_severity:<new-severity>", "correct_predicate:<new-kind>", "correct_kwargs_positive">

RULES:
1. Every PASS 1 invariant MUST appear in EITHER `confirmed` OR `flagged`
   (not both, not neither). If you're unsure, place in `confirmed` (the
   bar for flagging is "obviously wrong against the source").
2. Flag reasons must be CONCRETE - cite the source line that contradicts
   the invariant, or note the specific shape mismatch.
3. NO markdown code fences. NO commentary outside the YAML.
4. First character must be `c` (from `confirmed:`).

CRITERIA FOR FLAGGING:
- Severity wrong: source has `raise ValueError` but invariant says
  `severity: dormant`; or source has `minor_issues[...] = ...` but
  invariant says `severity: error`.
- Predicate wrong: source has `if X.field not in {a, b}: raise`
  but invariant emits `predicate_kind: exact`.
- Kwargs_positive wrong: source's predicate is "value < 0 raises" but
  invariant's kwargs_positive shows `field: 1` (which would NOT trigger).
- Hallucinated: invariant references a field name that does NOT appear
  in the source.

Emit the YAML now:

```

### Response

```
confirmed:
- tensorrt_build_cache_max_records_lt_1
```
