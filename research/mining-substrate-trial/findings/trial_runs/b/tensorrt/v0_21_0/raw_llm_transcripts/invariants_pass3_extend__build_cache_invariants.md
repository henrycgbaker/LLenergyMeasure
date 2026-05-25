# invariants_pass3_extend extraction transcript: build_cache_invariants

- chunk_description: tensorrt_llm.BuildCache.__init__ (classic if X: raise pattern)
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 11.74
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v0.21.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_build_cache_max_records_lt_1
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_records:
        <: 1
  invariant_under_test: BuildCache.__init__ flags max_records < 1


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

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



OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: tensorrt
  library: tensorrt
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. tensorrt.GenerationConfig or tensorrt.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source>'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

PREDICATE FORMS (use the EXACT keys shown):
- Exact value:         `tensorrt_llm.field: value`
- Not in list:         `tensorrt_llm.field: {present: true, not_in: [a, b]}`
- Greater than:        `tensorrt_llm.field: {'>': value}`
- Less than:           `tensorrt_llm.field: {'<': value}`
- Type not in:         `tensorrt_llm.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `tensorrt_llm.field: {present: true}`

CRITICAL RULES FOR PASS 3:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   First character must be `i` (from `invariants:`).
2. ONLY emit invariants that are NOT already in PASS 1's list (check by
   ID and by (field, predicate_kind) tuple).
3. Look ESPECIALLY for these PASS 1 blind spots:
   a. PER-FIELD invariants where PASS 1 emitted ONE entry covering many
      fields. Example: PASS 1 emitted `transformers_temperature_set_when_do_sample_false`
      but the source has SIMILAR if-blocks for top_p, top_k, min_p,
      typical_p, epsilon_cutoff, eta_cutoff. Emit ONE per field PASS 1
      missed.
   b. Multi-clause `if A and B and C:` invariants where PASS 1 keyed on
      only A. Emit one invariant per (A, B, C) tuple if each is independent.
   c. Type-check in
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_build_cache_max_records_gt_0
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BuildCache.__init__ checks max_records > 0
  severity: error
  native_type: tensorrt.BuildCacheConfig
  miner_source:
    path: build_cache.py
    method: __init__
    line_at_scan: 23
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_records:
        {'>': 0}
  kwargs_positive:
    max_records: 1
  kwargs_negative:
    max_records: 0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'max_records should be greater than 0'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
