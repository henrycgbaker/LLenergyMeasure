# e6 vllm v0_7_3 observations

- pattern: e6
- engine: vllm
- version_slug: v0_7_3
- wall_clock_sec: 1048.5
- energy_wh: 0.00
- schema_recall: 0.9704
- schema_precision: 0.8506
- invariant_recall: 0.3077
- invariant_precision: 0.1739
- schema_ref_count: 135
- schema_cell_count: 154
- schema_intersection: 131
- invariant_ref_count: 26
- invariant_cell_count: 46
- invariant_intersection: 8
- schema_failure_mode: none
- invariant_failure_mode: none

## Run observations

- pattern=e6 engine=vllm schema_wall=516.9s invariants_wall=531.2s total_wall=1048.5s
- e6_field_anchor: 15 classes, 249 declared fields
- invariants chunk 'sampling_params_invariants': emitted 18 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'guided_decoding_params_invariants': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_tokenizer_mode': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_quantization': emitted 2 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_cuda_graph': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_bnb_config': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'cache_config_invariants': emitted 3 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'scheduler_config_invariants': emitted 8 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'parallel_config_invariants': emitted 3 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'lora_prompt_adapter_invariants': emitted 8 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']