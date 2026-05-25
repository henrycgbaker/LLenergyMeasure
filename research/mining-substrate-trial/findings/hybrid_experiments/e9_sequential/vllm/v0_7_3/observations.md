# e9 vllm v0_7_3 observations

- pattern: e9
- engine: vllm
- version_slug: v0_7_3
- wall_clock_sec: 1025.5
- energy_wh: 0.00
- schema_recall: 0.9704
- schema_precision: 0.8506
- invariant_recall: 0.3462
- invariant_precision: 0.1915
- schema_ref_count: 135
- schema_cell_count: 154
- schema_intersection: 131
- invariant_ref_count: 26
- invariant_cell_count: 47
- invariant_intersection: 9
- schema_failure_mode: none
- invariant_failure_mode: none

## Run observations

- pattern=e9 engine=vllm schema_wall=515.9s invariants_wall=509.2s total_wall=1025.5s
- invariants chunk 'sampling_params_invariants': emitted 20 unique (cumulative total so far: 20)
- invariants chunk 'guided_decoding_params_invariants': emitted 1 unique (cumulative total so far: 21)
- invariants chunk 'model_config_verify_tokenizer_mode': emitted 1 unique (cumulative total so far: 22)
- invariants chunk 'model_config_verify_quantization': emitted 2 unique (cumulative total so far: 24)
- invariants chunk 'model_config_verify_cuda_graph': emitted 1 unique (cumulative total so far: 25)
- invariants chunk 'model_config_verify_bnb_config': emitted 1 unique (cumulative total so far: 26)
- invariants chunk 'cache_config_invariants': emitted 3 unique (cumulative total so far: 29)
- invariants chunk 'scheduler_config_invariants': emitted 7 unique (cumulative total so far: 36)
- invariants chunk 'parallel_config_invariants': emitted 3 unique (cumulative total so far: 39)
- invariants chunk 'lora_prompt_adapter_invariants': emitted 8 unique (cumulative total so far: 47)