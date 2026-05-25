# e9 transformers v4_57_3 observations

- pattern: e9
- engine: transformers
- version_slug: v4_57_3
- wall_clock_sec: 901.8
- energy_wh: 0.00
- schema_recall: 0.8304
- schema_precision: 0.9894
- invariant_recall: 0.3333
- invariant_precision: 0.4062
- schema_ref_count: 112
- schema_cell_count: 94
- schema_intersection: 93
- invariant_ref_count: 39
- invariant_cell_count: 32
- invariant_intersection: 13
- schema_failure_mode: none
- invariant_failure_mode: none

## Run observations

- pattern=e9 engine=transformers schema_wall=497.4s invariants_wall=403.9s total_wall=901.8s
- invariants chunk 'generation_config_init_invariants': emitted 0 unique (cumulative total so far: 0)
- invariants chunk 'bitsandbytes_config_invariants': emitted 12 unique (cumulative total so far: 12)
- invariants chunk 'validate_section_00_1._Validation_of_individual_attributes': emitted 1 unique (cumulative total so far: 13)
- invariants chunk 'validate_section_01_1.1._Decoding_attributes': emitted 3 unique (cumulative total so far: 16)
- invariants chunk 'validate_section_02_1.2._Cache_attributes': emitted 1 unique (cumulative total so far: 17)
- invariants chunk 'validate_section_03_1.3._Performance_attributes': emitted 1 unique (cumulative total so far: 18)
- invariants chunk 'validate_section_04_1.4._Watermarking_attributes': emitted 0 unique (cumulative total so far: 18)
- invariants chunk 'validate_section_05_2._Validation_of_attribute_combinations': emitted 6 unique (cumulative total so far: 24)
- invariants chunk 'validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl': emitted 1 unique (cumulative total so far: 25)
- invariants chunk 'validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode': emitted 1 unique (cumulative total so far: 26)
- invariants chunk 'validate_section_08_2.4._check_num_return_sequences': emitted 1 unique (cumulative total so far: 27)
- invariants chunk 'validate_section_09_2.5._check_cache_related_arguments': emitted 3 unique (cumulative total so far: 30)
- invariants chunk 'validate_section_10_2.6._other_incorrect_combinations': emitted 1 unique (cumulative total so far: 31)
- invariants chunk 'validate_section_11_3._Check_common_issue_passing_generate_arguments_inside_the_': emitted 1 unique (cumulative total so far: 32)