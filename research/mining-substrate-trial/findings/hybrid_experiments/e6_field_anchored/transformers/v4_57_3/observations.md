# e6 transformers v4_57_3 observations

- pattern: e6
- engine: transformers
- version_slug: v4_57_3
- wall_clock_sec: 1256.0
- energy_wh: 0.00
- schema_recall: 0.8304
- schema_precision: 0.9894
- invariant_recall: 0.5641
- invariant_precision: 0.3860
- schema_ref_count: 112
- schema_cell_count: 94
- schema_intersection: 93
- invariant_ref_count: 39
- invariant_cell_count: 57
- invariant_intersection: 22
- schema_failure_mode: none
- invariant_failure_mode: none

## Run observations

- pattern=e6 engine=transformers schema_wall=497.5s invariants_wall=758.0s total_wall=1256.0s
- e6_field_anchor: 3 classes, 82 declared fields
- invariants chunk 'generation_config_init_invariants': emitted 21 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'bitsandbytes_config_invariants': emitted 12 unique; anchor_classes=['BitsAndBytesConfig']
- invariants chunk 'validate_section_00_1._Validation_of_individual_attributes': emitted 8 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_01_1.1._Decoding_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_02_1.2._Cache_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_03_1.3._Performance_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_04_1.4._Watermarking_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_05_2._Validation_of_attribute_combinations': emitted 2 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl': emitted 5 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_08_2.4._check_num_return_sequences': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_09_2.5._check_cache_related_arguments': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_10_2.6._other_incorrect_combinations': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_11_3._Check_common_issue_passing_generate_arguments_inside_the_': emitted 1 unique; anchor_classes=['GenerationConfig']