# Phase 2.5 multi-pass calibration - transformers v4_57_3 (b)

- model: `llama3.1:70b` (num_ctx=32768)
- multi-pass: verify=on extend=on
- invariants_wall: 1249.8s (total wall 1250.2s)
- energy: 69.66 Wh

## Schema (carried over from round 3 - unchanged)
- recall: **83.0%** (93/112)
- precision: 93.9%
- type_accuracy: 57.0%

## Invariants (multi-pass + Phase 2.5 rubric fix)
- recall: **53.8%** (21/39)
- precision: 30.9% (21/68)
- severity_accuracy: 76.2%
- failure_mode: none

## Observations

- chunk 'validate_section_01_1.1._Decoding_attributes' pass2 flag (non-applied): id='transformers_pad_token_id_lt_zero' reason='Source raises minor issue for pad_token_id < 0, but invariant severity is warning.' fix='correct_severity:error'
- invariants chunk 'validate_section_01_1.1._Decoding_attributes' pass3_extend: extraction failed; modes=['parse_failure_after_retries']
- multipass summary: pass2_dropped=1, pass3_added=45, total_invariants=68