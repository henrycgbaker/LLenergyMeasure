# Calibration round 2 - transformers v4.57.3

- model: `llama3.1:70b` (num_ctx=32768)
- max_retries: 2
- wall_clock: 1117.9s (schema 512.5s + invariants 599.5s)
- energy: 50.30 Wh

## Schema
- recall: **83.0%** (93/112)
- precision: 93.9% (93/99)
- type_accuracy: 57.0%
- failure_mode: none

## Invariants
- recall: **32.1%** (9/28)
- precision: 64.3% (9/14)
- severity_accuracy: 77.8%
- failure_mode: none

## Observations

- invariants chunk 'bitsandbytes_config_invariants': extraction failed; modes=['parse_failure_after_retries']
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=512.5s, invariants_wall=599.5s