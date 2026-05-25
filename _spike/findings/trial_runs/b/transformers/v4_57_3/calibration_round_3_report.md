# Calibration round 3 - transformers v4.57.3

- model: `llama3.1:70b` (num_ctx=32768)
- max_retries: 2
- wall_clock: 889.5s (schema 511.6s + invariants 372.1s)
- energy: 37.05 Wh

## Schema
- recall: **83.0%** (93/112)
- precision: 93.9% (93/99)
- type_accuracy: 57.0%
- failure_mode: none

## Invariants
- recall: **60.7%** (17/28)
- precision: 65.4% (17/26)
- severity_accuracy: 88.2%
- failure_mode: none

## Observations

- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=511.6s, invariants_wall=372.1s