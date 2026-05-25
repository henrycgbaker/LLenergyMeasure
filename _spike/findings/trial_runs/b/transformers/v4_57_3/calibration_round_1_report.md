# Calibration round 1 - transformers v4.57.3

- model: `llama3.1:70b` (num_ctx=32768)
- max_retries: 2
- wall_clock: 480.0s (schema 211.0s + invariants 263.1s)
- energy: 21.71 Wh

## Schema
- recall: **51.8%** (58/112)
- precision: 92.1% (58/63)
- type_accuracy: 60.3%
- failure_mode: none

## Invariants
- recall: **32.1%** (9/28)
- precision: 64.3% (9/14)
- severity_accuracy: 77.8%
- failure_mode: none

## Observations

- strategy_b: schema_chunks=3, invariants_chunks=13, schema_wall=211.0s, invariants_wall=263.1s