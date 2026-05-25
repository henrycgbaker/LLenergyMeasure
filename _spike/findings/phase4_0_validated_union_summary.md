# Phase 4.0 - validated-union ground truth summary

_generated at 2026-05-25T15:21:13.274112+00:00_

Per the DECISIONS_LOG entry framing: the corrected ground truth for the mining-substrate trial is the UNION of every strategy's mined invariants, filtered to entries that pass runtime validation. (a)'s output is one input among several; equal weight in the union.

## Per-cell union sizes (active cells)

| engine | version | unique_unioned | validated_both | infra_errors | a_alone_ref | union_delta_vs_a |
|---|---|---|---|---|---|---|
| transformers | v4_57_3 | 108 | 56 | 5 | 41 | +15 |
| vllm | v0_7_3 | 106 | 39 | 20 | 26 | +13 |
| tensorrt | v0_21_0 | 70 | 11 | 19 | 35 | -24 |

## Per-strategy contributor breakdown (active cells)

Which strategies contributed at least once to the validated union for each cell.

| engine | version | a | b | d-ab | h2 | h3 | h6 | e6 | e9 |
|---|---|---|---|---|---|---|---|---|---|
| transformers | v4_57_3 | 37 | 35 | 37 | 37 | 32 | 10 | 32 | 22 |
| vllm | v0_7_3 | 26 | 19 | 26 | 23 | 19 | 0 | 17 | 17 |
| tensorrt | v0_21_0 | 11 | 5 | 11 | 11 | 5 | 0 | 0 | 0 |

## Unique-contributor count per strategy (active cells)

Number of validated union entries where this strategy is the ONLY
contributor. High counts indicate distinctive coverage; zero counts
indicate the strategy's outputs were entirely captured by other strategies.

| engine | version | a | b | d-ab | h2 | h3 | h6 | e6 | e9 |
|---|---|---|---|---|---|---|---|---|---|
| transformers | v4_57_3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| vllm | v0_7_3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 |
| tensorrt | v0_21_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Per-strategy aggregate deltas: (a)-as-reference -> validated-union

Recall + precision computed against the new reference. Strategies that beat (a) on the union (find things (a) missed) shift down on (a)-relative recall but reveal their true coverage.

| strategy | cells | inv_recall_a | inv_recall_vu | delta | inv_precision_a | inv_precision_vu | delta |
|---|---|---|---|---|---|---|---|
| a | 15/15 | 52.8% | 46.6% | -6.2pp | 54.3% | 32.7% | -21.6pp |
| b | 15/15 | 34.4% | 42.3% | +8.0pp | 21.0% | 27.6% | +6.6pp |
| b_8b | 1/1 | 35.7% | 25.0% | -10.7pp | 16.1% | 22.6% | +6.5pp |
| c | 0/1 | 0.0% | 0.0% | +0.0pp | 0.0% | 0.0% | +0.0pp |
| d-ab | 15/15 | 100.0% | 77.6% | -22.4pp | 93.6% | 73.6% | -20.0pp |
| e6 | 2/2 | 43.6% | 50.4% | +6.8pp | 28.0% | 46.5% | +18.6pp |
| e9 | 2/2 | 34.0% | 41.4% | +7.5pp | 29.9% | 52.5% | +22.6pp |
| h6 | 1/1 | 12.8% | 17.9% | +5.0pp | 31.2% | 62.5% | +31.2pp |

## Validation infrastructure errors

Per Phase 1 Day 2 tensorrt finding: type-blind probe synthesis leaves
entries that pass static identity comparison but fail when the live
library tries to construct from kwargs. These DON'T count as failed
union entries; they count as `validation_error` (separate bucket).

- transformers/v4_57_3: 5 infra errors out of 108 unique unioned entries (4.6%)
- vllm/v0_7_3: 20 infra errors out of 106 unique unioned entries (18.9%)
- tensorrt/v0_21_0: 19 infra errors out of 70 unique unioned entries (27.1%)

## Cells where the union shifted the picture

Cells where vu scoring changed the headline recall by >5pp (positive or negative). Positive shift = strategy looks BETTER under vu (it found things (a) missed); negative shift = strategy looks WORSE (its prior recall was inflated by being measured against (a)'s narrow reference).

| strategy | engine | version | inv_recall_a | inv_recall_vu | delta |
|---|---|---|---|---|---|
| a | transformers | v4_57_3 | 100.0% | 66.1% | -33.9pp |
| d-ab | transformers | v4_55_4 | 100.0% | 66.1% | -33.9pp |
| d-ab | transformers | v4_56_2 | 100.0% | 66.1% | -33.9pp |
| d-ab | transformers | v4_57_3 | 100.0% | 66.1% | -33.9pp |
| d-ab | transformers | v4_57_6 | 100.0% | 66.1% | -33.9pp |
| d-ab | transformers | v5_9_0 | 100.0% | 66.1% | -33.9pp |
| a | vllm | v0_7_3 | 100.0% | 66.7% | -33.3pp |
| d-ab | vllm | v0_19_1 | 100.0% | 66.7% | -33.3pp |
| d-ab | vllm | v0_6_0 | 100.0% | 66.7% | -33.3pp |
| d-ab | vllm | v0_6_6_post1 | 100.0% | 66.7% | -33.3pp |
| d-ab | vllm | v0_7_3 | 100.0% | 66.7% | -33.3pp |
| d-ab | vllm | v0_9_2 | 100.0% | 66.7% | -33.3pp |
| a | transformers | v4_56_2 | 48.7% | 33.9% | -14.8pp |
| a | transformers | v4_57_6 | 43.6% | 32.1% | -11.4pp |
| b_8b | transformers | v4_57_3 | 35.7% | 25.0% | -10.7pp |
| b | transformers | v4_56_2 | 59.0% | 53.6% | -5.4pp |
| h6 | transformers | v4_57_3 | 12.8% | 17.9% | +5.0pp |
| e9 | transformers | v4_57_3 | 33.3% | 39.3% | +6.0pp |
| b | transformers | v4_57_3 | 56.4% | 62.5% | +6.1pp |
| e9 | vllm | v0_7_3 | 34.6% | 43.6% | +9.0pp |
| b | vllm | v0_7_3 | 38.5% | 48.7% | +10.3pp |
| b | vllm | v0_9_2 | 30.8% | 41.0% | +10.3pp |
| b | vllm | v0_6_0 | 34.6% | 46.2% | +11.5pp |
| e6 | vllm | v0_7_3 | 30.8% | 43.6% | +12.8pp |
| b | tensorrt | v1_0_0 | 22.6% | 36.4% | +13.8pp |
| b | tensorrt | v1_2_1 | 19.4% | 36.4% | +17.0pp |
| b | tensorrt | v0_21_0 | 25.8% | 45.5% | +19.6pp |
| b | tensorrt | v0_19_0 | 16.1% | 36.4% | +20.2pp |
| b | tensorrt | v0_20_0 | 16.1% | 36.4% | +20.2pp |
