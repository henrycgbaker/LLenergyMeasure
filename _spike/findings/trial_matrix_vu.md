# Empirical trial matrix - validated-union scoring

_generated at 2026-05-25T15:21:13.270787+00:00; score files: 51_

Per the trial DECISIONS_LOG entry on the validated-union ground truth: this matrix scores every cell against the UNION of all strategies' outputs filtered to runtime-validated entries, NOT against (a)'s output. See `_spike/findings/phase4_0_validated_union_summary.md` for the methodology + per-cell union sizes.

## Per-strategy aggregate (vu scoring)

| strategy | cells | inv_recall_mean | inv_precision_mean | schema_recall_mean | wall_mean_s |
|---|---|---|---|---|---|
| a | 15 | 46.6% | 32.7% | 60.0% | 1.9 |
| b | 15 | 42.3% | 27.6% | 60.6% | 3411.4 |
| b_8b | 1 | 25.0% | 22.6% | 85.7% | 412.6 |
| d-ab | 15 | 77.6% | 73.6% | 100.0% | 254.8 |
| e6 | 2 | 50.4% | 46.5% | 90.0% | 1152.3 |
| e9 | 2 | 41.4% | 52.5% | 90.0% | 963.6 |
| h6 | 1 | 17.9% | 62.5% | 75.0% | 526.4 |

## Per-cell vu scores

| strategy | engine | version | bump | inv_recall | inv_prec | inv_ref | inv_cell | inv_int | failure_modes |
|---|---|---|---|---|---|---|---|---|---|
| a | tensorrt | v0_19_0 | v-2 | 100.0% | 35.5% | 11 | 31 | 11 | none |
| a | tensorrt | v0_20_0 | v-1 | 100.0% | 35.5% | 11 | 31 | 11 | none |
| a | tensorrt | v0_21_0 | active | 100.0% | 35.5% | 11 | 31 | 11 | none |
| a | tensorrt | v1_0_0 | v+1 | 100.0% | 35.5% | 11 | 31 | 11 | none |
| a | tensorrt | v1_2_1 | v+major | 100.0% | 35.5% | 11 | 31 | 11 | none |
| a | transformers | v4_55_4 | v-2 | 0.0% | 0.0% | 56 | 0 | 0 | detectable |
| a | transformers | v4_56_2 | v-1 | 33.9% | 54.3% | 56 | 35 | 19 | none |
| a | transformers | v4_57_3 | active | 66.1% | 94.9% | 56 | 39 | 37 | none |
| a | transformers | v4_57_6 | v+1 | 32.1% | 64.3% | 56 | 28 | 18 | none |
| a | transformers | v5_9_0 | v+major | 0.0% | 0.0% | 56 | 0 | 0 | detectable |
| a | vllm | v0_19_1 | v+major | 0.0% | 0.0% | 39 | 0 | 0 | detectable |
| a | vllm | v0_6_0 | v-2 | 0.0% | 0.0% | 39 | 0 | 0 | detectable |
| a | vllm | v0_6_6_post1 | v-1 | 0.0% | 0.0% | 39 | 0 | 0 | detectable |
| a | vllm | v0_7_3 | active | 66.7% | 100.0% | 39 | 26 | 26 | none |
| a | vllm | v0_9_2 | v+1 | 0.0% | 0.0% | 39 | 0 | 0 | detectable |
| b | tensorrt | v0_19_0 | v-2 | 36.4% | 10.8% | 11 | 37 | 4 | silent;none |
| b | tensorrt | v0_20_0 | v-1 | 36.4% | 12.1% | 11 | 33 | 4 | silent;none |
| b | tensorrt | v0_21_0 | active | 45.5% | 12.8% | 11 | 39 | 5 | none |
| b | tensorrt | v1_0_0 | v+1 | 36.4% | 10.5% | 11 | 38 | 4 | none |
| b | tensorrt | v1_2_1 | v+major | 36.4% | 10.0% | 11 | 40 | 4 | none;silent |
| b | transformers | v4_55_4 | v-2 | 55.4% | 41.9% | 56 | 74 | 31 | none |
| b | transformers | v4_56_2 | v-1 | 53.6% | 41.1% | 56 | 73 | 30 | none |
| b | transformers | v4_57_3 | active | 62.5% | 68.6% | 56 | 51 | 35 | none |
| b | transformers | v4_57_6 | v+1 | 58.9% | 64.7% | 56 | 51 | 33 | none |
| b | transformers | v5_9_0 | v+major | 44.6% | 35.2% | 56 | 71 | 25 | none |
| b | vllm | v0_19_1 | v+major | 0.0% | 0.0% | 39 | 4 | 0 | silent |
| b | vllm | v0_6_0 | v-2 | 46.2% | 28.1% | 39 | 64 | 18 | none |
| b | vllm | v0_6_6_post1 | v-1 | 33.3% | 22.4% | 39 | 58 | 13 | none |
| b | vllm | v0_7_3 | active | 48.7% | 28.8% | 39 | 66 | 19 | none |
| b | vllm | v0_9_2 | v+1 | 41.0% | 26.7% | 39 | 60 | 16 | none |
| b_8b | transformers | v4_57_3 | active | 25.0% | 22.6% | 56 | 62 | 14 | none |
| d-ab | tensorrt | v0_19_0 | v-2 | 100.0% | 32.4% | 11 | 34 | 11 | none |
| d-ab | tensorrt | v0_20_0 | v-1 | 100.0% | 32.4% | 11 | 34 | 11 | none |
| d-ab | tensorrt | v0_21_0 | active | 100.0% | 28.2% | 11 | 39 | 11 | none |
| d-ab | tensorrt | v1_0_0 | v+1 | 100.0% | 28.2% | 11 | 39 | 11 | none |
| d-ab | tensorrt | v1_2_1 | v+major | 100.0% | 31.4% | 11 | 35 | 11 | none |
| d-ab | transformers | v4_55_4 | v-2 | 66.1% | 90.2% | 56 | 41 | 37 | none |
| d-ab | transformers | v4_56_2 | v-1 | 66.1% | 90.2% | 56 | 41 | 37 | none |
| d-ab | transformers | v4_57_3 | active | 66.1% | 90.2% | 56 | 41 | 37 | none |
| d-ab | transformers | v4_57_6 | v+1 | 66.1% | 90.2% | 56 | 41 | 37 | none |
| d-ab | transformers | v5_9_0 | v+major | 66.1% | 90.2% | 56 | 41 | 37 | none |
| d-ab | vllm | v0_19_1 | v+major | 66.7% | 100.0% | 39 | 26 | 26 | none |
| d-ab | vllm | v0_6_0 | v-2 | 66.7% | 100.0% | 39 | 26 | 26 | none |
| d-ab | vllm | v0_6_6_post1 | v-1 | 66.7% | 100.0% | 39 | 26 | 26 | none |
| d-ab | vllm | v0_7_3 | active | 66.7% | 100.0% | 39 | 26 | 26 | none |
| d-ab | vllm | v0_9_2 | v+1 | 66.7% | 100.0% | 39 | 26 | 26 | none |
| e6 | transformers | v4_57_3 | active | 57.1% | 56.1% | 56 | 57 | 32 | none |
| e6 | vllm | v0_7_3 | active | 43.6% | 37.0% | 39 | 46 | 17 | none |
| e9 | transformers | v4_57_3 | active | 39.3% | 68.8% | 56 | 32 | 22 | none |
| e9 | vllm | v0_7_3 | active | 43.6% | 36.2% | 39 | 47 | 17 | none |
| h6 | transformers | v4_57_3 | active | 17.9% | 62.5% | 56 | 16 | 10 | none;silent |
