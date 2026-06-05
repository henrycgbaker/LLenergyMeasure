# Wave 1 cells re-scored against ground truth (experiment-queue step 0.4)

Each Wave 1 cell whose (engine, version) has an Opus-established ground truth, re-scored against the canonical GT. Columns: original validated-union (VU) recall/precision (from `findings/trial_scores/*__vu.json`) side by side with the new vs-GT numbers, both STRICT (locked-scorer identity) and TOLERANT (convention-insensitive: invariants on (field, coarse predicate bucket); schema on field name, namespace dropped).

TOLERANT is the defensible headline given the namespace + predicate-kind convention drift between GT and the mined catalogues; STRICT bounds quality from below. See `wave2_deviations.md` for the matching-method record.

Summary: 14 cells re-scored, 7 not re-scoreable (raw output absent).

## Re-scored cells

| cell | VU sch r/p | VU inv r/p | strict sch r/p | strict inv r/p | tol sch r/p | tol inv r/p | note |
|---|---|---|---|---|---|---|---|
| a/tensorrt/v0_21_0 | 1.000/1.000 | 1.000/0.355 | 0.344/1.000 | 0.131/0.258 | 0.386/1.000 | 0.190/0.400 |  |
| a/transformers/v4_57_3 | 1.000/1.000 | 0.661/0.949 | 0.354/0.920 | 0.050/0.154 | 0.451/0.991 | 0.149/0.607 |  |
| a/vllm/v0_7_3 | 1.000/1.000 | 0.667/1.000 | 0.494/0.978 | 0.275/0.846 | 0.615/0.978 | 0.316/0.960 |  |
| b_8b/transformers/v4_57_3 | 0.857/0.932 | 0.250/0.226 | 0.323/0.913 | 0.042/0.081 | 0.415/0.990 | 0.140/0.291 |  |
| b/tensorrt/v0_21_0 | 0.561/0.465 | 0.455/0.128 | 0.376/0.907 | 0.115/0.179 | 0.451/1.000 | 0.206/0.419 |  |
| b/tensorrt/v1_2_1 | 0.505/0.367 | 0.364/0.100 | 0.343/0.891 | 0.063/0.125 | 0.400/1.000 | 0.113/0.300 |  |
| b/transformers/v4_57_3 | 0.830/0.939 | 0.625/0.686 | 0.296/0.869 | 0.042/0.098 | 0.402/1.000 | 0.167/0.500 |  |
| b/vllm/v0_19_1 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 |  |
| b/vllm/v0_7_3 | 0.970/0.851 | 0.487/0.288 | 0.487/0.844 | 0.163/0.197 | 0.709/0.987 | 0.421/0.552 |  |
| d-ab/tensorrt/v0_21_0 | 1.000/1.000 | 1.000/0.282 | 0.344/1.000 | 0.131/0.205 | 0.386/1.000 | 0.190/0.316 |  |
| d-ab/tensorrt/v1_2_1 | 1.000/1.000 | 1.000/0.314 | 0.267/0.953 | 0.114/0.257 | 0.296/0.981 | 0.113/0.265 |  |
| d-ab/transformers/v4_57_3 | 1.000/1.000 | 0.661/0.902 | 0.354/0.920 | 0.050/0.146 | 0.451/0.991 | 0.149/0.567 |  |
| d-ab/vllm/v0_19_1 | 1.000/1.000 | 0.667/1.000 | 0.139/0.652 | 0.169/0.462 | 0.221/0.679 | 0.191/0.520 |  |
| d-ab/vllm/v0_7_3 | 1.000/1.000 | 0.667/1.000 | 0.494/0.978 | 0.275/0.846 | 0.615/0.978 | 0.316/0.960 |  |

## Not re-scoreable (raw output absent)

| cell | VU sch r/p | VU inv r/p | reason |
|---|---|---|---|
| a/tensorrt/v1_2_1 | 1.000/1.000 | 1.000/0.355 | not re-scoreable (raw output absent) |
| a/vllm/v0_19_1 | 0.000/0.000 | 0.000/0.000 | not re-scoreable (raw output absent) |
| e6/transformers/v4_57_3 | 0.830/0.989 | 0.571/0.561 | not re-scoreable (raw output absent) |
| e6/vllm/v0_7_3 | 0.970/0.851 | 0.436/0.370 | not re-scoreable (raw output absent) |
| e9/transformers/v4_57_3 | 0.830/0.989 | 0.393/0.688 | not re-scoreable (raw output absent) |
| e9/vllm/v0_7_3 | 0.970/0.851 | 0.436/0.362 | not re-scoreable (raw output absent) |
| h6/transformers/v4_57_3 | 0.750/0.944 | 0.179/0.625 | not re-scoreable (raw output absent) |
