# Delta: tensorrt-llm 1.2.1 ground-truth vs LLEM baseline

Compares this source-walked ground truth (`schema_ground_truth.json` +
`invariants_ground_truth.yaml`) against the LLEM baseline mining outputs for
the v1.2.1 pin.

## Baseline status: NO outputs produced

`engine_versions/tensorrt/v1_2_1/` contains the frozen producer machinery
(`producers/static_invariant_miner.py`, `producers/schema_introspector.py`) and
an `__init__.py` describing LANDMARKS, but **there is no `outputs/` directory**.
The miners were never run against the v1.2.1 pin; the baseline schema and
invariant artefacts (schema.discovered.json, invariants.proposed.yaml,
invariants.validated.yaml, curated.yaml) that existed for v0.21.0 do not exist
for v1.2.1.

Consequence: the baseline coverage for v1.2.1 is **zero**. Every entry in this
ground truth is a net addition relative to baseline.

| Surface                  | Baseline (v1.2.1) | Ground truth | Delta  |
|--------------------------|------------------:|-------------:|-------:|
| Schema entries (total)   |                 0 |          438 |   +438 |
| Invariants               |                 0 |           92 |    +92 |
| engine_envs              |                 0 |           55 |    +55 |
| Subconfig classes        |                 0 |           35 |    +35 |

Because the baseline is empty, the substantive comparison for this pin is the
**version delta vs the v0.21.0 ground truth** (the bump-pair), recorded
separately in `version_delta.md`. That is where the mining-recall signal lives:
it isolates what a substrate must newly recover when stepping the pin across the
0.x -> 1.x major boundary, rather than against a non-existent baseline.

## What the baseline producers would have to recover

The v1.2.1 producer machinery is the same shape as the v0.21.0 producers
(a static AST invariant miner + a pydantic schema introspector). Re-running them
against the v1.2.1 source would, by analogy with the v0.21.0 baseline-vs-GT
delta, still miss:

- The `engine_envs` namespace entirely (the static miner reads class-body AST,
  not `os.environ` reads) - 55 entries.
- The nested subconfig field surface unless the introspector recurses into
  pydantic sub-models (CudaGraphConfig, MoeConfig, the sparse-attention tree,
  the 9-class speculative tree).

It would, however, now recover **PluginConfig** far more easily than at v0.21,
because the v1.2.1 PluginConfig is a plain pydantic BaseModel rather than a
metaclass-generated property tree (see version_delta.md section 3). This is the
clearest "the upstream refactor changed mining difficulty" signal in the
bump-pair.

## Recommendation

Run the frozen v1.2.1 producers to populate
`engine_versions/tensorrt/v1_2_1/outputs/` so a true baseline-vs-GT delta can be
computed for this pin (matching the v0.21.0 deliverable). Until then, treat
`version_delta.md` as the primary comparison artefact for v1.2.1.
