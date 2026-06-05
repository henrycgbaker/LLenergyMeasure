# Wave 2.6 deliverable: bump-survivability per primitive (Axis 8)

The central self-update question (WAVE2_SCOPE): the production workflow must
respond to DYNAMIC engine changes (renames, removals, additions, new validator
surfaces) WITHOUT per-bump human intervention. This deliverable measures how the
deterministic substrates survive real upstream bumps, using the per-bump-pair GT
deltas + the substrate recall at both ends of each bump.

Source: per-engine `version_delta.md` (the GT deltas), `wave2_substrate_matrix.json`
(recall at v_old and v_new). Bumps measured:
- transformers v4.57.3 -> v5.6.2 (major)
- vllm v0.7.3 -> v0.19.1 (major refactor)
- tensorrt-llm v0.21.0 -> v1.2.1 (major)

## Recall across the bump (improved-det, tolerant inv recall vs each version's GT)

| engine | v_old recall | v_new recall | direction | GT inv (old->new) |
|---|---|---|---|---|
| transformers | 0.404 | 0.416 | flat (+0.012) | 114 -> 101 |
| vllm | 0.513 | **0.147** | COLLAPSE (-0.366) | 76 -> 68 |
| tensorrt | 0.270 | 0.400 | RISE (+0.130) | 63 -> 80 |

Critically: recall is measured against EACH version's own GT, so these track how
well the SAME fixed substrate handles a changed surface - the self-update signal.

## Findings

1. **Bump-survivability is engine-specific and driven by HOW the surface
   changed, not by churn magnitude.** All three bumps are major/high-churn, yet
   the substrate response ranges from collapse to improvement:
   - **vllm: catastrophic (-0.37).** v0.19.1 moved a large fraction of invariants
     to declarative `Field(ge/gt/le/Literal)` constraints + per-platform
     `check_and_update_config`. The substrate was built for imperative `raise`;
     it cannot parse declarative constraints, so it falls off a cliff exactly
     where upstream refactored. This is the dominant negative finding.
   - **tensorrt: improves (+0.13).** v1.2.1 migrated PluginConfig/BuildConfig/
     LoraConfig from dataclass/metaclass to plain pydantic and moved C++-only
     validators into Python. The surface became MORE statically visible, so the
     same substrate recalls more. Upstream churn worked IN the substrate's favour.
   - **transformers: flat.** The GenerationConfig lazy-default refactor hurts
     default-mining but the invariant SET stayed predominantly imperative
     `validate()` methods, so invariant recall held.

2. **The substrate does not "know what it doesn't know" across a bump.** When
   recall collapses (vllm), nothing in the deterministic output signals the
   collapse - it silently emits fewer invariants. A self-updating workflow CANNOT
   rely on the substrate to detect its own degradation; an external signal is
   required (the runtime-validate gate's acceptance rate, or an LLM diff-reviewer
   comparing v_old and v_new catalogues - the W-F diagnose role).

3. **Citation/landmark pinning does NOT survive any of these bumps.** Every
   vllm `config.py:NNNN` citation is stale (subpackage move); tensorrt class
   structure reorganised; transformers files moved to `integrations/*`. Any
   producer pinned to source locations or a hand-curated landmark tuple (W-A
   status quo) breaks wholesale. The improved-det primitives are
   pattern-matched, not location-pinned, so they degrade gracefully (partial
   recall) rather than crashing - a real self-update advantage over the status
   quo, even where recall drops.

4. **Direct implication for the self-update workflow design.** A purely-static
   floor is NOT bump-robust on its own (the vllm cliff proves it). Robustness
   requires EITHER (a) a new declarative-constraint primitive (closes the vllm
   failure mode mechanically - candidate "Primitive 8"), OR (b) an LLM tail that
   reads the diff and recovers the residual, OR both. The bump-survivability
   evidence favours W-G (improved-det floor + LLM extend) over W-B (pure
   universal substrate), because W-B inherits the cliff with no recovery path.

## Update (Wave 2.5): Primitive 8 measured - hypothesis CONFIRMED

The "add a declarative-`Field` primitive and re-measure the vllm cliff" experiment
below was BUILT and run (`a_improved_det_v2.py` = improved-det + Primitive 8;
results in `findings/wave2_primitive8_results.json`). Primitive 8 extracts pydantic
`Field(ge/gt/le/lt/...)` + `Annotated[..., Field/Meta(...)]` + `Literal`/enum field
types as invariants, and globs the `config/*.py` subpackage the flat-`config.py`-era
primitives never reached.

Measured recovery of the vllm v0.19.1 cliff (tolerant inv recall vs GT):
- v1 (improved-det) 0.147 -> v2 (improved-det + Primitive 8) **0.309**.
- Recovers **0.162 of the 0.366 cliff = 44%** on the conservative leaf+bucket
  metric; ~the full v0.7.3 floor (~0.52) leaf-level once GT's own coarse-bucket
  drift is allowed (GT buckets `Field(gt=0,le=1)` ranges as `membership` etc.).
- 11 NEW true-positive GT invariants on vllm v0.19.1, all declarative.
- GENERALISES: both tensorrt cells also lift (v0.21.0 0.270->0.349; v1.2.1
  0.400->0.500), with precision UP on tensorrt. No recall regressions anywhere.
- Cost: precision dips on vllm v0.19.1 (0.286->0.208) because Primitive 8 also
  emits ~55 declarative constraints on compile/distributed internals that GT
  deliberately excludes - breadth, not error; manageable by scoping Primitive 8 to
  caller-touchable config classes if precision is a hard constraint.

Conclusion: the declarative-`Field` primitive mechanically recovers ~half the bump
cliff with no LLM, confirming it as the highest-ROI engineering item for
bump-robustness. The remaining residual is genuine LLM/semantic territory.

## What would make this measurement stronger (Wave 3 / deferred)

- Scope Primitive 8 to caller-touchable config classes + re-measure precision.
- Measure framework-reflection across the bumps: reflection reads the resolved
  pydantic model, so it should be IMMUNE to the imperative->declarative shift
  that sinks the source-walkers (it sees the constraint as a field validator
  regardless of how it is declared). This is the highest-value deferred cell.
- bump-UPDATE cells (propose a producer/catalogue patch and check it passes the
  runtime gate without human edit) - the true self-update binary - need the
  per-version engine containers; deferred.
