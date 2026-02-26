# Codebase Propagation Audit: `.planning/codebase/` → `.product/`

**Date**: 2026-02-25
**Source**: `.planning/codebase/` (7 files: ARCHITECTURE.md, CONCERNS.md, CONVENTIONS.md, INTEGRATIONS.md, STACK.md, STRUCTURE.md, TESTING.md)
**Audited against**: `.product/decisions/`, `.product/designs/`, `.product/NEEDS_ADDRESSING.md`

---

## Summary

| Category | Count | Blocking? |
|----------|-------|-----------|
| Fully propagated | 20 | N/A |
| Partially propagated | 8 | 1 partial |
| Not propagated — blocking | 3 | YES |
| Not propagated — informational | 7 | No |
| Contradictions — blocking | 2 | YES |
| Contradictions — informational | 3 | No |

---

## Priority Action List (Must Address Before Implementation)

### 1. `_extends` YAML inheritance — KEEP OR CUT DECISION NEEDED

**Source**: `CONCERNS.md` — "Config Inheritance with Deep Merge"; `ARCHITECTURE.md` — ConfigLoader with `_extends`
**Gap**: Fully implemented in v1.x but absent from ALL v2.0 design specs. `decisions/config-architecture.md` has no mention. `designs/experiment-config.md` schema has no `_extends` field. `preservation_audit/N-X13` flags this as "Immediately Actionable" but the action was never taken.
**Risk**: Phase 5 implementors building from `designs/experiment-config.md` will silently lose this feature.
**Action**: Add explicit keep/cut decision to `decisions/config-architecture.md`.

### 2. pynvml + vLLM thread safety race

**Source**: `CONCERNS.md` — "pynvml Thread Safety with vLLM"
**Gap**: GPU utilisation sampler may fail silently when vLLM initialises CUDA context. The "single NVML session owner" rule in `.product/` doesn't cover this race condition (rule is about Zeus vs NVML poller, not about vLLM CUDA context interference).
**Action**: Add to `designs/energy-backends.md` § NVML single-session owner — specify behaviour when vLLM initialises CUDA mid-measurement.

### 3. Memory peak measurement window semantics

**Source**: `CONCERNS.md` — "Memory Peak Statistics Reset Timing"
**Gap**: `torch.cuda.reset_peak_memory_stats()` placement determines whether `peak_memory_mb` reflects model load or inference only. Not specified anywhere in `.product/`.
**Action**: Specify in `designs/result-schema.md` what `peak_memory_mb` measures and where reset is called.

### 4. `_extends` security boundary (path traversal)

**Source**: `CONCERNS.md` — "Config File Path Traversal Prevention"
**Gap**: `_extends` has cycle detection but no directory boundary check. Inherited configs can load arbitrary files.
**Action**: If `_extends` is kept (see #1), document security boundary in `decisions/config-architecture.md`.

### 5. tqdm vs Rich Live conflict

**Source**: `STACK.md` — tqdm dependency; `preservation_audit/N-X03` — "tqdm ProgressTracker cannot run inside Rich Live context"
**Gap**: `designs/observability.md` uses Rich for study-level display. Per-experiment inference loops use tqdm. These conflict in Rich Live context. No resolution in any `.product/` doc.
**Action**: Resolve in `designs/observability.md` — tqdm for subprocess-internal, Rich Progress for orchestrator.

### 6. Q5 output contract stale in experiment-study-architecture.md

**Source**: Cross-doc inconsistency found during audit
**Gap**: `decisions/experiment-study-architecture.md` Q5 says "Single → flat JSON" but `decisions/output-storage.md` was revised 2026-02-25 to "always subdirectory".
**Action**: Add superseded annotation to Q5 in `experiment-study-architecture.md`.

---

## Should Address (Reduces Implementation Risk)

### 7. `init_cmd.py` and `questionary` removal not recorded

v1.x has full `init_cmd.py` + `questionary` dependency. `llem init` was explicitly rejected in `decisions/installation.md`. But neither `init_cmd.py` deletion nor `questionary` removal is in the dead code removal list.

### 8. SSOT introspection complexity — no decision recorded

`CONCERNS.md` recommends evaluating static dict vs 807-line runtime `introspection.py`. `preservation_audit/P-03` keeps it but the design critique has no response in `decisions/`.

### 9. `python-on-whales` removal not recorded

v1.x uses it for Docker orchestration. v2.0 design uses `subprocess.run`. The dependency removal is not captured.

### 10. `schedule` library — no cut decision

v1.x has `schedule >=1.2.2` for scheduled execution. No mention anywhere in v2.0 scope. Needs explicit cut.

### 11. Webhook notifications (P-13) — still Pending

`preservation_audit/P-13` marks webhook/httpx as "Pending — keep or cut?" Still unresolved.

---

## Partially Propagated (8 items)

1. Config SSOT introspection complexity — kept but no design review
2. Detection module boundaries (4 overlapping detection modules) — partially in preservation audit
3. Launcher.py launch mode complexity — addressed by new subprocess model but strategy pattern unresolved
4. Extended metrics null handling — kept but nullable semantics unspecified
5. Streaming latency thread safety — kept but fragility unaddressed
6. vLLM multiprocessing env var hacks — version-specific workaround strategy missing
7. GPU topology/MIG detection — kept but energy attribution limitations uncommunicated
8. CI has no automated testing — target designed but current-state gap not noted

---

## Fully Propagated (20 items)

Docker GPU passthrough, CUDA init ordering, ephemeral containers, campaign.py dead code,
vLLM/TRT process incompatibility, PyTorch model_kwargs bug, resume/checkpoint, subprocess
isolation redesign, exception hierarchy, library-first architecture, backend extras,
two-tier testing, config inheritance (flagged), output storage, state machine, HF_TOKEN
auth, privileged Docker mode, loguru logging, multi-GPU support, installation design.
