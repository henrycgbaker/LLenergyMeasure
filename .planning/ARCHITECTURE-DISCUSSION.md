# Architecture Discussion: Execution Model & Campaign State

**Date:** 2026-02-03
**Context:** Issues raised during Phase 2/2.1 UAT that need resolution before Phase 3
**Status:** RESOLVED — decisions captured 2026-02-03

---

## Overview

Three interconnected issues need resolution:

1. **Container/execution architecture** — Is the current detection and dispatch model correct?
2. **Campaign resume feature** — How should state persistence and resumption work?
3. **Cycle detection bug** — Experiments don't know they're part of a multi-cycle campaign

These are symptoms of an underlying question: **What is the relationship between campaigns, experiments, and containers?**

---

## Issue 1: Container Architecture & Detection Logic

### Current Implementation

The codebase has:
- **Detection modules** (`config/docker_detection.py`, `config/backend_detection.py`) — created in Phase 2.1
- **Dispatch logic** (`cli/campaign.py`) — uses `docker compose run --rm` per experiment
- **Named volumes** — persist HF cache, TensorRT engines, experiment state

### Questions to Resolve

#### Q1.1: Is the detection logic correct?

**Current behaviour:**
```python
# docker_detection.py
def should_use_docker_for_campaign(backends: list[str]) -> bool:
    # Already inside Docker → no nested containers
    if is_inside_docker():
        return False

    # Single backend → prefer local execution if available
    if len(backends) == 1:
        backend = backends[0]
        if is_local_python_install() and is_backend_available(backend):
            return False  # Run locally

    # Multi-backend OR backend not installed locally → use Docker
    return True
```

**Your concern:** "It seems to try to detect if it's running locally or in container... I don't think this is correct?"

**Research finding:** This logic IS correct for the execution model described. The orchestrator runs on host and needs to know:
1. Am I already inside a container? (Don't nest containers)
2. Is the required backend available locally? (Don't use Docker if not needed)
3. Is this a multi-backend campaign? (Docker provides isolation between backends)

**But:** The detection only matters if we're deciding host vs container execution. If campaigns ALWAYS dispatch to containers, we don't need most of this detection.

#### Q1.2: Orchestrator on host — is this industry standard?

**Research finding: YES**

From Phase 2 RESEARCH:
> "Orchestrator runs as native Python on host. Backend containers receive experiment dispatch via `docker compose exec`. No Docker-in-Docker, no socket mounting."
>
> "Matches how MLflow/BentoML work. Researchers shouldn't debug container networking."

From container strategy research (MLPerf):
> "MLPerf uses one-off containers for actual benchmark runs. Persistent storage via volume mounts (not long-running containers)."

**Conclusion:** Host orchestrator + container dispatch is the correct architecture.

#### Q1.3: Could the orchestrator itself run in a container?

**Possible but adds complexity:**
- Would need Docker socket mount (`/var/run/docker.sock`) — security concern
- Would need to handle volume path translation (host paths vs container paths)
- GPU access from orchestrator not typically needed (it's just dispatching)
- No ML tool does this — all run orchestrator on host

**Recommendation:** Keep orchestrator on host. This is industry standard and simpler.

#### Q1.4: `run --rm` vs `up + exec` — which is correct?

**Research evolution:**
1. Phase 2 RESEARCH initially recommended `up + exec` (long-running containers)
2. During implementation, this was changed to `run --rm` (ephemeral containers)
3. Debug research document (`container-strategy-research.md`) explains the rationale

**Why `run --rm` is correct for this workload:**

| Factor | `run --rm` | `up + exec` |
|--------|------------|-------------|
| Container overhead | 3-5s per experiment | 0s per experiment |
| Campaign overhead | ~24s for 6 experiments | ~10s (startup + teardown) |
| Experiment duration | 1-10 minutes | 1-10 minutes |
| **Relative overhead** | **1.3%** | **0.6%** |
| GPU memory isolation | Perfect (fresh container) | Must manually clear |
| Error recovery | Automatic (fresh state) | Manual restart logic |
| Implementation | 5 lines | 300+ lines (ContainerManager) |

**Conclusion:** `run --rm` is correct. The 0.7% overhead savings from `up + exec` doesn't justify 300+ lines of lifecycle management code.

#### Q1.5: Should we offer `up + exec` as a user option?

**Your suggestion:** "Offer users an option to either run each experiment as separately spun up container (default + recommended for proper isolation), or if they're in a rush they can opt for docker up + exec."

**Assessment:**
- **Pro:** User choice, potentially faster for very short experiments
- **Con:** Maintaining two code paths, complexity, potential bugs
- **Break-even:** `up + exec` only justified if experiments average <30 seconds each (ours are 1-10 minutes)

**Recommendation:** **No** — not worth the complexity. If this becomes a bottleneck, revisit later. For now, `run --rm` covers 99% of use cases optimally.

---

## Issue 2: Campaign Resume Feature

### Current State

Campaign has manifest tracking (`campaign_manifest.json`) but:
- Resume is basic (ask user, skip completed, retry failed)
- No `lem resume` top-level command
- Interrupt handling (Ctrl+C) not well-defined
- Experiments don't know they're part of a campaign

### Requirements (from your message)

1. **Resume check at CLI entry point** — BEFORE any config processing
2. **Interrupt handling** — Ctrl+C correctly stops experiment/campaign
3. **State identification** — Resume identifies if it was experiment or campaign
4. **Campaign precedence** — Campaign is higher organising principle
5. **`lem resume` command** — Shows stats, offers options (resume/restart/wipe)
6. **New experiment/campaign detection** — Same flow as resume

### Proposed Design

```
lem experiment config.yaml
    │
    ├── Check: Is there interrupted experiment state?
    │   └── Yes → Offer resume prompt (before any config loading)
    │
    └── No → Proceed with experiment

lem campaign config.yaml
    │
    ├── Check: Is there interrupted campaign state?
    │   ├── Yes → Show stats (progress %, completed/remaining table)
    │   │   └── Options: Resume / Restart / Wipe state
    │   │
    │   └── No → Proceed with campaign
    │
    └── Campaign starts (Ctrl+C saves state, exits cleanly)

lem resume
    │
    ├── Check for interrupted state (campaign or experiment)
    │   ├── Campaign found → Show campaign resume UI
    │   ├── Experiment found → Show experiment resume UI
    │   └── Nothing found → "No interrupted work found"
    │
    └── User chooses action → Routes to appropriate command
```

### State Persistence Model

```
.lem-state/
├── active_campaign.json     # Currently running campaign (if any)
├── active_experiment.json   # Currently running experiment (if any)
└── campaigns/
    └── <campaign_id>/
        ├── manifest.json    # Full manifest with all experiment status
        └── checkpoints/     # Per-experiment checkpoints (optional)
```

**Key insight:** Campaign owns experiments. If an experiment is interrupted mid-campaign, the campaign manifest tracks this. The experiment doesn't need separate state — it's recorded in the campaign.

### Questions to Resolve

#### Q2.1: Where should state live?

**Options:**
- A) `.lem-state/` in project root (alongside `.planning/`, `results/`)
- B) `.cache/llenergymeasure/` in project root
- C) `~/.cache/llenergymeasure/` in user home (global)
- D) Within `results/` directory (alongside result files)

**Recommendation:** Option A — `.lem-state/` in project root. Clear, discoverable, gitignore-able, project-scoped.

#### Q2.2: Should we add a user preferences config?

**Your suggestion:** "An initial install/config workflow that creates a config.json file with these sort of user preferences saved. Similar to how GSD works."

**Potential settings:**
- Default backend preference
- Default results directory
- Docker image preferences
- Thermal gap defaults
- Notification webhook URL

**Recommendation:** **Yes, but minimal scope.** Create `.lem-config.yaml` in project root (or `~/.config/llenergymeasure/config.yaml` for global). Start with:
- `default_backend: pytorch`
- `results_dir: ./results`
- `docker.auto_build: false`

Expand based on user feedback, not speculation.

---

## Issue 3: Cycle Detection Bug

### Symptom

```
Campaign Configuration
  Cycles: 3

Experiment: vllm-gpt2-float16
  Single cycle: confidence intervals and robustness metrics require >= 3 cycles (--cycles 3)
```

Campaign has 3 cycles, but individual experiment inside container doesn't know this.

### Root Cause

When campaign dispatches to container:
```bash
docker compose run --rm vllm lem experiment config.yaml --dataset alpaca -n 100
```

The experiment command runs in isolation. It doesn't know:
- That it's part of a campaign
- What cycle it's in (1 of 3)
- That other cycles will run

The "single cycle" message comes from experiment code checking its own `--cycles` argument, which defaults to 1.

### Fix Options

#### Option A: Pass campaign context to experiment

```bash
docker compose run --rm vllm lem experiment config.yaml \
    --campaign-id f3829eb0 \
    --cycle 1 \
    --total-cycles 3
```

Experiment then knows it's part of a campaign and suppresses the warning.

**Pro:** Clean, explicit
**Con:** More arguments to thread through

#### Option B: Environment variables

```bash
docker compose run --rm \
    -e LEM_CAMPAIGN_ID=f3829eb0 \
    -e LEM_CYCLE=1 \
    -e LEM_TOTAL_CYCLES=3 \
    vllm lem experiment config.yaml
```

Experiment checks env vars before displaying warning.

**Pro:** No CLI changes, easy to add
**Con:** Hidden context, harder to debug

#### Option C: Campaign-level CI computation only

Experiments never compute CI — that's campaign's responsibility. Experiment just saves raw results. Campaign aggregates across cycles and computes CI.

**Pro:** Clean separation of concerns (already how aggregation works)
**Con:** Experiment still displays misleading message

**Recommendation:** **Combination of A + C**
- Pass `--campaign-context` flag (or env var) to suppress experiment-level messages
- Compute CI at campaign level only (already the design)
- Experiment output says "Part of campaign f3829eb0, cycle 1/3" instead of warning

---

## Issue 4: Other Bugs Mentioned

### 4.1: TensorRT backend not available

```
llenergymeasure.exceptions.ConfigurationError: Backend 'tensorrt' is not available on this system.
```

**Analysis:** Experiment ran in `base` container, not `tensorrt` container. Docker dispatch logic is routing incorrectly.

**Likely cause:** Service name mismatch. Campaign config says `tensorrt`, but container service is named something else (or image not built).

**Fix:** Verify container routing logic in `campaign.py` matches docker-compose.yml service names.

### 4.2: Default GPU wait times too long

**Current:** 1 minute between experiments

**Your feedback:** "A lot for r[...]" (message truncated, assuming "a lot for routine use")

**Recommendation:** Add to user preferences config:
```yaml
thermal_gaps:
  between_experiments: 30  # seconds (default: 60)
  between_cycles: 180      # seconds (default: 300)
```

### 4.3: Multi-GPU terminology

**Current:** `num_processes` (PyTorch/accelerate specific)

**Your question:** Whether this applies to other backends

**Answer:** `num_processes` is accelerate-specific. Other backends have different parallelism:
- vLLM: `tensor_parallel_size`
- TensorRT: `tensor_parallelism`

The current code uses `accelerate launch --num_processes` for all backends, which may not be correct for vLLM/TensorRT.

**Recommendation:** Phase 3 parameter completeness should audit parallelism handling per backend.

---

## Proposed Resolution: New Phases

Based on this discussion, I recommend adding:

### Phase 2.2: Campaign Execution Model (Architecture Fix)

**Goal:** Fix container routing, cycle context propagation, and clarify the execution model.

**Scope:**
- Fix TensorRT container routing bug
- Add campaign context to experiment dispatch (cycle awareness)
- Ensure CI computed at campaign level only
- Document the execution model clearly

**Estimated:** 2-3 plans

### Phase 2.3: Campaign State & Resume

**Goal:** Robust campaign state persistence with graceful interrupt handling and resume capability.

**Scope:**
- State directory structure (`.lem-state/`)
- `lem resume` command with full UI
- Interrupt handler (Ctrl+C saves state cleanly)
- Resume check at CLI entry points (before config loading)
- Basic user preferences file (`.lem-config.yaml`)

**Estimated:** 4-5 plans

### Deferred to Phase 4 (Polish)

- GPU wait time configurability
- Multi-GPU parallelism audit
- Documentation refresh for execution model

---

## Questions for User

Before proceeding with planning, please confirm:

1. **Orchestrator location:** Agree that host-based orchestrator is correct? (Research strongly supports this)

2. **Container strategy:** Agree that `run --rm` only (no `up + exec` option) is sufficient for now?

3. **State location:** `.lem-state/` in project root acceptable? Or prefer `results/.state/` or global `~/.cache/`?

4. **User preferences:** Start minimal (defaults only) or include more options from the start?

5. **Phase ordering:** Address Phase 2.2 (architecture fixes) before Phase 2.3 (resume), or combine them?

6. **Cycle bug priority:** Fix immediately (quick patch) or as part of Phase 2.2?

---

---

## Decisions (2026-02-03)

### D1: State Directory Location

**Decision:** `.lem-state/` in project root

**Rationale:** Clear, discoverable, project-scoped, gitignore-able. Alongside `results/` and `.planning/`.

### D2: User Preferences Config

**Decision:** Full-featured from start

**Scope includes:**
- Default backend preference
- Results directory
- Thermal gap settings (between experiments, between cycles)
- Docker preferences (container strategy, auto-build)
- Notification webhook URL

**File:** `.lem-config.yaml` in project root

### D3: Container Strategy

**Decision:** Offer both `run --rm` (default) and `up + exec` as user option

**Implementation:**
- `run --rm` remains default (recommended for isolation)
- User can configure `docker.strategy: persistent` in `.lem-config.yaml` to use `up + exec`
- Campaign CLI flag: `--container-strategy [ephemeral|persistent]`

**Rationale:** User choice. Some users may prefer faster execution over perfect isolation for rapid iteration.

### D4: Phase Structure

**Decision:** Two phases

- **Phase 2.2:** Campaign Execution Model (architecture fixes)
  - Container routing fix (TensorRT bug)
  - Cycle context propagation
  - Dual container strategy implementation
  - CI computation at campaign level only

- **Phase 2.3:** Campaign State & Resume
  - State directory structure (`.lem-state/`)
  - `lem resume` command
  - Interrupt handling (Ctrl+C)
  - User preferences config (`.lem-config.yaml`)
  - Resume check at CLI entry points

---

## Next Steps

1. Update ROADMAP.md with Phase 2.2 and 2.3
2. `/gsd:discuss-phase 2.2` — Gather context for execution model phase
3. `/gsd:plan-phase 2.2` — Create plans
4. Execute Phase 2.2
5. Repeat for Phase 2.3

---

## References

- `.planning/debug/container-strategy-research.md` — Full `run --rm` vs `up + exec` analysis
- `.planning/phases/02-RESEARCH.md` — Campaign orchestration research
- `.planning/phases/02.1-RESEARCH.md` — Install experience research
- `.planning/phases/02-CONTEXT.md` — Phase 2 decisions
- `.planning/phases/02.1-CONTEXT.md` — Phase 2.1 decisions
