# Stack Research — Audit & Challenge

**Project:** LLenergyMeasure v2.0
**Mode:** AUDIT — reviewing and challenging existing decisions
**Researched:** 2026-02-25
**Confidence:** HIGH (energy/versions) | MEDIUM (backends, CLI) | LOW (visualisation future)

---

## Scope

This audit reviews every existing stack decision in `.planning/codebase/STACK.md` and
`.product/decisions/` against current evidence. Where a prior decision holds, this is stated.
Where it does not, this document says so explicitly.

**Prior stack (from `.planning/codebase/STACK.md`, audited 2026-02-05):**
Python 3.10+ · Pydantic v2 · Typer · PyTorch + Transformers · vLLM · TensorRT-LLM ·
CodeCarbon · nvidia-ml-py · Rich · Loguru · Poetry

---

## 1. Energy Measurement: CodeCarbon + Zeus + NVML

### Existing Decision

`.product/decisions/architecture.md` (Sub-decision G):
> "CO2 decoupled from energy. Base = NVML polling. `[zeus]` = accurate energy. `[codecarbon]` = CO2."

`.product/research/08-energy-plugin-architecture.md` recommends Zeus as P0, CodeCarbon as P0
fallback, DCGM as P1, direct RAPL/Scaphandre as P2.

### Current Version Landscape

| Tool | Existing Pin | Current Version | Notes |
|------|-------------|-----------------|-------|
| CodeCarbon | `>=2.8.0` | **3.2.2** (Feb 2026) | 3.x is a significant version jump |
| Zeus (zeus-ml → zeus) | `[optional]` | **0.13.1** (Nov 2025) | Package renamed from `zeus-ml` to `zeus` |
| nvidia-ml-py | `>=12.0.0` | **13.590.48** (Jan 2026) | Per `.planning/codebase/STACK.md` bump |

**CRITICAL — CodeCarbon version gap:** The existing `pyproject.toml` pins `>=2.8.0`.
Current release is 3.2.2, a major version jump. The `_prepare_emissions_data()` private API call
identified in `.product/research/08-energy-plugin-architecture.md` as fragile across versions is
a live risk. Must verify API compatibility with 3.x before shipping.

**CRITICAL — Zeus package rename:** The PyPI package is now `zeus`, not `zeus-ml`.
The old `zeus-ml` package stopped at v0.11.0. Installing `zeus-ml` gets stale code.
`pyproject.toml` extras must use `zeus` (the current package).

### Should We Add PowerJoular / Scaphandre / RAPL Direct?

**PowerJoular** (Ada, external process monitoring by PID):
- Process-level CPU+GPU energy via RAPL + nvidia-smi
- Language-agnostic — monitors any process externally
- No Python in measurement path (lower interference)
- **Gap:** nvidia-smi sampling interval (~1s), not NVML counter accuracy
- **Assessment:** Useful for system-level cross-validation only. Not a replacement for Zeus.

**Scaphandre** (Rust agent, Prometheus exporter):
- RAPL-based CPU/DRAM only; no GPU energy
- Per-process attribution via `/proc`
- Research (Arxiv 2511.05597) notes "only provides an estimate based on RAPL without quantifying accuracy"
- Prometheus endpoint approach adds operational complexity (daemon required)
- **Assessment:** REJECT for primary use. Useful only in CPU-only or cross-validation scenarios.
  Zeus already reads RAPL directly with better integration.

**RAPL direct** (`/sys/class/powercap/intel-rapl/`):
- Zeus already wraps RAPL internally for CPU/DRAM measurement
- Building a separate RAPL backend would duplicate Zeus's CPU code path
- **Assessment:** REJECT — Zeus already handles this better.

**Intel RAPL via `pyRAPL`:**
- Library abandoned; last commit 2021.
- **Assessment:** REJECT — dead project.

**NVIDIA DCGM:**
- Enterprise GPU monitoring service (`nv-hostengine` daemon required)
- Very high accuracy for GPU metrics; no CPU/DRAM
- Relevant for HPC/cluster deployments where DCGM is pre-deployed
- **Assessment:** RETAIN as P1 (post v2.0). Not needed for researchers' workstations.

### Is the CO2 / Energy Separation Well-Grounded?

**YES — this decision holds and is well-supported.**

Evidence:
1. Zeus (the reference implementation for ML.ENERGY Leaderboard) explicitly does not compute CO2.
2. The ML.ENERGY Benchmark uses Zeus for energy + a separate carbon intensity lookup.
3. PMC comparison paper found up to 400% variation between tools measuring the same workload
   (different component scoping), confirming that energy and CO2 are genuinely different concerns.
4. Scaphandre-based tools (MELODI framework, 2025) also separate GPU energy from CO2 estimation.

**One nuance:** The `[codecarbon]` extra is positioned as the CO2 path, but CodeCarbon 3.x
now pins `pydantic<2` in some internal components — check compatibility with our Pydantic v2
config models. This must be verified before CodeCarbon is marketed as the CO2 extra.

### Recommendation

| Component | Decision | Version to Pin | Confidence |
|-----------|----------|----------------|------------|
| Zeus | KEEP as primary energy backend | `zeus>=0.13.1` (note: package is `zeus`, not `zeus-ml`) | HIGH |
| CodeCarbon | KEEP for CO2 only; verify 3.x private API compatibility | `codecarbon>=3.2.2` | MEDIUM |
| nvidia-ml-py | KEEP; bump minimum | `nvidia-ml-py>=13.590.48` | HIGH |
| PowerJoular | REJECT for primary use | — | HIGH |
| Scaphandre | REJECT for primary use | — | HIGH |
| RAPL direct | REJECT (Zeus handles) | — | HIGH |
| DCGM | P1 future optional | — | MEDIUM |

**Action required:** Update `pyproject.toml` extras from `zeus-ml` to `zeus`. Verify
`codecarbon>=3.0.0` does not break the private API calls in `codecarbon.py`.

---

## 2. Inference Backends: PyTorch + vLLM + TensorRT-LLM

### Existing Decision

Three backends at v2.0: PyTorch (Transformers), vLLM, TensorRT-LLM.
Priority ordering for additions: SGLang > llama.cpp > HF TGI (`.product/decisions/additional-backends.md`).

### Current Version Landscape

| Backend | Old Pin | Current Stable | Change |
|---------|---------|----------------|--------|
| PyTorch | `>=2.5.0` | **2.10.0** (Jan 2026) | Active; 2.11 planned Feb 2026 |
| Transformers | `>=4.49.0` | **v5.x** (now weekly minor releases) | **MAJOR VERSION** |
| vLLM | `>=0.6.0` | **0.15.1** | 15 minor versions ahead of pin |
| TensorRT-LLM | `>=0.12.0` | **1.3.0rc4** (Feb 2026) | Approaching 1.0 stable |

**CRITICAL — Transformers v5:** HuggingFace released Transformers v5 (first major release in
5 years). It removes long-due deprecations and refactors APIs. The old `>=4.49.0` pin allows
v4.x. For v2.0 we should pin `>=5.0.0`. The `.planning/codebase/STACK.md` already shows
`5.0.0` as the tested version, so this is partially addressed — but the `pyproject.toml`
`>=4.49.0` pin must be updated.

**CRITICAL — vLLM 0.15.1 vs pinned 0.6.0:** A 9-minor-version gap is substantial. vLLM v0.9.x
introduced a constraint: `transformers < 4.54.0` for that version range. Version drift between
vLLM and Transformers is a known pain point. Pinning strategy must be tested against current
vLLM+Transformers compatibility matrix.

**TensorRT-LLM approaching 1.0:** The `1.3.0rc` series suggests API stabilisation. The old
`>=0.12.0` pin is far behind. The 1.x API is likely breaking from 0.12 given NVIDIA's
typical versioning.

### Should We Add SGLang?

**SGLang evidence (2026-02-25):**
- Joined PyTorch ecosystem (official endorsement)
- Running on 400,000+ GPUs worldwide; xAI uses it for Grok 3
- 29% throughput advantage over optimised vLLM on H100 (H100 + Llama 3.1 8B benchmark)
- RadixAttention (automatic KV cache reuse) produces meaningfully different energy profiles
  from PagedAttention — this is directly relevant to parameter space measurement
- 2026 Q1 roadmap issue is open on GitHub, indicating active development trajectory

**Assessment: SGLang should move from "v3.x tentative" to "v2.2 candidate."**

Rationale: If the tool's purpose is measuring the *effect of implementation choices*, SGLang vs
vLLM is an especially high-value comparison because both run the same models but with
fundamentally different KV cache strategies. The energy profiles will differ non-trivially. This
is not just adding another backend — it is adding a backend with a genuinely different
implementation parameter space. No other energy benchmarking tool offers SGLang + energy.

**Process isolation:** SGLang likely conflicts with TRT-LLM at the CUDA driver level (confirmed
assumption, not yet empirically tested). Docker isolation required for multi-backend studies.

### Should We Add llama.cpp / GGUF?

**llama.cpp evidence:**
- 50K+ stars; runs natively as C++ subprocess (natural process isolation)
- CPU inference enables researchers without A100s — significant audience segment
- `optimum-benchmark` supports it via `llama-cpp-python`
- Energy profiles fundamentally different (CPU-dominated vs GPU-dominated)

**Complication:** CPU-inference energy profiles are not directly comparable to GPU-inference
profiles on a per-token basis. The tool's core claim is measuring the effect of implementation
choices while holding the model constant — llama.cpp users typically run quantised models
(GGUF), which also changes the model. This conflates quantisation effects with backend effects.

**Assessment: Keep llama.cpp as priority backend but DEFER to v3.x.**
Reason: The comparability concern requires a design decision (how to label and present CPU vs GPU
results) that should not be rushed into v2.0. The audience argument is valid but does not change
the priority ordering.

### Should We Add ExLlamaV2?

**Existing decision:** Rejected 2026-02-19 — niche GPTQ-focused use case.

**New evidence:** ExLlamaV2 v0.3.2 (mid-2025) added paged attention via Flash Attention 2.5.7+
and dynamic batching. It has broadened beyond GPTQ to EXL2 (2-8 bit) quantisation. However:
- Still consumer-GPU / small-batch focused
- Target audience overlaps with llama.cpp (researchers without data centre GPUs)
- Adds GPTQ quantisation as a variable, conflating model change with backend change

**Assessment: RETAIN rejection.** ExLlamaV2's use case is subsumed by llama.cpp for our
purposes. Neither should be added before llama.cpp is properly scoped.

### Should We Consider HF TGI?

**Assessment: RETAIN low priority.** TGI is PagedAttention-based (same as vLLM). Overlap is
high, differentiation is low. The only use case is measuring TGI-specific overhead (HTTP serving
layer) vs raw vLLM, which is a valid but niche research question. Defer past v4.0.

### Recommendation

| Backend | Decision | Version to Pin | Confidence |
|---------|----------|----------------|------------|
| PyTorch | KEEP | `>=2.9.0` (tested against 2.10.0) | HIGH |
| Transformers | KEEP; update pin | `>=5.0.0` (breaking from 4.x) | HIGH |
| vLLM | KEEP; update pin | `>=0.15.0` (verify Transformers compat) | HIGH |
| TensorRT-LLM | KEEP; update pin | `>=1.0.0` (1.3.0rc4 is pre-release) | MEDIUM |
| SGLang | ACCELERATE to v2.2 candidate | `>=0.4.0` (check current stable) | MEDIUM |
| llama.cpp | KEEP at v3.x; resolve comparability design | `llama-cpp-python>=0.3.0` | MEDIUM |
| ExLlamaV2 | RETAIN rejection | — | HIGH |
| HF TGI | RETAIN low priority | — | HIGH |

**Action required:** Audit `pyproject.toml` against current Transformers v5 + vLLM 0.15.x
compatibility. The version matrix must be tested, not assumed.

---

## 3. Configuration: Pydantic v2

### Existing Decision

`.planning/codebase/STACK.md`: Pydantic v2 for config validation and domain models.
Confirmed across multiple sessions.

### Should We Switch to Hydra?

**Pydantic v2 (current):**
- Fast Rust-backed validation
- Rich type support including constrained types, discriminated unions
- `ValidationError` passes through unchanged (confirmed in memory)
- Used by vLLM, FastAPI, Zeus, and most of the Python ML ecosystem
- Version 2.10+ is current (training data says ~2.10; HIGH confidence it's active)

**Hydra:**
- Dynamic hierarchical config composition
- CLI override syntax (`+`, `~`, `=`)
- Config groups for sweep composition
- Used by Facebook Research ML tools (FAIR), MMDetection, Hydra-based training frameworks
- **Cons:** Adds a non-trivial dependency with its own config schema system; runtime config
  assembly at CLI level vs parse-time resolution that the project has already committed to
  (`.product/decisions/experiment-study-architecture.md` specifies sweep resolution at
  YAML parse time, before Pydantic)

**Combined Pydantic + Hydra** (pattern seen in some research codebases):
- Hydra for config assembly and CLI overrides; Pydantic for validation
- This is a common pattern but adds two config systems

**Assessment: KEEP Pydantic v2. REJECT Hydra.**

Rationale: The project's config architecture (already decided) resolves sweeps at YAML parse
time and passes a resolved `StudyConfig` to the runner. Hydra's value is dynamic composition at
CLI invocation time — this is the opposite of the "parse-time resolution" model. Hydra would
require either (a) re-architecting the config resolution approach or (b) being used only as a
config loader (wasting most of its value). The existing Pydantic model is already well-designed
and validated against peer tools (see `.product/research/10-sweep-validation-patterns.md`).

**Note on `[codecarbon]` compatibility:** CodeCarbon has historically had internal Pydantic v1
dependencies in some versions. With CodeCarbon 3.2.2, this must be verified. If CodeCarbon 3.x
still conflicts, it must be handled with either an isolation boundary or a version-specific extra.

### Recommendation

| Choice | Decision | Version | Confidence |
|--------|----------|---------|------------|
| Pydantic v2 | KEEP | `>=2.9.0` | HIGH |
| Hydra | REJECT | — | HIGH |
| pydantic-settings | RETAIN for Layer 1 user config | `>=2.0` | HIGH |

---

## 4. CLI Framework: Typer

### Existing Decision

`.planning/codebase/STACK.md`: Typer `>=0.15.0`. Used across v1.x codebase.

### Should We Switch to cyclopts or Click?

**Typer (current):**
- Based on Click; type hints as CLI definition
- FastAPI team maintenance (active)
- Latest: v0.15.x (active as of Feb 2026)
- **Known issue:** Since v0.22.0, `typer-slim` no longer exists separately; installs full Typer
- Does not support Union types in parameter definitions
- Complex nested command scenarios can be awkward
- Well-documented; large community

**cyclopts:**
- Newer (2023+); inspired by Typer but addresses its limitations
- 38% less code than Typer for equivalent CLIs (per cyclopts docs comparison)
- Supports Union types, Literal types natively
- Docstring-driven help generation
- Migration path documented (`migrating-from-typer.html`)
- v3.16+ active (Feb 2026)
- Much smaller community; less ecosystem validation

**Click (direct):**
- Click is what Typer wraps; using Click directly gives more control
- More verbose; less type-hint-native
- lm-eval, MLflow, and most peer tools use Click directly
- Zero magic: full control over parameter parsing

**Assessment: KEEP Typer for v2.0. EVALUATE cyclopts at v3.0.**

Rationale: The v2.0 CLI is 2 commands + 1 flag (`llem run`, `llem config`, `--version`). This
is an extremely simple command surface — the existing Typer codebase handles it adequately. The
Union type limitation is not a blocker for this command set. Migrating CLI frameworks mid-version
introduces risk with minimal payoff for a 2-command CLI.

At v3.0, when additional commands may be added (lm-eval integration), re-evaluate cyclopts. The
38% code reduction and Union type support become more relevant with a larger command surface.

**Risk: Typer slow updates.** The Typer repository is maintained by the FastAPI team and has
historically been slower to adopt Click API changes. Monitor for regressions.

### Recommendation

| Choice | Decision | Version | Confidence |
|--------|----------|---------|------------|
| Typer | KEEP for v2.0 | `>=0.15.0` | HIGH |
| cyclopts | EVALUATE for v3.0+ | — | MEDIUM |
| Click (direct) | REJECT for v2.0 — too much rewrite | — | MEDIUM |

---

## 5. Testing: GPU-Dependent Code

### Existing Decision

`.planning/codebase/STACK.md`: pytest `>=8.0`, pytest-cov `>=4.0`. No GPU-specific strategy
documented in existing stack.

### What Do Peer Tools Do?

**HuggingFace Transformers:** `@pytest.mark.gpu` mark for GPU tests; CPU-only in PR CI;
GPU tests run on separate hardware. GPU availability via `torch.cuda.is_available()` fixture.

**InnerEye / Microsoft ML:** Marks: `@pytest.mark.gpu` and `@pytest.mark.cpu_and_gpu`.
All unmarked tests run on CPU CI; GPU tests require separate runner.

**vLLM:** Large mock/stub layer for GPU-free unit testing. Integration tests require GPU hardware.

**Zeus:** `pytest-mock`, `pytest-xdist` for parallelism. CPU-only unit tests; GPU tests require
hardware (flagged in CI).

**Pattern across peer tools:**
1. Unit tests: mock all NVML/CUDA calls; test on CPU CI
2. Integration tests: require GPU hardware; run on separate CI runner or nightly
3. Mark decorators: `@pytest.mark.require_gpu` or `@pytest.mark.slow`

### Recommended Testing Stack

| Tool | Version | Purpose | When |
|------|---------|---------|------|
| pytest | `>=8.0` | Test runner | All tests |
| pytest-cov | `>=4.0` | Coverage | All tests |
| pytest-mock | `>=3.12` | Mock NVML, backends | Unit tests |
| pytest-xdist | `>=3.5` | Parallel test execution | CI unit tests |
| `@pytest.mark.require_gpu` | custom mark | Gate GPU tests | GPU CI runner only |
| `unittest.mock.patch` | stdlib | Patch nvidia-ml-py, codecarbon | Unit tests |

**What NOT to do:** Do not attempt to test actual energy measurement accuracy in unit tests.
Energy measurement accuracy requires real hardware and controlled conditions — this belongs in
integration tests on real hardware, not in mocked unit tests. Unit tests verify code paths
(protocol compliance, error handling, result schema construction), not measurement accuracy.

**CI structure (recommended):**
```
Unit tests (CPU-only, fast):     all PRs, github-actions standard runner
Integration tests (GPU required): nightly, separate GPU runner
Hardware benchmarks:              manual trigger only, tagged GPU runner
```

### Recommendation

| Choice | Decision | Version | Confidence |
|--------|----------|---------|------------|
| pytest + pytest-cov | KEEP | `>=8.0`, `>=4.0` | HIGH |
| pytest-mock | ADD | `>=3.12` | HIGH |
| pytest-xdist | ADD | `>=3.5` | HIGH |
| GPU test marking | ADOPT `@pytest.mark.require_gpu` | custom | HIGH |
| Mocking NVML | ADOPT unittest.mock.patch pattern | stdlib | HIGH |

---

## 6. Visualisation

### Existing Decision

No explicit visualisation decision recorded in `.product/decisions/`. The v1.x codebase uses
Rich for terminal output. CodeCarbon's Carbonboard (Plotly Dash) was noted in research as a
local dashboard option.

### What Do Peer Tools Use?

**ML.ENERGY Leaderboard:** Node.js frontend (Vite + TypeScript + Tailwind) for the web
platform; Python-side results stored as JSON. No Python visualisation library.

**CodeCarbon:** Plotly Dash for Carbonboard (local CSV visualisation).

**optimum-benchmark:** No built-in visualisation; results as JSON/CSV; users bring their own.

**lm-eval:** No built-in visualisation; JSON results; community builds dashboards externally.

**Pattern:** No peer tool bundles a full visualisation framework. Results as JSON/CSV; users
handle visualisation. The web platform (v4.0) handles the public-facing display.

### Recommendations by Phase

**v2.0 (CLI/library only):**
- Rich for terminal output (already decided; KEEP)
- No Python visualisation library — output JSON/CSV; users bring their own
- Rationale: Adding a visualisation dependency (Plotly, Matplotlib) at v2.0 is premature.
  It adds weight to the base install and constrains the web platform design.

**v2.x local results navigation** (if `.product/decisions/local-result-navigation.md` requires
interactive exploration):
- Plotly (not Dash) for standalone HTML export — zero server required, shareable
- `plotly>=5.0` as an optional extra `[viz]`
- Rationale: Plotly's interactive HTML output is self-contained. Streamlit/Panel require a
  server process — inappropriate for a CLI tool's "local results" feature.

**v4.0 web platform:**
- Separate technology decision; not Python; likely TypeScript + Plotly.js or Vega-Lite

| Choice | Phase | Decision | Rationale |
|--------|-------|----------|-----------|
| Rich | v2.0 | KEEP | Terminal output; already decided |
| Matplotlib | v2.0 | REJECT | Static only; poor interactivity for parameter sweeps |
| Plotly | v2.x optional | CONDITIONALLY ADD as `[viz]` | Interactive HTML; zero server dependency |
| Streamlit | Any | REJECT | Server dependency; not appropriate for CLI tool |
| Panel | Any | REJECT | Server dependency; complex for parameter exploration |
| Dash | Any | REJECT | Server dependency; overkill for local results |

**Confidence:** MEDIUM — visualisation roadmap depends on decisions not yet made
(`.product/decisions/local-result-navigation.md`).

---

## 7. Package Management: Poetry vs uv

### Existing Decision

`.planning/codebase/STACK.md`: Poetry 2.x. `poetry.lock` present.

### Evidence (2026-02-25)

**Poetry:**
- v2.3.2 released February 2026 — actively maintained
- ~66 million monthly downloads on PyPI
- 7+ years production history; battle-tested
- Handles publishing, dependency groups, lock files
- Slow vs uv: ~11s lock from cold vs ~3s for uv

**uv:**
- ~75 million monthly downloads on PyPI — now *exceeds* Poetry in downloads
- Rust-backed; 10-100x faster than pip/Poetry for most operations
- Manages Python versions itself
- Early adopters report frequent API changes and breaking updates
- Less than 2 years old; rapidly maturing
- MLOps Community reports successful large-scale migrations from Poetry to uv

**Community signal:** In the data science/MLops community (2025-2026), uv adoption is
accelerating sharply. The 100x speed improvement is noticeable for a project with complex
GPU dependency trees (pytorch, vllm, tensorrt).

**The binary dependency problem:** uv handles binary packages (PyTorch, CUDA-compiled wheels)
through pip's wheel index. This is generally fine for pre-compiled wheels (which PyTorch, vLLM,
and TRT-LLM publish). The "binary deps" concern is primarily for packages requiring compilation
from source, which none of our main dependencies require.

### Assessment: MIGRATE from Poetry to uv for v2.0

The migration cost is bounded (well-documented migration path; `uv` is `pyproject.toml`
compatible). The payoff is significant:
1. Faster CI install times (GPU CI runners are expensive; faster installs matter)
2. Python version management built-in (simplifies Docker build setup)
3. Cleaner multi-extras install UX (`uv pip install "llenergymeasure[pytorch,zeus]"` — same
   syntax; just faster)
4. Growing community standard — new researchers will expect `uv` workflows

**Risk:** uv's API has had breaking changes. Pin uv itself: use `uv>=0.5.0` (stable API era).

### Recommendation

| Choice | Decision | Version | Confidence |
|--------|----------|---------|------------|
| Poetry | MIGRATE AWAY for v2.0 | — | MEDIUM |
| uv | ADOPT as primary package manager | `>=0.5.0` | MEDIUM |
| pip (direct) | RETAIN as fallback for users | — | HIGH |

**Action required:** Migrate `poetry.lock` → `uv.lock`. Update CI scripts. Update
Docker build from `poetry install` to `uv sync`. Document in README.

**Alternative view:** If migration risk is judged too high for v2.0, keep Poetry for v2.0
and plan uv migration for v2.2. The performance argument is strongest for CI pipelines,
not end-user installs (end users run `pip install`, not `poetry install`).

---

## 8. Statistical Libraries: Bootstrap CI, Warmup Detection

### Existing Decision

`.planning/codebase/STACK.md`: numpy for statistics. `.product/decisions/warmup-strategy.md`
specifies fixed-count warmup (n=5, 2 tokens max); 30s thermal floor.

No peer tool implements CV-convergence warmup (`.product/research/14-flops-warmup-lora-multiGPU.md`
confirms: "No CV warmup in any peer").

### What Should We Use?

**`scipy.stats.bootstrap` (current as of SciPy 1.17.0):**
- BCa method (bias-corrected and accelerated) is the gold standard for CI computation
- Percentile method available but "rarely used in practice" per SciPy docs
- `scipy.stats.bootstrap(data, statistic_fn, confidence_level=0.95, method='BCa', n_resamples=9999)`
- Already likely in the transitive dependency tree (numpy → scipy natural)

**`statsmodels`:**
- Heavier dependency; used for regression, ANOVA, time series models
- Bootstrap support exists but less ergonomic than scipy.stats for simple CI
- **Assessment:** REJECT for bootstrap CI — scipy.stats is sufficient and lighter.
  Consider adding statsmodels only if significance testing across backends is needed (future).

**Custom implementation:**
- Acceptable only for simple percentile bootstrap
- For BCa: non-trivial to implement correctly (jackknife bias correction)
- **Assessment:** REJECT — use scipy.stats.bootstrap

**numpy alone:**
- Sufficient for percentile bootstrap (simple resampling)
- Insufficient for BCa without additional code
- **Assessment:** REJECT for production CI — use scipy.stats.bootstrap

### Warmup Detection Libraries

The `.product/decisions/warmup-strategy.md` uses fixed-count warmup (5 runs). If a CV-based
approach is ever added (flagged as "no peer does this, but scientifically motivated"):
- **ruptures** (`pip install ruptures`): Change point detection in time series
  Gaussian kernel + Pelt algorithm for detecting steady-state onset from power/latency timeline
- **Assessment:** LOW priority. Fixed-count warmup is simpler and adequate for v2.0.
  Reserve ruptures evaluation for v2.x statistical enhancement milestone.

### Recommendation

| Choice | Decision | Version | Confidence |
|--------|----------|---------|------------|
| numpy | KEEP for basic stats | `>=1.26` | HIGH |
| `scipy.stats.bootstrap` | ADD for CI | `scipy>=1.12` | HIGH |
| statsmodels | REJECT for v2.0 | — | HIGH |
| ruptures | LOW PRIORITY (v2.x) | — | MEDIUM |

**Note:** scipy is likely already a transitive dependency (via sklearn, which Zeus depends on
for pre-Volta power monitoring). Adding `scipy` explicitly makes the dependency explicit.

---

## 9. Supporting Libraries — Unchanged Decisions

The following stack elements have been reviewed and no change is warranted:

| Library | Decision | Notes |
|---------|----------|-------|
| **Loguru** | KEEP | Structured logging; widely used in Python ML tooling; no superior alternative |
| **Rich** | KEEP | Terminal tables, progress, panels; used by Zeus itself; no superior alternative |
| **python-dotenv** | KEEP | `.env` loading for HF_TOKEN; standard pattern |
| **python-on-whales** | KEEP | Docker orchestration; rationale already confirmed in prior STACK.md |
| **psutil** | KEEP | CPU/memory metadata; stdlib supplement |
| **datasets (HuggingFace)** | KEEP | AIEnergyScore dataset loading; no alternative |
| **peft** | KEEP | LoRA adapter support confirmed for v2.0 |
| **bitsandbytes** | KEEP for `[pytorch]` | 4-bit/8-bit quantisation for PyTorch backend |

**Removed from v2.0 scope** (present in v1.x, not needed in v2.0):
- `schedule` — scheduled execution deferred
- `questionary` — no `llem init`; progressive disclosure instead
- `fastapi`, `uvicorn`, `sqlalchemy`, `asyncpg`, `alembic`, `python-jose` — API backend deferred to v4.0

---

## 10. Summary: What Is Wrong or Questionable

| Item | Severity | Issue | Action |
|------|----------|-------|--------|
| `zeus-ml` package name | **CRITICAL** | Package renamed to `zeus`; `zeus-ml` is stale | Update `pyproject.toml` extras immediately |
| CodeCarbon version pin `>=2.8.0` | **HIGH** | Current is 3.2.2; private API may have changed | Verify `_prepare_emissions_data()` in 3.x |
| Transformers pin `>=4.49.0` | **HIGH** | Current is v5.x; major version change | Update pin to `>=5.0.0` |
| vLLM pin `>=0.6.0` | **HIGH** | Current is 0.15.1; 9 minor versions ahead | Update pin; verify Transformers compat |
| TRT-LLM pin `>=0.12.0` | **HIGH** | Current is 1.3.0rc4; approaching 1.0 with likely breaking changes | Update pin to `>=1.0.0` when stable |
| No pytest-mock / GPU marking | **MEDIUM** | Testing strategy incomplete for GPU-dependent code | Add `pytest-mock`, `pytest-xdist`, GPU marks |
| SGLang deferral to v3.x | **MEDIUM** | Accelerate to v2.2 candidate; production-proven; unique energy profile | Update roadmap |
| Poetry (package manager) | **MEDIUM** | uv now more popular; faster CI builds | Evaluate migration to uv for v2.0 |
| CodeCarbon CO2 accuracy on v2.0 claim | **MEDIUM** | CodeCarbon is estimation (~±15%); presenting it alongside Zeus (±5%) is misleading without labelling | Add accuracy metadata to output schema |
| scipy not explicit in deps | **LOW** | scipy.stats.bootstrap should be explicit dep | Add `scipy>=1.12` |
| No visualisation strategy | **LOW** | Not a blocker; defer until local-result-navigation decision | Address in v2.x planning |

---

## 11. Recommended Stack (v2.0)

### Core Technologies

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | `>=3.10` | Application code | TRT-LLM lower bound; 3.10+ type syntax |
| Pydantic v2 | `>=2.9.0` | Config validation, domain models | Industry standard; Zeus uses it; fast Rust backend |
| Typer | `>=0.15.0` | CLI framework | Adequate for 2-command surface; defer cyclopts to v3.0 |
| PyTorch | `>=2.9.0` | Inference backend + tensor ops | Current; tested against 2.10.0 |
| Transformers | `>=5.0.0` | HuggingFace model loading | v5 is current stable; breaking from 4.x |
| vLLM | `>=0.15.0` | High-throughput inference | Current; verify Transformers compat |
| TensorRT-LLM | `>=1.0.0` | Compiled NVIDIA inference | Pin to 1.0 stable when released |

### Energy Stack

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `zeus` | `>=0.13.1` | Accurate energy measurement (±5%) | Hardware counter access; NeurIPS D&B 2025; ML.ENERGY uses it |
| `codecarbon` | `>=3.2.2` | CO2 estimation only | Decoupled from energy; separate optional extra |
| `nvidia-ml-py` | `>=13.590.48` | Base NVML polling | Official NVIDIA; deprecated pynvml removed |

### Supporting Libraries

| Library | Version | Purpose | When |
|---------|---------|---------|------|
| Loguru | `>=0.7.0` | Structured logging | Always |
| Rich | `>=13.0` | Terminal output | Always |
| numpy | `>=1.26` | Numeric operations | Always |
| scipy | `>=1.12` | `stats.bootstrap` BCa CI | Statistical reporting |
| psutil | `>=6.1` | CPU/memory metadata | Environment snapshot |
| datasets | `>=3.0` | AIEnergyScore dataset | Default workload |
| peft | `>=0.18.1` | LoRA adapters | `[pytorch]` extra |
| bitsandbytes | `>=0.45.0` | Quantisation | `[pytorch]` extra |
| python-dotenv | `>=1.0.0` | `.env` credential loading | Always |
| python-on-whales | `>=0.70.0` | Docker orchestration | `[v2.2]` multi-backend |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv (or Poetry 2.x) | Package management | uv preferred for speed; Poetry if migration risk judged too high |
| Ruff `>=0.8.0` | Linting + formatting | 100 char line length; pre-commit hook |
| mypy `>=1.0` | Type checking | Strict mode recommended |
| pytest `>=8.0` | Test runner | |
| pytest-cov `>=4.0` | Coverage | |
| pytest-mock `>=3.12` | GPU/backend mocking | Unit tests without hardware |
| pytest-xdist `>=3.5` | Parallel tests | CI speed |

### What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `zeus-ml` (PyPI) | Abandoned; last version 0.11.0; project moved to `zeus` | `zeus>=0.13.1` |
| `pynvml` | Deprecated by NVIDIA; PyTorch 25.09 containers emit warnings | `nvidia-ml-py>=13.590.48` |
| `codecarbon<3.0` | Old API; private method usage fragile | `codecarbon>=3.2.2` |
| `transformers<5.0` | Major version behind; deprecations removed in v5 | `transformers>=5.0.0` |
| Hydra | Config assembly at CLI time contradicts parse-time resolution architecture | Pydantic v2 |
| Streamlit/Panel/Dash | Server dependency; CLI tool should not require a web server for local results | Plotly standalone HTML (optional) |
| PowerJoular/Scaphandre | Lower accuracy than Zeus; RAPL estimation without NVML counter access | Zeus |
| `statsmodels` | Overkill for bootstrap CI; heavy dep | `scipy.stats.bootstrap` |
| PyPI `schedule` library | Scheduled execution deferred; adds footgun for v2.0 users | Remove from v2.0 deps |
| `questionary` | No `llem init`; peer tools don't have it | Remove from v2.0 deps |

---

## 12. M2 Study Execution: New Capabilities Stack

**Updated:** 2026-02-27 — focused stack for M2 (study/sweep execution). No new dependencies
required; all capabilities use Python stdlib + existing project dependencies.

### Subprocess Isolation (NEW in M2)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `multiprocessing.get_context("spawn")` | Python stdlib (3.10+) | Spawn isolated child process per experiment | PyTorch docs mandate `spawn` (not `fork`) for CUDA; `get_context()` avoids global side-effects on other libraries |
| `multiprocessing.Pipe(duplex=False)` | Python stdlib (3.10+) | Return `ExperimentResult` from child to parent | Faster than Queue (Queue adds lock overhead); single producer/consumer — Pipe is the correct primitive |
| `multiprocessing.Queue` | Python stdlib (3.10+) | Progress events from child to parent, concurrent with `p.join()` | Supports daemon thread drain while parent blocks; sentinel pattern handles SIGKILL path cleanly |
| `threading.Thread` | Python stdlib (3.10+) | Drain progress Queue in parent while blocked at `p.join()` | Main thread blocks at `p.join()`; daemon thread enables live progress updates |

**Confirmed peer pattern:** optimum-benchmark (`launcher=process`) sets multiprocessing start
method to `spawn` — confirmed from GitHub issue logs showing `[process][INFO] - + Setting
multiprocessing start method to spawn`. This is the de facto industry standard for GPU
subprocess isolation.

**Critical: use `get_context("spawn")`, NOT `set_start_method("spawn")`:** The context form
is scoped to `StudyRunner` only. The global form conflicts with Datasets, Accelerate, and other
libraries that also call `set_start_method` — raises `RuntimeError: context already set`.

**Critical: `daemon=False` on worker process:** With `daemon=True`, `Ctrl+C` kills the daemon
immediately and CUDA contexts do not teardown cleanly. With `daemon=False`, `StudyRunner` calls
`p.join()` explicitly — orphaned processes are prevented in normal operation, and CUDA teardown
is clean on the signal path.

**Critical: `p.kill()` not `p.terminate()` for timeouts:** SIGTERM (`p.terminate()`) may be
ignored by a GPU kernel stuck in a CUDA operation. CUDA signal handling is not re-entrant.
SIGKILL (`p.kill()`) guarantees termination.

**Pipe buffer concern:** Linux OS pipe buffer is ~64KB. If `ExperimentResult.model_dump_json()`
exceeds 64KB before parent reads it, sender blocks → deadlock with `p.join()`. Current
`ExperimentResult` is well under 64KB for v2.0 fields. The `_send_result()` fallback (write to
temp file >1MB, send path via Pipe) handles the rare large-result case. No reader thread needed.

**What NOT to use:**

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `multiprocessing.fork` (Linux default) | CUDA runtime is not fork-safe — `RuntimeError: Cannot re-initialize CUDA in forked subprocess` | `multiprocessing.get_context("spawn")` |
| `subprocess.run(["llem", "experiment", ...])` | CLI re-entry (v1.x pattern): no timeout, exit-code-only errors, filesystem-only IPC, CLI startup overhead | `mp_ctx.Process(target=_run_experiment_worker, ...)` |
| `ProcessPoolExecutor` | Pool reuses processes — CUDA state bleeds between experiments | Fresh `mp_ctx.Process` per experiment |
| `daemon=True` on worker | Killed on Ctrl+C before CUDA teardown | `daemon=False` (default) + explicit `p.join()` |
| `p.terminate()` for timeouts | Ignored by hung CUDA kernels | `p.kill()` (SIGKILL) |

### Sweep Grid Expansion (NEW in M2)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `itertools.product` | Python stdlib (3.10+) | Cartesian product over sweep dimensions | Zero dependency; lazy iterator; correct; standard for grid sweeps in all ML frameworks |

**Implementation:** `itertools.product(*[values for values in sweep_dict.values()])` with
`zip(sweep_dict.keys(), combo)` to reconstruct named parameter dicts. Dotted key parsing uses
`key.split(".", 1)` (first dot only) to handle nested backend params.

**No external library:** ConfigSpace has declarative constraint syntax but 0 of 5 peer tools
use it (W&B, Hydra, Ray Tune, Optuna, lm-eval all implement constraints via their own
validators). Pydantic `@model_validator` is the standard pattern. itertools.product handles all
grid expansion; constraints handled at ExperimentConfig construction time.

### Manifest Checkpointing (NEW in M2)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pydantic.BaseModel.model_dump_json()` | 2.12.5 (installed) | Serialise `StudyManifest` to JSON | Already project dependency; consistent with ExperimentResult serialisation |
| `os.replace()` | Python stdlib (3.10+) | Atomic manifest write (POSIX rename) | Reader always sees complete JSON, never partial write |
| `tempfile.mkstemp(dir=path.parent)` | Python stdlib (3.10+) | Temp file for atomic rename | Same-directory temp ensures rename stays on same filesystem (no cross-device copy) |

**Atomic write pattern (verified locally):**
```python
import os, tempfile
from pathlib import Path

def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, str(path))   # POSIX atomic rename(2)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise
```

**Not needed:** `atomicwrites` PyPI package adds `fsync` before rename (crash-safe on power
failure). Benchmark manifests do not require power-failure guarantees — if the host loses power
mid-study, the study re-runs. stdlib `os.replace()` is sufficient.

### Thermal Gap Management (EXISTING PATTERN, NEW WIRING in M2)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `time.sleep()` | Python stdlib (3.10+) | Fixed thermal gap between experiments | Simplest correct implementation; parent deliberately blocks during gap |

**Implementation:** Single `time.sleep(gap_seconds)` in parent between `_run_one()` calls.
Rich progress display during gap uses a spinner or countdown — the existing
`display.show_thermal_gap(gap)` call handles this; implementation can use 1s sleep loop
with Rich `Progress` updates if live countdown is desired.

**No adaptive logic in v2.0:** Gap values are fixed from `execution.config_gap_seconds`
(default 60s) and `execution.cycle_gap_seconds` (default 300s). GPU-temperature-based adaptive
gaps are a v2.x research question. `psutil` (already installed, 7.2.1) provides CPU/system
temperature if adaptive gaps are ever added.

### New Module Layout (M2, no new dependencies)

```
src/llenergymeasure/
└── study/                          ← NEW package (M2)
    ├── __init__.py
    ├── runner.py                    ← StudyRunner (multiprocessing.Process per experiment)
    ├── grid.py                      ← expand_grid() (itertools.product + dotted key parsing)
    └── manifest.py                  ← StudyManifest, ManifestWriter (atomic JSON writes)
```

**Integration points:**
- `StudyRunner._run_one()` calls `ExperimentOrchestrator(config).run()` inside child — M1
  single-experiment path, unchanged
- `StudyConfig` in `config/models.py` gains `execution: ExecutionConfig` field (M1 stub exists)
- `run_study()` in `_api.py` currently raises `NotImplementedError` — M2 implements via `StudyRunner`
- `ManifestWriter` writes to `results/{study_name}_{timestamp}/study_manifest.json`

### Version Compatibility (M2 capabilities)

| Package | Installed | Min Required | Notes |
|---------|-----------|-------------|-------|
| Python | 3.12.12 | 3.10 (pyproject.toml) | `get_context("spawn")` available since 3.4 |
| pydantic | 2.12.5 | ≥2.0 | `model_dump_json()` is Pydantic v2 API — verified working |
| rich | 14.2.0 | ≥10.0 | `Progress`, `Live`, spinner stable |
| loguru | 0.7.3 | any | Study lifecycle logging |
| pyarrow | 23.0.0 | ≥14.0 | Parquet export for study results |

**No new pyproject.toml entries required for M2.** All subprocess isolation, grid expansion,
manifest checkpointing, and thermal gap capabilities use Python stdlib + existing dependencies.

---

## Sources

- [PyTorch Multiprocessing Best Practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) — `spawn` mandate for CUDA — HIGH confidence
- [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html) — Pipe/Queue semantics, buffer characteristics — HIGH confidence
- [optimum-benchmark GitHub](https://github.com/huggingface/optimum-benchmark) — `launcher=process` spawn pattern — MEDIUM confidence (README + issue logs)
- [Python itertools docs](https://docs.python.org/3/library/itertools.html) — `product()` semantics — HIGH confidence
- Local verification — `get_context("spawn")`, `os.replace()`, `itertools.product`, `model_dump_json()` verified in Python 3.12.12 — HIGH confidence
- `.product/designs/experiment-isolation.md` — Full `StudyRunner` pattern — HIGH confidence
- `.product/designs/study-yaml.md` — Sweep grammar, dotted key notation — HIGH confidence
- `.product/designs/study-resume.md` — `StudyManifest` schema, `ManifestWriter` — HIGH confidence
- Zeus v0.13.1 — [https://github.com/ml-energy/zeus](https://github.com/ml-energy/zeus) — HIGH confidence
- CodeCarbon 3.2.2 — [https://pypi.org/project/codecarbon/](https://pypi.org/project/codecarbon/) — HIGH confidence
- vLLM 0.15.1 — [https://pypi.org/project/vllm/](https://pypi.org/project/vllm/) — HIGH confidence
- TensorRT-LLM 1.3.0rc4 — [https://pypi.org/project/tensorrt-llm/](https://pypi.org/project/tensorrt-llm/) — HIGH confidence
- scipy.stats.bootstrap — [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — HIGH confidence

---

*Stack audit for: LLenergyMeasure v2.0*
*Audited: 2026-02-25 (original); M2 addendum: 2026-02-27*
