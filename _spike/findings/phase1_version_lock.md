# Phase 1 Day 3 - Version lock for the mining-substrate empirical trial

**Status:** Locked (PyPI + NVIDIA index probed 2026-05-25).
**Cross-refs:** `.planning/mining-substrate-empirical-trial.md` § Experimental design / Matrix (table now reflects these picks); `_spike/findings/trial_epistemic_framing.md` § "What 'maximal' actually means dimensionally" (brittleness is first-class).
**Scope:** locks the 15 (engine, version) cells of the 5x3 matrix; downstream Phases 1-Day-4 (reference construction) and 3 (matrix execution) consume this lock.

---

## 15-cell summary

| Engine | Slot | Concrete version | Release date | Wheel availability (cp312, x86_64) | Install size (per venv, top-level pkg only) | Venv path | Container reuse |
|---|---|---|---|---|---|---|---|
| transformers | v-2 | `4.55.4` | 2025-08-22 | py3-none-any (`transformers-4.55.4-py3-none-any.whl`, ~11 MB) | ~80 MB base, ~3-4 GB with torch | `/tmp/trial_transformers_4_55_4_venv/` | none |
| transformers | v-1 | `4.56.2` | 2025-09-19 | py3-none-any (~11 MB) | ~80 MB base, ~3-4 GB with torch | `/tmp/trial_transformers_4_56_2_venv/` | none |
| transformers | active | `4.57.3` | 2025-11-25 | py3-none-any (~12 MB) | already installed in project venv | project venv (use directly) | `llenergymeasure:transformers-4.57.3` |
| transformers | v+1 | `4.57.6` | 2026-01-16 | py3-none-any (~12 MB) | ~80 MB base, ~3-4 GB with torch | `/tmp/trial_transformers_4_57_6_venv/` (or reuse container) | `llenergymeasure:transformers-4.57.6` |
| transformers | v+major | `5.9.0` | 2026-05-20 | py3-none-any (~11 MB) | ~80 MB base, ~3-4 GB with torch | `/tmp/trial_transformers_5_9_0_venv/` | none |
| vllm | v-2 | `0.6.0` | 2024-09-05 | `cp38-abi3-manylinux1_x86_64` (~163 MB) | ~5-7 GB (CUDA torch + flash-attn pulled by deps) | `/tmp/trial_vllm_0_6_0_venv/` (host glibc 2.31 ok) | none |
| vllm | v-1 | `0.6.6.post1` | 2024-12-27 | `cp38-abi3-manylinux1_x86_64` (~192 MB) | ~5-7 GB | `/tmp/trial_vllm_0_6_6_post1_venv/` (host ok) | none |
| vllm | active | `0.7.3` | 2025-02-20 | `cp38-abi3-manylinux1_x86_64` (~252 MB) | ~5-7 GB (already inside container) | use container | `llenergymeasure:vllm-v0.7.3` |
| vllm | v+1 | `0.9.2` | 2025-07-08 | `cp38-abi3-manylinux1_x86_64` (~366 MB) | ~5-7 GB | `/tmp/trial_vllm_0_9_2_venv/` (host glibc 2.31 ok - manylinux1) | none |
| vllm | v+major | `0.19.1` | 2026-04-18 | `cp38-abi3-manylinux_2_31_x86_64` (~413 MB) | ~5-7 GB | use container | `vllm/vllm-openai:v0.19.1` |
| tensorrt | v-2 | `0.19.0` | 2025-04-30 (approx via NVIDIA index timestamps) | `cp312-cp312-linux_x86_64` from NVIDIA index (~1.95 GB) | ~8-10 GB | `/tmp/trial_tensorrt_0_19_0_venv/` (CUDA wheels - NVIDIA index needed) | none |
| tensorrt | v-1 | `0.20.0` | 2025-06-04 (approx) | `cp312-cp312-linux_x86_64` from NVIDIA index (~3.53 GB) | ~10-12 GB | `/tmp/trial_tensorrt_0_20_0_venv/` | none |
| tensorrt | active | `0.21.0` | 2025-08-04 (approx) | `cp312-cp312-linux_x86_64` from NVIDIA index (~3.75 GB) | ~10-12 GB | `/tmp/trial_tensorrt_0_21_0_venv/` | none (the existing `llenergymeasure:tensorrt` image targets a different version; check before reuse - see § Per-engine investigation below) |
| tensorrt | v+1 | `1.0.0` | 2025-11-15 (approx) | `cp312-cp312-linux_x86_64` from NVIDIA index (~3.39 GB) | ~10-12 GB | `/tmp/trial_tensorrt_1_0_0_venv/` | none |
| tensorrt | v+major | `1.2.1` | 2026-04-20 (approx) | `cp312-cp312-linux_x86_64` from NVIDIA index (~2.40 GB) | ~10-12 GB | use container | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` |

(Install sizes are estimated; the upper bound assumes torch+CUDA installs. For strategy (b)/(c) "pure LLM" cells we only need source code, not a working install, so a fresh venv is unnecessary - source can be extracted from the wheel zip and read directly. See § Venv allocation below for the install-vs-source split.)

---

## Per-engine investigation

### transformers (PyPI: https://pypi.org/pypi/transformers/json)

Available versions probed: 4.55.0-4.55.4, 4.56.0-4.56.2, 4.57.0-4.57.6, 5.0.0rc{0,1,2,3}, 5.0.0, 5.1.0-5.9.0.

**Picks and reasoning:**
- v-2 `4.55.4` (latest 4.55.x patch, 2025-08-22) - second-most-prior minor line; locked patch level for max stability at that minor.
- v-1 `4.56.2` (latest 4.56.x patch, 2025-09-19) - the minor just before active 4.57.
- active `4.57.3` (locked).
- v+1 `4.57.6` (latest 4.57.x patch, 2026-01-16). **Anomaly**: no `4.58.x` / `4.59.x` was ever released. The 4.x track ended at 4.57.6 and the project jumped straight to 5.0.0. The closest "minor bump" within the 4.x track is therefore a patch bump within 4.57. The brittleness signal at v+1 is patch-level, not minor-level; the minor-bump signal effectively collapses into the v+major cell.
- v+major `5.9.0` (latest 5.x at probe time, 2026-05-20). Pre-existing Bake-off D used this exact version - convenient continuity for the reference set (Phase 1 Day 4 can reuse Bake-off D's mined output as a seed).

**Wheel compatibility:** all transformers releases ship `py3-none-any.whl` (pure-Python). cp312 manylinux x86_64 compatibility is automatic.

**Adjacent anomalies surfaced:** 5.0.0rc0-rc3 sit between 4.57.3 and 4.57.4 chronologically (rc0 was 2025-12-01, 4.57.4 was 2026-01-13). This means the 4.57.4-6 patches happened AFTER 5.0.0rc{0,1,2,3} - HF was actively backporting fixes to 4.57 even after starting the 5.x rc cycle. The v+1 cell is therefore "active line that continued maturing during the major-version rc cycle" - potentially interesting for the brittleness story (the patch backports may have been API-stabilising for the v4 surface).

### vllm (PyPI: https://pypi.org/pypi/vllm/json)

Available versions probed: 0.1.0-0.21.0 (full sequence; major chunks: 0.5.x, 0.6.x x14 releases, 0.7.x x4, 0.8.x x6, 0.9.x x4, 0.10.x x4, 0.11.x x3, 0.12.x x1, 0.13.x x1, 0.14.x x2, 0.15.x x2, 0.16.x x1, 0.17.x x2, 0.18.x x2, 0.19.x x2, 0.20.x x3, 0.21.x x1).

**Picks and reasoning:**
- v-2 `0.6.0` (2024-09-05) - earliest 0.6.x release. Picked over later 0.6.x to maximise distance from v-1 (0.6.6.post1).
- v-1 `0.6.6.post1` (2024-12-27) - latest 0.6.x. Sits ~1 month before 0.7.0 (2025-01-27).
- active `0.7.3` (locked).
- v+1 `0.9.2` (2025-07-08) - "latest 0.8-0.9" per plan. 0.9.2 is the last 0.9.x; chosen over 0.8.5.post1 because the v+1 slot wants the most recent reachable minor (the plan range goes up to 0.9). 0.9.x sits one major-feature-release ahead of 0.7 (cuda-graph rework, V1 engine path).
- v+major `0.19.1` (2026-04-18) - matches pre-existing `vllm/vllm-openai:v0.19.1` image. "latest 0.16-0.19" per plan; 0.19.1 is the highest in that range.

Note: the major-bump candidate at probe time COULD be 0.21.0 (current latest); the plan explicitly capped at 0.16-0.19 (i.e. acknowledged 0.19 as the architectural-shift target, with newer 0.20-0.21 being further evolution that's out of scope for the trial's range probe).

**Wheel compatibility:**
- All vllm releases ship `cp38-abi3` (stable ABI). Forward-compatible to cp312.
- 0.6.x and 0.7.x and 0.8.x and 0.9.x: `manylinux1_x86_64` (very permissive baseline, installs on any glibc 2.5+).
- 0.10.x-0.11.x: `manylinux1_x86_64`.
- 0.12.x: `manylinux1_x86_64`.
- 0.13.x-0.19.x: `manylinux_2_31_x86_64` (needs glibc >=2.31; **host glibc is exactly 2.31** - installable directly on host).
- 0.20.x: `manylinux_2_35_x86_64` (needs glibc >=2.35 - **NOT installable on host**, container only).
- 0.21.x: `manylinux_2_24_x86_64` (oddly back to a lower baseline - installable on host).

All four locked vllm picks (0.6.0, 0.6.6.post1, 0.9.2, 0.19.1) install on host glibc 2.31. Active 0.7.3 lives inside its container - no host install needed.

**Adjacent observations:** vllm has had three glibc-baseline shifts (0.13 -> 2.31, 0.20 -> 2.35, 0.21 -> 2.24) within a year, suggesting CI matrix shifts driven by which AMI ships in their build pipeline rather than a deliberate ABI strategy. None of the locked picks land in the 0.20.x glibc-2.35 trap.

### tensorrt_llm (PyPI stub + NVIDIA index: https://pypi.nvidia.com/tensorrt-llm/)

**Important wheel-source detail:** PyPI hosts only stub source distributions for tensorrt-llm (1-10 KB sdists with no wheels). The real wheels come from the NVIDIA index. Any venv install MUST add `--extra-index-url https://pypi.nvidia.com`.

Available versions probed (stable cp312 wheels only, x86_64): 0.13.0, 0.14.0, 0.15.0, 0.16.0, 0.17.0.post1, 0.18.0-0.18.2, 0.19.0, 0.20.0, 0.21.0, 1.0.0, 1.1.0, 1.2.0, 1.2.1 (+ many dev/rc tags between).

cp312 wheels begin at 0.13.0; earlier releases (0.5-0.12) only ship cp310 wheels. None of the locked picks fall in the cp310-only range.

**Picks and reasoning:**
- v-2 `0.19.0` (NVIDIA index timestamp 2025-04-30 approx) - only stable 0.19.x release. "latest 0.19.x" per plan trivially.
- v-1 `0.20.0` - only stable 0.20.x release. "latest 0.20.x" per plan trivially.
- active `0.21.0` (locked).
- v+1 `1.0.0` (2025-11-15 approx). **Anomaly**: no `0.22.x` ever released; the 0.x track ended at 0.21.0 and the project jumped to 1.0.0. The plan's "latest 0.2x after 0.21" range is empty. v+1 is therefore filled by the FIRST 1.x release - this tests the immediate architectural shift moment. The brittleness signal at v+1 vs v+major is therefore "early-major" vs "settled-major" rather than "minor" vs "major".
- v+major `1.2.1` - latest stable 1.x at probe time. Matches pre-existing `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` container.

**Wheel compatibility:**
- All five locked picks ship `cp312-cp312-linux_x86_64` wheels from the NVIDIA index. (Note: not `manylinux` - they are raw `linux_x86_64`, meaning glibc compatibility is not platform-tagged; in practice they need glibc >=2.28 or thereabouts. Host glibc 2.31 is fine.)
- Wheels are CUDA-bound (link against specific CUDA + cuDNN versions). The 0.21.0 wheel targets CUDA 12.x; 1.x wheels target CUDA 12.4+. Per § Tactical context the project has `aimehub/pytorch-2.5.1-aime-cuda12.1.1` as the CUDA base - this works for tensorrt 0.21 but may need a newer CUDA base for tensorrt 1.x venvs. For the 1.2.1 cell the pre-existing `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` container ships the matching CUDA stack and should be used rather than installing the wheel into a venv.

**Adjacent observation - existing `llenergymeasure:tensorrt` image (54.3 GB):** present on host but its installed version isn't tagged in the image name. If it ships tensorrt_llm 0.21.0 then it covers the active cell; if it ships an older version it doesn't. **Action for downstream:** the Phase 3 trial-runner should `docker run llenergymeasure:tensorrt python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"` once to identify which cell (if any) it covers, then either reuse for that cell or fall back to fresh venv install.

**Adjacent observation - tensorrt_llm install graph is heavy:** per `pip download` metadata the install pulls in `tensorrt`, `tensorrt-cu12`, `polygraphy`, `nvidia-modelopt`, `mpi4py`, and ~40 other CUDA/TRT bindings. Per-venv size 8-12 GB is conservative.

---

## Wheel compatibility per cell

All 15 cells have working cp312 x86_64 wheels at the locked version. Detailed compatibility:

| Cell | Wheel tag | Glibc requirement | Host installable? | Container alternative |
|---|---|---|---|---|
| transformers 4.55.4 | py3-none-any | any | yes | - |
| transformers 4.56.2 | py3-none-any | any | yes | - |
| transformers 4.57.3 | py3-none-any | any | yes | `llenergymeasure:transformers-4.57.3` |
| transformers 4.57.6 | py3-none-any | any | yes | `llenergymeasure:transformers-4.57.6` |
| transformers 5.9.0 | py3-none-any | any | yes | - |
| vllm 0.6.0 | cp38-abi3 manylinux1_x86_64 | >=2.5 | yes | - |
| vllm 0.6.6.post1 | cp38-abi3 manylinux1_x86_64 | >=2.5 | yes | - |
| vllm 0.7.3 | cp38-abi3 manylinux1_x86_64 | >=2.5 | yes (but use container) | `llenergymeasure:vllm-v0.7.3` |
| vllm 0.9.2 | cp38-abi3 manylinux1_x86_64 | >=2.5 | yes | - |
| vllm 0.19.1 | cp38-abi3 manylinux_2_31_x86_64 | >=2.31 | yes (host exactly 2.31) | `vllm/vllm-openai:v0.19.1` |
| tensorrt 0.19.0 | cp312-cp312 linux_x86_64 (NVIDIA idx) | ~>=2.28 | yes (CUDA 12.x runtime needed) | - |
| tensorrt 0.20.0 | cp312-cp312 linux_x86_64 (NVIDIA idx) | ~>=2.28 | yes | - |
| tensorrt 0.21.0 | cp312-cp312 linux_x86_64 (NVIDIA idx) | ~>=2.28 | yes | possibly `llenergymeasure:tensorrt` (verify version first) |
| tensorrt 1.0.0 | cp312-cp312 linux_x86_64 (NVIDIA idx) | ~>=2.28 | yes (CUDA 12.4+ likely needed) | - |
| tensorrt 1.2.1 | cp312-cp312 linux_x86_64 (NVIDIA idx) | ~>=2.28 | yes | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` |

No unlockable cells. Every slot has a concrete pick + working wheel.

---

## Disk budget total + breakdown

Two scenarios. The plan target is <100 GB transient disk.

**Scenario A: install every cell into a fresh venv** (worst case; no container reuse):

| Engine | Cells | Per-venv est. | Subtotal |
|---|---|---|---|
| transformers | 5 | 1 GB (no torch) / 4 GB (with torch) | 5-20 GB |
| vllm | 5 | 6 GB | 30 GB |
| tensorrt_llm | 5 | 10 GB | 50 GB |
| **Total** | **15** | - | **85-100 GB** |

This hits the 100 GB ceiling under the upper estimate. Tight but within budget.

**Scenario B: reuse pre-existing containers + minimise venvs for source-only strategies** (realistic):

Strategy (b)/(c) "pure LLM" cells need only source code, not a working install. Source can be extracted from the wheel zip in <1 GB transient per cell:

```
unzip -d /tmp/trial_src_<engine>_<version>/ <wheel.whl>
```

Strategy (a) "pure mining" cells need a working install (dynamic miner imports the package). These are the 15 expensive installs.

Strategy (d) hybrid follows whichever sub-strategy it uses.

Container reuse covers 5/15 of the (a) cells (active + v+major for each engine, plus transformers v+1):

| Cell | Source | Disk impact |
|---|---|---|
| transformers 4.57.3 | project venv (already installed) | 0 GB additional |
| transformers 4.57.6 | container `llenergymeasure:transformers-4.57.6` | 0 GB additional (8.79 GB image already pulled) |
| vllm 0.7.3 | container `llenergymeasure:vllm-v0.7.3` | 0 GB additional (16.7 GB image already pulled) |
| vllm 0.19.1 | container `vllm/vllm-openai:v0.19.1` | 0 GB additional (22.6 GB image already pulled) |
| tensorrt 1.2.1 | container `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` | 0 GB additional (37.3 GB image already pulled) |

Remaining 10 cells needing fresh venv install (for strategy (a) dynamic miner) or wheel-source extraction (for strategy (b)/(c)):

| Cell | Strategy (a) venv | Strategy (b)/(c) source only |
|---|---|---|
| transformers 4.55.4 | ~1-4 GB | ~50 MB |
| transformers 4.56.2 | ~1-4 GB | ~50 MB |
| transformers 5.9.0 | ~1-4 GB | ~50 MB |
| vllm 0.6.0 | ~5-7 GB | ~200 MB (wheel extracted) |
| vllm 0.6.6.post1 | ~5-7 GB | ~250 MB |
| vllm 0.9.2 | ~5-7 GB | ~450 MB |
| tensorrt 0.19.0 | ~10-12 GB | ~2.5 GB (wheel extracted) |
| tensorrt 0.20.0 | ~10-12 GB | ~4.5 GB |
| tensorrt 0.21.0 | ~10-12 GB | ~4.8 GB |
| tensorrt 1.0.0 | ~10-12 GB | ~4.3 GB |

Realistic disk total (Scenario B, all strategies executable):

- Strategy (a) requires fresh venvs for 10 cells (no container coverage): ~58-76 GB.
- Strategy (b)/(c) source-only extractions for the same 10 cells: ~17 GB.
- Containers reused: 0 GB additional (already on disk).

**Total transient disk: ~75-95 GB** depending on how torch-heavy the transformers venvs end up. Well within the <100 GB plan target.

**Optimisations available if disk is tight:**
- Skip torch install in transformers venvs unless dynamic miner is being run on that cell (torch is 3 GB / venv on its own; saving for transformers means staying ~5 GB lighter).
- For tensorrt cells where (a) dynamic miner isn't critical (per Phase 1 Day 2: vllm + tensorrt static miners are being extended, dynamic miners may be added LATER per § Per-strategy infrastructure needs), skip the venv install entirely - read source from the wheel via `unzip`. Saves ~40-50 GB.
- Sequential rather than parallel cell execution: build venv -> run -> tear down. Cuts peak disk to per-cell maximum (~12 GB).

---

## Venv allocation: paths + install order

**Per-cell venv paths** (use these exactly in `_spike/scripts/trial_runner.py` when it's built):

```
/tmp/trial_transformers_4_55_4_venv/
/tmp/trial_transformers_4_56_2_venv/
# transformers_4_57_3 -> project venv (or container llenergymeasure:transformers-4.57.3)
/tmp/trial_transformers_4_57_6_venv/  # or container llenergymeasure:transformers-4.57.6
/tmp/trial_transformers_5_9_0_venv/

/tmp/trial_vllm_0_6_0_venv/
/tmp/trial_vllm_0_6_6_post1_venv/
# vllm_0_7_3 -> container llenergymeasure:vllm-v0.7.3
/tmp/trial_vllm_0_9_2_venv/
# vllm_0_19_1 -> container vllm/vllm-openai:v0.19.1

/tmp/trial_tensorrt_0_19_0_venv/
/tmp/trial_tensorrt_0_20_0_venv/
/tmp/trial_tensorrt_0_21_0_venv/
/tmp/trial_tensorrt_1_0_0_venv/
# tensorrt_1_2_1 -> container nvcr.io/nvidia/tensorrt-llm/release:1.2.1
```

**Source-only extraction paths** (for strategy (b)/(c) where install isn't needed):

```
/tmp/trial_src_transformers_<ver>/
/tmp/trial_src_vllm_<ver>/
/tmp/trial_src_tensorrt_<ver>/
```

**Slug convention:** version dots and `.postN` suffixes become underscores. `0.6.6.post1` -> `0_6_6_post1`. `4.57.3` -> `4_57_3`. Consistent across paths and any per-cell artefact filenames.

**Install order** (suggested for trial_runner; ordered cheapest-first to maximise progress against disk budget):

1. transformers cells first (light installs, no GPU):
   - 4.55.4 (~30 sec install, ~1 GB)
   - 4.56.2 (~30 sec)
   - 4.57.6 (~30 sec; or skip and use container)
   - 5.9.0 (~30 sec)
2. vllm host-installable cells (manylinux1 baseline):
   - 0.6.0 (~5 min install incl. dependencies; ~6 GB)
   - 0.6.6.post1 (~5 min; ~6 GB)
   - 0.9.2 (~5 min; ~6 GB)
3. tensorrt cells (heaviest; require NVIDIA index + CUDA):
   - 0.19.0 (~10-15 min install; ~10 GB)
   - 0.20.0 (~10-15 min; ~10 GB)
   - 0.21.0 (~10-15 min; ~10 GB; possibly covered by `llenergymeasure:tensorrt` - check first)
   - 1.0.0 (~10-15 min; ~10 GB; may need newer CUDA base)
4. Containers (no install needed; pull-on-demand if not present):
   - `llenergymeasure:transformers-4.57.3`, `llenergymeasure:transformers-4.57.6`
   - `llenergymeasure:vllm-v0.7.3`, `vllm/vllm-openai:v0.19.1`
   - `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`

All five containers are already on disk per `docker images` probe; no pulls needed.

**venv build command template** (use `uv` for speed):

```bash
# transformers cell:
uv venv /tmp/trial_transformers_<slug>_venv --python 3.12
source /tmp/trial_transformers_<slug>_venv/bin/activate
uv pip install transformers==<version>
# (skip torch unless strategy (a) dynamic miner needed on this cell)

# vllm cell:
uv venv /tmp/trial_vllm_<slug>_venv --python 3.12
source /tmp/trial_vllm_<slug>_venv/bin/activate
uv pip install vllm==<version>

# tensorrt cell:
uv venv /tmp/trial_tensorrt_<slug>_venv --python 3.12
source /tmp/trial_tensorrt_<slug>_venv/bin/activate
uv pip install tensorrt_llm==<version> --extra-index-url https://pypi.nvidia.com
```

---

## Container reuse summary

5 cells are covered by pre-existing docker images (no fresh venv build needed for those cells):

| Cell | Image | Image size on disk |
|---|---|---|
| transformers 4.57.3 (active) | `llenergymeasure:transformers-4.57.3` | 8.79 GB |
| transformers 4.57.6 (v+1) | `llenergymeasure:transformers-4.57.6` | 8.79 GB |
| vllm 0.7.3 (active) | `llenergymeasure:vllm-v0.7.3` | 16.7 GB |
| vllm 0.19.1 (v+major) | `vllm/vllm-openai:v0.19.1` | 22.6 GB |
| tensorrt 1.2.1 (v+major) | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` | 37.3 GB |

Total existing-image disk: ~94 GB (already accounted for; the trial doesn't add to this).

Possible additional reuse: `llenergymeasure:tensorrt` (54.3 GB) may cover tensorrt 0.21.0 if its installed version is 0.21.0. Trial-runner should detect this at startup before deciding to build a 0.21.0 venv.

---

## Unlockable cells

**None.** All 15 cells have:
- a concrete PyPI version chosen,
- a working cp312 x86_64 wheel available,
- a clear install or container-reuse path.

Caveats documented above (transformers v+1 is patch-not-minor; tensorrt v+1 is early-major-not-minor) are matrix-shape observations rather than lock failures. The slot is filled in every case; the BRITTLENESS interpretation at those cells will need to account for the bump-distance being smaller (transformers) or different-axis (tensorrt) than originally framed.

---

## Items for downstream phases

These follow from the lock but are not part of Day 3's deliverable:

1. **Phase 1 Day 4 (reference construction)**: now has a concrete cell list to build references against. transformers 5.9.0 reference can seed from Bake-off D output (already produced). transformers 4.57.3 reference is the existing mature `engine_versions/transformers/v4_57_3/outputs/`.
2. **Phase 2 (LLM infra)**: source-only extraction (no install) is sufficient for strategy (b)/(c). The trial_runner's LLM-side path can avoid the venv-build overhead by using `unzip -d /tmp/trial_src_<engine>_<ver>/ <wheel>` and pointing the chunker at the extracted source tree.
3. **Phase 3 (matrix execution)**: container coverage tells the runner WHERE to execute each cell (host venv vs docker exec). The five container-covered cells run as `docker exec <container> python -m _spike.trial.run_cell ...`; the ten venv cells run as `source /tmp/<venv>/bin/activate && python -m _spike.trial.run_cell ...`.
4. **One verification still owed**: the existing `llenergymeasure:tensorrt` image's installed version - run `docker run --rm llenergymeasure:tensorrt python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"` once at Phase 3 startup to decide whether to reuse it for the tensorrt 0.21.0 cell.

---

## Verification log (commands run at lock time)

For traceability if a downstream phase wants to re-verify a pick. All commands ran cleanly on the host.

```bash
# Per-engine version listings:
pip index versions transformers
pip index versions vllm
pip index versions tensorrt_llm
pip index versions tensorrt_llm --extra-index-url https://pypi.nvidia.com

# Per-version wheel metadata:
curl -s https://pypi.org/pypi/<pkg>/<ver>/json | python3 -c "..."

# NVIDIA index listing:
curl -s https://pypi.nvidia.com/tensorrt-llm/ | grep -oE 'tensorrt_llm-[^"]+\.whl'

# Per-wheel HEAD size probe (NVIDIA-hosted wheels):
curl -sI "https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-<ver>-cp312-cp312-linux_x86_64.whl"

# Resolution check for non-trivial wheel tags (vllm 0.19.1 / 0.21.0):
pip download vllm==<ver> --no-deps --dest /tmp/probe-vllm-<ver>/ \
    --python-version 3.12 --only-binary :all: \
    --platform manylinux_2_<N>_x86_64 --abi abi3 --implementation cp

# Host glibc check:
ldd --version

# Existing docker images check:
docker images
```

No commands wrote into `src/`; the only file edits made during this lock pass are this report and the matrix table in `.planning/mining-substrate-empirical-trial.md`.
