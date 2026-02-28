# Phase 18: Docker Pre-flight - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Docker pre-flight checks catch misconfigured host environments before any container is launched, giving users actionable error messages. Covers: NVIDIA Container Toolkit detection, GPU visibility validation inside containers, and CUDA/driver compatibility checking. Does NOT include per-image CUDA version management, platform-specific install automation, or standalone CLI surface.

</domain>

<decisions>
## Implementation Decisions

### Error messaging style
- Actionable guidance: name the problem, explain why it matters, suggest the fix (with upstream NVIDIA doc link)
- Use existing `LlemError` hierarchy — new `PreFlightError` subclass
- Multiple failures: grouped summary (numbered list of all failures within the current tier, then abort)
- Silent on success — no output when all checks pass, just proceed to container launch

### Check execution model
- Tiered execution: tier-1 checks (Docker exists, NVIDIA Container Toolkit installed) must all pass before tier-2 checks (GPU visibility inside container, CUDA/driver compat) run
- Report all failures within each tier before aborting
- Runs once per study, before the first Docker dispatch — not per-experiment
- Internal only — no separate CLI surface (no `llem config --check-docker`)
- No caching needed — single invocation per study

### CUDA/driver mismatch handling
- Hard error with version numbers when host driver is too old for container CUDA (e.g. "Host driver 525.x < required 530+ for CUDA 12.x")
- `--skip-preflight` flag (CLI) and `docker.skip_preflight: true` (config YAML) to bypass — CLI overrides config
- Detection: both tiers — host `nvidia-smi` parse (tier 1), then lightweight container probe with `nvidia-smi` inside (tier 2)
- Granularity: "does GPU work in Docker at all" check, NOT per-image CUDA version requirements
- No host `nvidia-smi` found: warn but attempt container launch (supports remote Docker daemon with GPUs)

### Recovery guidance
- Generic guidance only — link to upstream NVIDIA docs, no platform-specific install commands (no apt/yum detection)
- Do not reference our own documentation in error messages — always point upstream
- Container probe failure (tier 2, after host nvidia-smi passed): generic "NVIDIA Container Toolkit not configured correctly" with link to NVIDIA toolkit docs
- Keep error messages lean — we are not an NVIDIA support tool

### Claude's Discretion
- Exact tier-1 vs tier-2 check partitioning (which checks go in which tier)
- Container probe image choice (e.g. `nvidia/cuda:base` or `ubuntu` with `--gpus`)
- Internal module structure (single module vs check-per-file)
- Exact `nvidia-smi` output parsing approach
- PreFlightError subclass design (single class vs per-check subclasses)

</decisions>

<specifics>
## Specific Ideas

- Remote Docker daemon scenario: user may have no local NVIDIA driver but a remote Docker host with GPUs — pre-flight must not hard-block on missing host nvidia-smi
- Force/skip mechanism serves both CUDA mismatch bypass and the remote daemon case
- Tier model means a user with no Docker installed sees "Docker not found" rather than a cascade of GPU-related errors that don't help

</specifics>

<deferred>
## Deferred Ideas

- Per-backend-image CUDA version requirements (e.g. vLLM needs 12.1+, TRT-LLM needs 12.4+) — could be added later if users hit confusing errors after pulling large images
- Platform-specific install guidance (apt/yum/brew detection) — maintenance burden outweighs value for now

</deferred>

---

*Phase: 18-docker-pre-flight*
*Context gathered: 2026-02-28*
