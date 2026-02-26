# Release Process

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

> **Note on status:** Marked Proposed pending user confirmation. Applies from v2.0.0 upon
> confirmation.
>
> **Updated (2026-02-25):** Docker image publishing updated from v2.2 to v2.0 to reflect
> collapsed versioning roadmap. See [versioning-roadmap.md](versioning-roadmap.md).

---

## Context

The project needs a release process that covers PyPI publishing, Docker image distribution
(from v2.0), release cadence policy, changelog management, and version string management.
Each of these is a distinct sub-decision that can be made independently.

The target audience is ML researchers and practitioners — a community that values
reproducibility (pinnable versions, image digests) and stability over frequent releases.
The release process must integrate with the SemVer and deprecation policy in
`backward-compatibility.md`.

---

## M1 — PyPI Publishing

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **GitHub Actions + OIDC trusted publisher, triggered by git tag — Chosen** | No stored API token; no token rotation; used by lm-eval, sentence-transformers, most modern Python packages; version-tag trigger is unambiguous | Requires initial PyPI trusted publisher setup (one-time) |
| Manual `twine upload` from developer machine | Simple initial setup | Token management risk; human error on version strings; not reproducible |
| GitHub Actions + stored API token | Automated | Token rotation burden; token exposure risk in CI logs |

### Decision

We will use a GitHub Actions workflow triggered by git tag `v{version}`, publishing via
OIDC trusted publisher (no stored API token).

Workflow:
1. Tag `v2.0.0` on `main` → triggers release workflow
2. Build: `python -m build` (sdist + wheel)
3. Check: `twine check dist/*`
4. Publish: `twine upload` to PyPI via OIDC (no stored API token — GitHub Actions trusted publisher)
5. Create GitHub Release: auto-generated changelog from conventional commits since last tag

**OIDC trusted publisher** (not API token): PyPI supports GitHub Actions as a trusted
publisher since 2023. No secret management required, no token rotation. Peer: lm-eval,
sentence-transformers, and most modern Python packages use this pattern.

**Version source**: `pyproject.toml` `version = "2.0.0"`. Must match the git tag.
CI validates match before publishing — mismatch = pipeline failure, not silent skip.

**Pre-release builds**: `2.0.0.dev0`, `2.0.0.a1`, `2.0.0.b1`, `2.0.0.rc1`.
Published to PyPI with pre-release flag. `pip install llenergymeasure` never installs
pre-releases. `pip install llenergymeasure==2.0.0.rc1` installs explicitly.

### Consequences

Positive: No token management; reproducible builds; CI validates version consistency.
Negative / Trade-offs: One-time PyPI trusted publisher setup required.
Neutral: Pre-release versions must use PEP 440 pre-release suffixes.

---

## M2 — Docker Hub (Backend Images)

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`llenergymeasure/` org on Docker Hub, auto-built on release tag — Chosen** | Consistent with PyPI release trigger; image digests in release notes for reproducibility; `latest` tag auto-managed | Build matrix (6 images per release) adds CI time |
| Users build locally from Dockerfile | No CI complexity | Poor UX; no reproducibility; version drift |
| GitHub Container Registry (ghcr.io) | Integrated with GitHub | Less discoverable than Docker Hub for researchers; requires GitHub auth for pulls |

### Decision

We will publish `llenergymeasure/` organisation images on Docker Hub, auto-built on the
same release tag as PyPI (`v2.0.0` or later — Docker images ship at v2.0).

Build matrix automated via GitHub Actions: `{pytorch, vllm, tensorrt} × {cuda11, cuda12}`.

See [future-versions.md](future-versions.md) § "Docker Image Strategy" for full naming
conventions and base image choices.

**Image digest in release notes**: Every release records the image digests for
reproducibility. Users can pin: `llenergymeasure/pytorch@sha256:abc123...`

**Latest tag**: `llenergymeasure/pytorch:latest` always points to the most recent
stable release. `llenergymeasure/pytorch:2.2.0-cuda12` pins to a specific release.

### Consequences

Positive: Reproducible image pinning via digest; auto-managed `latest`; consistent with PyPI release trigger.
Negative / Trade-offs: 6-image build matrix adds CI time per release; Docker Hub org must be claimed.
Neutral: Docker image publishing begins at v2.0.

---

## M3 — Release Cadence

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Milestone-driven (no fixed schedule) — Chosen** | Matches research tool context; avoids pressure to ship half-complete milestones; consistent with lm-eval and Zeus | No predictable release timeline for users |
| Time-driven (e.g., monthly) | Predictable for users | Creates pressure to ship incomplete milestones; CodeCarbon example shows it still slips |
| On-demand (no policy) | Maximum flexibility | No discipline; releases can be delayed indefinitely |

### Decision

We will use milestone-driven releases with no fixed schedule.

Rationale: This is a research tool used by ML researchers and practitioners. Regular releases
on a fixed cadence would create pressure to ship half-complete milestones. Peer comparison:
lm-eval (milestone-driven), Zeus (milestone-driven), CodeCarbon (roughly quarterly but slips),
Pydantic (major versions milestone-driven; patch releases frequent).

**Patch releases** (e.g. `2.0.1`): Cut immediately for P0/P1 bugs. No waiting.
**Minor releases** (e.g. `2.1.0`): Cut when the milestone is complete.
**Major releases** (e.g. `3.0.0`): Cut when the breaking-change milestone is complete.

### Consequences

Positive: Releases ship when complete, not when a calendar says so; matches lm-eval/Zeus behaviour.
Negative / Trade-offs: No predictable timeline for users planning around release dates.
Neutral: Patch releases for P0/P1 bugs remain time-sensitive regardless of cadence policy.

---

## M4 — Changelog

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`CHANGELOG.md` + conventional commits + human curation — Chosen** | Auto-gen as starting point; human curation removes low-quality entries ("fix typo", "wip"); conventional commits are already enforced | Requires human curation step per release |
| Fully automated changelog (no curation) | Zero effort | Low-quality entries; auto-gen often misses the "why" |
| Manual changelog only | Highest quality | Time-consuming; easy to forget |

### Decision

We will maintain `CHANGELOG.md` in the repo root. Auto-generation from conventional commits
since last tag provides a starting draft; a human curates before publishing.

Structure:
```markdown
## v2.1.0 — YYYY-MM-DD

### Added
- Zeus energy backend (`pip install llenergymeasure[zeus]`)
- Bootstrap CI for experiment results

### Changed
- `run_study()` CLI effective default `n_cycles` changed from 1 to 3

### Deprecated (removed in v2.2.0)
- `ExperimentConfig.model_name` → use `model`

### Fixed
- vLLM shm-size not propagated to Docker containers (#42)
```

**Auto-generation**: GitHub Release description auto-generated from conventional commits
since last tag. Human curates before publishing — auto-gen is a starting point, not the
final changelog. Reason: auto-gen often produces low-quality entries ("fix typo", "wip").

### Consequences

Positive: Changelog quality controlled by human; conventional commits provide structure for auto-gen.
Negative / Trade-offs: Human curation step adds to release checklist.
Neutral: Deprecation entries in CHANGELOG must match the deprecation policy in `backward-compatibility.md`.

---

## M5 — Versioning Source of Truth

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`pyproject.toml` only — Chosen** | Used by Pydantic, httpx, lm-eval; no duplication; `importlib.metadata` exposes at runtime | Requires `importlib.metadata` import for runtime version access (standard library) |
| `__version__.py` | Simple runtime access | Duplication with `pyproject.toml`; version drift risk |
| Both `pyproject.toml` and `__version__.py` | Belt-and-suspenders | Two sources to keep in sync; CI must validate both |

### Decision

`pyproject.toml` is the single source of truth for the version string. No `__version__.py`,
no version duplication. Hatch and Poetry both support `pyproject.toml`-only versioning.

```toml
[project]
version = "2.0.0"
```

Available at runtime via `importlib.metadata`:
```python
from importlib.metadata import version
__version__ = version("llenergymeasure")
```

This is the pattern used by Pydantic, httpx, and lm-eval. Do not hardcode the version
string in Python source — it creates drift.

### Consequences

Positive: Single version source; no drift; standard library import for runtime access.
Negative / Trade-offs: N/A — `importlib.metadata` is standard library from Python 3.8+.
Neutral: CI must validate that `pyproject.toml` version matches the git tag before publishing.

---

## Related

- [backward-compatibility.md](backward-compatibility.md): SemVer policy, deprecation window
- [future-versions.md](future-versions.md): Docker image naming convention (M2 above)
- [../designs/testing.md](../designs/testing.md): CI workflow that gates releases
