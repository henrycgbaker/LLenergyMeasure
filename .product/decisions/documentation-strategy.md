# Documentation Strategy

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

> **Note on status:** Marked Proposed pending user confirmation. Applies from v2.0.0 upon confirmation.

## Decision

MkDocs Material + mkdocstrings, hosted on ReadTheDocs. Google-style docstrings. All user docs in `docs/`; short README linking out. Versioned docs from v2.0.0 (`/stable`, `/latest`, `/v2.0.0`). Guides (how-to) and reference (field/flag lists) strictly separated.

---

## Context

LLenergyMeasure needs user-facing docs covering installation, config, library API, backends, and energy measurement methodology. Docs must be versioned (SemVer), support PR preview builds, and auto-generate API reference from docstrings (`__init__.py` exports per `backward-compatibility.md`).

The project is a Python CLI library starting in 2024 — MkDocs Material is the dominant standard for modern Python tools; Sphinx is the legacy choice for pre-2020 tools.

---

## N1 — Documentation Tool

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **MkDocs + Material theme — Chosen** | Dominant modern standard (lm-eval, Zeus, optimum-benchmark, Pydantic, FastAPI); Markdown source matches repo prose style; search, versioning, admonitions, code tabs, dark mode built in; `mkdocstrings[python]` for API auto-gen | N/A for a new project |
| Sphinx (RST) | Large ecosystem; mature | Legacy choice for pre-2020 tools; reStructuredText adds friction; vLLM and CodeCarbon use it because they predate the shift |
| Docusaurus | Modern; React-based | Adds frontend build complexity for a Python tool; not standard in Python ecosystem |
| Plain GitHub wiki | Zero setup | No versioning; no API reference generation; disconnected from code; not used by any peer |

Peers: lm-eval, Zeus, optimum-benchmark, Pydantic, FastAPI all use MkDocs Material. vLLM and CodeCarbon use Sphinx (predates the shift).

### Consequences

Positive: Modern toolchain; Markdown source; auto-generated API reference; no RST friction.
Negative / Trade-offs: N/A for a new project choosing between these options.
Neutral: Docstring format must be chosen consistently (see N4).

---

## N2 — Hosting

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **ReadTheDocs — Chosen** | Free for open-source; automatic versioned docs (`/stable`, `/latest`, `/v2.0.0`); PR preview builds; MkDocs natively supported; lm-eval, CodeCarbon, sentence-transformers, huggingface/datasets all use it | None for a standard open-source project |
| GitHub Pages + `mike` (MkDocs versioning plugin) | Integrated with GitHub | Manual versioning setup; no automatic PR preview builds; no automatic SSL for custom domains |
| Self-hosted | Full control | Ops burden; not justified for docs |

Host at `llenergymeasure.readthedocs.io`. Claim the URL early, even if content is sparse at v2.0.

### Consequences

Positive: Automatic versioning; PR preview builds; free SSL; zero ops burden.
Negative / Trade-offs: N/A — RTD is strictly better than alternatives for this use case.
Neutral: URL must be claimed before v2.0 release even if content is minimal.

---

## N3 — Documentation Structure

### Decision

We will use the following structure:

```
docs/
  index.md            # What it is, quick example, install
  quickstart.md       # 5-minute getting started (zero-config → first result)
  guides/
    experiments.md    # llem run (single): config options, flags, output
    studies.md        # llem run (study YAML): sweep grammar, execution block
    config.md         # llem config: reading env output, user config file
    library.md        # run_experiment() + run_study() with code examples
    backends.md       # Backend install, capabilities, parameter coverage
    energy.md         # Energy measurement: what's measured, how, accuracy
  reference/
    api.md            # Auto-generated from docstrings (mkdocstrings)
    experiment-config.md  # ExperimentConfig full field reference
    study-config.md       # StudyConfig full field reference
    result-schema.md      # ExperimentResult + StudyResult field reference
    cli.md                # Auto-generated from Typer (typer-cli or click-man)
  contributing.md     # How to contribute, local dev setup
  changelog.md        # Symlink to CHANGELOG.md in repo root
```

**Key principle**: Guides explain *how to do things*. Reference explains *what fields/flags
exist*. Never mix them — lm-eval and Pydantic both make this mistake in places.

### Consequences

Positive: Clear separation between guides and reference prevents the mixing error seen in lm-eval and Pydantic.
Negative / Trade-offs: Requires discipline to maintain the separation as docs grow.
Neutral: N/A

---

## N4 — API Reference Auto-Generation

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`mkdocstrings[python]` + Google-style docstrings — Chosen** | Integrates directly with MkDocs; Google style used by Pydantic, FastAPI, httpx | N/A |
| Sphinx autodoc | Large ecosystem | Requires Sphinx; inconsistent with MkDocs choice |
| Manual API reference | Full control | Unmaintainable as API grows; drifts from code |

### Decision

We will use `mkdocstrings[python]` for API reference auto-generation. Docstring format:
Google style (consistent with Pydantic, FastAPI, httpx).

```python
def run_experiment(config: str | Path | ExperimentConfig, **kwargs) -> ExperimentResult:
    """Run a single inference efficiency experiment.

    Args:
        config: Path to a YAML config file, an ExperimentConfig object,
            or keyword arguments for zero-config mode.
        **kwargs: Zero-config mode — passed directly to ExperimentConfig.
            Must include at least ``model`` and ``backend``.

    Returns:
        ExperimentResult with energy, throughput, and latency metrics.

    Raises:
        ConfigError: If the config is invalid or the file is not found.
        BackendError: If the backend is not installed or fails to load.
        PreFlightError: If pre-flight validation fails.

    Example:
        >>> result = run_experiment("experiment.yaml")
        >>> result = run_experiment(model="meta-llama/Llama-3.1-8B", backend="pytorch")
    """
```

**Pydantic model docs**: Fields auto-documented from `Field(description="...")`.
Every ExperimentConfig field must have a description. This is enforced in CI via
a custom check: `grep -r 'Field()' llenergymeasure/config/` returns empty.

### Consequences

Positive: API reference stays in sync with code; CI-enforced field descriptions.
Negative / Trade-offs: Every ExperimentConfig field requires a `Field(description=...)` — CI will fail without it.
Neutral: Google-style docstrings must be enforced as a contribution standard.

---

## N5 — Versioned Docs from v2.0

### Decision

We will ship versioned docs from the v2.0.0 release. RTD automatically creates `/stable`
(latest release) and `/latest` (main branch) aliases. `mike` (MkDocs versioning plugin)
manages explicit version selectors in the UI.

Version selector shows: `v2.0.0`, `v3.0.0`, `latest`, `stable`.
Old versions stay available indefinitely — research reproducibility requires this.

### Consequences

Positive: Researchers can reference the exact docs version for their installed version.
Negative / Trade-offs: Old version docs consume RTD storage indefinitely (negligible in practice).
Neutral: `mike` plugin must be configured at project setup, not retrofitted later.

---

## N6 — In-Repo vs Hosted Docs Split

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **All user docs in `docs/`; short README linking out — Chosen** | Single place to update; versioned; Pydantic and FastAPI model; README stays maintainable | Users who find via GitHub get fewer docs at-a-glance |
| Long tutorials in README | Discoverable via GitHub | Two places to update; README becomes unmaintainable; not versioned |
| GitHub wiki | Supplementary docs | No versioning; disconnected from code; rejected for same reason as hosting choice |

**Rejected:**
- **Mixing tutorials into README**: `**Rejected (2026-02-19):** GitHub-only discoverability at the cost of maintainability — two places to update.`
- **Wiki**: `**Rejected (2026-02-19):** No versioning, disconnected from code.`

### Decision

We will keep all user-facing docs in `docs/`. README is short (< 100 lines): install
command, one-line description, link to docs. No long tutorials in README — send users
to the hosted docs. Peer: Pydantic, FastAPI.

`CONTRIBUTING.md`: Dev setup, test running, PR process, how to add a backend.

`CLAUDE.md` / `.planning/`: Internal project context (not user-facing). Not published to docs.

### Consequences

Positive: Single place to update; versioned alongside releases; README stays maintainable.
Negative / Trade-offs: Users who discover via GitHub README get minimal information and must follow the docs link.
Neutral: CONTRIBUTING.md must cover enough for new contributors without requiring docs access.

---

## Related

- [release-process.md](release-process.md): Release workflow that triggers docs builds
- [../designs/testing.md](../designs/testing.md): CI that validates doc builds on PRs
- [backward-compatibility.md](backward-compatibility.md): Versioned docs mirror SemVer policy
