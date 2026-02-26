# Access Control & Authentication

**Status:** Accepted
**Date decided:** 2026-02-25
**Last updated:** 2026-02-25 (revised: delegate to `huggingface_hub` auth chain; `.env` approach superseded)
**Research:** N/A

## Decision

- Delegate authentication to `huggingface_hub`'s built-in auth chain: `huggingface-cli login` → `~/.cache/huggingface/token`, `HF_TOKEN` env var, or programmatic `login()`. No `.env` file pattern.
- Auth fields in experiment/study YAML → hard error at parse time.
- Pre-flight checks report missing credentials with exact fix commands referencing `huggingface-cli login`.

---

## Context

LLenergyMeasure needs access to gated Hugging Face models (e.g. Llama) that require authentication. The tool must handle auth credentials without inadvertently exposing them via config files that might be version-controlled.

The target v2.0 deployment is a single-user local machine or a shared server where each researcher uses their own OS account. Multi-tenancy and fine-grained access control are not v2.0 concerns.

The 12-factor app pattern (store credentials in the environment, not config files) is the established industry standard for this class of problem. `.env` files are the standard ergonomic layer on top of this pattern — used by Docker Compose, python-dotenv, and most modern frameworks.

## Considered Options

### Credential Storage Mechanism

| Option | Pros | Cons |
|--------|------|------|
| **Delegate to `huggingface_hub` auth chain (chosen, revised 2026-02-25)** | ML researchers already use `huggingface-cli login`; zero new patterns to learn; `huggingface_hub` handles token storage, env var, and programmatic login; no `.env` file needed | Adds `huggingface_hub` as dependency (already required by transformers) |
| `.env` file with `.gitignore` + `.env.example` template | Standard dotenv pattern; familiar to web developers | Unfamiliar to ML researchers who use `huggingface-cli login`; adds a pattern foreign to the audience; requires `.gitignore` discipline |
| Shell environment variables only | Session-scoped, not filesystem-scoped. Works with external secrets managers. | Requires users to set env vars in their shell profile; unfamiliar to non-developers. |
| Config file field (`~/.config/llenergymeasure/config.yaml`) | Convenient; single place for all config | High risk of accidental version-control exposure. Requires encryption or gitignore discipline that users routinely fail. Rejected by all peer tools. |
| OS keyring integration | Secure storage; GUI-friendly | Adds OS-specific dependency (keyring library). YAGNI at v2.0. Platform differences (GNOME Keyring, macOS Keychain, Windows Credential Store) add complexity. |

> **Superseded (2026-02-25):** Shell-env-only was the original decision (2026-02-19). Briefly revised to `.env` file, then superseded again by `huggingface_hub` delegation. ML researchers already use `huggingface-cli login` — introducing a `.env` pattern is unnecessary friction for this audience.
>
> **Rejected (2026-02-25): `.env` file pattern.** Over-engineered for the v2.0 use case. The only auth requirement is `HF_TOKEN`, which `huggingface_hub` handles natively. `.env` is a web development pattern unfamiliar to ML researchers. See `.planning/research/DECISION-AUDIT.md` §2.26.

### Config File Validation

| Option | Pros | Cons |
|--------|------|------|
| **Hard error at parse time if auth fields present in experiment/study YAML** | Prevents accidental version-control exposure. Consistent with maximally-explicit principle in `cli-ux.md` and `extra = "forbid"` in `config-architecture.md`. Error message directs user to `.env`. | Breaks on any field name that pattern-matches auth terms; false positives possible. |
| No validation | Simpler | Silently accepts credentials in config; high exposure risk. |
| Warning only | Non-breaking | Users may ignore warnings; security hygiene degrades silently. |

### Pre-flight Auth Checks

| Option | Pros | Cons |
|--------|------|------|
| **Pre-flight check with exact fix command in error message** | User sees precisely what is missing and what to run. Consistent with general pre-flight check pattern in `architecture.md`. | Requires detecting which models are gated (HF API call or local heuristic). |
| Generic "auth failed" error at runtime | Simpler pre-flight | Poor UX; user must interpret backend error messages to understand what went wrong. |
| No pre-flight; fail at model load | Zero overhead | Cryptic backend error; no actionable guidance. |

### Shared Server Multi-Tenancy

| Option | Pros | Cons |
|--------|------|------|
| **Standard OS user-environment isolation** | Each user manages their own `.env` or shell environment. No additional tool mechanism needed. Sufficient for v2.0 shared-server use case. | No enforcement: a misconfigured shared account can leak credentials via process environment. |
| Built-in multi-tenancy (per-user credential isolation) | Enterprise-grade isolation | Significant complexity; no peer tool has this. Out of scope for research tool. |

### Rejected Options

**Rejected (2026-02-19): OS keyring integration** — YAGNI. Shell environment is sufficient for the target user. Keyring adds a platform-specific dependency (GNOME Keyring, macOS Keychain, Windows Credential Store) with non-trivial platform differences. No peer tool uses it.

**Rejected (2026-02-19): Auth fields in experiment/study YAML** — High risk of accidental git exposure. Experiment and study YAML files are version-controlled by design. Auth credentials must live in `.env` or shell environment only. This is consistent with `extra = "forbid"` in `config-architecture.md` — any undeclared field in experiment/study YAML is already a hard error.

## Decision

Authentication is delegated to `huggingface_hub`'s built-in auth chain. The tool does not manage credentials itself.

Auth resolution order (handled by `huggingface_hub`):
1. `HF_TOKEN` environment variable (CI/CD, containers)
2. `~/.cache/huggingface/token` (written by `huggingface-cli login`)
3. Programmatic `huggingface_hub.login()` (library users)

Supporting infrastructure:
- Experiment/study YAML validation hard-errors if auth fields are present (consistent with `extra = "forbid"` in `config-architecture.md`)
- Pre-flight checks report missing credentials with exact fix command: `huggingface-cli login`
- No `.env` file, no `.env.example`, no `python-dotenv` dependency

Rationale: ML researchers already use `huggingface-cli login` — this is the established auth pattern for the target audience. `huggingface_hub` is already a transitive dependency (via `transformers`). Introducing a `.env` file pattern adds friction and unfamiliar tooling for zero benefit when the only v2.0 auth requirement is `HF_TOKEN`.

## Consequences

Positive:
- Zero new patterns for ML researchers — `huggingface-cli login` is already standard practice
- No credential files in project directory — no `.gitignore` discipline required
- `HF_TOKEN` env var works for CI/CD and Docker without changes
- Pre-flight errors are actionable with exact fix command

Negative / Trade-offs:
- If future auth requirements emerge beyond HF (e.g., API keys for cloud energy APIs), this approach may need extending
- OS keyring integration would be more secure but is deferred

Neutral / Follow-up decisions triggered:
- Fine-grained model access control deferred (no version target)
- Multi-user server isolation beyond OS user-environment deferred
- Auth usage audit logging deferred

## Auth Requirements

| Purpose | How supplied | Required when |
|---------|-------------|--------------|
| HuggingFace model access | `huggingface-cli login` or `HF_TOKEN` env var | Gated models (Llama, certain Mistral variants, etc.) |

## Pre-flight Output (missing auth)

```
  Models
  ✗ meta-llama/Llama-3-70B   gated model — not authenticated
                              → run: huggingface-cli login
                              → or set: export HF_TOKEN=hf_...
```

## Config Validation Rule

Experiment and study YAML configs raise `ConfigError` at parse time if any field name matches an auth-related pattern. This is complementary to `extra = "forbid"` in `ExperimentConfig` (see `config-architecture.md`) — auth-like field names are caught with a specific error message directing the user to `huggingface-cli login`, rather than the generic "extra fields not permitted" message.

## Deferred

- Fine-grained model access control — deferred (no version target)
- Multi-user server isolation beyond OS user-environment — deferred
- Auth usage audit logging — deferred

## Related

- [`cli-ux.md`](cli-ux.md) — maximally-explicit UX principle; pre-flight check pattern
- [`architecture.md`](architecture.md) — pre-flight check system (implicit, not `llem check`)
- [`config-architecture.md`](config-architecture.md) — `extra = "forbid"` on ExperimentConfig; auth fields excluded
- [`../designs/experiment-config.md`](../designs/experiment-config.md) — `ExperimentConfig` fields; no auth fields permitted
