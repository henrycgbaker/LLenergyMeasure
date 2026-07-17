---
title: Release process
description: How to cut a release of LLenergyMeasure, from version bump to PyPI publication.
---

# Release process

LLenergyMeasure uses semantic versioning in the `0.x` range while the project
is pre-1.0. This page documents the versioning scheme, the two files that must
stay in sync, and the step-by-step release procedure.

---

## Versioning scheme

- Pre-1.0: minor version bumps (`0.1.0`, `0.2.0`, ..., `0.N.0`) mark significant
  capability milestones. Patch versions (`0.N.1`) are reserved for hotfixes.
- `1.0.0` is reserved for the production-ready release. It is not tied to a
  calendar date and will be cut when the project meets the production-readiness bar.
- Version management is manual - no automated bump tooling is used.

Feature work lands on `main` at the current version. Version numbers do not
change on feature PRs - only on release commits.

---

## Version sources

Two files hold the version and must stay in sync:

| File | Field | Example |
|------|-------|---------|
| `pyproject.toml` | `[project] version = "..."` | `version = "0.11.0"` |
| `src/llenergymeasure/_version.py` | `__version__ = "..."` | `__version__ = "0.11.0"` |

A mismatch between these two files is a bug. The build system reads
`pyproject.toml`; the runtime API (`llem --version`, `llenergymeasure.__version__`)
reads `__version__`, which is defined in `src/llenergymeasure/_version.py` and
re-exported from `__init__.py`.

---

## Release steps

1. **Bump the version in `pyproject.toml`**

   Update the `version` field under `[project]`:

   ```toml
   [project]
   name = "llenergymeasure"
   version = "0.12.0"
   ```

2. **Bump the version in `_version.py`**

   Update `__version__` in `src/llenergymeasure/_version.py`:

   ```python
   __version__ = "0.12.0"
   ```

   Both files must show the same version string.

3. **Update `CHANGELOG.md`**

   Add an entry for the new version at the top of the changelog. Note key
   changes shipped since the previous release. The format is free-form but
   should be scannable.

4. **Commit**

   Commit the three changed files with the message:

   ```
   chore: release 0.12.0
   ```

   This commit goes directly to `main` via the normal PR flow.

5. **Create and push the git tag**

   ```bash
   git tag v0.12.0
   git push origin v0.12.0
   ```

   The tag must use the `v` prefix and match the version string exactly.
   Pushing the tag triggers the CI publication workflow.

6. **CI publishes to PyPI**

   The `release.yml` workflow (`.github/workflows/release.yml`) triggers on
   `push: tags: ["v*"]`. It runs lint, type-check, tests, and version
   validation, then the `release` job builds the sdist and wheel with
   `uv build` and the `publish-pypi` job uploads them to PyPI. Publishing uses
   [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/): the
   runner mints a short-lived identity token that PyPI verifies against a
   configured trusted publisher, so no API token or secret is stored in the
   repository. No manual upload is required.

   Trusted publishing requires a one-time setup on pypi.org: the project's
   trusted publisher must name this repository, the `release.yml` workflow, and
   the `pypi` environment. Until that publisher is configured, the
   `publish-pypi` job fails at the publish step (the build and GitHub Release
   still succeed).

---

## Verifying a release

After CI completes:

```bash
pip install --upgrade llenergymeasure
llem --version   # should show the new version
```

Check the GitHub Releases page - the tag should appear there automatically
once the workflow completes.

---

## Pre-1.0 vs post-1.0 policy

The current `0.x` scheme treats every release as potentially breaking. The
stability contract documented in `src/llenergymeasure/__init__.py` applies:
names in `__all__` follow semver, internal names may change without notice.

When the project reaches `1.0.0`, the guarantee tightens: breaking changes to
public API require a major version bump. Until then, `0.x` minor bumps may
include breaking changes with a one-release deprecation window where practical.

---

## See also

- [Contributing: development](/contributing/development) - local dev setup
- [Reference: CLI](/reference/cli) - verifying installed version with `llem --version`
