# Pip-dispatch dependency isolation - 2026-07-18

**Status:** RATIFIED 2026-07-18 (maintainer chose "fix properly and cleanly, release held";
approach A selected over B/C after code-map review). Supersedes nothing; extends the
container-dispatch design shipped in #830/#835/#837.
**Trigger:** v0.13.0 pre-tag exit gate (ds01, 2026-07-18) FAILED 2/3 engines for a fresh
`pip install` + docker dispatch, deterministic on retry. vLLM passed end-to-end; transformers
and TensorRT-LLM crashed on import inside their containers.

## Root cause

`docker_runner.py::append_package_dispatch` bind-mounts `_resolve_package_parent_dir()`
(a fixed `parent.parent.parent` walk from `docker_runner.py`) at `/llem-src:ro`, and
`container_entrypoint.sh` puts `/llem-src` first on `PYTHONPATH`.

- Editable/checkout install: parent dir is `src/`, containing only `llenergymeasure/`. Safe.
- Wheel (pip) install: parent dir is the venv's entire `site-packages/`. Every host
  third-party package shadows the container's own copies, because `PYTHONPATH` entries
  always precede container site-packages on `sys.path`.

Observed failures:
1. transformers: host venv py3.12 `pydantic_core` C extension shadows the image's py3.11
   copy; llenergymeasure itself fails to import in-container.
2. tensorrt: `pyproject.toml` pins `huggingface-hub>=0.20` with no upper bound; a fresh
   install today resolves 1.24.0, which shadows the NGC image's 0.36.2 and is rejected by
   the image's own transformers version guard. Hits every fresh install post hf-hub 1.0.

The dep-priming probe (`container_entrypoint.sh`) checks `importlib.metadata.distribution`
presence only, and the mounted host dist-info satisfies it, so priming never repairs either
case. The whole-parent mount was never a documented design choice - it is an artifact of the
3-hop resolver. No test asserted the mount excludes non-llem packages (the coverage gap).

## Decision: narrow the mount (approach A)

Mount ONLY the package directory itself, at a nested target:

- Resolve the package dir from the package's own `__file__` (2 hops: `infra/` -> package).
- Mount `-v {pkg_dir}:/llem-src/llenergymeasure:ro`.
- `PYTHONPATH` construction unchanged (`/llem-src` first): `/llem-src` now exposes only
  `llenergymeasure`, so nothing can shadow container-native deps.
- One uniform code path: identical for editable and wheel installs; no editable-detection.
- No dist-info mounting: the editable flow has always run without llem dist-info
  in-container, proving container-side code does not need it.
- Priming mechanism unchanged: genuinely-missing deps still install into the
  `/llem-runtime-deps/py{minor}` cache (correct ABI, resolved by the container's pip).
  With the mount narrowed, the probe's metadata check is sound again (no foreign
  dist-info can satisfy it).

Invariant to record in code (docstring) and enforce in tests: `/llem-src` must expose the
`llenergymeasure` package and nothing else - never host site-packages siblings.

## Rejected alternatives

- **B (reorder PYTHONPATH):** not viable pure - `PYTHONPATH` entries precede container
  site-packages regardless of their order among themselves, so host copies still shadow
  native ones. Viable only bundled with probe repair plus priming of shadowed deps, which
  primes packages the images already bundle (drift risk against pinned engine stacks) and
  keeps all host packages readable in-container. Strictly more moving parts than A.
- **C (pip install the wheel in-container):** the SOTA pattern for general remote execution
  (Ray runtime_env, SageMaker script mode) but wrong here: an in-container resolver run is
  a mutation vector on the measured environment, whose byte-identity to the recorded image
  digest is the product's provenance anchor. Also adds per-dispatch network/latency and
  cuts against the stated bind-mount goal (src/ edits without rebuild/reinstall).

## Riders (ratified same session)

1. Probe hardening: upgrade the entrypoint dep probe from dist-info presence to a real
   import check. Belt-and-braces once A lands; separate PR (same files as the mount fix).
2. `llem run -o`: honored only in the resume branch of `api/_impl.py::run_study` today,
   silently ignored for fresh runs despite help text. Fix: CLI `-o` overrides
   `output.results_dir` for fresh runs too; resume behaviour preserved.
3. Stale-local-image warning: `get_default_image()` silently prefers a bare
   `llenergymeasure:{engine}` local tag over the version-pinned default; months-stale dev
   tags hijack resolution. Keep the precedence (intentional for dev iteration) but log a
   warning naming the tag, the bypassed default, and the remedy.

## Explicitly not doing

- `huggingface-hub` upper bound: unnecessary once isolation lands (the container never
  sees the host copy); host-side llem works with hf-hub 1.x. Do not add pins we do not need.
- Scoping the priming requirements list to container-side imports only (typer/rich/pyyaml/
  python-dotenv/filelock are primed but never imported in-container): a safe
  over-approximation today; a hand-maintained second list is a drift surface. Revisit in F6
  if priming cost ever matters.
- Baking llem into upstream engine images: contradicts the upstream-cache-first ruling.

## Verification bar

- Unit: regression test constructing a site-packages-like parent (llenergymeasure + a foreign
  package + foreign dist-info) asserting the mount exposes only the package dir; existing
  TestMountPivot suite updated; PYTHONPATH still not set at the docker level.
- Live: wheel-install repro on ds01 (fresh HOME, scratch venv, wheel install, docker dispatch
  with ghcr transformers image) must pass where it deterministically failed pre-fix; then a
  full 3-engine exit-gate re-run before any tag.

## Related findings logged, out of this doc's scope

- CLI default `n_cycles=3` (intentional, documented; no action).
- Schema handshake UNVERIFIED for current images (missing fingerprint label at image build
  time; runtime fallback probe confirms engine versions; image-publish pipeline concern).
- Version/renumbering strategy under separate consideration (blast-radius audit running).
