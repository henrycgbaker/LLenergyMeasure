.PHONY: help setup docker-setup dev install clean pre-pr
.PHONY: format lint lint-fix typecheck check
.PHONY: test test-unit test-integration test-all
.PHONY: docs-all docs-check docs-generate docs-serve docs-build docs-clean
.PHONY: discover-schema discover-schemas-all scaffold-snapshot promote-schemas
.PHONY: check-citations probe-candidates analyst-cold-read absorb rules-coverage check-corpus-literals
.PHONY: check-plugin-kwargs
.PHONY: package-check
.PHONY: docker-smoke
.PHONY: ci ci-all ci-docker
.PHONY: gpu-ci
.PHONY: docker-builder-setup docker-builder-rm
.PHONY: docker-build docker-seed-transformers
.PHONY: docker-pull docker-images docker-check
.PHONY: llem docker-shell docker-build-dev docker-dev
.PHONY: llem-clean-cache

# per-dev extension hook for personal targets; not git tracked.
-include Makefile.local

# Default target prints help when `make` is invoked with no arguments.
.DEFAULT_GOAL := help

# PUID/PGID for correct file ownership on bind mounts (LinuxServer.io pattern)
export PUID := $(shell id -u)
export PGID := $(shell id -g)

# =============================================================================
# Help
#   Targets with a `## description` suffix are listed by `make help`.
#   Anything without is treated as internal (still invokable, just hidden).
# =============================================================================

help: ## Show available make targets and their descriptions
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z][a-zA-Z0-9_-]*:.*?## / {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

# =============================================================================
# Quick Start
#   Local:  make setup         (uv sync --dev + pre-commit)
#   Docker: make docker-setup  (above + docker compose build)
#   Dev:    make dev            (uv sync --dev + pre-commit)
# =============================================================================

setup: ## Install local dev environment (uv sync --dev + pre-commit hooks)
	uv sync --dev
	pre-commit install
	@echo "Dev environment ready. Run: llem --help"

docker-setup: setup ## setup + docker compose build (transformers engine image)
	docker compose build
	@echo "Docker environment ready. Run: llem run <config.yaml>"
	@echo "Tip: run 'make docker-builder-setup' for a BuildKit builder with larger cache limits"

# =============================================================================
# Local Development
# =============================================================================

format: ## Auto-format src/, tests/, and the knowledge-production machinery with ruff
	uv run ruff format src/ tests/ scripts/engine_producers/

lint: ## Run ruff check, ruff format --check, and import-linter
	uv run ruff check src/ tests/ scripts/engine_producers/
	uv run ruff format --check src/ tests/ scripts/engine_producers/
	uv run lint-imports

lint-fix: ## Auto-fix lint issues (ruff check --fix + format)
	uv run ruff check src/ tests/ scripts/engine_producers/ --fix
	uv run ruff format src/ tests/ scripts/engine_producers/

typecheck: ## Run mypy on src/, tests/, and the knowledge-production machinery
	uv run mypy src/ tests/ scripts/engine_producers/

check: lint typecheck ## lint + typecheck (no tests)

test: ## Host tests (excludes gpu, docker, and slow markers)
	uv run pytest tests/ -m "not gpu and not docker and not slow" -n auto -x -q --tb=short

test-unit: ## Unit tests with verbose output
	uv run pytest tests/unit/ -n auto -v

test-integration: ## Integration tests with verbose output
	uv run pytest tests/integration/ -n auto -v

test-all: ## All tests (excludes tests/runtime/)
	uv run pytest tests/ -n auto -v --ignore=tests/runtime/

install: ## uv sync (runtime deps only, no dev tooling)
	uv sync

dev: ## uv sync --dev + pre-commit install (same as setup)
	uv sync --dev
	uv run pre-commit install

clean: ## Remove local caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Generated documentation
#   docs-all regenerates every SSOT-derived artefact under docs/. Generators
#   must be deterministic (no wall-clock timestamps); git history is the
#   timestamp.
#   docs-check runs docs-all and fails if any committed generated doc drifts.
#   docs-generate is the smaller, website-only subset depended on by serve/build.
# =============================================================================

docs-all: ## Regenerate every SSOT-derived doc (CLI, config, schema, curation, invalid combos, API)
	uv run python scripts/generate_invalid_combos_doc.py
	uv run python scripts/generate_config_docs.py --output docs/reference/study-config.md
	uv run python scripts/generate_cli_reference.py --output docs/reference/cli.md
	@for engine in transformers vllm tensorrt; do \
		uv run python scripts/generate_schema_doc.py --engine $$engine --out docs/reference/engines/schema-$$engine.md; \
	done
	uv run python scripts/generate_curation_doc.py
	uv run python scripts/generate_api_docs.py
	@echo "All generated docs refreshed"

docs-check: docs-all ## Regenerate SSOT docs and fail if the committed copies drift
	@git diff --exit-code -- docs/reference/study-config.md docs/reference/cli.md \
		'docs/reference/engines/schema-*.md' 'docs/reference/engines/curation-*.md' \
		docs/reference/engines/invalid-combos.md \
		|| (echo "Generated docs are stale. Commit the regenerated files above." && exit 1)
	@echo "Generated docs are up to date"

# Rediscover a vendored engine schema by running introspection inside the
# engine's Docker image. Writes to src/llenergymeasure/engines/<engine>/schema.discovered.json
# and prints the git diff. Committing (or not) is the review gate.
discover-schema: ## Rediscover one engine schema (ENGINE=vllm|tensorrt|transformers)
	@test -n "$(ENGINE)" || (echo "Usage: make discover-schema ENGINE={vllm|tensorrt|transformers}" && exit 1)
	./scripts/refresh_discovered_schemas.sh $(ENGINE)

discover-schemas-all: ## Rediscover all three engine schemas in sequence
	./scripts/refresh_discovered_schemas.sh vllm
	./scripts/refresh_discovered_schemas.sh tensorrt
	./scripts/refresh_discovered_schemas.sh transformers

# Scaffold the per-version snapshot outputs dir a bump needs before codegen can
# go green (engine_versions/<engine>/v<safe>/outputs/). Derives v<safe> from the
# current.yaml pin via engine_versions/_outputs.py (the one place that mangling
# lives). Creating the dir is all this does - the maintainer still drops the
# mined schema.discovered.json + curated.yaml into it.
scaffold-snapshot: ## Scaffold the snapshot outputs dir from the pin (ENGINE=vllm|tensorrt|transformers)
	@test -n "$(ENGINE)" || (echo "Usage: make scaffold-snapshot ENGINE={vllm|tensorrt|transformers}" && exit 1)
	@ver=$$(python3 -c "import yaml; print(yaml.safe_load(open('engine_versions/$(ENGINE)/current.yaml'))['library']['current_version'])"); \
	outdir=$$(python3 -c "from engine_versions import _outputs; print(_outputs.outputs_dir('$(ENGINE)', '$$ver'))"); \
	mkdir -p "$$outdir"; \
	echo "Scaffolded $$outdir (drop schema.discovered.json + curated.yaml here, then run config-codegen)"

# Promote the versioned discovered-schema snapshots into the packaged src copies:
# a byte-copy of engine_versions/<engine>/v<safe>/outputs/schema.discovered.json
# -> src/llenergymeasure/engines/<engine>/schema.discovered.json at each engine's
# pin. This is the ONLY writer of the src copies (no transformation); the refresh
# script runs it automatically after discovery. Pass ENGINE=<e> for one engine.
# The CI surface-equality guard is the drift tripwire for this promotion.
promote-schemas: ## Byte-copy versioned schema snapshots into the src copies (ENGINE=<e> optional)
	python3 scripts/promote_schemas.py $(if $(ENGINE),--engine $(ENGINE),)

# Proposer S2: LLM analyst cold read of the pinned engine source into candidate
# rules for the verification ladder. Needs a local Ollama daemon (owns the GPU)
# and SRC = the engine package directory at the pinned version. Writes to the
# version-scoped candidate pool, never to the shipped corpora.
analyst-cold-read: ## Cold-read one engine's source into candidates (ENGINE=vllm SRC=path/to/source [ARGS=...])
	@test -n "$(ENGINE)" && test -n "$(SRC)" || (echo "Usage: make analyst-cold-read ENGINE=vllm SRC=engine-src/ [ARGS='--samples 1']" && exit 1)
	uv run python scripts/analyst_cold_read.py --engine $(ENGINE) --source-root "$(SRC)" $(ARGS)

# Verification-ladder tier 1: confirm each proposed candidate rule's citation
# resolves against a pinned engine source tree. Used by the absorb workflow and
# CI. Obtaining SRC (the engine source tree) is the caller's concern.
check-citations: ## Verify candidate citations (CANDIDATES=file.yaml SRC=path/to/source)
	@test -n "$(CANDIDATES)" && test -n "$(SRC)" || (echo "Usage: make check-citations CANDIDATES=candidates.yaml SRC=engine-src/" && exit 1)
	uv run python scripts/check_citations.py "$(CANDIDATES)" --source-root "$(SRC)"

# Verification-ladder tiers 2-3: run construction/identity probes against the
# real engine inside its Docker image. ARGS forwards to the kernel (e.g. --out).
probe-candidates: ## Probe candidate rules in-engine (ENGINE=vllm|tensorrt|transformers CANDIDATES=file.yaml [ARGS=...])
	@test -n "$(ENGINE)" && test -n "$(CANDIDATES)" || (echo "Usage: make probe-candidates ENGINE={vllm|tensorrt|transformers} CANDIDATES=candidates.yaml [ARGS='--out /tmp/verdicts.yaml']" && exit 1)
	./scripts/probe_candidates.sh $(ENGINE) --candidates "$(CANDIDATES)" $(ARGS)

# The absorb conductor: one command per engine-version bump that drives the whole
# knowledge-refresh loop (cold read -> pool union -> recall interrogation ->
# verification ladder -> promotion into the shipped rules corpus -> review delta).
# SRC = the engine package at the pinned version. --dry-run reports the delta and
# writes no shipped corpus. ARGS forwards flags (--skip-cold-read, --clean-room).
absorb: ## Absorb one engine-version bump into the shipped rules (ENGINE=vllm SRC=path/to/source [ARGS='--dry-run'])
	@test -n "$(ENGINE)" && test -n "$(SRC)" || (echo "Usage: make absorb ENGINE=vllm SRC=engine-src/ [ARGS='--dry-run --skip-cold-read']" && exit 1)
	uv run python scripts/absorb.py --engine $(ENGINE) --source-root "$(SRC)" $(ARGS)

# The completeness counterpart to absorb: after a bump, report validator sites in
# the engine source that no shipped rule covers (advisory - exit 0 by default).
# SRC = the same engine package at the pinned version absorb takes. ARGS forwards
# flags (--fail-on-uncovered to exit non-zero when any gap remains).
rules-coverage: ## Report uncovered engine validator sites (ENGINE=vllm SRC=path/to/source [ARGS='--fail-on-uncovered'])
	@test -n "$(ENGINE)" && test -n "$(SRC)" || (echo "Usage: make rules-coverage ENGINE=vllm SRC=engine-src/ [ARGS='--fail-on-uncovered']" && exit 1)
	uv run python scripts/rules_coverage.py --engine $(ENGINE) --source-root "$(SRC)" $(ARGS)

# Standing consistency check between the two knowledge products: every string
# literal the shipped rules corpus asserts must be expressible in the discovered
# schema type (directly, or via a verified runtime_literals entry). A finding
# means a corpus rule references a value the generated typed config would reject.
check-corpus-literals: ## Report corpus rule literals inexpressible in discovered schema types
	uv run python -m scripts.engine_producers._runtime_literals --census

# Standing lint between the glue code and the mined knowledge: every
# constructor-kwarg name an engine plugin's translation layer hand-types as a
# string literal must exist in that engine's discovered schema at the current
# pin (or carry an allowlist rationale in the script). Catches upstream kwarg
# renames the schema already knows about but the hand-written plugin missed.
check-plugin-kwargs: ## Lint hand-typed plugin constructor kwargs against the mined schema
	uv run python scripts/check_plugin_kwargs.py

# Build wheel + validate package install + check version consistency
package-check: ## Build wheel, validate install, and check pyproject/_version sync
	uv build --wheel
	@python3 -m venv /tmp/pkg-check-local 2>/dev/null || true
	@/tmp/pkg-check-local/bin/pip install dist/*.whl --quiet --force-reinstall
	@/tmp/pkg-check-local/bin/python -c "from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult; print('Package install OK')"
	@PYPROJECT_VER=$$(python3 -c "import tomllib; f=open('pyproject.toml','rb'); print(tomllib.load(f)['project']['version'])"); \
	 VERSION_VER=$$(python3 -c "import re; s=open('src/llenergymeasure/_version.py').read(); print(re.search(r'__version__[^=]*=\s*\"([^\"]+)\"', s).group(1))"); \
	 echo "pyproject.toml: $$PYPROJECT_VER"; \
	 echo "_version.py:    $$VERSION_VER"; \
	 [ "$$PYPROJECT_VER" = "$$VERSION_VER" ] || { echo "ERROR: Version mismatch"; exit 1; }
	@echo "Package validation OK"

# =============================================================================
# Docker smoke + CI helpers
#   docker-smoke runs `llem --version` and `llem doctor` against each
#   first-party compose-managed engine image, in an isolated compose project
#   that is torn down on exit. Assumes the image is already built (use
#   `make docker-build` first). To add new engines as they get first-party
#   wrappers, append further `docker compose ... run --rm <engine> ...` lines.
# =============================================================================

docker-smoke: ## Smoke-test compose-managed engine image(s) with guaranteed teardown
	@PROJECT=smoke-llem-$$$$; \
	trap "docker compose -p $$PROJECT down -v >/dev/null 2>&1" EXIT; \
	docker compose -p $$PROJECT run --rm transformers llem --version && \
	docker compose -p $$PROJECT run --rm transformers llem doctor

# CI targets - run the same checks as GitHub Actions ci.yml
ci: lint typecheck test package-check docs-check ## Local equivalent of GitHub Actions ci.yml

pre-pr: ci ## Run the local CI suite before opening a PR (alias of ci)

ci-all: ci docker-smoke ## ci + docker smoke

# Run CI in a clean container matching GitHub Actions (ubuntu + Python 3.12 + uv)
# Catches "works on my machine" issues before pushing
CI_IMAGE := llenergymeasure-ci-env:local
define CI_DOCKERFILE
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY . .
ENV UV_FROZEN=true UV_NO_PROGRESS=1
RUN uv sync --dev --extra codecarbon --extra zeus
endef
export CI_DOCKERFILE
ci-docker: ## Run ci inside a clean Ubuntu container (matches GitHub Actions environment)
	echo "$$CI_DOCKERFILE" | docker build -t $(CI_IMAGE) -f - .
	docker run --rm $(CI_IMAGE) sh -c '\
		uv run ruff check src/ tests/ && \
		uv run ruff format --check src/ tests/ && \
		uv run lint-imports && \
		uv run mypy src/ tests/ && \
		uv run pytest tests/ -m "not gpu and not docker" -x -q --tb=short && \
		echo "=== CI-docker: all checks passed ==="'
	@docker rmi $(CI_IMAGE) 2>/dev/null || true

# =============================================================================
# GPU CI - mirrors .github/workflows/gpu-ci.yml
#   Requires: Docker, NVIDIA GPUs, nvidia-container-toolkit. When additional
#   engines get first-party Dockerfiles, fan out the build + test steps below
#   (or extract a per-engine sub-recipe).
# =============================================================================

gpu-ci: ## GPU integration tests (mirrors gpu-ci.yml; transformers engine)
	docker build -f docker/Dockerfile.transformers -t llenergymeasure-ci:transformers .
	docker rm llem-ci-setup 2>/dev/null || true
	docker run --name llem-ci-setup llenergymeasure-ci:transformers pip install --no-cache-dir pytest pytest-xdist
	docker commit llem-ci-setup llenergymeasure-ci:transformers
	docker rm llem-ci-setup
	mkdir -p results/
	docker run --rm --gpus all \
		-v "$(CURDIR)/tests":/app/tests:ro \
		-v "$(CURDIR)/results":/app/results \
		llenergymeasure-ci:transformers \
		python3 -m pytest tests/ -v --tb=short -o "addopts="
	docker run --rm --gpus all \
		-v "$(CURDIR)/tests":/app/tests:ro \
		-v "$(CURDIR)/results":/app/results \
		llenergymeasure-ci:transformers \
		bash tests/integration/sigint_verify.sh
	docker rmi llenergymeasure-ci:transformers 2>/dev/null || true

# =============================================================================
# Docker - first-party image builds and registry pulls
#   Only the transformers engine has a first-party image. vLLM and TensorRT-LLM
#   run inside upstream images (vllm/vllm-openai, nvcr.io/nvidia/tensorrt-llm/release)
#   with the llenergymeasure source bind-mounted at runtime.
# =============================================================================

# Builder name read by docker compose / buildx via BUILDX_BUILDER for
# registry-cached builds. The docker-container driver is required to import
# cache_from registry refs (the default `docker` driver cannot).
BUILDER_NAME := llem-builder

docker-builder-setup: ## Create the BuildKit builder with tuned cache limits (200 GiB)
	@if docker buildx inspect $(BUILDER_NAME) >/dev/null 2>&1; then \
		echo "Builder '$(BUILDER_NAME)' already exists"; \
	else \
		echo "Creating builder '$(BUILDER_NAME)' with 200 GiB cache limit..."; \
		docker buildx create \
			--name $(BUILDER_NAME) \
			--driver docker-container \
			--buildkitd-config docker/buildkitd.toml \
			--bootstrap; \
		echo "Builder '$(BUILDER_NAME)' created. Use with: BUILDX_BUILDER=$(BUILDER_NAME) docker compose build"; \
	fi

docker-builder-rm: ## Remove the BuildKit builder (e.g. to recreate with new config)
	docker buildx rm $(BUILDER_NAME) 2>/dev/null || true

# Build first-party engine images. When additional engines get first-party
# Dockerfiles, append further `scripts/docker_build_with_cache_report.sh <engine>`
# lines.
docker-build: ## Build first-party engine images (currently: transformers)
	@echo "First build pulls cache layers from ghcr.io; warm rebuilds < 5 min."
	scripts/docker_build_with_cache_report.sh transformers

# Seed the transformers image from a local machine with sufficient RAM
# (the FA3 Hopper compile, ~30 min, needs more memory than CI hosted
# runners have). Requires: docker login ghcr.io, llem-builder buildx
# builder. Uses Dockerfile default MAX_JOBS=32 - matches local layer
# cache so FA3 is not recompiled if already built locally.
#
# One push, producing the two refs the promotion path consumes:
#   - transformers-cache:transformers-<VER> - the runnable promotion source
#     (<VER> = library.current_version from
#     engine_versions/transformers/current.yaml). publish-engine-image.yml
#     tag-copies it to the canonical tags when a bump lands on main.
#   - transformers-cache:transformers-<VER>-buildcache - the mode=max
#     BuildKit cache manifest (includes the FA3 layer). This is THE cache;
#     warm re-seeds import it via --cache-from.
#
# Run this during a transformers bump session, before or alongside the bump
# PR - a missing seed fails the merge-time promotion run. The seed never
# writes the canonical transformers:latest or transformers:<version> tags,
# and never writes cache to them: those are tag-copies owned by
# publish-engine-image.yml (merge) and docker-publish.yml (release).
docker-seed-transformers: ## Seed transformers promotion source image + mode=max build cache
	@engver=$$(python3 -c "import yaml; print(yaml.safe_load(open('engine_versions/transformers/current.yaml'))['library']['current_version'])"); \
	cacheref=ghcr.io/henrycgbaker/llenergymeasure/transformers-cache; \
	echo "Seeding promotion source $$cacheref:transformers-$$engver"; \
	docker buildx build \
	  --builder $(BUILDER_NAME) \
	  -f docker/Dockerfile.transformers \
	  --target runtime \
	  --build-arg TRANSFORMERS_VERSION=$$engver \
	  --cache-from type=registry,ref=$$cacheref:transformers-$$engver-buildcache \
	  --cache-from type=registry,ref=$$cacheref:transformers-$$engver \
	  --cache-from type=registry,ref=ghcr.io/henrycgbaker/llenergymeasure/transformers:latest \
	  --cache-to   type=registry,ref=$$cacheref:transformers-$$engver-buildcache,mode=max \
	  --push \
	  --tag $$cacheref:transformers-$$engver \
	  .

docker-pull: ## Pull the versioned transformers image from GHCR
	@version=$$(python3 -c "from llenergymeasure._version import __version__; print(__version__)" 2>/dev/null || echo "latest"); \
	echo "Pulling ghcr.io/henrycgbaker/llenergymeasure/transformers:v$$version"; \
	docker pull "ghcr.io/henrycgbaker/llenergymeasure/transformers:v$$version"

docker-images: ## Show which images llem will use (local vs registry)
	@python3 -c "from llenergymeasure.infra.image_registry import show_image_resolution; show_image_resolution()"

docker-check: ## Validate the docker-compose config parses cleanly
	@docker compose config -q || (echo "Error: Invalid docker-compose config"; exit 1)
	@echo "Docker config OK"

# =============================================================================
# llem in Docker
#   `make llem CMD=...` runs the llem CLI inside the transformers container.
#   The current llem subcommand surface is: run, study, doctor, report-gaps.
#
# Examples:
#   make llem CMD="--help"
#   make llem CMD="doctor"
#   make llem CMD="run configs/example-study-full.yaml"
# =============================================================================

CMD ?= --help

llem: docker-check ## Run any llem command in the transformers container (use CMD="...")
	docker compose run --rm transformers llem $(CMD)

docker-shell: ## Interactive bash shell in the transformers container
	docker compose run --rm transformers /bin/bash

# =============================================================================
# Dev shells - transformers-dev profile (source bind-mounted, dev tooling)
# =============================================================================

docker-build-dev: ## Build the transformers-dev image (dev profile)
	docker compose --profile dev build transformers-dev

docker-dev: ## Interactive dev shell with source bind-mounted (transformers-dev)
	docker compose --profile dev run --rm transformers-dev

# =============================================================================
# Volume management
#   The only named volume in docker-compose.yml is llem-hf-cache. Experiment
#   state (.state/) and the TensorRT engine cache are bind mounts on the host,
#   so there are no named volumes to clear for them.
# =============================================================================

llem-clean-cache: ## Remove the HuggingFace model cache volume (forces re-download)
	@if docker volume inspect llem-hf-cache >/dev/null 2>&1; then \
		docker volume rm llem-hf-cache && echo "Cleared HuggingFace cache"; \
	else \
		echo "No HuggingFace cache volume to clear"; \
	fi

# =============================================================================
# Docs site (Docusaurus)
#   Source content lives in docs/; site infra lives in website/.
#   For a complete SSOT regen (config, CLI, schema, curation,
#   invalid combos, API) use `make docs-all`. docs-generate covers only the
#   API reference subset the website depends on at serve/build time.
# =============================================================================

docs-generate: ## Regenerate API docs only (subset of docs-all that website needs)
	uv run python scripts/generate_api_docs.py

docs-serve: docs-generate ## Serve docs site locally with auto-reload
	cd website && npm start

docs-build: docs-generate ## Build docs site for production
	cd website && npm run build

docs-clean: ## Remove docs site build artefacts
	rm -rf website/node_modules website/build website/.docusaurus website/.cache-loader \
	       docs/reference/api
