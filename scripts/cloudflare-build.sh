#!/usr/bin/env bash
# Cloudflare Pages build entrypoint.
#
# Invoked by CF Pages dashboard (Build command: `bash scripts/cloudflare-build.sh`).
# Mirrors the build sequence in .github/workflows/docs.yml so previews
# render the same site GH Pages publishes from main:
#
#   1. install llenergymeasure (base deps only) so generate_api_docs.py
#      can `import llenergymeasure` to render the API reference page
#   2. run the API docs generator (writes docs/api/llenergymeasure.md,
#      which is gitignored — must be regenerated on every build)
#   3. install Docusaurus deps + build the static site to website/build/
#
# Runtime versions are pinned in .nvmrc (Node) and .python-version
# (Python); the CF V2 build image auto-activates them via nvm/pyenv.
# CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID are not used here — direct
# GitHub App integration handles auth, this script only builds.

set -euo pipefail

# CF Pages serves at root (*.pages.dev), unlike GH Pages which serves
# under the /llenergymeasure/ subpath. Override docusaurus.config.ts's
# default baseUrl here rather than via dashboard env vars — the new
# Workers Builds UI doesn't expose a clean Production/Preview-scoped
# build-time env-var panel. Hardcoding here is unambiguous and only
# affects CF builds (GH Pages uses .github/workflows/docs.yml, which
# doesn't invoke this script and inherits the config default).
export DOCUSAURUS_BASE_URL=/

echo "::group::Python environment"
python --version
pip --version
echo "::endgroup::"

echo "::group::Install llenergymeasure (base deps)"
# -e . is fine in CF's ephemeral build env; gives the API doc generator
# access to llenergymeasure.* without copying into site-packages.
pip install -e .
echo "::endgroup::"

echo "::group::Generate Python API reference"
python scripts/generate_api_docs.py
echo "::endgroup::"

echo "::group::Node environment"
node --version
npm --version
echo "::endgroup::"

echo "::group::Install Docusaurus deps"
cd website
npm ci
echo "::endgroup::"

echo "::group::Build Docusaurus site"
# onBrokenLinks: 'throw' in docusaurus.config.ts fails the build on
# broken internal links — same gate as the GH Pages production build.
npm run build
echo "::endgroup::"
