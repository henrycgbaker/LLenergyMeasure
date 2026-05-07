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
