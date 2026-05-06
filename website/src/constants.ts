// Shared site constants. Imported by docusaurus.config.ts (build-time)
// and by React pages (src/pages/*) so cross-page CTAs can't drift from
// each other when sidebar entry points change.

export const SITE_TITLE = 'LLenergyMeasure';

export const ROUTES = {
  userGuide: '/docs/installation',
  methodology: '/methodology/methodology',
  api: '/api/llenergymeasure',
  architecture: '/architecture/architecture-overview',
} as const;

export const GITHUB_REPO = 'https://github.com/henrycgbaker/llenergymeasure';
export const EDIT_URL = `${GITHUB_REPO}/tree/main/website/`;
