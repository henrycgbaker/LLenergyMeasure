// Shared site constants. Imported by docusaurus.config.ts (build-time)
// and by React pages (src/pages/*) so cross-page CTAs can't drift from
// each other when sidebar entry points change.

export const SITE_TITLE = 'LLenergyMeasure';

// Route entry points per pillar. Each value is the URL of the FIRST
// page in that sidebar - clicking the corresponding navbar item or
// landing-page card lands there.
export const ROUTES = {
  getStarted: '/get-started/install',
  tutorials: '/tutorials/first-measurement',
  howTo: '/how-to/install',
  reference: '/reference/cli',
  explanation: '/explanation/overview',
  contributing: '/contributing/development',
  // Deep route used by the landing-page hero secondary CTA.
  why: '/explanation/why',
} as const;

export const GITHUB_REPO = 'https://github.com/henrycgbaker/llenergymeasure';
export const EDIT_URL = `${GITHUB_REPO}/tree/main/website/`;
