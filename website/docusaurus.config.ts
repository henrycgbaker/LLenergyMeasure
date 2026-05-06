import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Runs in Node.js — no client-side code (no browser APIs, no JSX).
//
// Three-plugin IA: docs/user/ (technical user guide + auto-generated
// reference), docs/methodology/ (measurement methodology + interpretation),
// docs/architecture/ (pipeline internals + contributor onboarding).
//
// Content lives in docs/ at the repo root; this Docusaurus instance lives
// in website/ and points its plugins at ../docs/<subdir>. .product/ is
// gitignored project-decision storage and is deliberately not referenced.

const config: Config = {
  title: 'llenergymeasure',
  tagline: 'Energy & efficiency measurement for LLM inference',

  future: {
    v4: true,
  },

  url: 'https://henrycgbaker.github.io',
  baseUrl: '/llenergymeasure/',

  organizationName: 'henrycgbaker',
  projectName: 'llenergymeasure',

  onBrokenLinks: 'throw',

  markdown: {
    // Treat .md files as plain CommonMark (not MDX). Docs in this repo use
    // standard markdown features only — HTML comments, GFM tables, code
    // blocks — and don't need JSX. CommonMark mode is more permissive.
    format: 'md',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs/user',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/henrycgbaker/llenergymeasure/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'methodology',
        path: '../docs/methodology',
        routeBasePath: 'methodology',
        sidebarPath: './sidebarsMethodology.ts',
        editUrl:
          'https://github.com/henrycgbaker/llenergymeasure/tree/main/website/',
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'architecture',
        path: '../docs/architecture',
        routeBasePath: 'architecture',
        sidebarPath: './sidebarsArchitecture.ts',
        editUrl:
          'https://github.com/henrycgbaker/llenergymeasure/tree/main/website/',
      },
    ],
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        docsDir: ['../docs/user', '../docs/methodology', '../docs/architecture'],
        docsRouteBasePath: ['docs', 'methodology', 'architecture'],
        indexBlog: false,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'llenergymeasure',
      items: [
        {to: '/docs/installation', label: 'User Guide', position: 'left'},
        {to: '/methodology/methodology', label: 'Methodology', position: 'left'},
        {
          to: '/architecture/architecture-overview',
          label: 'Architecture',
          position: 'left',
        },
        {
          href: 'https://github.com/henrycgbaker/llenergymeasure',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Use',
          items: [
            {label: 'Installation', to: '/docs/installation'},
            {label: 'CLI reference', to: '/docs/cli-reference'},
            {label: 'Troubleshooting', to: '/docs/troubleshooting'},
          ],
        },
        {
          title: 'Learn',
          items: [
            {label: 'Methodology', to: '/methodology/methodology'},
            {label: 'Energy measurement', to: '/methodology/energy-measurement'},
            {
              label: 'Architecture overview',
              to: '/architecture/architecture-overview',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/henrycgbaker/llenergymeasure',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} llenergymeasure contributors`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
