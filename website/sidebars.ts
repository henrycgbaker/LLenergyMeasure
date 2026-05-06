import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'installation',
    {
      type: 'category',
      label: 'Getting started',
      items: [
        'getting-started',
        'getting-started-policy',
        'docker-setup',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['cli-reference', 'study-config'],
    },
    'troubleshooting',
    {
      type: 'category',
      label: 'Reference (auto-generated)',
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'Per-engine schema (discovered)',
          items: [
            'generated/schema-transformers',
            'generated/schema-vllm',
            'generated/schema-tensorrt',
          ],
        },
        {
          type: 'category',
          label: 'Per-engine curated parameters',
          items: [
            'generated/curation-transformers',
            'generated/curation-vllm',
            'generated/curation-tensorrt',
          ],
        },
        {
          type: 'category',
          label: 'Per-engine invariants (mined)',
          // generated/invariants-vllm pending #445 (vllm vendor gate)
          items: [
            'generated/invariants-transformers',
            'generated/invariants-tensorrt',
          ],
        },
        'generated/invalid-combos',
      ],
    },
  ],
};

export default sidebars;
