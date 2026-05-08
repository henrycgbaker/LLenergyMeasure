import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebarsReference: SidebarsConfig = {
  referenceSidebar: [
    'cli',
    'study-config',
    'dataset-format',
    'results-schema',
    'invariants-corpus-format',
    'schema-discovered-format',
    {
      type: 'category',
      label: 'Library API (pre-1.0, unstable)',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Entry points',
          collapsed: false,
          items: [
            'library/run_experiment',
            'library/run_study',
            'library/ExperimentConfig',
            'library/StudyConfig',
            'library/ExperimentResult',
          ],
        },
        'api/llenergymeasure',
      ],
    },
    {
      type: 'category',
      label: 'Engines',
      collapsed: false,
      items: [
        'engines/configuration',
        {
          type: 'category',
          label: 'Schema (discovered)',
          items: [
            'engines/schema-transformers',
            'engines/schema-vllm',
            'engines/schema-tensorrt',
          ],
        },
        {
          type: 'category',
          label: 'Curated parameters',
          items: [
            'engines/curation-transformers',
            'engines/curation-vllm',
            'engines/curation-tensorrt',
          ],
        },
        {
          type: 'category',
          label: 'Invariants (mined)',
          items: [
            'engines/invariants-transformers',
            'engines/invariants-vllm',
            'engines/invariants-tensorrt',
          ],
        },
        'engines/invalid-combos',
      ],
    },
  ],
};

export default sidebarsReference;
