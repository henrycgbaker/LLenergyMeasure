import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebarsExplanation: SidebarsConfig = {
  explanationSidebar: [
    'overview',
    {
      type: 'category',
      label: 'Methodology',
      collapsed: false,
      items: [
        'methodology/methodology',
        'methodology/what-we-measure',
        'methodology/energy-measurement',
        'methodology/measurement-warnings',
        'methodology/comparison-context',
        'methodology/dataset-context',
        {
          type: 'category',
          label: 'Empirical grounding',
          collapsed: true,
          link: {type: 'doc', id: 'methodology/empirical-grounding/index'},
          items: [
            'methodology/empirical-grounding/min-window-duration',
            'methodology/empirical-grounding/loaded-thermal-equilibrium',
            'methodology/empirical-grounding/probe-window-rank',
            'methodology/empirical-grounding/open-loop-arrivals',
          ],
        },
        'methodology/glossary',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      collapsed: false,
      items: [
        'architecture/architecture-overview',
        'architecture/pipeline-architecture',
        'architecture/server-measurement',
        'architecture/ci-architecture',
        {
          type: 'category',
          label: 'Plugin model and extensibility',
          collapsed: true,
          items: [
            'architecture/harness-plugin',
            'architecture/engine-extensibility',
          ],
        },
        {
          type: 'category',
          label: 'Config knowledge',
          collapsed: true,
          items: [
            'architecture/parameter-discovery',
            'architecture/parameter-curation',
          ],
        },
      ],
    },
    'why',
    'ecosystem',
  ],
};

export default sidebarsExplanation;
