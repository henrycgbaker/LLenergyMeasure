import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebarsArchitecture: SidebarsConfig = {
  architectureSidebar: [
    'architecture-overview',
    'pipeline-architecture',
    'miner-pipeline',
    'extending-miners',
    'engines',
    'development',
    'schema-refresh',
    'validation-rule-corpus',
    // TODO: append 'Python API reference (auto-generated)' category once
    // pdoc-driven API docs land. See sequencing in /docs/architecture/.
  ],
};

export default sidebarsArchitecture;
