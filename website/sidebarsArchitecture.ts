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
    // PR 3 (pdoc) appends a 'Python API reference' category here.
  ],
};

export default sidebarsArchitecture;
