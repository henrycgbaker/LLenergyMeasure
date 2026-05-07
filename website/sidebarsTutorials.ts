import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebarsTutorials: SidebarsConfig = {
  tutorialsSidebar: [
    'first-measurement',
    // 'multi-backend-study' added in PR 3 (flagship tutorial)
    {
      type: 'category',
      label: 'Non-technical',
      collapsed: true,
      items: ['non-technical/what-llem-shows'],
    },
  ],
};

export default sidebarsTutorials;
