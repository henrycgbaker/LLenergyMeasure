import React from 'react';
import Layout from '@theme/Layout';
import ChangelogContent from '../../../CHANGELOG.md';

export default function ChangelogPage(): React.JSX.Element {
  return (
    <Layout title="Changelog" description="All notable changes to LLenergyMeasure">
      <main className="container margin-vert--lg">
        <ChangelogContent />
      </main>
    </Layout>
  );
}
