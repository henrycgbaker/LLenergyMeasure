import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import {ROUTES, SITE_TITLE} from '../constants';
import styles from './index.module.css';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <main className={styles.hero}>
        <div className="container">
          <div className={styles.inner}>
            <Heading as="h1" className={styles.title}>
              {SITE_TITLE}
            </Heading>

            <p className={styles.lead}>
              Energy &amp; efficiency measurement for LLM inference.
            </p>

            <p className={styles.description}>
              A CLI-first benchmarking framework for evaluation of LLM
              inference efficiency across heterogeneous runtimes. Supports
              energy, throughput, and FLOPs measurement, with
              engine-introspected parameter spaces, intelligent experiment
              deduplication and sampling, and structured sweep design.
            </p>

            <div className={styles.cta}>
              <Link
                className="button button--primary button--lg"
                to={ROUTES.getStarted}>
                Get Started →
              </Link>
            </div>

            <div className={styles.actions}>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.tutorials}>
                Tutorials
              </Link>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.howTo}>
                How-to
              </Link>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.reference}>
                Reference
              </Link>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.explanation}>
                Explanation
              </Link>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
