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
              CLI-first measurement framework for energy, throughput, and
              FLOPs across multiple inference engines (Transformers, vLLM,
              TensorRT-LLM). Reproducible study configs, curated
              measurement-relevant parameters, and engine-introspected
              schemas.
            </p>

            <div className={styles.actions}>
              <Link
                className="button button--primary button--lg"
                to={ROUTES.userGuide}>
                Docs
              </Link>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.methodology}>
                Methodology
              </Link>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.api}>
                API
              </Link>
              <Link
                className="button button--secondary button--lg"
                to={ROUTES.architecture}>
                Architecture
              </Link>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
