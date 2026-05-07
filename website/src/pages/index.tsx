import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import {ROUTES, GITHUB_REPO, SITE_TITLE} from '../constants';
import styles from './index.module.css';

// ---------------------------------------------------------------------------
// Hero
// ---------------------------------------------------------------------------

function Hero(): ReactNode {
  return (
    <section className={styles.hero}>
      <div className="container">
        <div className={styles.heroInner}>
          <Heading as="h1" className={styles.wordmark}>
            {SITE_TITLE}
          </Heading>

          <p className={styles.tagline}>
            Measure how implementation choices drive LLM inference efficiency.
          </p>

          <p className={styles.subTag}>
            Multi-engine, methodology-first, parametrically extensible.
          </p>

          <div className={styles.heroCtas}>
            <Link
              className="button button--primary button--lg"
              to={ROUTES.getStarted}>
              Get Started
            </Link>
          </div>

          <div className={styles.heroSecondary}>
            <Link className="button button--secondary" href={GITHUB_REPO}>
              GitHub
            </Link>
            <Link
              className="button button--secondary"
              to="/explanation/methodology/methodology">
              Read the methodology
            </Link>
            <Link
              className="button button--secondary"
              to="/contributing/citation">
              Cite this work
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Disambiguation strip
// ---------------------------------------------------------------------------

function DisambiguationStrip(): ReactNode {
  return (
    <div className={styles.disambig}>
      <div className="container">
        <p className={styles.disambigText}>
          Zeus and CodeCarbon are integrated as samplers, not replaced.
          {' '}lm-evaluation-harness measures capability; this measures
          efficiency, and the two pair naturally.
          {' '}MLPerf is a fixed-rule benchmark; this is researcher-extensible
          measurement whose outputs inform benchmark design.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// What is this?
// ---------------------------------------------------------------------------

function WhatIsThis(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionNarrow}>
          <Heading as="h2" className={styles.sectionHeading}>
            What is this?
          </Heading>
          <p className={styles.pitch}>
            LLenergyMeasure is an open research tool for measuring LLM inference
            efficiency. It stitches energy samplers (Zeus, CodeCarbon, NVML),
            inference engines (transformers, vLLM, TensorRT-LLM), and a
            methodology-rigorous measurement harness into a coherent research
            tool that takes researchers&#39; specs and runs them. The tool
            discovers and exposes engine parameters programmatically, then uses
            invariant mining and deduplication to keep the resulting parameter
            space tractable for sweeps. It can be used for any LLM efficiency
            research; the primary question it was built to answer is: how do
            implementation choices drive efficiency?
          </p>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Code preview
// ---------------------------------------------------------------------------

function CodePreview(): ReactNode {
  return (
    <section className={styles.sectionAlt}>
      <div className="container">
        <div className={styles.sectionNarrow}>
          <Heading as="h2" className={styles.sectionHeading}>
            First measurement
          </Heading>
          <Tabs groupId="surface">
            <TabItem value="cli" label="CLI">
              <pre className={styles.codeBlock}><code>{`$ llem run --model gpt2 --engine transformers

engine:      transformers
model:       gpt2
energy:      12.43 J
throughput:  47.2 tok/s
duration:    3.18 s`}</code></pre>
            </TabItem>
            <TabItem value="python" label="Python">
              <pre className={styles.codeBlock}><code>{`from llenergymeasure import run_experiment

result = run_experiment(model="gpt2", engine="transformers")

print(result.total_energy_j)            # 12.43
print(result.avg_tokens_per_second)     # 47.2`}</code></pre>
            </TabItem>
          </Tabs>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Methodology specifics
// ---------------------------------------------------------------------------

function MethodologyBlock(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionNarrow}>
          <Heading as="h2" className={styles.sectionHeading}>
            Methodology
          </Heading>
          <ul className={styles.methodologyList}>
            <li>
              GPU power sampled via NVML at 100 ms intervals (default sampler)
            </li>
            <li>
              Baseline idle-power subtracted via two-container baseline measurement
            </li>
            <li>
              Warmup convergence required (CV {'<'} 0.05) before the measurement
              window opens
            </li>
            <li>
              Energy reported in joules; reproducibility notes attached to every
              result
            </li>
            <li>
              Sampler plugins: NVML, Zeus, CodeCarbon
            </li>
            <li>
              Engine plugins: transformers, vLLM, TensorRT-LLM
            </li>
          </ul>
          <p>
            <Link to="/explanation/methodology/methodology">
              Read the full methodology -&gt;
            </Link>
          </p>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// What we are NOT
// ---------------------------------------------------------------------------

function BoundariesBlock(): ReactNode {
  return (
    <section className={styles.sectionAlt}>
      <div className="container">
        <div className={styles.sectionNarrow}>
          <Heading as="h2" className={styles.sectionHeading}>
            Explicit boundaries
          </Heading>
          <ul className={styles.boundaryList}>
            <li>
              We measure inference, not training.
            </li>
            <li>
              We measure efficiency, not capability - pair with{' '}
              <a
                href="https://github.com/EleutherAI/lm-evaluation-harness"
                target="_blank"
                rel="noreferrer">
                lm-evaluation-harness
              </a>{' '}
              for the capability side.
            </li>
            <li>
              We are not a fixed-rule benchmark - pair with MLPerf for
              standardised comparison.
            </li>
            <li>
              Our outputs inform benchmark design and policy; we are not the
              benchmark.
            </li>
          </ul>
          <p>
            <Link to="/explanation/ecosystem">
              Ecosystem positioning -&gt;
            </Link>
          </p>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Audience cards
// ---------------------------------------------------------------------------

interface AudienceCardProps {
  title: string;
  description: string;
  ctaLabel: string;
  to: string;
}

function AudienceCard({title, description, ctaLabel, to}: AudienceCardProps): ReactNode {
  return (
    <div className={styles.card}>
      <Heading as="h3" className={styles.cardTitle}>
        {title}
      </Heading>
      <p className={styles.cardDescription}>{description}</p>
      <Link className="button button--outline button--primary" to={to}>
        {ctaLabel}
      </Link>
    </div>
  );
}

const AUDIENCE_CARDS: AudienceCardProps[] = [
  {
    title: 'Researcher',
    description:
      'Design and run a multi-engine implementation-parameter study. ' +
      'Control engine, sampler, and sweep axes from a single YAML config.',
    ctaLabel: 'Multi-engine study tutorial',
    to: '/tutorials/multi-engine-study',
  },
  {
    title: 'Engineer',
    description:
      'Run llem in your CI pipeline with Docker and vLLM. ' +
      'Reproducible containers; no host CUDA dependency.',
    ctaLabel: 'Run with Docker + vLLM',
    to: '/how-to/run-with-docker-vllm',
  },
  {
    title: 'Policy reader',
    description:
      'Understand what LLenergyMeasure measures, what it does not ' +
      'measure, and how to read its outputs.',
    ctaLabel: 'Plain-language overview',
    to: '/get-started/for-policy-readers',
  },
  {
    title: 'Citing',
    description:
      'Using LLenergyMeasure in published research? ' +
      'Copy the BibTeX reference from the citation page.',
    ctaLabel: 'Cite this work',
    to: '/contributing/citation',
  },
];

function AudienceCards(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <Heading as="h2" className={styles.sectionHeading}>
          Where to start
        </Heading>
        <div className={styles.cardGrid}>
          {AUDIENCE_CARDS.map((card) => (
            <AudienceCard key={card.title} {...card} />
          ))}
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Pillar cards (Diátaxis)
// ---------------------------------------------------------------------------

interface PillarCardProps {
  title: string;
  tagline: string;
  to: string;
}

function PillarCard({title, tagline, to}: PillarCardProps): ReactNode {
  return (
    <Link to={to} className={styles.pillarCard}>
      <Heading as="h3" className={styles.pillarCardTitle}>
        {title}
      </Heading>
      <p className={styles.pillarCardTagline}>{tagline}</p>
    </Link>
  );
}

const PILLAR_CARDS: PillarCardProps[] = [
  {
    title: 'Tutorials',
    tagline: 'Learn by doing',
    to: ROUTES.tutorials,
  },
  {
    title: 'How-to',
    tagline: 'Solve a specific problem',
    to: ROUTES.howTo,
  },
  {
    title: 'Reference',
    tagline: 'Look up a specific fact',
    to: ROUTES.reference,
  },
  {
    title: 'Explanation',
    tagline: 'Understand the why',
    to: ROUTES.explanation,
  },
];

function PillarCards(): ReactNode {
  return (
    <section className={styles.sectionAlt}>
      <div className="container">
        <Heading as="h2" className={styles.sectionHeading}>
          Documentation
        </Heading>
        <div className={styles.cardGrid}>
          {PILLAR_CARDS.map((card) => (
            <PillarCard key={card.title} {...card} />
          ))}
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Contributor footer + citation block
// ---------------------------------------------------------------------------

const BIBTEX = `@software{baker2026llenergymeasure,
  author    = {Baker, Henry C. G.},
  title     = {{LLenergyMeasure}: Energy and efficiency measurement for LLM inference},
  year      = {2026},
  version   = {0.9.0},
  url       = {https://github.com/henrycgbaker/llenergymeasure},
  note      = {Pre-1.0 release. See GitHub releases for the current version.
               CLI-first measurement framework for LLM inference efficiency
               across heterogeneous runtimes.}
}`;

function ContributorFooter(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionNarrow}>
          <p className={styles.attribution}>
            Grew from a master&#39;s thesis on LLM energy efficiency by{' '}
            <a
              href="https://henrycgbaker.github.io/research/llm-energy-efficiency/"
              target="_blank"
              rel="noreferrer">
              Henry Baker
            </a>
            .
          </p>

          <details className={styles.citationDetails}>
            <summary className={styles.citationSummary}>BibTeX citation</summary>
            <pre className={styles.codeBlock}><code>{BIBTEX}</code></pre>
            <p>
              <Link to="/contributing/citation">Full citation page -&gt;</Link>
            </p>
          </details>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Page root
// ---------------------------------------------------------------------------

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <main>
        <Hero />
        <DisambiguationStrip />
        <WhatIsThis />
        <CodePreview />
        <MethodologyBlock />
        <BoundariesBlock />
        <AudienceCards />
        <PillarCards />
        <ContributorFooter />
      </main>
    </Layout>
  );
}
