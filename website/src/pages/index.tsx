import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import {ROUTES, SITE_TITLE} from '../constants';
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
            A CLI-first framework that supports energy, throughput, and FLOPs
            measurement - with extensible multi-engine-introspected parameter
            spaces, structured sweep logic, experiment deduplication and
            sampling.
          </p>

          <div className={styles.heroCtas}>
            <Link
              className="button button--primary button--lg"
              to={ROUTES.getStarted}>
              Get Started
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="/explanation/why">
              Why this exists
            </Link>
          </div>
        </div>
      </div>
    </section>
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
              Per-engine Docker isolation - dependency stacks pinned and
              isolated; runs reproduce bit-identically against the same
              pinned versions
            </li>
            <li>
              Sampler plugins:{' '}
              <a
                href="https://docs.nvidia.com/deploy/nvml-api/"
                target="_blank"
                rel="noreferrer">
                NVML
              </a>
              ,{' '}
              <a
                href="https://ml.energy/zeus/"
                target="_blank"
                rel="noreferrer">
                Zeus
              </a>
              ,{' '}
              <a
                href="https://codecarbon.io/"
                target="_blank"
                rel="noreferrer">
                CodeCarbon
              </a>
            </li>
            <li>
              Engine plugins (extensible via{' '}
              <Link to="/explanation/architecture/engine-extensibility">
                plugin protocol
              </Link>
              ; ships with{' '}
              <a
                href="https://huggingface.co/docs/transformers"
                target="_blank"
                rel="noreferrer">
                transformers
              </a>
              ,{' '}
              <a
                href="https://docs.vllm.ai/"
                target="_blank"
                rel="noreferrer">
                vLLM
              </a>
              ,{' '}
              <a
                href="https://nvidia.github.io/TensorRT-LLM/"
                target="_blank"
                rel="noreferrer">
                TensorRT-LLM
              </a>
              )
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
// Built for a moving target (extensibility / introspection / sweep tractability)
// ---------------------------------------------------------------------------

function ExtensibilityBlock(): ReactNode {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionNarrow}>
          <Heading as="h2" className={styles.sectionHeading}>
            Built for a moving target
          </Heading>
          <p className={styles.pitch}>
            The inference stack moves quickly: vLLM, TensorRT-LLM, and
            SGLang ship new versions on a monthly cadence, each adding
            tens to hundreds of configuration parameters. Manual curation
            does not keep up.
          </p>
          <p className={styles.pitch}>
            llem handles this through three coupled mechanisms:
          </p>
          <ul className={styles.methodologyList}>
            <li>
              <strong>Programmatic introspection.</strong> Each engine&#39;s
              full parameter surface is discovered automatically from
              Pydantic schemas and class signatures, not from a
              hand-curated list.
            </li>
            <li>
              <strong>Invariant mining.</strong> Validation constraints -
              which parameter combinations the engine rejects, warns on, or
              silently normalises - are mined automatically from each
              library&#39;s source via AST walking of validator methods and
              dynamic probing. Mined rules deduplicate across engines by
              fingerprint.
            </li>
            <li>
              <strong>Renovate-driven refresh.</strong> When an engine
              version bumps upstream, CI re-runs introspection and mining
              inside the new image, regenerates the schema and invariants
              artefacts, and posts the diff for review.
            </li>
          </ul>
          <p className={styles.pitch}>
            Together these make sweeps tractable across an otherwise-vast
            parameter space. A study can request thousands of cells;
            deduplication collapses semantically-identical configs; the
            resolved plan is reviewable before any GPU time is spent.
          </p>
          <p>
            <Link to="/explanation/architecture/engine-introspection-pipelines">
              Engine introspection pipelines -&gt;
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
            What this isn&#39;t
          </Heading>
          <ul className={styles.boundaryList}>
            <li>
              Inference, not training. Training accounting belongs elsewhere.
            </li>
            <li>
              Efficiency, not capability - pair with{' '}
              <a
                href="https://github.com/EleutherAI/lm-evaluation-harness"
                target="_blank"
                rel="noreferrer">
                lm-evaluation-harness
              </a>{' '}
              for the capability side.
            </li>
            <li>
              Measurement, not benchmark - pair with{' '}
              <a
                href="https://mlcommons.org/benchmarks/inference-datacenter/"
                target="_blank"
                rel="noreferrer">
                MLPerf Inference
              </a>{' '}
              for standardised inter-model comparison.
            </li>
            <li>
              Outputs inform benchmark design and policy; llem is not the
              benchmark itself.
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
        <WhatIsThis />
        <CodePreview />
        <MethodologyBlock />
        <ExtensibilityBlock />
        <BoundariesBlock />
        <AudienceCards />
        <PillarCards />
        <ContributorFooter />
      </main>
    </Layout>
  );
}
