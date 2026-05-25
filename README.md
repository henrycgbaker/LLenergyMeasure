# LLenergyMeasure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-henrycgbaker.github.io-blue)](https://henrycgbaker.github.io/llenergymeasure/)

📖 **Documentation site:** <https://henrycgbaker.github.io/llenergymeasure/>

Measure the energy efficiency of LLM inference across different implementation configurations.

LLenergyMeasure is a Python framework for measuring the energy consumption, throughput, and computational cost (FLOPs) of LLM inference across different deployment configurations. It helps researchers compare the energy efficiency of different models, inference engines, and a wide range of implementation decisions - reproducibly and at publication quality.

---

## Key Features

- **Multi-engine inference** - Transformers, vLLM, TensorRT-LLM, SGLang (planned)
- **GPU energy measurement** - NVML, Zeus, CodeCarbon, others 
- **Smart sweep system** - define parameter grids, run Cartesian product experiments automatically; intelligently managed sweep hierarchy scopes available config fields to appropriate engine/component, and ensures invalid combinations are removed
- **Docker isolation** - launches per-experiment containers with full GPU passthrough; latest docker images for each engine in registry with full runner configurability and local mode also available. Every study pre-flight now verifies that each image's `ExperimentConfig` schema fingerprint matches the host's, aborting with an actionable rebuild hint on drift (`llem doctor` for a one-shot check).
- **Reproducibility** - fixed seeds, cycle ordering, thermal management, environment snapshots, effective config recorded (add others)
- **Built-in datasets** - AI Energy Score benchmark prompts included; custom JSONL datasets also supported

---

## Quick Install

```bash
pip install llenergymeasure
```

Engine code (Transformers, vLLM, TensorRT-LLM) runs inside per-engine Docker images; the host package is the orchestrator. See [docs/contributing/development.md](docs/contributing/development.md) for the build/run pattern.

Run your first measurement (host dispatches the appropriate engine container):

```bash
llem run --model gpt2 --engine transformers
```

See the [documentation site](https://henrycgbaker.github.io/llenergymeasure/) for the full guide - tutorials, how-to recipes, reference (CLI, study config, library API, engines), conceptual explanation (methodology, energy measurement, architecture), and a contributing guide for internals.

---

## Contributing

Contributions welcome. See the [development install](docs/how-to/install.md#install-from-source-development) instructions to set up a local environment, plus the [contributing guide](docs/contributing/development.md).

---

## Research artefacts

Long-running research artefacts (trials, methodology explorations) live under `research/` and are not part of the production codebase. Casual contributors who don't need the research corpus can opt out via `bash scripts/setup-research-optin.sh`. See [`research/README.md`](research/README.md) for the catalogue.

---

## License

[MIT](LICENSE)
