import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# CLI Reference

`llem` is the command-line interface for LLenergyMeasure. It exposes four
commands - `run`, `config`, `doctor`, and `report-gaps` - plus a `--version`
flag. Every operation available from the CLI is also available through the
[Python library API](/reference/library/run_experiment); the CLI is a thin
driver over the same underlying harness.

`llem` has four commands (`run`, `config`, `doctor`, `report-gaps`) and one flag (`--version`).

```
llem run [CONFIG] [OPTIONS]   # run an experiment or study
llem config [OPTIONS]         # show environment and configuration status
llem doctor                   # verify Docker images match the host schema
llem report-gaps [OPTIONS]    # propose corpus rules from runtime observations
llem --version                # print version and exit
```

---

<!-- Auto-generated sections from scripts/generate_cli_reference.py — regenerate with: uv run python scripts/generate_cli_reference.py -->

## `llem run`

Run an experiment or study. Detects study mode automatically when the YAML config contains `sweep:` or `experiments:` keys.

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `config` | path | no | Path to experiment or study YAML config |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | str | — | Model name or HuggingFace path |
| `--engine` | `-e` | str | — | Inference engine (`transformers`, `vllm`, `tensorrt`) |
| `--dataset` | `-d` | str | — | Dataset source (`aienergyscore` or `.jsonl` file path) |
| `-n` | | int | — | Number of prompts to run (`dataset.n_prompts`) |
| `--batch-size` | | int | — | Batch size (Transformers engine only) |
| `--dtype` | `-p` | str | — | Model dtype (`float32`, `float16`, `bfloat16`) |
| `--output` | `-o` | str | — | Output directory for results |
| `--dry-run` | | flag | false | Validate config and estimate VRAM without running |
| `--quiet` | `-q` | flag | false | Suppress progress bars |
| `--verbose` | `-v` | flag | false | Show detailed output and tracebacks |
| `--cycles` | | int | — | Number of measurement cycles (study mode) |
| `--order` | | str | — | Cycle ordering: `sequential`, `interleave`, `shuffle` (study mode) |
| `--no-gaps` | | flag | false | Disable thermal gaps between experiments (study mode) |
| `--skip-preflight` | | flag | false | Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility) |
| `--resume` | | flag | false | Resume most recent interrupted study |
| `--resume-dir` | | path | — | Resume a specific study directory |
| `--fail-fast` | | flag | false | Abort study on first failure (circuit breaker threshold=1) |
| `--no-circuit-breaker` | | flag | false | Disable circuit breaker entirely |
| `--timeout` | | float | — | Study wall-clock timeout in hours (e.g. `24`, `1.5`) |
| `--no-lock` | | flag | false | Disable GPU lock files (advanced) |

**CLI effective defaults for study mode** (applied when neither the YAML `study_execution:` block nor a CLI flag specifies the value):

- `--cycles` defaults to `3` (Pydantic model default is `1`)
- `--order` defaults to `shuffle` (Pydantic model default is `sequential`)

These defaults are applied at the CLI layer to give better statistical coverage out of the box. To use the conservative model defaults, set them explicitly in the YAML `study_execution:` block.

**Exit codes:** `0` success, `1` experiment/engine/preflight error, `2` config validation error, `130` interrupted (Ctrl-C).

---

## `llem config`

Show environment and configuration status. Always exits `0` — this command is informational only.

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--verbose` | `-v` | flag | false | Show driver version, engine versions, and full config diff |

**Example output:**

```
GPU
  NVIDIA A100-SXM4-80GB  80.0 GB
Engines
  transformers: installed
  vllm: not installed  (runs in Docker — see docs/development.md)
  tensorrt: not installed  (runs in Docker — see docs/development.md)
Energy
  Energy: nvml
Config
  Path: /home/user/.config/llenergymeasure/config.yaml
  Status: using defaults (no config file)
Python
  3.12.0
```

---

## `llem doctor`

Verify that every engine's resolved Docker image matches the host's
`ExperimentConfig` schema. Compares the `llem.expconf.schema.fingerprint`
OCI label baked into each image against a fingerprint computed from the
host's current `ExperimentConfig.model_json_schema()`.

Image resolution follows the same chain as `llem run`: local build
(`llenergymeasure:{engine}`) first, then the versioned GHCR tag
(`ghcr.io/henrycgbaker/llenergymeasure/{engine}:v{version}`).

**Exit codes:** `0` when every reachable image matches (or labels are absent
on legacy images); `1` when at least one image's schema fingerprint differs
from the host.

**Columns:**

| Column | Meaning |
|--------|---------|
| `Engine` | Engine identifier (`transformers`, `vllm`, `tensorrt`) |
| `Image` | Resolved image tag (local or GHCR) |
| `Pkg ver` | `org.opencontainers.image.version` label (LLenergyMeasure release) |
| `Img FP` | First 12 chars of `llem.expconf.schema.fingerprint` label |
| `Host FP` | First 12 chars of host `ExperimentConfig` fingerprint |
| `Status` | `OK` / `MISMATCH` / `UNVERIFIED` (pre-handshake image) / `UNREACHABLE` (no such image) |

**Bypass:** `LLEM_SKIP_IMAGE_CHECK=1` disables the runtime handshake in
`llem run`; when set, `llem doctor` still reports the true status but prints a
warning in the footer.

**Example:**

```bash
llem doctor
```

```
Engine     Image                          Pkg ver     Img FP          Host FP         Status
---------------------------------------------------------------------------------------------
transformers llenergymeasure:transformers        0.9.0       a1b2c3d4e5f6    a1b2c3d4e5f6    OK
vllm        llenergymeasure:vllm           0.9.0       9988776655ff    a1b2c3d4e5f6    MISMATCH
            └─ repull: docker pull vllm/vllm-openai:0.7.3
tensorrt    llenergymeasure:tensorrt       0.9.0       a1b2c3d4e5f6    a1b2c3d4e5f6    OK

Host llenergymeasure version: 0.9.0
Host ExperimentConfig fingerprint: a1b2c3d4e5f6…
```

---

## `llem report-gaps`

Propose new invariant-corpus entries from observations captured during a study run. Use this when a researcher hits a runtime warning or unexpected outcome that the corpus doesn't yet describe — `report-gaps` synthesises a YAML fragment you can review and merge into the validated corpus.

```bash
llem report-gaps \
  --study-dir results/<study-name>_<timestamp> \
  --out proposed-rules.yaml \
  [--source runtime-warnings] \
  [--engine vllm] \
  [--include-exceptions]
```

**Options:**

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--study-dir PATH` | yes | — | Study directory to scan. Repeat the flag to pass multiple study directories. |
| `--out PATH` | yes | — | Output path for proposed YAML fragments (one YAML document per gap, separated by `---`). |
| `--source NAME` | no | `runtime-warnings` | Feedback source to scan. Only `runtime-warnings` is wired in this release. |
| `--engine NAME` | no | (all) | Filter: only propose rules for this engine (`transformers` / `vllm` / `tensorrt`). |
| `--include-exceptions` | no | false | Also propose rules from runtime *exceptions*. Off by default — exceptions ship through a different review path. |

**Output:** writes a multi-document YAML file at `--out`. Each document is a candidate corpus entry with `# TODO: human` markers on placeholder fields the reviewer must fill in. The file is *not* automatically merged into the corpus — review and apply manually.

**See also:** [Architecture > parameter discovery](/explanation/architecture/parameter-discovery) for the runtime-observation pipeline; [Architecture > parameter curation](/explanation/architecture/parameter-curation) for the curation review flow.

---

## `llem --version`

Print version and exit.

```bash
llem --version
```

Example output:

```
llem v0.9.0
```

---

## Examples

### Single experiment

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
llem run --model gpt2 -e transformers
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment(model="gpt2", engine="transformers")
```

</TabItem>
</Tabs>

### Single experiment via YAML

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
llem run experiment.yaml
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment("experiment.yaml")
```

</TabItem>
</Tabs>

### Dry run (validate config, estimate VRAM)

```bash
llem run experiment.yaml --dry-run
```

Dry-run is a CLI-only path; `run_study` in the Python API does not
currently take a `dry_run` flag.

### Study with cycle override

```bash
# Run 5 cycles in interleave order instead of the CLI default (3 shuffle)
llem run study.yaml --cycles 5 --order interleave
```

### Suppress thermal gaps (testing only)

```bash
llem run study.yaml --no-gaps
```

### Skip Docker pre-flight (when Docker daemon is on a remote host)

```bash
llem run study.yaml --skip-preflight
```

### Resume an interrupted study

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
# Auto-detect most recent resumable study
llem run study.yaml --resume

# Resume a specific study directory
llem run study.yaml --resume-dir results/full-suite-all-engines_20260329_1716/
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_study

# Auto-detect most recent resumable study
study_result = run_study("study.yaml", resume=True)

# Resume a specific study directory
study_result = run_study("study.yaml", resume_dir="results/full-suite-all-engines_20260329_1716/")
```

</TabItem>
</Tabs>

### Fail-fast mode (abort on first failure)

```bash
llem run study.yaml --fail-fast
```

### Set a wall-clock timeout

```bash
# Abort after 24 hours, mark remaining experiments as skipped
llem run study.yaml --timeout 24
```

### Environment check

```bash
llem config
llem config --verbose
```
