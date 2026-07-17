# cli/ - Command-line Interface

Typer-based CLI for llenergymeasure. Layer 6 (top layer) in the six-layer architecture.

## Purpose

The CLI is a thin client over the `api/` layer - it handles argument parsing, user-facing formatting, and error presentation. `llem run` runs experiments and studies, `llem doctor` is the environment health check (GPU, engines, energy samplers, Docker, credentials, resolved config, and the image schema handshake), `llem report-gaps` proposes engine rule-set entries, and `llem study` writes and prepares study files.

## Modules

| Module | Description |
|--------|-------------|
| `__init__.py` | `app` Typer instance, logging setup, command registration |
| `run.py` | `llem run` command |
| `doctor_cmd.py` | `llem doctor` command (environment health check) |
| `_display.py` | Output formatting helpers (headers, result tables, errors) |
| `_vram.py` | VRAM estimation for pre-run model size hints |

## Commands

### llem run

```bash
llem run [CONFIG]           # run from YAML config
llem run --model gpt2       # inline model spec
llem run --model gpt2 --engine transformers --dataset aienergyscore -n 100
llem run --dry-run          # validate config without running
llem run -v                 # verbose logging (INFO)
llem run -vv                # debug logging
```

Key options:

| Option | Description |
|--------|-------------|
| `CONFIG` | Path to experiment or study YAML |
| `--model / -m` | Model name or HuggingFace path |
| `--engine / -e` | Inference engine (`pytorch`, `vllm`, `tensorrt`) |
| `--dataset / -d` | Dataset name or JSONL file path |
| `-n` | Number of prompts |
| `--dry-run` | Validate config, print plan, exit |
| `--skip-preflight` | Skip Docker/CUDA pre-flight checks |
| `-v / -vv` | Verbosity (INFO / DEBUG) |

### llem doctor

```bash
llem doctor          # sectioned environment health check
llem doctor --check  # exit 0=ok, 1=warnings, 2=errors (CI/scripting)
llem doctor --json   # full report as machine-readable JSON
```

The environment health check. Each line is prefixed `[ok]`/`[warn]`/`[fail]`
with a `->` fix hint on anything that is not ready. Sections: GPU / driver,
Engines (importable locally or via Docker, image cache state), Energy
measurement (NVML / Zeus / CodeCarbon), Docker (CLI, daemon, NVIDIA Container
Toolkit), Credentials (HF_TOKEN presence - the value is never printed),
Configuration (user-config status and resolved runner/sampler/gap settings with
their provenance), and the Image schema handshake.

The handshake reads the `llem.expconf.schema.fingerprint` OCI label from each
engine image and compares it to a fingerprint computed from the host's current
`ExperimentConfig.model_json_schema()`. A `MISMATCH` is the one condition that
makes `llem doctor` exit non-zero by default (exit 1); `--check` grades all
severities (0/1/2). Set `LLEM_SKIP_IMAGE_CHECK=1` to bypass the runtime
handshake in `llem run` (doctor still reports the true status).

### llem --version

```bash
llem --version
```

## Logging

Verbosity is controlled per-run via `-v` / `-vv`. The `llenergymeasure` logger hierarchy is configured in `__init__.py`:

| Flag | Level |
|------|-------|
| (none) | WARNING (or `LLEM_LOG_LEVEL` env var) |
| `-v` | INFO |
| `-vv` | DEBUG |

## Layer constraints

- Layer 6 - top layer, may import from all layers below
- Nothing imports from `cli/` except the entrypoint defined in `pyproject.toml`
- CLI is a client of `api/`, not the reverse

## Related

- See `../api/` for the library API the CLI delegates to
- See `../config/README.md` for full YAML config reference
