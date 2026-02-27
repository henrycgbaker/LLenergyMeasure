---
status: complete
phase: 07-cli
source: [07-01-SUMMARY.md, 07-02-SUMMARY.md, 07-03-SUMMARY.md]
started: 2026-02-27T01:00:00Z
updated: 2026-02-27T01:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. llem --help shows both commands
expected: `llem --help` prints help text listing both `run` and `config` as available commands. No Rich formatting — plain text only. Exits 0.
result: pass

### 2. llem --version prints version
expected: `llem --version` prints `llem v2.0.0` (or current version) and exits 0.
result: pass

### 3. llem run --help shows all flags
expected: `llem run --help` lists: `[CONFIG]`, `--model/-m`, `--backend/-b`, `--dataset/-d`, `-n`, `--batch-size`, `--precision/-p`, `--output/-o`, `--dry-run`, `--quiet/-q`, `--verbose/-v`. Exits 0.
result: pass

### 4. llem run with no args exits 2
expected: `llem run` with no arguments prints a ConfigError message to stderr and exits with code 2. Should indicate that either a config file or --model is required.
result: pass

### 5. llem run --dry-run validates and shows VRAM
expected: `llem run --model gpt2 --backend pytorch --dry-run` shows a "Config (resolved)" section with model/backend/precision, a "VRAM estimate" section (or "unavailable" if no network), and exits 0 without running inference.
result: pass

### 6. llem run --dry-run --verbose adds source annotations
expected: `llem run --model gpt2 --backend pytorch --dry-run --verbose` shows the same as above but with source annotations indicating which values are defaults vs explicitly set.
result: pass

### 7. llem config shows environment
expected: `llem config` prints sections for GPU (name + VRAM or "No GPU detected"), Backends (installed/not installed with install hints), Energy (selected backend), User Config (path + status), Python version. Always exits 0.
result: pass

### 8. llem config --verbose adds detail
expected: `llem config --verbose` shows everything from basic config plus: NVIDIA driver version (if GPU), per-backend version strings (if installed), and non-default user config values.
result: pass

### 9. ValidationError gets friendly formatting
expected: `llem run --model gpt2 --backend pytorh` (misspelled backend) prints a friendly error header "Config validation failed (1 error):" with a did-you-mean suggestion ("pytorch"), not a raw Pydantic traceback. Exits code 2.
result: pass

### 10. Number formatting (3 significant figures)
expected: Running `python -c "from llenergymeasure.cli._display import _sig3; print(_sig3(312.4), _sig3(3.12), _sig3(0.00312), _sig3(0))"` outputs `312 3.12 0.00312 0`.
result: pass

### 11. Duration formatting
expected: Running `python -c "from llenergymeasure.cli._display import _format_duration; print(_format_duration(4.2), _format_duration(272), _format_duration(3900))"` outputs `4.2s 4m 32s 1h 05m`.
result: pass

### 12. All unit tests pass
expected: `pytest tests/unit/test_cli_display.py tests/unit/test_cli_run.py tests/unit/test_cli_config.py -v` — all 37 tests pass with no failures or errors.
result: pass

## Summary

total: 12
passed: 12
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
