# P-07: Parameter Provenance Tracking

**Module**: `src/llenergymeasure/config/provenance.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: `designs/reproducibility.md` lists reproducibility as a goal but does not mention provenance tracking; at risk of being dropped in the config refactor.

---

## What Exists in the Code

**Primary file**: `src/llenergymeasure/config/provenance.py` (227 lines)
**Key classes/functions**:
- `ParameterSource` enum (line 23) — `PYDANTIC_DEFAULT`, `PRESET`, `CONFIG_FILE`, `CLI`
- `ParameterProvenance` (line 39) — `path: str`, `value: Any`, `source: ParameterSource`, `source_detail: str | None` (e.g., preset name, config file path)
- `ResolvedConfig` (line 66) — pairs a fully-resolved `ExperimentConfig` with its complete provenance dict:
  - `get_provenance(path: str) -> ParameterProvenance | None` (line 93)
  - `get_parameters_by_source(source: ParameterSource)` (line 104)
  - `get_non_default_parameters()` (line 115)
  - `get_cli_overrides()` (line 123)
  - `to_summary_dict()` (line 131) — serialises to dict for embedding in result files
- Utilities: `flatten_dict()`, `unflatten_dict()`, `compare_dicts()` (lines 148–226)

The config loader (`config/loader.py`, `load_config_with_provenance()`, lines 351–488) populates provenance in four passes: pydantic defaults → preset → config file → CLI overrides. Each pass records the source with enough detail to audit exactly where each parameter value came from.

**Integration with results**: `RawProcessResult` and `AggregatedResult` carry `parameter_provenance: dict[str, dict]` and `preset_chain: list[str]` (domain/experiment.py, lines 84–91).

## Why It Matters

A researcher can inspect a result file and know: "this parameter came from the preset `throughput`, this one was overridden from the config file, this one was a CLI flag." This is critical for experiment reproducibility audits and debugging unexpected parameter values. It also enables detecting configuration drift across a multi-cycle study.

## Planning Gap Details

`designs/reproducibility.md` mentions reproducibility goals — environment snapshots, config hashing, `measurement_methodology` field — but **does not document parameter provenance at all**. There is a risk that the config refactor drops the `ResolvedConfig` wrapper and the provenance dict without realising its value. The result schema fields (`parameter_provenance`, `preset_chain`) will be lost if not explicitly called out.

## Recommendation for Phase 5

Add to `designs/reproducibility.md` under a new "Parameter Provenance" section:

> **Parameter Provenance**: Each `ExperimentResult` includes `parameter_provenance: dict[str, dict]`
> and `preset_chain: list[str]`. These track the source of every configuration parameter through
> the four-layer resolution chain: pydantic default → preset → config file → CLI override.
>
> Implementation: `config/provenance.py` → `ResolvedConfig.to_summary_dict()`.
>
> This enables researchers to audit exactly how a run was configured. Example use: debugging why
> `batch_size=8` appears in results when the config file specifies `batch_size=4` (CLI override).

Ensure the config refactor returns a `ResolvedConfig` (not bare `ExperimentConfig`) from the loader.
