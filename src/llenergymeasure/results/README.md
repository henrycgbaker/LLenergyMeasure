# results/ - Results Persistence

Saves and loads a single experiment's results to and from the filesystem.
There is no repository object, aggregation command, or export layer here; the
package is one module, `persistence.py`.

## persistence.py

Public functions:

| Function | Purpose |
|----------|---------|
| `save_result(result, output_dir, ...)` | Write an `ExperimentResult` to a collision-safe per-experiment directory. Returns the `result.json` path. |
| `load_result(path)` | Read an `ExperimentResult` back from a `result.json` path, re-attaching sidecars. |
| `save_config_sidecar(...)` | Write the resolved-config `config.json` sidecar (carries `observed_config_hash`). |
| `save_environment(snapshot, dir)` | Write the `environment.json` sidecar. |

Writes are atomic (temp file plus `os.replace`) and directory names are made
collision-free before writing.

## Per-experiment layout

`save_result` creates one directory per run under `output_dir`:

```
{output_dir}/
  [{index}_]c{cycle}_{model}-{engine}_{hash}/
    result.json          # ExperimentResult, pydantic model_dump_json (the only typed dump)
    config.json          # resolved-config sidecar (save_config_sidecar); its
                         # provenance section records per-field override sources
    environment.json     # environment snapshot (save_environment)
    timeseries.parquet   # GPU power/thermal/memory series (copied in when present)
```

`load_result` reads `result.json` and best-effort re-attaches the
`timeseries.parquet` and `environment.json` sidecars; a missing or corrupt
sidecar degrades gracefully (warning, not error) rather than failing the load.

Study-level artefacts (the `_study-artefacts/` directory, `manifest.json`,
`equivalence_groups.json`) are written by the `study/` and `api/` layers, not
here. Artefact filenames are defined in `domain/bundle_artefacts.py`.

## Related

- `../domain/experiment.py` - the `ExperimentResult` model this package serialises.
- `../domain/bundle_artefacts.py` - shared bundle filename constants.
