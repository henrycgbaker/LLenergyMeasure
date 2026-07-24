# results/ - Results Persistence

Saves and loads a single experiment's results to and from the filesystem.
There is no repository object, aggregation command, or export layer here. The
package is two modules: `bundle.py`, which owns the per-experiment
results-bundle write and read policy, and `persistence.py`, which holds the
low-level filesystem primitives it builds on.

## bundle.py

`BundleWriter` is the single owner of the assembly policy that turns a completed
experiment into an on-disk bundle: the collision-free experiment directory,
`result.json` (with runner provenance folded in), `system.json`, the
`config.json` sidecar move and patch, and the timeseries attach, all driven by
the `domain.bundle_artefacts.ARTEFACTS` registry. `BundleReader` is its
read-side counterpart: given a bundle directory it discovers the artefacts via
the same registry and returns a `LoadedBundle`. Both are stamped with the single
`bundle_version` (`"2.0"`).

## persistence.py

Holds the low-level filesystem primitives `BundleWriter` delegates to (atomic
writes, collision-safe directory naming, the sidecar writers). `load_result` is
a thin wrapper over `BundleReader.read`.

Public functions:

| Function | Purpose |
|----------|---------|
| `save_result(result, output_dir, ...)` | Write an `ExperimentResult` to a collision-safe per-experiment directory. Returns the `result.json` path. |
| `load_result(path)` | Read an `ExperimentResult` back from a `result.json` path, re-attaching sidecars. |
| `save_config_sidecar(...)` | Write the resolved-config `config.json` sidecar (carries `observed_config_hash`). |
| `save_system(snapshot, dir)` | Write the `system.json` sidecar. |

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
    system.json          # system snapshot (save_system)
    timeseries.parquet   # GPU power/thermal/memory series (copied in when present)
```

`load_result` reads `result.json` and best-effort re-attaches the
`timeseries.parquet` and `system.json` sidecars; a missing or corrupt
sidecar degrades gracefully (warning, not error) rather than failing the load.

Study-level artefacts (the `_study-artefacts/` directory, `manifest.json`,
`equivalence_groups.json`) are written by the `study/` layer (the `ManifestWriter`
in `study/orchestration.py`), not here. Artefact filenames are defined in
`domain/bundle_artefacts.py`.

## Related

- `../domain/experiment.py` - the `ExperimentResult` model this package serialises.
- `../domain/bundle_artefacts.py` - shared bundle filename constants.
