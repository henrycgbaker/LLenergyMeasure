# engine_versions workspace

This directory is the single source of truth for what llem knows about each
supported inference engine at each version it supports. It is a workspace: the
files here are the inputs that later steps read to generate the typed
configuration models and validation rules that ship inside the package. The
directory lives at the repository root, outside `src/`, so it is never bundled
into the built wheel.

## Layout

```
engine_versions/
  <engine>/
    current.yaml                     which version is deployed right now
    v<safe_version>/
      outputs/
        schema.discovered.json       full config surface for this version
        curated.yaml                 which of those fields llem exposes
```

`<engine>` is one of `vllm`, `tensorrt`, `transformers`. `<safe_version>` is
the version number with dots turned into underscores and a leading `v`, so
`0.7.3` becomes `v0_7_3`. A version directory is a snapshot: several can exist
side by side, one per version the project keeps knowledge for.

### The two mined files

- `schema.discovered.json` is the complete configuration surface for one
  engine version: every field the engine accepts, with its type, bounds, and
  allowed values. It is produced by inspecting the engine and is treated as
  data, not hand-edited.
- `curated.yaml` is the maintainer's allowlist. Of all the fields the schema
  discovered, it names the subset llem promotes to first-class typed fields.
  Curation is an exposure decision made here; discovery never narrows.

Both files carry the engine version they describe in an `engine_version`
field, so a snapshot is self-identifying.

### current.yaml

`current.yaml` records the single version of each engine that is deployed
today, under `library.current_version`. It is the pointer; the version
directories are the snapshots it can point at. Renovate updates this pointer
when a new engine release lands.

## Active snapshots

| engine       | versions in the workspace |
| ------------ | ------------------------- |
| vllm         | 0.7.3, 0.19.1             |
| tensorrt     | 0.21.0, 1.0.0             |
| transformers | 5.7.0                     |

## Resolving paths in code

`_outputs.py` is the one module that knows this layout. It is a plain path
resolver with no engine imports and no dynamic loading:

```python
from engine_versions import _outputs

_outputs.schema_path("vllm", "0.7.3")     # .../vllm/v0_7_3/outputs/schema.discovered.json
_outputs.curated_path("vllm", "0.7.3")    # .../vllm/v0_7_3/outputs/curated.yaml
_outputs.workspace_versions("vllm")       # ["0.7.3", "0.19.1"]
```

`workspace_versions` reports which versions are present by scanning for
snapshots that carry both mined files, so the directory tree is the source of
truth and there is no list to keep in sync by hand.
