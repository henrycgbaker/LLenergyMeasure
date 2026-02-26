# Designs

**SSOT for**: How to implement decisions — detailed specs, schemas, code examples, diagrams.
**Upstream**: [../decisions/](../decisions/) (what + why), [../research/](../research/) (peer code + evidence)
**Downstream**: GSD planning phases (implementation)

**Last updated**: 2026-02-25
**Status**: v2.0 target designs. Not yet implemented.

## Writing Rules for This Directory

- **Reference decisions, don't repeat rationale**: "Why X?" → cite `decisions/topic.md`. Don't re-explain here.
- **Reference research, don't duplicate**: Peer code examples and benchmark data belong in `research/`. Link to them.
- **Detail is appropriate here**: YAML shapes, Pydantic field types, code stubs, diagrams — all welcome.
- **DRY across design docs**: If two design docs share a concept (e.g., config hash), the canonical definition lives in one file; others reference it.
- **Grounded**: Any design choice not obvious from first principles must cite a peer pattern or `research/` file.

---

## Files

| File | Contents | Version | Last Updated |
|------|----------|---------|--------------|
| [architecture.md](architecture.md) | Module layout, public API, call graph, config model, state machine, NVML session ownership | v2.0 | 2026-02-25 |
| [cli-commands.md](cli-commands.md) | 2 commands + 1 flag; full signatures; zero-config invocation; rename `lem`→`llem` | v2.0 | 2026-02-19 |
| [library-api.md](library-api.md) | `run_experiment`/`run_study` overloads; side-effect-free principle; peer comparison | v2.0 | 2026-02-25 |
| [experiment-config.md](experiment-config.md) | Full `ExperimentConfig` schema; composition architecture; backend sections | v2.0 | 2026-02-25 |
| [config-model.md](config-model.md) | SSOT field placement; data flow; hash semantics; precedence chains | v2.0 | 2026-02-25 |
| [study-yaml.md](study-yaml.md) | `StudyConfig` schema; `sweep:` grammar; `execution:` block | v2.0 | 2026-02-25 |
| [result-schema.md](result-schema.md) | `ExperimentResult`/`StudyResult` schema; 3 new fields; Parquet export; peer comparison | v2.0 | 2026-02-19 |
| [user-config.md](user-config.md) | User config (`~/.config/llenergymeasure/config.yaml`); runner mappings; execution defaults | v2.0 | 2026-02-19 |
| [packaging.md](packaging.md) | `pyproject.toml` structure; optional extras; entry points | v2.0 | 2026-02-19 |
| [dataset.md](dataset.md) | Built-in datasets; `load_dataset()` dispatcher; synthetic generation | v2.0 | 2026-02-19 |
| [experiment-isolation.md](experiment-isolation.md) | Subprocess isolation pattern; `spawn` context; Pipe communication | v2.0 | 2026-02-19 |
| [testing.md](testing.md) | Test strategy; backend mocking; fixture patterns | v2.0 | 2026-02-19 |
| [observability.md](observability.md) | Rich progress display; structured logging; experiment lifecycle events | v2.0 | 2026-02-19 |
| [reproducibility.md](reproducibility.md) | `config_hash`; environment snapshot; determinism guarantees | v2.0 | 2026-02-19 |
| [carbon-co2.md](carbon-co2.md) | CO2 estimation; `CO2Estimate` model; regional intensity data | v2.0 | 2026-02-19 |
| [docker-execution.md](docker-execution.md) | Docker lifecycle; volume sharing; cold-start pattern | v2.0 (later milestone) | 2026-02-25 |
| [study-resume.md](study-resume.md) | Study resume from manifest; checkpoint pattern | v2.0 (later milestone) | 2026-02-25 |
| [schema-migration.md](schema-migration.md) | Result schema migration path; forward compatibility | v2.0 | 2026-02-19 |
| [energy-backends.md](energy-backends.md) | Energy backend plugin system; backend registry; accuracy table; NVML conflict | v2.0 | 2026-02-25 |
| [web-platform.md](web-platform.md) | Static leaderboard (v4.0), dynamic API (v4.1), live features (v4.2); tech stack | v4.0+ | 2026-02-19 |

---

## Rejected / Superseded Designs

| File | Contents | Reason |
|------|----------|--------|
| *(none yet)* | | |
