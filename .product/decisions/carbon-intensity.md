# Carbon Intensity & CO2 Estimation

**Status:** Accepted
**Date decided:** 2026-02-25
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

- Carbon intensity lookup delegated to CodeCarbon (`[codecarbon]` optional extra). Base package has no static lookup table. 
- Three paths to CO2: (1) user-specified `grid_carbon_intensity_gco2_kwh` → base arithmetic, (2) CodeCarbon installed → its lookup, (3) neither → `co2_estimate: None`. Tool never invents a CO2 number.

---

## Context

LLenergyMeasure measures joules (energy). CO2 is a derived quantity, not a direct measurement: `co2_grams = energy_kwh × grid_carbon_intensity_gco2_kwh`. To compute it, the tool needs a grid carbon intensity value for the user's location.

Key constraints shaping this decision:
- Target environments include air-gapped HPC clusters — no network calls at experiment time.
- Scientific honesty requires clearly labelling CO2 as an estimate, not a measurement.
- The base package must remain lightweight — no heavy external API dependencies.
- CodeCarbon (`[codecarbon]` optional extra) already handles its own CO2 calculation with regional lookup tables and IP-based location detection. We should not duplicate this.

This is already reflected in the architecture decision: "CO2 decoupled from energy measurement" — CodeCarbon provides CO2 *estimates* (`[codecarbon]` optional extra); the base package provides raw NVML energy. These are independent concerns.

## Considered Options

### Source of grid carbon intensity values

| Option | Pros | Cons |
|--------|------|------|
| **Delegate lookup to CodeCarbon; base package supports user-specified override only** | No maintenance burden for lookup tables. CodeCarbon already has granular regional data. Base package stays minimal. Users who know their grid intensity can still get CO2 without CodeCarbon. | No automatic CO2 without `[codecarbon]` installed — users must either install CodeCarbon or manually specify intensity. |
| Real-time API (Electricity Maps, WattTime) | Most accurate current intensity. | Breaks in air-gapped HPC. Network call during experiment introduces latency and failure mode. |
| Static lookup table built into base package | No network dependency. Works in air-gapped HPC. | Duplicates CodeCarbon's functionality. Maintenance burden for data updates. Values are annual averages — not real-time. Regional granularity limited. |

> **Superseded (2026-02-25):** Static lookup table in base package was the original decision (2026-02-19). Revised to delegate lookup entirely to CodeCarbon — avoids duplicating functionality that CodeCarbon already provides with better regional granularity. User-specified `grid_carbon_intensity_gco2_kwh` override is retained in the base package (just arithmetic, no lookup table needed).

### User override behaviour

| Option | Pros | Cons |
|--------|------|------|
| **User-specified `grid_carbon_intensity_gco2_kwh` always takes highest priority** | User knows their actual grid (renewable contract, university cluster PPA) better than any lookup. Works without CodeCarbon — just arithmetic. | None — highest precision source should always win. |
| Lookup always used; user value is advisory | Consistent methodology across runs. | Produces wrong CO2 figures for users with accurate intensity data. |

### Storing CO2 in results

| Option | Pros | Cons |
|--------|------|------|
| **`CO2Estimate` sub-model with explicit `estimation_method` field** | Consumers can filter by method. `"user_specified"` vs `"codecarbon"` are distinguishable. | Slightly more verbose result schema. |
| Single `co2_grams` float field | Simpler. | No way to know which methodology produced the figure — cross-study comparison breaks. |
| Omit CO2 from base result entirely | Keeps base package minimal. | Users who specify intensity manually get no CO2 figure. |

### Behaviour when no intensity source is available

| Option | Pros | Cons |
|--------|------|------|
| **`co2_estimate: None` — field omitted** | Scientific honesty — tool never invents a CO2 number without a basis. | Users without CodeCarbon or manual intensity get no CO2 figure. |
| Use global average silently | Always produces a CO2 figure. | Silent fabrication — users may not realise CO2 is based on a global average. Violates scientific honesty requirement. |
| Global average with warning | Produces a figure; warns user. | Warning fatigue. Still invents a figure the user may not want. |

### Rejected Options

**Rejected (2026-02-25): Static lookup table in base package** — Duplicates CodeCarbon's functionality with less regional granularity. Maintenance burden for data updates each release. CodeCarbon already provides this with better coverage and automatic location detection.

**Rejected (2026-02-19): Real-time API (Electricity Maps, WattTime)** — Breaks air-gapped HPC. Network call during experiment introduces latency and failure mode.

## Decision

Carbon intensity lookup is delegated entirely to CodeCarbon (`[codecarbon]` optional extra). The base package does **not** ship a static lookup table.

The base package retains only:
- Simple `energy × intensity × PUE` arithmetic when the user manually specifies `grid_carbon_intensity_gco2_kwh` in user config or via `LLEM_CARBON_INTENSITY` env var
- The `CO2Estimate` result model with `estimation_method` field
- `co2_estimate: None` when no intensity source is available

Three paths to a CO2 figure:
1. **User-specified intensity** — set `grid_carbon_intensity_gco2_kwh` in user config. Base package does the arithmetic. No CodeCarbon needed.
2. **CodeCarbon installed** — CodeCarbon handles lookup and location detection automatically. Its figure is stored in `result.codecarbon_co2_grams`.
3. **Neither** — `co2_estimate: None`. The tool never invents a CO2 number.

Rationale: CodeCarbon already maintains regional carbon intensity data with better coverage than we could provide. Duplicating this functionality adds maintenance burden with no benefit. Users who know their grid intensity can still get CO2 figures from the base package via simple arithmetic.

## Consequences

Positive:
- No lookup table to maintain — eliminates per-release data update burden.
- CodeCarbon provides better regional granularity than our static table would have.
- Base package stays minimal; CO2 arithmetic for user-specified intensity is trivial.
- `estimation_method` field enables downstream filtering and cross-study comparability.
- Scientific honesty preserved — tool never silently invents a CO2 number.

Negative / Trade-offs:
- No automatic CO2 without CodeCarbon or manual intensity — users must take one extra step.
- CodeCarbon's IP-based detection requires network access — does not work in air-gapped HPC.
  Air-gapped users must specify `grid_carbon_intensity_gco2_kwh` manually.

Neutral / Follow-up decisions triggered:
- `estimation_method` vocabulary must be kept stable (or versioned) for cross-study consumers.

## Config Integration

Infrastructure context fields in user config (`~/.config/llenergymeasure/config.yaml`):

```yaml
measurement:
  carbon_intensity_gco2_kwh: 385    # gCO2/kWh — user-specified grid intensity
  datacenter_pue: 1.2               # power usage effectiveness (default: 1.0)
```

**Resolution order:**
1. `grid_carbon_intensity_gco2_kwh` explicit value (user config or env var) — base package arithmetic
2. CodeCarbon lookup (if installed) — CodeCarbon handles location detection
3. `co2_estimate: None` when neither is available

**PUE adjustment:**
```
effective_energy_kwh = measured_energy_kwh × pue
co2_grams = effective_energy_kwh × grid_carbon_intensity_gco2_kwh
```

## Result Fields

See `designs/result-schema.md` for the full `CO2Estimate` model definition. Summary:

- `co2_grams: float`
- `grid_carbon_intensity_gco2_kwh: float` — value actually used
- `pue: float` — value actually used
- `estimation_method: str` — `"user_specified"` | `"codecarbon"`

Stored in `ExperimentResult.co2_estimate: CO2Estimate | None`. `None` when no intensity source is available.

When CodeCarbon is installed, its figure is also stored separately:
- `codecarbon_co2_grams: float | None` — CodeCarbon's own calculation

Both are preserved when both sources are available — they use different methodologies and are useful for cross-validation.

## Deferred

- Real-time carbon intensity API (Electricity Maps, WattTime) — deferred indefinitely.
  Would break air-gapped HPC.

## Related

- [architecture.md](architecture.md) — CO2 decoupled from energy measurement (Sub-decision G)
- [installation.md](installation.md) — `[codecarbon]` as optional extra
- [../designs/carbon-co2.md](../designs/carbon-co2.md) — CO2 estimation design (`CO2Estimate` model, calculation)
- [../designs/result-schema.md](../designs/result-schema.md) — `CO2Estimate` model in `ExperimentResult`
- [../designs/user-config.md](../designs/user-config.md) — `measurement.carbon_intensity_gco2_kwh` field
- [../designs/energy-backends.md](../designs/energy-backends.md) — energy backend plugin system (separate concern)
