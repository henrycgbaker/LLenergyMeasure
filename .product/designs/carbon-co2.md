# Carbon & CO2 Estimation Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/carbon-intensity.md](../decisions/carbon-intensity.md)
**Status**: Revised — static lookup table removed; lookup delegated to CodeCarbon (2026-02-25)

---

## Core Framing

CO2 estimation is a **derived quantity**, not a measurement. The tool measures joules (energy).
CO2 is then: `co2_grams = energy_kwh × grid_carbon_intensity_gco2_kwh × pue`.

This distinction matters for scientific honesty. Results clearly label CO2 as an estimate
derived from grid intensity data — not a direct measurement.

The tool never silently invents a CO2 number. If no intensity source is available,
`co2_estimate` is `None`.

---

## Three Paths to a CO2 Figure

1. **User-specified intensity** — set `measurement.carbon_intensity_gco2_kwh` in user config
   or `LLEM_CARBON_INTENSITY` env var. Base package does the arithmetic. No CodeCarbon needed.
2. **CodeCarbon installed** (`pip install llenergymeasure[codecarbon]`) — CodeCarbon handles
   regional lookup and location detection automatically. Its figure is stored separately.
3. **Neither** — `co2_estimate: None`. No CO2 figure produced.

---

## Config Fields

Infrastructure context fields live in user config (`~/.config/llenergymeasure/config.yaml`),
not in experiment/study YAML (experiment configs are infrastructure-agnostic):

```yaml
# ~/.config/llenergymeasure/config.yaml
measurement:
  carbon_intensity_gco2_kwh: 385    # gCO2/kWh for your electricity grid
  datacenter_pue: 1.2               # power usage effectiveness (default: 1.0)
```

`carbon_intensity_gco2_kwh` is required for base-package CO2 estimation (without CodeCarbon).
`datacenter_pue` defaults to 1.0 (no overhead) — only needs setting for datacentre/HPC.

Env var overrides: `LLEM_CARBON_INTENSITY`, `LLEM_DATACENTER_PUE`.

---

## CO2Estimate Result Model

```python
# src/llenergymeasure/domain/co2.py

from pydantic import BaseModel
from typing import Literal

EstimationMethod = Literal[
    "user_specified",   # user set carbon_intensity_gco2_kwh directly
    "codecarbon",       # CodeCarbon handled lookup + location detection
]


class CO2Estimate(BaseModel):
    co2_grams: float
    grid_carbon_intensity_gco2_kwh: float   # value actually used
    pue: float                               # value actually used
    estimation_method: EstimationMethod
```

Stored in `ExperimentResult.co2_estimate: CO2Estimate | None`.
`None` when no intensity source is available.

---

## Calculation (Base Package)

```python
# src/llenergymeasure/domain/co2.py

def estimate_co2(
    energy_joules: float,
    carbon_intensity_gco2_kwh: float | None,
    pue: float = 1.0,
) -> CO2Estimate | None:
    """Returns None if carbon intensity is not specified."""
    if carbon_intensity_gco2_kwh is None:
        return None

    energy_kwh = energy_joules / 3_600_000
    effective_energy_kwh = energy_kwh * pue
    co2_grams = effective_energy_kwh * carbon_intensity_gco2_kwh

    return CO2Estimate(
        co2_grams=co2_grams,
        grid_carbon_intensity_gco2_kwh=carbon_intensity_gco2_kwh,
        pue=pue,
        estimation_method="user_specified",
    )
```

This is intentionally simple — just arithmetic. All lookup complexity is delegated to
CodeCarbon.

---

## Relationship to CodeCarbon

When `llenergymeasure[codecarbon]` is installed and enabled, CodeCarbon computes its own CO2
figure using its built-in regional lookup tables and automatic IP-based location detection.

Both figures are stored separately in `ExperimentResult`:
- `co2_estimate: CO2Estimate | None` — base package calculation (user-specified intensity)
- `codecarbon_co2_grams: float | None` — CodeCarbon's calculation (if installed)

Both are preserved when available — they use different methodologies and are useful for
cross-validation.

```python
class ExperimentResult(BaseModel):
    ...
    co2_estimate: CO2Estimate | None = None       # base package (user-specified intensity)
    codecarbon_co2_grams: float | None = None     # CodeCarbon estimate (if installed)
```

---

## What Is NOT Supported

- **Static lookup table in base package** — removed (2026-02-25). CodeCarbon handles this
  with better regional granularity. See decision rationale in `decisions/carbon-intensity.md`.
- **Real-time carbon intensity API** (Electricity Maps, WattTime) — breaks air-gapped HPC.
- **`datacenter_location` field in base package** — removed. Location-based lookup is
  CodeCarbon's responsibility. The base package only does arithmetic with user-specified values.

---

## Related

- [../decisions/carbon-intensity.md](../decisions/carbon-intensity.md): Decision rationale
- [user-config.md](user-config.md): `measurement.carbon_intensity_gco2_kwh` and `datacenter_pue` fields
- [result-schema.md](result-schema.md): `co2_estimate` in ExperimentResult
