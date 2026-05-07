---
title: Citation
description: How to cite LLenergyMeasure in research.
---

# Citation

## How to cite

If you use LLenergyMeasure in research, please cite it as:

```bibtex
@software{baker2026llenergymeasure,
  author    = {Baker, Henry C. G.},
  title     = {{LLenergyMeasure}: Energy and efficiency measurement for LLM inference},
  year      = {2026},
  version   = {0.10.0},
  url       = {https://github.com/henrycgbaker/llenergymeasure},
  note      = {Pre-1.0 release. CLI-first benchmarking framework for LLM inference
               efficiency across heterogeneous runtimes.}
}
```

For a plain-text reference:

> Baker, H. C. G. (2026). *LLenergyMeasure: Energy and efficiency measurement
> for LLM inference* (v0.10.0). https://github.com/henrycgbaker/llenergymeasure

---

## Citing the bundled AIEnergyScore dataset

LLenergyMeasure ships with the AIEnergyScore prompt dataset as its default
measurement corpus. If your results use the default dataset, also cite the
upstream source:

```bibtex
@misc{lottick2019energy,
  title        = {Energy Usage Reports: Environmental awareness as part of
                  algorithmic accountability},
  author       = {Lottick, Kadan and Susai, Silvia and Friedler, Sorelle A.
                  and Wilson, Jonathan P.},
  year         = {2019},
  howpublished = {NeurIPS 2019 Workshop on Tackling Climate Change with ML},
  url          = {https://arxiv.org/abs/1910.08235}
}
```

For more about the dataset format and provenance, see
[Reference: dataset format](/reference/dataset-format).
