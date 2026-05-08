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
  title     = {{LLenergyMeasure}: Measure how implementation choices drive
                LLM inference efficiency},
  year      = {2026},
  version   = {0.9.0},
  url       = {https://github.com/henrycgbaker/llenergymeasure},
  note      = {Pre-1.0 release. See GitHub releases for the current version.
                Multi-engine, methodology-first measurement framework for
                LLM inference efficiency.}
}
```

For a plain-text reference:

> Baker, H. C. G. (2026). *LLenergyMeasure: Measure how implementation choices
> drive LLM inference efficiency* (v0.9.0).
> https://github.com/henrycgbaker/llenergymeasure

---

## Citing the bundled AIEnergyScore dataset

LLenergyMeasure ships with the AIEnergyScore prompt dataset as its default
measurement corpus. If your results use the default dataset, also cite the
upstream project:

```bibtex
@misc{aienergyscore,
  title        = {{AIEnergyScore}: A standardised approach for evaluating
                  the energy efficiency of AI model inference},
  author       = {Luccioni, Alexandra Sasha and {Hugging Face}},
  year         = {2024},
  howpublished = {\url{https://huggingface.github.io/AIEnergyScore/}}
}
```

The methodology that informs AIEnergyScore is set out in:

```bibtex
@inproceedings{luccioni2024power,
  title     = {Power Hungry Processing: {Watts} Driving the Cost of AI Deployment?},
  author    = {Luccioni, Alexandra Sasha and Jernite, Yacine and Strubell, Emma},
  booktitle = {Proceedings of the 2024 ACM Conference on Fairness,
                Accountability, and Transparency (FAccT)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.16863}
}
```

For more about the dataset format and provenance, see
[Reference: dataset format](/reference/dataset-format).
