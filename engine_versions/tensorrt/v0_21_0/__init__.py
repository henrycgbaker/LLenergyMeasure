"""Frozen TensorRT-LLM 0.21.0 N-1 window: landmarks.yaml + outputs/ corpus only.

Prior-pin snapshot retained as the decay re-gate + surface-trend window
(see scripts/engine_producers/_current.previous_pin_outputs_dir). The
producers/ walker was removed in the producer dedup: the current pin's
walker is version-agnostic and would mine this surface from landmarks.yaml
if it ever needed to. Nothing imports this package; it is read by path
(the landmarks.yaml loader + the outputs reader).
"""
