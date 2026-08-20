"""Serving layer: engine-server placement, launch, readiness and transport.

Owns everything that is identical for every OpenAI-compatible inference server,
so no engine adapter re-implements it and no consumer above reaches into the
container plumbing to get it. Split by concern, and imported per concern - there
is deliberately no package-level re-export, so taking the vocabulary does not
drag in the mechanism:

- :mod:`llenergymeasure.serving.types` - the vocabulary. Placement, handle, probe
  shape and the four lifecycle errors, with no mechanism attached. Import this to
  name a server in an annotation, a constructor call or an ``except`` clause.
- :mod:`llenergymeasure.serving.lifecycle` - the mechanism. Port allocation,
  container/process launch, the readiness wait, and leak-free shutdown.
- :mod:`llenergymeasure.serving.transport` - the wire. One request out, one
  streamed completion back, for the load issuer above to schedule.

Per-engine knowledge (the serve command, the probe body) stays in the engine
adapters, which compose these primitives through the ``ServerCapable`` protocol
extension.
"""
