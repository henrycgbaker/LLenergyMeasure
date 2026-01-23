"""Base classes and protocol for energy backends.

Re-exports the EnergyBackend protocol from protocols.py for convenience.
"""

from llenergymeasure.protocols import EnergyBackend

__all__ = ["EnergyBackend"]
