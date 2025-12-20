"""Base classes and protocol for energy backends.

Re-exports the EnergyBackend protocol from protocols.py for convenience.
"""

from llm_energy_measure.protocols import EnergyBackend

__all__ = ["EnergyBackend"]
