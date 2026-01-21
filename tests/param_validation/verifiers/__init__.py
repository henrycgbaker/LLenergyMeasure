"""Verifiers package for parameter validation.

Provides verification strategies for different test types:
- Passthrough: Verify param value reaches backend config
- Behaviour: Verify observable output/performance change
- Introspection: Inspect model/engine state
- Mock: CI-safe verification via patching
"""

from .behaviour import BehaviourVerifier
from .introspection import IntrospectionVerifier
from .mock_verifiers import MockVerifier
from .passthrough import PassthroughVerifier

__all__ = [
    "BehaviourVerifier",
    "IntrospectionVerifier",
    "MockVerifier",
    "PassthroughVerifier",
]
