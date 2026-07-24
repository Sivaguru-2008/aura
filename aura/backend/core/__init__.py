"""Core platform services: intake, routing, and the primitives both depend on.

``core`` knows nothing about any specific imaging domain. It may import from
``backend.models`` and ``backend.engines.base`` (the contract), but never from a
concrete engine — that direction of dependency is what keeps engines pluggable.
"""
