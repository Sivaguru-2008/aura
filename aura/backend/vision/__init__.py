"""AURA perception layer.

Sibling of :mod:`aura.backend.foundation`, and the split between them is the point:
``foundation`` makes a study trustworthy and says what was done to it; ``vision`` looks
at the pixels. One module today — :mod:`aura.backend.vision.brain`, the NeuroMind Brain
Vision Engine. A thorax counterpart would live beside it; the chest stack's existing
models are not moved here, for the reasons ``backend/README.md`` gives about relocating
working code.
"""
