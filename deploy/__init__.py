"""Deployment helpers — preflight checks, model fetch, and post-deploy smoke test.

A package rather than loose scripts so they can be run as ``python -m deploy.preflight``
from the repository root. Run that way the root is ``sys.path[0]``, which is what makes
``import aura`` resolve without a PYTHONPATH entry or any sys.path manipulation.
"""
