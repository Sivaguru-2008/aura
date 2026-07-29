"""AURA — multimodal clinical decision-support platform.

This is the project's single import root. Every first-party module is addressed
under it (``aura.common.config``, ``aura.gateway.app``, ``aura.backend.…``), so the
package resolves identically whether it is run from a checkout::

    python -m aura.aura_cli serve

installed into an environment, or copied into a container. Nothing is imported
here on purpose: the subpackages pull in torch, cv2 and the quantum stack, and
paying that cost just to reach ``aura.common.config`` would make CLI start-up and
the container health check needlessly slow.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
