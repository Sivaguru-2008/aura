"""HTTP surface for the routing layer.

Mounted onto the existing gateway app rather than served separately, so the router
inherits the gateway's audit middleware, security enforcement, and CORS posture
without duplicating any of it.
"""

from .routes import build_router

__all__ = ["build_router"]
