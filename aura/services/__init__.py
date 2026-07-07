"""AURA intelligence engines. Each subpackage is independently replaceable and
implements a small, typed contract from `schemas`. No engine imports another;
they compose only through the gateway pipeline and the event bus.
"""
