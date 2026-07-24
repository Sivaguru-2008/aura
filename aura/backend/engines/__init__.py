"""Analysis engines. One package per imaging domain.

An engine is the *only* place clinical interpretation happens. The router selects
one; the dispatch service runs it; nothing else in the backend knows what a lung or a
ventricle is.

Engines never import each other. They depend only on ``engines.base`` (the contract)
and on whatever domain code they own — which for ``thorax`` means the pre-existing,
unmodified ``services.*`` / ``gateway.pipeline`` stack.
"""
