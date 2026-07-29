#!/usr/bin/env bash
# AURA container entrypoint.
#
#   serve      (default) preflight, then uvicorn on 0.0.0.0:${PORT:-8000}
#   preflight  run the checks and exit
#   train      train fusion + vision, then exit
#   <other>    exec'd verbatim, so `docker run aura bash` still works
set -euo pipefail

cd /app

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Written as explicit if-blocks rather than `[ test ] && arr+=(...)`: bash
# exempts a failing non-final command in an && list from `set -e`, so the short
# form does work here, but it is exactly the kind of subtlety that breaks the
# moment someone moves it into a function.
preflight_args=()
if [ "${AURA_SKIP_BRAIN:-0}" = "1" ]; then
    preflight_args+=(--skip-brain)
fi
# AURA_PREFLIGHT=warn keeps a degraded container up (useful for debugging a bad
# deploy); anything else means a failed check stops the container.
if [ "${AURA_PREFLIGHT:-strict}" = "warn" ]; then
    preflight_args+=(--warn-only)
fi

run_preflight() {
    if [ "${AURA_PREFLIGHT:-strict}" = "skip" ]; then
        echo "[entrypoint] AURA_PREFLIGHT=skip - not verifying the served models."
        return 0
    fi
    python -m deploy.preflight "${preflight_args[@]}"
}

case "${1:-serve}" in
    serve)
        run_preflight
        echo "[entrypoint] starting uvicorn on ${HOST}:${PORT}"
        # --workers is deliberately 1. The gateway keeps per-process state in a
        # module-level dict and binds the mock PACS listener to :11112 during
        # lifespan; a second worker would race on both and fail to bind.
        exec python -m uvicorn aura.gateway.app:app \
            --app-dir /app \
            --host "${HOST}" \
            --port "${PORT}" \
            --workers 1 \
            --timeout-keep-alive 75 \
            --log-level "${AURA_LOG_LEVEL:-info}"
        ;;
    preflight)
        exec python -m deploy.preflight "${preflight_args[@]}"
        ;;
    train)
        cd /app
        python -m aura.aura_cli train
        exec python -m aura.aura_cli bench
        ;;
    *)
        exec "$@"
        ;;
esac
