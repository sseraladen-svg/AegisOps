#!/bin/sh
# Container entrypoint: make `docker compose up` a genuine one-command run.
#
# The API is useless against an empty database — every endpoint 404s or 503s —
# so if no telemetry exists yet, generate it and run the detectors before
# serving. The check is on the database file, not a flag, so a rebuild over an
# existing volume doesn't redo ~20 seconds of work on every restart.
set -e

if [ ! -f /app/telemetry.db ]; then
  echo "no telemetry.db found — running the pipeline before serving"
  # --no-investigate: the agent stage needs an API key, and a container that
  # refused to start over a missing credential would be worse than one that
  # starts with no reports. Run `make investigate` (or the pipeline with a key)
  # to add them.
  python -m app.pipeline --no-investigate
else
  echo "telemetry.db present — skipping the pipeline"
fi

exec "$@"
