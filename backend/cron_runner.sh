#!/bin/bash
# Scheduled post loop — runs agent_runner.py at regular intervals
# Interval is controlled by CRON_INTERVAL_SEC (default: 7200 = 2 hours)

INTERVAL="${CRON_INTERVAL_SEC:-7200}"

echo "=== cron_runner started (interval=${INTERVAL}s) ==="

while true; do
    echo "[$(date -Iseconds)] Running agent_runner.py --full ..."
    python agent_runner.py --full
    echo "[$(date -Iseconds)] Done. Sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
done
