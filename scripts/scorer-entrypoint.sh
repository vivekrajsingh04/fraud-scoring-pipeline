#!/usr/bin/env bash
# The scorer runs two things in one container: the HTTP API (for synchronous
# scoring and metrics) and the Kafka consumer (for the streaming path). They
# share one loaded ONNX session's worth of memory but are separate processes, so
# a slow HTTP client cannot stall stream consumption.
set -euo pipefail

python -m uvicorn fraudpipe.scorer.app:app --host 0.0.0.0 --port 8000 --workers 1 &
api_pid=$!

python -m fraudpipe.scorer.consumer &
consumer_pid=$!

# If either process dies, take the container down so the orchestrator restarts
# it, rather than silently serving with half the service missing.
trap 'kill -TERM $api_pid $consumer_pid 2>/dev/null || true' TERM INT
wait -n $api_pid $consumer_pid
exit $?
