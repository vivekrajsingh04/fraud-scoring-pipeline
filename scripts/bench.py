#!/usr/bin/env python3
"""Measure sustained throughput and end-to-end latency percentiles.

Produces the two tables the README quotes. Both numbers are measured, not
estimated:

* **Throughput** -- the replayer runs with ``--no-sleep`` so it publishes as
  fast as the broker accepts, and we measure how quickly decisions accumulate in
  Postgres. That makes the pipeline, not the replayer, the bottleneck.
* **End-to-end latency** -- ``scored_at - event_ts``, read from Postgres. This
  spans replayer publish, Kafka, the Spark micro-batch, the scorer's inference
  and the database write. Quoting inference latency alone would be flattering
  and meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# Both the repo root (for `scripts.`) and src/ (for `fraudpipe.`), so this runs
# as `python scripts/bench.py` from anywhere without an editable install.
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from fraudpipe.config import SETTINGS  # noqa: E402
from scripts.smoke_compose import fetch_decisions, make_events, publish  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def main() -> int:
    p = argparse.ArgumentParser(prog="bench")
    p.add_argument("--events", type=int, default=20_000)
    p.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    p.add_argument("--topic", default=SETTINGS.topic_raw)
    p.add_argument("--dsn", default=SETTINGS.postgres_dsn)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--out", default="artifacts/bench.json")
    args = p.parse_args()

    run_id = f"bench{int(time.time())}"
    events = make_events(args.events, run_id, seed=1234)

    print(f"publishing {args.events} events as fast as the broker accepts...")
    t_publish_start = time.monotonic()
    publish(events, args.bootstrap, args.topic)
    publish_s = time.monotonic() - t_publish_start
    print(
        f"  published in {publish_s:.1f}s "
        f"({args.events / max(publish_s, 1e-9):,.0f} events/sec into Kafka)"
    )

    deadline = time.monotonic() + args.timeout
    first_seen = None
    rows: list[dict] = []
    while time.monotonic() < deadline:
        rows = fetch_decisions(args.dsn, run_id)
        if rows and first_seen is None:
            first_seen = time.monotonic()
        if len(rows) >= args.events:
            break
        time.sleep(2.0)
        print(f"  {len(rows):,}/{args.events:,} decisions", flush=True)

    drain_s = time.monotonic() - (first_seen or t_publish_start)
    latencies = [r["e2e_ms"] for r in rows]

    result = {
        "events_published": args.events,
        "decisions_observed": len(rows),
        "completeness": round(len(rows) / args.events, 4),
        "publish_seconds": round(publish_s, 2),
        "publish_events_per_sec": round(args.events / max(publish_s, 1e-9), 1),
        "pipeline_drain_seconds": round(drain_s, 2),
        "sustained_events_per_sec": round(len(rows) / max(drain_s, 1e-9), 1),
        "e2e_latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "p99": round(percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1) if latencies else None,
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if len(rows) < args.events:
        print(
            f"WARNING: only {len(rows)}/{args.events} decisions observed before "
            f"the {args.timeout:.0f}s timeout -- the numbers above understate "
            f"latency and overstate completeness"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
