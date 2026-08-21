#!/usr/bin/env python3
"""Compose smoke test: publish N events, assert N decisions land in Postgres.

This is the test that proves the wiring, not the logic. It runs in CI after
``docker compose up`` and fails the build if a single event is dropped between
Kafka, Spark, the scorer and Postgres.

It asserts three things beyond the row count, because "100 rows exist" is a
weaker claim than it looks:

1. The decision ids are exactly the ids published -- not 100 arbitrary rows.
2. Every decision carries a score in [0, 1] and a valid decision label.
3. Republishing the same events does not create duplicates, which exercises the
   at-least-once + ON CONFLICT path deliberately rather than hoping it is never
   hit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudpipe.common.kafka import ensure_topics, make_producer  # noqa: E402
from fraudpipe.config import SETTINGS  # noqa: E402
from fraudpipe.schemas import Transaction  # noqa: E402

CITIES = [(12.9716, 77.5946), (19.0760, 72.8777), (28.6139, 77.2090)]
MCCS = ["5411", "5812", "5541", "7995"]


def make_events(n: int, run_id: str, seed: int = 42) -> list[Transaction]:
    rng = random.Random(seed)
    now_ms = int(time.time() * 1000)
    rows = []
    for i in range(n):
        lat, lon = CITIES[rng.randrange(len(CITIES))]
        rows.append(
            Transaction(
                txn_id=f"{run_id}-{i:05d}",
                card_id=f"{run_id}-card-{i % 12:02d}",
                ts_ms=now_ms + i * 10,
                amount=round(rng.lognormvariate(4.0, 1.0), 2),
                mcc=rng.choice(MCCS),
                merchant_id=f"m{rng.randrange(20):02d}",
                lat=lat,
                lon=lon,
                label=1 if rng.random() < 0.05 else 0,
            )
        )
    return rows


def publish(events: list[Transaction], bootstrap: str, topic: str) -> None:
    ensure_topics(bootstrap, [topic])
    producer = make_producer(bootstrap)
    for txn in events:
        producer.send(topic, key=txn.card_id, value=txn.to_json())
    producer.flush()
    producer.close()


def fetch_decisions(dsn: str, run_id: str) -> list[dict]:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT txn_id, score, decision, "
            "EXTRACT(EPOCH FROM (scored_at - event_ts)) * 1000 "
            "FROM decisions WHERE txn_id LIKE %s",
            (f"{run_id}-%",),
        )
        return [
            {"txn_id": r[0], "score": float(r[1]), "decision": r[2], "e2e_ms": float(r[3])}
            for r in cur.fetchall()
        ]


def wait_for(dsn: str, run_id: str, expected: int, timeout_s: float) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    rows: list[dict] = []
    while time.monotonic() < deadline:
        try:
            rows = fetch_decisions(dsn, run_id)
        except Exception as exc:
            print(f"  waiting for postgres: {exc}", flush=True)
        if len(rows) >= expected:
            return rows
        time.sleep(2.0)
        print(
            f"  {len(rows)}/{expected} decisions after "
            f"{timeout_s - (deadline - time.monotonic()):.0f}s",
            flush=True,
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(prog="smoke_compose")
    p.add_argument("--events", type=int, default=100)
    p.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    p.add_argument("--topic", default=SETTINGS.topic_raw)
    p.add_argument("--dsn", default=SETTINGS.postgres_dsn)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    run_id = f"smoke{int(time.time())}"
    events = make_events(args.events, run_id)

    print(f"publishing {len(events)} events to {args.topic} (run_id={run_id})")
    publish(events, args.bootstrap, args.topic)

    rows = wait_for(args.dsn, run_id, args.events, args.timeout)

    failures = []
    if len(rows) != args.events:
        failures.append(f"expected {args.events} decisions, got {len(rows)}")

    got_ids = {r["txn_id"] for r in rows}
    want_ids = {t.txn_id for t in events}
    if got_ids != want_ids:
        missing = sorted(want_ids - got_ids)[:5]
        extra = sorted(got_ids - want_ids)[:5]
        failures.append(f"id mismatch; missing={missing} unexpected={extra}")

    bad = [r for r in rows if not (0.0 <= r["score"] <= 1.0)]
    if bad:
        failures.append(f"{len(bad)} decisions have out-of-range scores")

    bad_labels = [r for r in rows if r["decision"] not in ("APPROVE", "REVIEW")]
    if bad_labels:
        failures.append(f"{len(bad_labels)} decisions have an invalid label")

    # Redelivery must be absorbed, not duplicated.
    print("republishing the same events to exercise the idempotent sink")
    publish(events, args.bootstrap, args.topic)
    time.sleep(20.0)
    after = fetch_decisions(args.dsn, run_id)
    if len(after) != args.events:
        failures.append(f"replay created duplicates: {len(after)} rows for {args.events} events")

    if rows:
        lat = sorted(r["e2e_ms"] for r in rows)
        summary = {
            "decisions": len(rows),
            "review_rate": round(sum(r["decision"] == "REVIEW" for r in rows) / len(rows), 4),
            "e2e_p50_ms": round(lat[len(lat) // 2], 1),
            "e2e_p95_ms": round(lat[int(len(lat) * 0.95)], 1),
            "e2e_p99_ms": round(lat[int(len(lat) * 0.99)], 1),
        }
        print(json.dumps(summary, indent=2))

    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"SMOKE TEST PASSED: {args.events} events -> {len(rows)} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
