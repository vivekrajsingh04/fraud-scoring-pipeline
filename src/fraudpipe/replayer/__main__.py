"""Replayer service: real labelled CSV -> Kafka ``txns.raw``, time-compressed.

The dataset is real and labelled; only the *arrival timing* is simulated, and it
is simulated faithfully: the gap between consecutive events is the dataset's own
gap divided by ``--speedup``. A 3-month dataset at 100x replays in ~22 hours; a
burst in the data is still a burst in the stream.

Records are keyed by ``card_id`` so every event for a card lands in one
partition, which is what keeps per-card state local in the Spark job.
"""

from __future__ import annotations

import argparse
import logging
import time

from fraudpipe.common.logging import log_with, setup_logging
from fraudpipe.config import SETTINGS
from fraudpipe.replayer.loaders import load_sorted

log = logging.getLogger("replayer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fraudpipe-replayer")
    p.add_argument("--dataset", default="ieee", choices=["ieee", "sparkov", "normalized"])
    p.add_argument("--path", required=True, help="path to the source CSV")
    p.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    p.add_argument("--topic", default=SETTINGS.topic_raw)
    p.add_argument(
        "--speedup",
        type=float,
        default=SETTINGS.replay_speedup,
        help="wall-clock compression factor; 100 means 100x faster than reality",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = replay everything")
    p.add_argument(
        "--max-sleep",
        type=float,
        default=5.0,
        help="cap on a single inter-arrival sleep, so overnight gaps in the data "
        "do not stall a demo",
    )
    p.add_argument("--partitions", type=int, default=6)
    p.add_argument("--no-sleep", action="store_true", help="throughput benchmark mode")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(SETTINGS.log_level)

    rows = load_sorted(args.dataset, args.path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        log.error("no rows loaded from %s", args.path)
        return 1

    span_s = (rows[-1].ts_ms - rows[0].ts_ms) / 1000.0
    log_with(
        log,
        logging.INFO,
        "replay starting",
        rows=len(rows),
        dataset=args.dataset,
        fraud_rate=round(sum(1 for r in rows if r.label) / len(rows), 5),
        dataset_span_hours=round(span_s / 3600.0, 2),
        speedup=args.speedup,
        projected_replay_minutes=round(span_s / max(args.speedup, 1e-9) / 60.0, 2),
    )

    # Imported here, not at module scope, so `parse_args` and the loaders stay
    # unit-testable without a Kafka client installed.
    from fraudpipe.common.kafka import ensure_topics, make_producer

    ensure_topics(args.bootstrap, [args.topic], partitions=args.partitions)
    producer = make_producer(args.bootstrap)

    sent = 0
    started = time.monotonic()
    prev_ts = rows[0].ts_ms
    try:
        for txn in rows:
            if not args.no_sleep:
                gap_s = (txn.ts_ms - prev_ts) / 1000.0 / args.speedup
                if gap_s > 0:
                    time.sleep(min(gap_s, args.max_sleep))
            prev_ts = txn.ts_ms
            # key = card_id -> partition affinity -> local per-card state
            producer.send(args.topic, key=txn.card_id, value=txn.to_json())
            sent += 1
            if sent % 10_000 == 0:
                elapsed = time.monotonic() - started
                log_with(
                    log,
                    logging.INFO,
                    "replay progress",
                    sent=sent,
                    eps=round(sent / max(elapsed, 1e-9), 1),
                )
    finally:
        producer.flush()
        producer.close()

    elapsed = time.monotonic() - started
    log_with(
        log,
        logging.INFO,
        "replay complete",
        sent=sent,
        seconds=round(elapsed, 2),
        eps=round(sent / max(elapsed, 1e-9), 1),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
