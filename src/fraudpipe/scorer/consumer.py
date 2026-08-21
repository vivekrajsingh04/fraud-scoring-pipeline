"""Kafka scoring worker: ``txns.featurized`` -> ONNX -> ``txns.scored`` + Postgres.

Delivery semantics, stated plainly because an interviewer will ask:

* Offsets are committed **after** the Postgres write and the Kafka produce, so
  the pipeline is at-least-once, never at-most-once.
* The Postgres insert is ``ON CONFLICT (txn_id) DO NOTHING``, so a redelivery
  after a crash is absorbed. Net effect: effectively-once *in the decisions
  table*, without needing Kafka transactions.
* Records are batched (default 200) because ONNX inference on a batch is far
  cheaper per row than one-at-a-time, and the 200ms SLO is measured end to end,
  not per call -- a 50ms batching window buys a large throughput multiple.
"""

from __future__ import annotations

import argparse
import logging
import time

from fraudpipe.common.kafka import ensure_topics, make_consumer, make_producer
from fraudpipe.common.logging import log_with, setup_logging
from fraudpipe.config import SETTINGS
from fraudpipe.schemas import Decision, FeatureVector
from fraudpipe.scorer.db import insert_decisions, make_pool
from fraudpipe.scorer.model import FraudModel

log = logging.getLogger("scorer.consumer")


def score_batch(model: FraudModel, threshold: float, fvs: list[FeatureVector]) -> list[Decision]:
    if not fvs:
        return []
    t0 = time.perf_counter()
    scores = model.score_batch([fv.as_model_input() for fv in fvs])
    per_row_ms = (time.perf_counter() - t0) * 1000.0 / len(fvs)
    now_ms = int(time.time() * 1000)
    return [
        Decision(
            txn_id=fv.txn_id,
            card_id=fv.card_id,
            ts_ms=fv.ts_ms,
            score=s,
            threshold=threshold,
            decision="REVIEW" if s >= threshold else "APPROVE",
            label=fv.label,
            model_version=model.version,
            scored_at_ms=now_ms,
            latency_ms=per_row_ms,
        )
        for fv, s in zip(fvs, scores, strict=True)
    ]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - compose smoke
    p = argparse.ArgumentParser(prog="fraudpipe-scorer-consumer")
    p.add_argument("--bootstrap", default=SETTINGS.kafka_bootstrap)
    p.add_argument("--group", default="fraudpipe-scorer")
    p.add_argument("--batch-max", type=int, default=200)
    p.add_argument("--poll-ms", type=int, default=50)
    args = p.parse_args(argv)
    setup_logging(SETTINGS.log_level)

    ensure_topics(args.bootstrap, [SETTINGS.topic_featurized, SETTINGS.topic_scored])
    model = FraudModel(SETTINGS.model_path, SETTINGS.model_version)
    pool = make_pool(SETTINGS.postgres_dsn)
    producer = make_producer(args.bootstrap)
    consumer = make_consumer(
        args.bootstrap,
        [SETTINGS.topic_featurized],
        args.group,
        max_poll_records=args.batch_max,
    )

    total = 0
    log.info("scorer consumer started threshold=%s", SETTINGS.decision_threshold)
    try:
        while True:
            polled = consumer.poll(timeout_ms=args.poll_ms, max_records=args.batch_max)
            fvs = [
                FeatureVector.from_json(msg.value) for records in polled.values() for msg in records
            ]
            if not fvs:
                continue

            decisions = score_batch(model, SETTINGS.decision_threshold, fvs)
            for d in decisions:
                producer.send(SETTINGS.topic_scored, key=d.card_id, value=d.to_json())
            producer.flush()
            insert_decisions(pool, decisions)
            # Only now is the work durable in both sinks; safe to advance offsets.
            consumer.commit()

            total += len(decisions)
            if total % 1000 < len(decisions):
                log_with(log, logging.INFO, "scored", total=total)
    except KeyboardInterrupt:
        log.info("shutting down after %d decisions", total)
    finally:
        consumer.close()
        producer.close()
        pool.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
