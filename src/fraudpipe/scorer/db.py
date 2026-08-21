"""Postgres sink for decisions.

Writes are idempotent on ``txn_id`` (``ON CONFLICT DO NOTHING``): the Kafka
consumer commits offsets only after the write, so at-least-once redelivery is
possible and must not duplicate a decision.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import psycopg
from psycopg_pool import ConnectionPool

from fraudpipe.schemas import Decision

log = logging.getLogger("scorer.db")

INSERT_SQL = """
INSERT INTO decisions (
    txn_id, card_id, event_ts, score, threshold, decision,
    label, model_version, scored_at, latency_ms
) VALUES (
    %s, %s, to_timestamp(%s / 1000.0), %s, %s, %s,
    %s, %s, to_timestamp(%s / 1000.0), %s
)
ON CONFLICT (txn_id) DO NOTHING
"""


def make_pool(dsn: str, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    return ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)


def insert_decisions(pool: ConnectionPool, decisions: Sequence[Decision]) -> int:
    if not decisions:
        return 0
    rows = [
        (
            d.txn_id,
            d.card_id,
            d.ts_ms,
            d.score,
            d.threshold,
            d.decision,
            d.label,
            d.model_version,
            d.scored_at_ms,
            d.latency_ms,
        )
        for d in decisions
    ]
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
        conn.commit()
    return len(rows)


def count_decisions(dsn: str) -> int:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM decisions")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def wait_for_decisions(dsn: str, expected: int, timeout_s: float = 120.0) -> int:
    """Poll until ``expected`` rows land. Used by the compose smoke test."""
    import time

    deadline = time.monotonic() + timeout_s
    seen = 0
    while time.monotonic() < deadline:
        try:
            seen = count_decisions(dsn)
        except Exception as exc:  # pragma: no cover - startup race
            log.debug("postgres not ready: %s", exc)
        if seen >= expected:
            return seen
        time.sleep(1.0)
    return seen


def iter_decision_ids(dsn: str) -> Iterable[str]:  # pragma: no cover - debug aid
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT txn_id FROM decisions ORDER BY scored_at")
        for (txn_id,) in cur:
            yield txn_id
