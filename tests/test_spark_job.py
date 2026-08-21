"""Drive the real Spark job as a real streaming query on ``local[2]``.

This is not a mock and not a batch stand-in. ``applyInPandasWithState`` is
*only* valid in a streaming query -- Spark rejects it outright on a batch
DataFrame -- so testing it means running an actual Structured Streaming job.
The fixture stream is written out as JSON files and consumed through the file
source with ``maxFilesPerTrigger=1``, which forces one micro-batch per file.

That last detail is what gives the test its teeth: per-card state has to be
checkpointed, serialised, and reloaded between every micro-batch, exactly as it
would be in production. It catches the class of bug the pure-Python parity test
cannot -- state-schema serialisation errors, Arrow conversion problems, and any
accidental dependency on driver-side state.

Skipped automatically when PySpark is unavailable (it does not support every
Python version); CI pins Python 3.11 so it always runs there.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from fraudpipe.training.build_dataset import replay_offline

pytest.importorskip("pyspark")

#: Number of files the fixture stream is split across -- and therefore the
#: number of micro-batches the state must survive.
N_BATCHES = 8


def _write_stream_files(txns, directory: Path, n_files: int = N_BATCHES) -> None:
    """Write the fixture stream as JSON files, in event-time order.

    Ordering across files is the whole point, and it needs forcing. Spark's
    ``FileStreamSource`` picks up files sorted by *modification time*, and eight
    small files written in a loop land in the same filesystem timestamp tick --
    so the tie is broken by directory-listing order, which is arbitrary. When
    that shuffles the batches, a card's events arrive out of event-time order and
    the features legitimately diverge from a strictly-ordered offline replay,
    because ``advance`` deliberately refuses to rewind a card's clock for late
    data.

    Stamping explicit, increasing mtimes makes the source order deterministic and
    reproduces the in-order-per-card delivery Kafka gives us.
    """
    directory.mkdir(parents=True, exist_ok=True)
    chunk = max(1, (len(txns) + n_files - 1) // n_files)
    base_mtime = time.time() - 3600
    for i in range(0, len(txns), chunk):
        part = txns[i : i + chunk]
        index = i // chunk
        path = directory / f"part-{index:04d}.json"
        with path.open("w") as fh:
            for t in part:
                # All-string fields, matching the schema the job parses out of
                # the Kafka value.
                fh.write(
                    json.dumps(
                        {
                            "txn_id": t.txn_id,
                            "card_id": t.card_id,
                            "ts_ms": str(t.ts_ms),
                            "amount": str(t.amount),
                            "mcc": t.mcc,
                            "merchant_id": t.merchant_id,
                            "lat": str(t.lat),
                            "lon": str(t.lon),
                            "label": str(t.label),
                        }
                    )
                    + "\n"
                )
        # One clear second between files, so the sort cannot tie.
        os.utime(path, (base_mtime + index, base_mtime + index))


def _run_streaming_job(spark, txns, tmp_path: Path, name: str) -> dict[str, dict]:
    """Run the production transformation as a streaming query; collect results."""
    from fraudpipe.streaming.job import RAW_SCHEMA, apply_features

    source = tmp_path / f"{name}-input"
    checkpoint = tmp_path / f"{name}-checkpoint"
    _write_stream_files(txns, source)

    stream = (
        spark.readStream.schema(RAW_SCHEMA)
        # One file per micro-batch: the state must be persisted and reloaded
        # between every batch rather than surviving in one task's memory.
        .option("maxFilesPerTrigger", 1)
        .json(str(source))
    )

    query = (
        apply_features(stream)
        .writeStream.format("memory")
        .queryName(name)
        .outputMode("append")
        .option("checkpointLocation", str(checkpoint))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination(timeout=300)

    rows = spark.sql(f"SELECT card_id, payload FROM {name}").collect()
    out = {}
    for row in rows:
        payload = json.loads(row["payload"])
        payload["_key"] = row["card_id"]
        out[payload["txn_id"]] = payload
    return out


def test_streaming_job_matches_offline_features(spark, stream, tmp_path):
    """The same byte-identity bar as the pure-Python parity test, through Spark."""
    online = _run_streaming_job(spark, stream, tmp_path, "parity")
    offline = {fv.txn_id: json.loads(fv.to_json()) for fv in replay_offline(stream)}

    assert len(online) == len(stream), "every input row must produce exactly one output"
    assert set(online) == set(offline)

    mismatched = []
    for txn_id, expected in offline.items():
        actual = {k: v for k, v in online[txn_id].items() if k != "_key"}
        if actual != expected:
            mismatched.append(txn_id)

    assert not mismatched, (
        f"{len(mismatched)} of {len(offline)} rows diverged between the offline "
        f"trainer and a genuine Spark streaming run across {N_BATCHES} micro-batches. "
        f"First 5: {mismatched[:5]}"
    )


def test_state_survives_micro_batch_boundaries(spark, stream, tmp_path):
    """Features that depend on history prove the state actually persisted.

    If per-card state were lost between micro-batches, every card's first
    transaction in each batch would look like a cold start. Asserting that far
    fewer rows are flagged as first-for-card than there are batches x cards is
    what makes this a real check rather than a smoke test.
    """
    online = _run_streaming_job(spark, stream, tmp_path, "statefulness")

    cold_starts = sum(1 for p in online.values() if p["features"]["is_first_txn_for_card"] == 1.0)
    distinct_cards = len({p["card_id"] for p in online.values()})

    assert cold_starts == distinct_cards, (
        f"expected exactly one cold start per card ({distinct_cards}), got "
        f"{cold_starts} -- state is being lost between micro-batches"
    )

    # And history genuinely accumulates: the last transaction of the busiest card
    # must have seen every one of its predecessors.
    by_card: dict[str, list[dict]] = {}
    for p in online.values():
        by_card.setdefault(p["card_id"], []).append(p)
    busiest = max(by_card.values(), key=len)
    busiest.sort(key=lambda p: p["ts_ms"])
    assert busiest[-1]["features"]["prior_txn_count"] == float(len(busiest) - 1)


def test_output_key_is_the_card_id(spark, stream, tmp_path):
    """A card must never be split across state instances or re-keyed downstream."""
    online = _run_streaming_job(spark, stream, tmp_path, "keying")
    for payload in online.values():
        assert payload["_key"] == payload["card_id"], (
            "output key must stay the card_id so downstream consumers keep per-card ordering"
        )
