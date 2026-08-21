"""Stateful feature computation: ``txns.raw`` -> ``txns.featurized``.

The interesting part is what this file *does not* contain: any feature maths.
Every number comes from ``fraudpipe.features.core.process``, the same function
the offline trainer calls. This module only handles Spark plumbing -- state
encoding, micro-batch ordering, and Kafka I/O.

PySpark's ``applyInPandasWithState`` is the Python binding for Scala's
``flatMapGroupsWithState``: arbitrary per-key state with a user-defined type,
which is what velocity/novelty/travel features need. Windowed SQL aggregations
alone cannot express "days since this card last used this MCC".
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout
from pyspark.sql.types import StringType, StructField, StructType

from fraudpipe.common.logging import setup_logging
from fraudpipe.config import SETTINGS
from fraudpipe.features.core import process
from fraudpipe.features.state import CardState
from fraudpipe.schemas import Transaction

log = logging.getLogger("streaming")

#: What we emit downstream -- one JSON blob per transaction, keyed by card_id.
OUTPUT_SCHEMA = StructType(
    [
        StructField("card_id", StringType(), False),
        StructField("payload", StringType(), False),
    ]
)

#: State is an opaque JSON string. Using one string column (rather than mirroring
#: CardState into a Spark StructType) means the state layout is owned by
#: ``features/state.py`` alone, so offline and online can never drift apart.
STATE_SCHEMA = StructType([StructField("blob", StringType(), True)])

#: Evict a card's state after this long with no traffic, bounding total state
#: size. The longest feature window is 24h, so anything older is already unused.
STATE_TTL_MS = 26 * 60 * 60 * 1000

RAW_SCHEMA = StructType(
    [
        StructField("txn_id", StringType(), False),
        StructField("card_id", StringType(), False),
        StructField("ts_ms", StringType(), False),
        StructField("amount", StringType(), False),
        StructField("mcc", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("lat", StringType(), True),
        StructField("lon", StringType(), True),
        StructField("label", StringType(), True),
    ]
)


def featurize_group(
    # Typed to match what applyInPandasWithState declares:
    # Callable[[Any, Iterable[DataFrame], GroupState], Iterable[DataFrame]].
    # Narrowing `key` to tuple[str, ...] or `pdfs` to Iterator is a
    # contravariance error even though both hold at runtime.
    key: Any,
    pdfs: Iterable[pd.DataFrame],
    state: GroupState,
) -> Iterator[pd.DataFrame]:
    """Advance one card's state over the rows in this micro-batch.

    Called once per (card_id, micro-batch). Because ``txns.raw`` is keyed by
    card_id, one card is confined to one partition and therefore to one instance
    of this function -- no shuffle, no cross-partition coordination.
    """
    card_id = key[0]

    if state.hasTimedOut:
        state.remove()
        return

    blob = state.get[0] if state.exists else None
    card_state = CardState.from_json(blob, card_id)

    # pandas types to_dict("records") as list[dict[Hashable, Any]]; every column
    # in RAW_SCHEMA is a StringType, so narrowing to str keys is safe and keeps
    # the sort key below honestly typed.
    rows: list[dict[str, str]] = []
    for pdf in pdfs:
        rows.extend({str(k): v for k, v in row.items()} for row in pdf.to_dict("records"))

    # Spark makes no ordering promise *within* a micro-batch, only across the
    # stream. Feature correctness depends on event-time order, so we sort here.
    # This is the online counterpart of the trainer's `load_sorted`.
    rows.sort(key=lambda r: (int(r["ts_ms"]), str(r["txn_id"])))

    out: list[tuple[str, str]] = []
    for r in rows:
        txn = Transaction.from_mapping(r)
        fv, card_state = process(card_state, txn)
        out.append((card_id, fv.to_json()))

    state.update((card_state.to_json(),))
    state.setTimeoutDuration(STATE_TTL_MS)

    if out:
        yield pd.DataFrame(out, columns=["card_id", "payload"])


def build_stream(spark: SparkSession, settings=SETTINGS) -> DataFrame:
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap)
        .option("subscribe", settings.topic_raw)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )
    parsed = raw.select(F.from_json(F.col("value").cast("string"), RAW_SCHEMA).alias("t")).select(
        "t.*"
    )
    return apply_features(parsed)


def apply_features(parsed: DataFrame) -> DataFrame:
    """Split out so tests can drive it with a batch DataFrame on ``local[2]``."""
    return parsed.groupBy("card_id").applyInPandasWithState(
        featurize_group,
        OUTPUT_SCHEMA,
        STATE_SCHEMA,
        "append",
        GroupStateTimeout.ProcessingTimeTimeout,
    )


def main() -> int:  # pragma: no cover - exercised by the compose smoke test
    setup_logging(SETTINGS.log_level)
    spark = (
        SparkSession.builder.appName("fraudpipe-features")
        .config("spark.sql.shuffle.partitions", "6")
        .config(
            "spark.sql.streaming.stateStore.providerClass",
            "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    query = (
        build_stream(spark)
        # Re-key by card_id downstream too, so the scorer's consumer group keeps
        # the same per-card affinity and per-card ordering.
        .selectExpr("CAST(card_id AS STRING) AS key", "CAST(payload AS STRING) AS value")
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", SETTINGS.kafka_bootstrap)
        .option("topic", SETTINGS.topic_featurized)
        .option("checkpointLocation", "/tmp/fraudpipe/checkpoints/features")
        .outputMode("append")
        .trigger(processingTime="1 second")
        .start()
    )
    query.awaitTermination()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
