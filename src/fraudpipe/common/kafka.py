"""Thin kafka-python wrappers.

Kept deliberately small: the interesting decision here is the *keying*, not the
client. Producers key every record by ``card_id`` so all events for a card land
in one partition -- see the README section "Why card_id is the partition key".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


def make_producer(bootstrap: str, **kwargs: Any) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        key_serializer=lambda k: k.encode() if isinstance(k, str) else k,
        value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
        acks="all",
        linger_ms=5,
        retries=5,
        enable_idempotence=True,
        **kwargs,
    )


def make_consumer(
    bootstrap: str, topics: Iterable[str], group_id: str, **kwargs: Any
) -> KafkaConsumer:
    return KafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        auto_offset_reset=kwargs.pop("auto_offset_reset", "earliest"),
        enable_auto_commit=False,  # we commit after the decision is durable
        max_poll_records=kwargs.pop("max_poll_records", 200),
        **kwargs,
    )


def ensure_topics(
    bootstrap: str, topics: Iterable[str], partitions: int = 6, replication: int = 1
) -> None:
    """Idempotently create topics. Partition count matters: it is the ceiling on
    per-card state parallelism, since a card never spans partitions."""
    admin = KafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        admin.create_topics(
            [NewTopic(t, partitions, replication) for t in topics],
            validate_only=False,
        )
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def iter_json(consumer: KafkaConsumer) -> Iterator[dict[str, Any]]:
    for msg in consumer:
        yield json.loads(msg.value)
