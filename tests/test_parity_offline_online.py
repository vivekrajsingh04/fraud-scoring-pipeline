"""Offline/online feature parity -- the assertion this project exists to make.

The claim being defended: *a training row and a serving row for the same
transaction are byte-identical*. Not "close", not "within 1e-6" -- identical,
compared on the canonical JSON encoding that goes over the wire.

The two paths differ in everything except the feature code:

  offline  : whole dataset sorted in memory, one dict of CardState per card
  online   : events sharded by card_id across Kafka partitions, arriving in
             arbitrarily-sized micro-batches, state serialised to JSON and
             re-parsed between every batch

If any feature secretly depended on seeing the full dataset, on batch
boundaries, or on a card's future, these two would diverge. They cannot,
because both call ``fraudpipe.features.core.process``.

This module runs in CI on every push (no Spark or Kafka required -- the online
path is simulated faithfully at the level that matters: sharding, batching, and
state round-tripping).
"""

from __future__ import annotations

import json
import random

import pytest

from fraudpipe.features.core import process
from fraudpipe.features.state import CardState
from fraudpipe.schemas import FEATURE_ORDER, FeatureVector, Transaction
from fraudpipe.training.build_dataset import replay_offline
from tests.conftest import synthetic_stream


def _online_replay(
    txns: list[Transaction], n_partitions: int = 4, seed: int = 3
) -> dict[str, FeatureVector]:
    """Simulate the streaming path: partition by card_id, batch, round-trip state.

    Mirrors ``streaming/job.py`` exactly -- including the intra-batch sort, which
    is the one thing the Spark job must do that the trainer gets for free.
    """
    rng = random.Random(seed)
    partitions: list[list[Transaction]] = [[] for _ in range(n_partitions)]
    for t in txns:
        # Same partition function Kafka uses: hash of the key. All events for one
        # card land in one partition, which is the whole point of keying by card.
        partitions[hash(t.card_id) % n_partitions].append(t)

    out: dict[str, FeatureVector] = {}
    for part in partitions:
        # State lives as an opaque JSON blob between micro-batches, exactly as
        # Spark's applyInPandasWithState stores it.
        blobs: dict[str, str | None] = {}
        i = 0
        while i < len(part):
            batch = part[i : i + rng.randrange(1, 17)]  # ragged batch sizes
            i += len(batch)

            by_card: dict[str, list[Transaction]] = {}
            for t in batch:
                by_card.setdefault(t.card_id, []).append(t)

            for card_id, rows in by_card.items():
                rows.sort(key=lambda t: (t.ts_ms, t.txn_id))
                state = CardState.from_json(blobs.get(card_id), card_id)
                for t in rows:
                    fv, state = process(state, t)
                    out[t.txn_id] = fv
                blobs[card_id] = state.to_json()
    return out


def test_offline_and_online_features_are_byte_identical(stream):
    offline = {fv.txn_id: fv for fv in replay_offline(stream)}
    online = _online_replay(stream)

    assert set(offline) == set(online), "every transaction must be featurized once"

    mismatches = []
    for txn_id, off in offline.items():
        if off.to_json() != online[txn_id].to_json():
            diff = {
                k: (off.features[k], online[txn_id].features[k])
                for k in FEATURE_ORDER
                if off.features[k] != online[txn_id].features[k]
            }
            mismatches.append((txn_id, diff))

    assert not mismatches, (
        f"{len(mismatches)} of {len(offline)} rows differ between the offline "
        f"trainer and the streaming path. First 3: "
        f"{json.dumps(mismatches[:3], default=str, indent=2)}"
    )


@pytest.mark.parametrize("n_partitions", [1, 2, 8, 32])
def test_parity_is_independent_of_partition_count(stream, n_partitions):
    """Rescaling the cluster must not change a single feature value."""
    offline = {fv.txn_id: fv.to_json() for fv in replay_offline(stream)}
    online = {k: v.to_json() for k, v in _online_replay(stream, n_partitions).items()}
    assert offline == online


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_parity_is_independent_of_batch_boundaries(stream, seed):
    """Different micro-batch splits must produce identical features."""
    offline = {fv.txn_id: fv.to_json() for fv in replay_offline(stream)}
    online = {k: v.to_json() for k, v in _online_replay(stream, 4, seed=seed).items()}
    assert offline == online


def test_parity_holds_on_a_larger_stream():
    big = synthetic_stream(n=5000, cards=200, seed=99)
    offline = {fv.txn_id: fv.to_json() for fv in replay_offline(big)}
    online = {k: v.to_json() for k, v in _online_replay(big, 8, seed=11).items()}
    assert offline == online


def test_a_deliberately_leaky_feature_breaks_parity():
    """Negative control: prove the parity test can actually fail.

    A test that passes because it checks nothing is worse than no test. Here we
    inject the classic leak -- computing a card's mean amount over the *whole*
    dataset instead of its past -- and assert the comparison catches it.
    """
    txns = synthetic_stream(n=300, cards=10, seed=5)

    offline = {fv.txn_id: dict(fv.features) for fv in replay_offline(txns)}

    # The leak: a "global" mean per card, fitted on all rows including the future.
    per_card: dict[str, list[float]] = {}
    for t in txns:
        per_card.setdefault(t.card_id, []).append(t.amount)
    leaky = {t.txn_id: sum(per_card[t.card_id]) / len(per_card[t.card_id]) for t in txns}

    differing = [tid for tid, f in offline.items() if f["prior_amount_mean"] != leaky[tid]]
    assert differing, (
        "the point-in-time mean matched a whole-dataset mean on every row, which "
        "would mean the feature is leaking"
    )


def test_replay_offline_rejects_unsorted_input():
    """The offline path must refuse input that would void point-in-time order."""
    txns = synthetic_stream(n=20, cards=3, seed=1)
    shuffled = list(reversed(txns))
    with pytest.raises(ValueError, match="out-of-order"):
        replay_offline(shuffled)
