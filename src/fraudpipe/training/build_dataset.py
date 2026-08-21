"""Offline replay: build the training matrix through the *streaming* code path.

This is the offline half of the point-in-time guarantee. We sort the whole
dataset by ``(ts_ms, txn_id)`` and push it through ``features.core.process`` one
event at a time, carrying per-card state exactly as the Spark job does. The
training row for a transaction at time T is therefore produced by a state object
that has only ever been shown events strictly before T.

Note what is *absent*: no ``groupby().transform()``, no ``rolling()`` over the
full frame, no target encoding fitted on all rows. Those are the constructs that
leak, and they are unavailable here by construction because we never hold more
than one card's past in scope.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fraudpipe.common.logging import log_with, setup_logging
from fraudpipe.features.core import process
from fraudpipe.features.state import CardState
from fraudpipe.replayer.loaders import load_sorted
from fraudpipe.schemas import FEATURE_ORDER, FeatureVector, Transaction

log = logging.getLogger("build_dataset")


def replay_offline(txns: list[Transaction]) -> list[FeatureVector]:
    """Feed already-sorted transactions through the shared feature functions."""
    states: dict[str, CardState] = {}
    out: list[FeatureVector] = []
    prev_ts = None
    for txn in txns:
        if prev_ts is not None and txn.ts_ms < prev_ts:
            raise ValueError(
                "offline replay received out-of-order input; sort by ts_ms first "
                "or point-in-time correctness is void"
            )
        prev_ts = txn.ts_ms
        st = states.get(txn.card_id)
        if st is None:
            st = states[txn.card_id] = CardState(card_id=txn.card_id)
        fv, states[txn.card_id] = process(st, txn)
        out.append(fv)
    return out


def to_frame(fvs: list[FeatureVector]) -> pd.DataFrame:
    df = pd.DataFrame([fv.features for fv in fvs], columns=list(FEATURE_ORDER))
    df.insert(0, "txn_id", [fv.txn_id for fv in fvs])
    df.insert(1, "card_id", [fv.card_id for fv in fvs])
    df.insert(2, "ts_ms", [fv.ts_ms for fv in fvs])
    df["label"] = [fv.label for fv in fvs]
    return df


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time, never at random.

    A random split lets the model train on a card's future and test on its past,
    which inflates every metric. Cut points are chosen on the timestamp axis so
    train < valid < test in event time, mirroring how the model would be
    deployed: fit on history, judge on what came after.
    """
    df = df.sort_values(["ts_ms", "txn_id"], kind="stable").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_valid = int(n * (train_frac + valid_frac))
    # Push the boundary forward so an identical timestamp never straddles a split.
    while 0 < i_train < n and df.ts_ms[i_train] == df.ts_ms[i_train - 1]:
        i_train += 1
    while 0 < i_valid < n and df.ts_ms[i_valid] == df.ts_ms[i_valid - 1]:
        i_valid += 1
    return df.iloc[:i_train], df.iloc[i_train:i_valid], df.iloc[i_valid:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fraudpipe-build-dataset")
    p.add_argument("--dataset", default="ieee", choices=["ieee", "sparkov", "normalized"])
    p.add_argument("--path", required=True)
    p.add_argument("--out", default="artifacts/features.parquet")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    setup_logging()

    txns = load_sorted(args.dataset, args.path)
    if args.limit:
        txns = txns[: args.limit]
    df = to_frame(replay_offline(txns))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log_with(
        log,
        logging.INFO,
        "dataset built",
        rows=len(df),
        cards=int(df.card_id.nunique()),
        fraud_rate=round(float(df.label.mean()), 5),
        out=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
