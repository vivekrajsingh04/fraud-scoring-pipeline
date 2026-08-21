"""Adapters that normalise a public labelled dataset into ``Transaction`` rows.

We do not synthesise transactions. Two real sources are supported:

* **IEEE-CIS Fraud Detection** (Kaggle, ~590k labelled transactions). It has no
  true card identifier or geo columns, so we derive them:
  ``card_id`` from the documented card/address tuple (``card1..card6``, ``addr1``,
  ``addr2``) which is the community-standard proxy for a card/account, and a
  stable pseudo-location from ``addr1``. The derivation is deterministic and
  documented; the *labels and amounts are the dataset's own*.
* **Sparkov** credit-card simulator output, which already carries ``cc_num``,
  ``merch_lat``/``merch_long``, ``category`` and ``is_fraud``.

Both yield rows sorted by event time -- the replayer relies on that.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from fraudpipe.schemas import Transaction

#: IEEE-CIS ``TransactionDT`` is seconds since an unpublished reference date.
#: Kaggle consensus places it at 2017-12-01; the absolute date is irrelevant to
#: the model (only deltas and hour-of-day matter) but we fix it for determinism.
IEEE_EPOCH = datetime(2017, 12, 1, tzinfo=timezone.utc)


def _stable_unit_float(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def _pseudo_geo(seed: str) -> tuple[float, float]:
    """Deterministic lat/lon for a region key.

    IEEE-CIS ships no coordinates. Rather than fabricate movement, we pin every
    transaction from the same billing region to the same point, so the
    impossible-travel feature only fires when a card's *region* changes -- which
    is a real signal in the data, not invented noise.
    """
    lat = -55.0 + _stable_unit_float("lat", seed) * 125.0
    lon = -170.0 + _stable_unit_float("lon", seed) * 340.0
    return round(lat, 5), round(lon, 5)


def load_ieee_cis(path: str | Path) -> Iterator[Transaction]:
    """Read ``train_transaction.csv`` from the IEEE-CIS dataset."""
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            card_id = "-".join(
                (row.get(c) or "NA") for c in ("card1", "card2", "card3", "card5", "addr1", "addr2")
            )
            region = (row.get("addr1") or "NA") + ":" + (row.get("addr2") or "NA")
            lat, lon = _pseudo_geo(region)
            dt_s = float(row["TransactionDT"])
            yield Transaction(
                txn_id=str(row["TransactionID"]),
                card_id=card_id,
                ts_ms=int((IEEE_EPOCH.timestamp() + dt_s) * 1000),
                amount=float(row["TransactionAmt"]),
                mcc=(row.get("ProductCD") or "NA"),
                merchant_id=(row.get("P_emaildomain") or "NA"),
                lat=lat,
                lon=lon,
                label=int(row["isFraud"]),
            )


def load_sparkov(path: str | Path) -> Iterator[Transaction]:
    """Read Sparkov simulator output (``fraudTrain.csv`` layout)."""
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            ts = datetime.fromisoformat(row["trans_date_trans_time"]).replace(tzinfo=timezone.utc)
            yield Transaction(
                txn_id=row["trans_num"],
                card_id=row["cc_num"],
                ts_ms=int(ts.timestamp() * 1000),
                amount=float(row["amt"]),
                mcc=row["category"],
                merchant_id=row["merchant"],
                lat=float(row["merch_lat"]),
                lon=float(row["merch_long"]),
                label=int(row["is_fraud"]),
            )


def load_normalized(path: str | Path) -> Iterator[Transaction]:
    """Read a CSV already in our own schema (used by tests and the smoke test)."""
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            yield Transaction.from_mapping(row)


LOADERS = {
    "ieee": load_ieee_cis,
    "sparkov": load_sparkov,
    "normalized": load_normalized,
}


def load_sorted(dataset: str, path: str | Path) -> list[Transaction]:
    """Load and sort by event time.

    Sorting is what makes the offline path point-in-time correct: the trainer
    replays in strict ``(ts_ms, txn_id)`` order, so a training row for a
    transaction at T is built from state containing only events before T.
    """
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(LOADERS)}")
    rows = list(LOADERS[dataset](path))
    rows.sort(key=lambda t: (t.ts_ms, t.txn_id))
    return rows
