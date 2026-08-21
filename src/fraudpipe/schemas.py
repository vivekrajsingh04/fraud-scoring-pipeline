"""Canonical wire and domain schemas.

.. note::
   These dataclasses deliberately do **not** use ``slots=True``, and this module
   avoids any API newer than Python 3.10.

   The reason is the executors: this module is imported by Spark, whose Python is
   whatever the Spark image ships (3.10 today, 3.8 on the non-java17 tags). The
   whole premise of the project is that the offline and online paths run *the
   same* module, so the shared code has to import cleanly on the oldest runtime
   any component uses -- which is also why ``[tool.ruff] target-version`` is
   pinned to py310 rather than py311, and why ``docker/Dockerfile.spark``
   imports this module at build time.


Every component -- replayer, Spark job, offline trainer, scorer -- speaks these
types. There is exactly one definition of a transaction and exactly one ordered
list of feature names (``FEATURE_ORDER``); the ONNX model's input vector is built
from that list in both training and serving, so a reordering can never silently
skew the model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Transaction:
    """One card transaction, as published to ``txns.raw``.

    ``ts_ms`` is the *event time* of the transaction (epoch milliseconds), not
    the time it was ingested. All feature windows are defined on event time.
    """

    txn_id: str
    card_id: str
    ts_ms: int
    amount: float
    mcc: str
    merchant_id: str
    lat: float
    lon: float
    label: int | None = None  # 1 = fraud, 0 = legit, None = unlabeled at serve time

    @staticmethod
    def from_mapping(d: Mapping[str, Any]) -> Transaction:
        label = d.get("label")
        return Transaction(
            txn_id=str(d["txn_id"]),
            card_id=str(d["card_id"]),
            ts_ms=int(d["ts_ms"]),
            amount=float(d["amount"]),
            mcc=str(d["mcc"]),
            merchant_id=str(d.get("merchant_id", "")),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            label=None if label is None else int(label),
        )

    @staticmethod
    def from_json(raw: str | bytes) -> Transaction:
        return Transaction.from_mapping(json.loads(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


#: The single source of truth for model input layout. Training builds its matrix
#: from this list; the scorer builds its input vector from this list. Adding a
#: feature means adding it here and retraining -- there is no third place to edit.
FEATURE_ORDER: tuple[str, ...] = (
    "amount",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "is_first_txn_for_card",
    # velocity
    "velocity_1m",
    "velocity_1h",
    "velocity_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    # amount anomaly
    "amount_zscore",
    "amount_ratio_to_mean",
    "prior_amount_mean",
    "prior_amount_std",
    "prior_txn_count",
    # merchant novelty
    "mcc_is_new",
    "mcc_days_since_last_seen",
    "card_distinct_mcc_count",
    "merchant_is_new",
    # impossible travel
    "seconds_since_last_txn",
    "distance_km_from_last",
    "implied_kmh",
    "impossible_travel_flag",
    # time-of-day deviation
    "hour_prior_count",
    "hour_probability",
    "hour_deviation",
    # hygiene
    "out_of_order_flag",
)


@dataclass(frozen=True)
class FeatureVector:
    """Features for one transaction, computed strictly from data before its ts."""

    txn_id: str
    card_id: str
    ts_ms: int
    label: int | None
    features: dict[str, float] = field(default_factory=dict)

    def as_model_input(self) -> list[float]:
        """Dense vector in ``FEATURE_ORDER``. Used identically offline and online."""
        return [float(self.features[name]) for name in FEATURE_ORDER]

    def to_json(self) -> str:
        return json.dumps(
            {
                "txn_id": self.txn_id,
                "card_id": self.card_id,
                "ts_ms": self.ts_ms,
                "label": self.label,
                "features": self.features,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(raw: str | bytes) -> FeatureVector:
        d = json.loads(raw)
        return FeatureVector(
            txn_id=d["txn_id"],
            card_id=d["card_id"],
            ts_ms=int(d["ts_ms"]),
            label=d.get("label"),
            features={k: float(v) for k, v in d["features"].items()},
        )


@dataclass(frozen=True)
class Decision:
    """Scorer output, written to ``txns.scored`` and to Postgres."""

    txn_id: str
    card_id: str
    ts_ms: int
    score: float
    threshold: float
    decision: str  # APPROVE | REVIEW
    label: int | None
    model_version: str
    scored_at_ms: int
    latency_ms: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
