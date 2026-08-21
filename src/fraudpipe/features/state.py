"""Per-card state carried between transactions.

This object is the *entire* memory of the feature pipeline. Spark stores it as an
opaque JSON string in ``applyInPandasWithState``; the offline trainer holds it in
a dict. Because both sides use the same (de)serialisation, a state checkpointed
online can be loaded offline and vice versa.

Every field is explicitly bounded so per-card state cannot grow without limit --
see ``prune`` and the ``MAX_*`` constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Longest velocity window we support. Timestamps older than this are dropped.
WINDOW_24H_MS = 24 * 60 * 60 * 1000
#: Rolling window for the amount mean/std used by the z-score feature.
AMOUNT_WINDOW = 50
#: Cap on distinct MCCs / merchants remembered per card (LRU by last-seen).
MAX_MCC = 64
MAX_MERCHANTS = 128

STATE_VERSION = 1


@dataclass
class CardState:
    """Mutable per-card aggregate. Advanced by exactly one function: ``advance``."""

    card_id: str
    #: Event timestamps (ms) of prior transactions within the last 24h, ascending.
    recent_ts: list[int] = field(default_factory=list)
    #: Amounts aligned 1:1 with ``recent_ts`` -- used for windowed amount sums.
    recent_amounts: list[float] = field(default_factory=list)
    #: Last ``AMOUNT_WINDOW`` amounts (not time-bounded) for the rolling z-score.
    amount_window: list[float] = field(default_factory=list)
    #: mcc -> last seen event ts (ms)
    mcc_last_seen: dict[str, int] = field(default_factory=dict)
    #: merchant_id -> last seen event ts (ms)
    merchant_last_seen: dict[str, int] = field(default_factory=dict)
    #: 24 buckets, counts of prior transactions by hour-of-day.
    hour_hist: list[int] = field(default_factory=lambda: [0] * 24)
    last_ts_ms: int | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    txn_count: int = 0

    # ------------------------------------------------------------------ pruning
    def prune(self, now_ms: int) -> None:
        """Drop everything outside the longest window, relative to ``now_ms``.

        Called before features are read so that a card that went quiet for a week
        does not carry stale 24h counts.
        """
        cutoff = now_ms - WINDOW_24H_MS
        keep_from = 0
        for i, ts in enumerate(self.recent_ts):
            if ts > cutoff:
                keep_from = i
                break
            keep_from = i + 1
        if keep_from:
            del self.recent_ts[:keep_from]
            del self.recent_amounts[:keep_from]
        if len(self.amount_window) > AMOUNT_WINDOW:
            del self.amount_window[: len(self.amount_window) - AMOUNT_WINDOW]
        _evict_lru(self.mcc_last_seen, MAX_MCC)
        _evict_lru(self.merchant_last_seen, MAX_MERCHANTS)

    # ------------------------------------------------------- (de)serialisation
    def to_json(self) -> str:
        return json.dumps(
            {
                "v": STATE_VERSION,
                "card_id": self.card_id,
                "recent_ts": self.recent_ts,
                "recent_amounts": self.recent_amounts,
                "amount_window": self.amount_window,
                "mcc_last_seen": self.mcc_last_seen,
                "merchant_last_seen": self.merchant_last_seen,
                "hour_hist": self.hour_hist,
                "last_ts_ms": self.last_ts_ms,
                "last_lat": self.last_lat,
                "last_lon": self.last_lon,
                "txn_count": self.txn_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(raw: str | bytes | None, card_id: str) -> CardState:
        if not raw:
            return CardState(card_id=card_id)
        d: dict[str, Any] = json.loads(raw)
        if d.get("v") != STATE_VERSION:
            # Forward-compat: an unknown state version is treated as a cold start
            # rather than silently producing features from a different schema.
            return CardState(card_id=card_id)
        return CardState(
            card_id=d["card_id"],
            recent_ts=[int(x) for x in d["recent_ts"]],
            recent_amounts=[float(x) for x in d["recent_amounts"]],
            amount_window=[float(x) for x in d["amount_window"]],
            mcc_last_seen={k: int(v) for k, v in d["mcc_last_seen"].items()},
            merchant_last_seen={k: int(v) for k, v in d["merchant_last_seen"].items()},
            hour_hist=[int(x) for x in d["hour_hist"]],
            last_ts_ms=None if d["last_ts_ms"] is None else int(d["last_ts_ms"]),
            last_lat=None if d["last_lat"] is None else float(d["last_lat"]),
            last_lon=None if d["last_lon"] is None else float(d["last_lon"]),
            txn_count=int(d["txn_count"]),
        )


def _evict_lru(m: dict[str, int], cap: int) -> None:
    if len(m) <= cap:
        return
    for key in sorted(m, key=lambda k: m[k])[: len(m) - cap]:
        del m[key]
