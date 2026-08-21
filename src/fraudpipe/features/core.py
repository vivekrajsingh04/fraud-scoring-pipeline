"""THE feature logic. Imported by the offline trainer and by the Spark job.

Point-in-time correctness is enforced *structurally*, not by discipline:

    features = compute_features(state, txn)   # reads state, never writes
    advance(state, txn)                       # writes state, never read from

``process`` is the only public entry point and it always calls them in that
order. A feature therefore cannot see its own transaction, and cannot see any
transaction that arrived later, because the state object simply does not contain
that information yet. There is no "training version" of this file.

Determinism note: everything here is pure-Python float64 with a fixed operation
order, so the offline and online paths produce bit-identical results. The parity
test in ``tests/test_parity_offline_online.py`` asserts exactly that, on the
canonical JSON encoding, and runs in CI.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from fraudpipe.features.geo import haversine_km
from fraudpipe.features.state import CardState
from fraudpipe.schemas import FEATURE_ORDER, FeatureVector, Transaction

WINDOW_1M_MS = 60 * 1000
WINDOW_1H_MS = 60 * 60 * 1000
WINDOW_24H_MS = 24 * 60 * 60 * 1000

#: Above this implied ground speed a card cannot physically have made both
#: transactions. ~900 km/h is commercial cruise; we use 1000 to leave headroom.
IMPOSSIBLE_KMH = 1000.0
#: Guards the implied-speed division when two transactions share a timestamp.
MIN_TRAVEL_SECONDS = 1.0
#: Laplace smoothing for the hour-of-day distribution, so a card's first
#: transaction in a given hour is "unusual" rather than "impossible".
HOUR_SMOOTHING = 1.0
#: Sentinel for "this card has never seen this MCC before". A real value can
#: never be negative, so the model can learn the split cleanly.
NEVER_SEEN = -1.0


def process(state: CardState, txn: Transaction) -> tuple[FeatureVector, CardState]:
    """Compute features for ``txn`` from ``state``, then fold ``txn`` into state.

    Returns the feature vector and the (mutated, returned for clarity) state.
    This is the single function both the trainer and the streaming job call.
    """
    fv = compute_features(state, txn)
    advance(state, txn)
    return fv, state


# --------------------------------------------------------------------- reading


def compute_features(state: CardState, txn: Transaction) -> FeatureVector:
    """Pure read of ``state``. MUST NOT mutate ``state`` in any way."""
    # Work on a pruned view relative to the *current* event time without
    # touching the caller's lists until `advance` runs.
    ts = txn.ts_ms
    recent_ts = state.recent_ts
    recent_amounts = state.recent_amounts

    f: dict[str, float] = {}

    dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    f["amount"] = txn.amount
    f["log_amount"] = math.log1p(max(0.0, txn.amount))
    f["hour_of_day"] = float(dt.hour)
    f["day_of_week"] = float(dt.weekday())
    f["is_first_txn_for_card"] = 1.0 if state.txn_count == 0 else 0.0

    # ---- velocity ---------------------------------------------------------
    # Counts are over *prior* transactions only; `txn` is not in `recent_ts` yet.
    c1m = c1h = c24h = 0
    s1h = s24h = 0.0
    for i in range(len(recent_ts) - 1, -1, -1):
        age = ts - recent_ts[i]
        if age >= WINDOW_24H_MS:
            break  # list is ascending, everything earlier is older still
        c24h += 1
        s24h += recent_amounts[i]
        if age < WINDOW_1H_MS:
            c1h += 1
            s1h += recent_amounts[i]
            if age < WINDOW_1M_MS:
                c1m += 1
    f["velocity_1m"] = float(c1m)
    f["velocity_1h"] = float(c1h)
    f["velocity_24h"] = float(c24h)
    f["amount_sum_1h"] = s1h
    f["amount_sum_24h"] = s24h

    # ---- amount anomaly ---------------------------------------------------
    win = state.amount_window
    n = len(win)
    if n >= 2:
        mean = math.fsum(win) / n
        var = math.fsum((x - mean) ** 2 for x in win) / (n - 1)  # sample variance
        std = math.sqrt(var)
        f["prior_amount_mean"] = mean
        f["prior_amount_std"] = std
        # A card with a perfectly flat spend history has std 0; a different
        # amount is then infinitely surprising, which is not a useful number.
        # Fall back to the mean as the scale, which is what an analyst would do.
        scale = std if std > 1e-9 else max(1.0, abs(mean))
        f["amount_zscore"] = (txn.amount - mean) / scale
        f["amount_ratio_to_mean"] = txn.amount / mean if abs(mean) > 1e-9 else 0.0
    else:
        f["prior_amount_mean"] = 0.0
        f["prior_amount_std"] = 0.0
        f["amount_zscore"] = 0.0
        f["amount_ratio_to_mean"] = 0.0
    f["prior_txn_count"] = float(state.txn_count)

    # ---- merchant / MCC novelty -------------------------------------------
    mcc_last = state.mcc_last_seen.get(txn.mcc)
    f["mcc_is_new"] = 0.0 if mcc_last is not None else 1.0
    f["mcc_days_since_last_seen"] = (
        NEVER_SEEN if mcc_last is None else (ts - mcc_last) / 86_400_000.0
    )
    f["card_distinct_mcc_count"] = float(len(state.mcc_last_seen))
    f["merchant_is_new"] = 0.0 if txn.merchant_id in state.merchant_last_seen else 1.0

    # ---- impossible travel -------------------------------------------------
    if state.last_ts_ms is None or state.last_lat is None or state.last_lon is None:
        f["seconds_since_last_txn"] = NEVER_SEEN
        f["distance_km_from_last"] = NEVER_SEEN
        f["implied_kmh"] = 0.0
        f["impossible_travel_flag"] = 0.0
    else:
        elapsed_s = (ts - state.last_ts_ms) / 1000.0
        dist = haversine_km(state.last_lat, state.last_lon, txn.lat, txn.lon)
        f["seconds_since_last_txn"] = elapsed_s
        f["distance_km_from_last"] = dist
        kmh = dist / (max(elapsed_s, MIN_TRAVEL_SECONDS) / 3600.0)
        f["implied_kmh"] = kmh
        f["impossible_travel_flag"] = 1.0 if kmh > IMPOSSIBLE_KMH else 0.0

    # ---- time-of-day deviation --------------------------------------------
    hour = dt.hour
    prior_hour_count = float(state.hour_hist[hour])
    total = float(sum(state.hour_hist))
    prob = (prior_hour_count + HOUR_SMOOTHING) / (total + HOUR_SMOOTHING * 24.0)
    f["hour_prior_count"] = prior_hour_count
    f["hour_probability"] = prob
    # Deviation relative to a uniform-hour card: >0 means rarer than uniform.
    f["hour_deviation"] = -math.log(prob) - math.log(24.0)

    # ---- hygiene ------------------------------------------------------------
    f["out_of_order_flag"] = 1.0 if state.last_ts_ms is not None and ts < state.last_ts_ms else 0.0

    missing = set(FEATURE_ORDER) - set(f)
    if missing:  # pragma: no cover - guards against an edit to FEATURE_ORDER only
        raise KeyError(f"feature(s) declared but not computed: {sorted(missing)}")

    return FeatureVector(
        txn_id=txn.txn_id,
        card_id=txn.card_id,
        ts_ms=ts,
        label=txn.label,
        features={name: f[name] for name in FEATURE_ORDER},
    )


# --------------------------------------------------------------------- writing


def advance(state: CardState, txn: Transaction) -> None:
    """Fold ``txn`` into ``state``. MUST NOT be called before ``compute_features``."""
    ts = txn.ts_ms

    # Keep `recent_*` ascending even if a late event slips through. Online this
    # is rare (Kafka preserves order per card_id, and card_id is the key);
    # offline it never happens because we replay in strict timestamp order.
    pos = len(state.recent_ts)
    while pos > 0 and state.recent_ts[pos - 1] > ts:
        pos -= 1
    state.recent_ts.insert(pos, ts)
    state.recent_amounts.insert(pos, txn.amount)

    state.amount_window.append(txn.amount)
    state.mcc_last_seen[txn.mcc] = max(ts, state.mcc_last_seen.get(txn.mcc, ts))
    if txn.merchant_id:
        state.merchant_last_seen[txn.merchant_id] = max(
            ts, state.merchant_last_seen.get(txn.merchant_id, ts)
        )
    state.hour_hist[datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).hour] += 1
    state.txn_count += 1

    # A late event must not rewind the card's position/clock, otherwise the next
    # in-order event would compute travel against a stale point.
    if state.last_ts_ms is None or ts >= state.last_ts_ms:
        state.last_ts_ms = ts
        state.last_lat = txn.lat
        state.last_lon = txn.lon

    state.prune(max(ts, state.last_ts_ms or ts))
