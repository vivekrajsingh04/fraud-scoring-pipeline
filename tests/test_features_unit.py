"""Hand-built fixtures for every feature. No randomness, no dataset needed."""

from __future__ import annotations

import math

import pytest

from fraudpipe.features.core import IMPOSSIBLE_KMH, compute_features, process
from fraudpipe.features.geo import haversine_km
from fraudpipe.features.state import AMOUNT_WINDOW, CardState
from fraudpipe.schemas import FEATURE_ORDER, Transaction

T0 = 1_700_000_000_000
BLR = (12.9716, 77.5946)
DEL = (28.6139, 77.2090)


def txn(
    offset_ms: int,
    amount: float = 100.0,
    mcc: str = "5411",
    merchant: str = "m1",
    loc: tuple[float, float] = BLR,
    tid: str = "",
) -> Transaction:
    return Transaction(
        txn_id=tid or f"t{offset_ms}",
        card_id="c1",
        ts_ms=T0 + offset_ms,
        amount=amount,
        mcc=mcc,
        merchant_id=merchant,
        lat=loc[0],
        lon=loc[1],
        label=0,
    )


def run(txns: list[Transaction]) -> list[dict[str, float]]:
    st = CardState(card_id="c1")
    out = []
    for t in txns:
        fv, st = process(st, t)
        out.append(fv.features)
    return out


def test_all_declared_features_are_produced():
    f = run([txn(0)])[0]
    assert set(f) == set(FEATURE_ORDER)
    assert list(f) == list(FEATURE_ORDER), "dict order must match FEATURE_ORDER"


def test_first_txn_sees_empty_history():
    f = run([txn(0)])[0]
    assert f["is_first_txn_for_card"] == 1.0
    assert f["velocity_1m"] == f["velocity_1h"] == f["velocity_24h"] == 0.0
    assert f["prior_txn_count"] == 0.0
    assert f["mcc_is_new"] == 1.0
    assert f["seconds_since_last_txn"] == -1.0  # NEVER_SEEN sentinel


def test_velocity_windows_are_half_open_and_exclude_current():
    # prior events at 0s, 30s, 90s, 2h, 2h30m; current at 3h
    offsets = [0, 30_000, 90_000, 7_200_000, 9_000_000, 10_800_000]
    fs = run([txn(o) for o in offsets])
    last = fs[-1]
    assert last["velocity_1m"] == 0.0
    # The 2h30m event is 30m old and counts. The 2h event is *exactly* 1h old,
    # and the window is half-open [t-1h, t), so it does not. This boundary is
    # pinned deliberately: an off-by-one here silently leaks one extra event
    # into every velocity feature.
    assert last["velocity_1h"] == 1.0
    assert last["velocity_24h"] == 5.0  # all five priors
    assert fs[2]["velocity_1h"] == 2.0


def test_velocity_window_boundary_is_exclusive_on_both_sides():
    """Pin the half-open [t-w, t) contract at the exact boundary tick.

    An event exactly one window old is *out*; one millisecond newer is *in*.
    Getting this backwards is the single most common way a velocity feature
    ends up counting the transaction that has not happened yet.
    """
    exactly_1m = run([txn(0), txn(60_000)])[-1]
    assert exactly_1m["velocity_1m"] == 0.0

    just_inside = run([txn(1), txn(60_000)])[-1]
    assert just_inside["velocity_1m"] == 1.0


def test_events_older_than_24h_leave_the_velocity_window():
    fs = run([txn(0), txn(25 * 3_600_000)])
    assert fs[1]["velocity_24h"] == 0.0
    assert fs[1]["prior_txn_count"] == 1.0  # lifetime count still remembers it


def test_velocity_never_counts_the_current_transaction():
    fs = run([txn(i * 1000) for i in range(5)])
    for i, f in enumerate(fs):
        assert f["velocity_1m"] == float(i), "count must be of priors only"


def test_amount_zscore_matches_hand_computation():
    amounts = [100.0, 100.0, 100.0, 100.0, 400.0]
    fs = run([txn(i * 60_000, amount=a) for i, a in enumerate(amounts)])
    last = fs[-1]
    prior = amounts[:-1]
    mean = sum(prior) / len(prior)
    assert last["prior_amount_mean"] == pytest.approx(mean)
    # zero-variance history: scale falls back to |mean|, documented in core.py
    assert last["prior_amount_std"] == pytest.approx(0.0)
    assert last["amount_zscore"] == pytest.approx((400.0 - 100.0) / 100.0)
    assert last["amount_ratio_to_mean"] == pytest.approx(4.0)


def test_amount_window_is_bounded():
    st = CardState(card_id="c1")
    for i in range(AMOUNT_WINDOW * 3):
        _, st = process(st, txn(i * 1000, amount=float(i)))
    assert len(st.amount_window) == AMOUNT_WINDOW


def test_mcc_novelty_and_days_since():
    fs = run(
        [
            txn(0, mcc="5411"),
            txn(86_400_000, mcc="5812"),  # new MCC, 1 day later
            txn(2 * 86_400_000, mcc="5411"),  # seen 2 days ago
        ]
    )
    assert fs[0]["mcc_is_new"] == 1.0 and fs[0]["mcc_days_since_last_seen"] == -1.0
    assert fs[1]["mcc_is_new"] == 1.0
    assert fs[1]["card_distinct_mcc_count"] == 1.0
    assert fs[2]["mcc_is_new"] == 0.0
    assert fs[2]["mcc_days_since_last_seen"] == pytest.approx(2.0)
    assert fs[2]["card_distinct_mcc_count"] == 2.0


def test_impossible_travel_fires_for_blr_to_del_in_one_minute():
    fs = run([txn(0, loc=BLR), txn(60_000, loc=DEL)])
    dist = haversine_km(*BLR, *DEL)
    assert fs[1]["distance_km_from_last"] == pytest.approx(dist, rel=1e-9)
    assert fs[1]["implied_kmh"] == pytest.approx(dist * 60.0, rel=1e-9)
    assert fs[1]["implied_kmh"] > IMPOSSIBLE_KMH
    assert fs[1]["impossible_travel_flag"] == 1.0


def test_normal_travel_does_not_fire():
    fs = run([txn(0, loc=BLR), txn(6 * 3_600_000, loc=DEL)])
    assert fs[1]["impossible_travel_flag"] == 0.0


def test_same_timestamp_does_not_divide_by_zero():
    fs = run([txn(0, loc=BLR), txn(0, loc=DEL, tid="tb")])
    assert math.isfinite(fs[1]["implied_kmh"])
    assert fs[1]["impossible_travel_flag"] == 1.0


def test_hour_of_day_deviation_rewards_habitual_hours():
    hour_ms = 3_600_000
    day = 24 * hour_ms
    # ten transactions in the same hour-of-day across ten days
    habitual = [txn(i * day, tid=f"h{i}") for i in range(10)]
    fs = run(habitual)
    assert fs[-1]["hour_prior_count"] == 9.0
    assert fs[-1]["hour_deviation"] < 0.0  # far more likely than uniform

    # then one at a different hour on day 11
    odd = run(habitual + [txn(10 * day + 7 * hour_ms, tid="odd")])[-1]
    assert odd["hour_prior_count"] == 0.0
    assert odd["hour_deviation"] > 0.0


def test_compute_features_does_not_mutate_state():
    st = CardState(card_id="c1")
    _, st = process(st, txn(0))
    before = st.to_json()
    compute_features(st, txn(1000))
    assert st.to_json() == before, "compute_features must be a pure read"


def test_out_of_order_event_is_flagged_and_does_not_rewind_position():
    st = CardState(card_id="c1")
    _, st = process(st, txn(600_000, loc=DEL))
    fv, st = process(st, txn(0, loc=BLR, tid="late"))
    assert fv.features["out_of_order_flag"] == 1.0
    assert st.last_ts_ms == T0 + 600_000
    assert st.last_lat == DEL[0]


def test_state_round_trips_through_json():
    st = CardState(card_id="c1")
    for i in range(20):
        _, st = process(st, txn(i * 60_000, amount=float(i), mcc=f"m{i % 4}"))
    assert CardState.from_json(st.to_json(), "c1").to_json() == st.to_json()


def test_unknown_state_version_cold_starts_rather_than_misreading():
    bad = '{"v":999,"card_id":"c1"}'
    st = CardState.from_json(bad, "c1")
    assert st.txn_count == 0 and st.last_ts_ms is None
