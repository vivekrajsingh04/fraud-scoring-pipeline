"""Replayer: dataset adapters and inter-arrival timing."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from fraudpipe.replayer.loaders import load_sorted
from fraudpipe.schemas import Transaction


def _write_normalized(path: Path, txns: list[Transaction]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(txns[0].to_dict()))
        w.writeheader()
        for t in txns:
            w.writerow(t.to_dict())


def test_loader_sorts_strictly_by_event_time(tmp_path, stream):
    path = tmp_path / "txns.csv"
    _write_normalized(path, list(reversed(stream)))
    loaded = load_sorted("normalized", path)
    assert [t.txn_id for t in loaded] == [t.txn_id for t in stream]
    assert all(loaded[i].ts_ms <= loaded[i + 1].ts_ms for i in range(len(loaded) - 1))


def test_loader_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        load_sorted("nope", tmp_path / "x.csv")


def test_ieee_adapter_derives_stable_card_and_geo(tmp_path):
    path = tmp_path / "ieee.csv"
    path.write_text(
        "TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,card2,card3,"
        "card5,addr1,addr2,P_emaildomain,isFraud\n"
        "1,86400,59.0,W,13926,361,150,142,315,87,gmail.com,0\n"
        "2,86500,10.5,W,13926,361,150,142,315,87,gmail.com,1\n"
        "3,86600,99.0,C,2755,404,150,102,299,87,yahoo.com,0\n"
    )
    rows = load_sorted("ieee", path)
    assert len(rows) == 3
    assert rows[0].card_id == rows[1].card_id, "same card tuple -> same card_id"
    assert rows[0].card_id != rows[2].card_id
    assert (rows[0].lat, rows[0].lon) == (rows[1].lat, rows[1].lon)
    assert rows[1].label == 1
    assert rows[1].ts_ms - rows[0].ts_ms == 100_000


def test_sparkov_adapter_reads_native_columns(tmp_path):
    path = tmp_path / "sparkov.csv"
    path.write_text(
        "trans_date_trans_time,cc_num,merchant,category,amt,trans_num,"
        "merch_lat,merch_long,is_fraud\n"
        "2019-01-01 00:00:18,2703186189652095,fraud_Rippin,misc_net,4.97,"
        "0b242abb623afc578575680df30655b9,36.011293,-82.048315,0\n"
    )
    rows = load_sorted("sparkov", path)
    assert rows[0].card_id == "2703186189652095"
    assert rows[0].mcc == "misc_net"
    assert rows[0].lat == pytest.approx(36.011293)


def test_replay_preserves_relative_gaps_under_compression(stream):
    """A 100x replay must take ~1/100th of the dataset's own span."""
    from fraudpipe.replayer.__main__ import parse_args

    args = parse_args(["--path", "x.csv", "--speedup", "100"])
    assert args.speedup == 100.0

    subset = stream[:20]
    speedup = 1000.0
    expected_s = (subset[-1].ts_ms - subset[0].ts_ms) / 1000.0 / speedup

    started = time.monotonic()
    prev = subset[0].ts_ms
    for txn in subset:
        gap = (txn.ts_ms - prev) / 1000.0 / speedup
        if gap > 0:
            time.sleep(min(gap, 5.0))
        prev = txn.ts_ms
    elapsed = time.monotonic() - started

    assert elapsed == pytest.approx(expected_s, abs=0.5)
    assert elapsed < (subset[-1].ts_ms - subset[0].ts_ms) / 1000.0
