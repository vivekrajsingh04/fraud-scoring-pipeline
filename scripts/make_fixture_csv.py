#!/usr/bin/env python3
"""Emit a fixture CSV in the normalized schema, for CI runs without Kaggle creds.

This is *not* training data for any reported metric. The README's model numbers
come from the real IEEE-CIS dataset; this file exists only so CI can exercise
the build-dataset -> train -> ONNX -> serve code path without credentials.
The distinction is stated in the README too.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudpipe.schemas import Transaction  # noqa: E402

CITIES = [
    (12.9716, 77.5946),
    (19.0760, 72.8777),
    (28.6139, 77.2090),
    (13.0827, 80.2707),
    (22.5726, 88.3639),
    (1.3521, 103.8198),
]
MCCS = ["5411", "5812", "5541", "7995", "4829", "5999", "5732", "4111"]


def generate(rows: int, cards: int, seed: int) -> list[Transaction]:
    rng = random.Random(seed)
    base = 1_700_000_000_000
    home = {f"card-{c:04d}": rng.randrange(len(CITIES)) for c in range(cards)}
    habit = {f"card-{c:04d}": rng.randrange(24) for c in range(cards)}
    out: list[Transaction] = []
    ts = base
    for i in range(rows):
        card = f"card-{rng.randrange(cards):04d}"
        ts += rng.randrange(500, 120_000)
        fraud = rng.random() < 0.025
        if fraud:
            # Fraud is made detectable through the features, not through a magic
            # column: unusual city, unusual hour, unusual amount, novel MCC.
            city = rng.randrange(len(CITIES))
            amount = round(rng.lognormvariate(6.2, 0.9), 2)
            hour_shift = (habit[card] + 12) % 24
        else:
            city = home[card]
            amount = round(rng.lognormvariate(3.9, 0.8), 2)
            hour_shift = habit[card]
        lat, lon = CITIES[city]
        day = ts // 86_400_000
        event_ts = day * 86_400_000 + hour_shift * 3_600_000 + rng.randrange(3_600_000)
        out.append(
            Transaction(
                txn_id=f"fx{i:07d}",
                card_id=card,
                ts_ms=event_ts,
                amount=amount,
                mcc=rng.choice(MCCS) if not fraud else rng.choice(MCCS[-3:]),
                merchant_id=f"m{rng.randrange(200):04d}",
                lat=lat + rng.gauss(0, 0.01),
                lon=lon + rng.gauss(0, 0.01),
                label=int(fraud),
            )
        )
    out.sort(key=lambda t: (t.ts_ms, t.txn_id))
    return out


def main() -> int:
    p = argparse.ArgumentParser(prog="make_fixture_csv")
    p.add_argument("--out", default="data/fixture.csv")
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--cards", type=int, default=300)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    txns = generate(args.rows, args.cards, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(txns[0].to_dict()))
        w.writeheader()
        for t in txns:
            w.writerow(t.to_dict())
    frauds = sum(t.label or 0 for t in txns)
    print(f"wrote {len(txns)} rows to {args.out} ({frauds} fraud, {frauds / len(txns):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
