from __future__ import annotations

import random
from pathlib import Path

import pytest

from fraudpipe.schemas import Transaction

FIXTURES = Path(__file__).parent / "fixtures"

CITIES = {
    "BLR": (12.9716, 77.5946),
    "BOM": (19.0760, 72.8777),
    "DEL": (28.6139, 77.2090),
    "SIN": (1.3521, 103.8198),
    "LHR": (51.4700, -0.4543),
}
MCCS = ["5411", "5812", "5541", "7995", "4829", "5999"]


def synthetic_stream(n: int = 400, cards: int = 25, seed: int = 7) -> list[Transaction]:
    """Deterministic stream used to *exercise* the pipeline in tests.

    This is test scaffolding, not training data. The model is trained on the real
    labelled dataset (see ``data/README.md``); these rows exist so unit tests are
    fast, hermetic and do not need a 590k-row Kaggle download in CI.
    """
    rng = random.Random(seed)
    base = 1_700_000_000_000
    rows: list[Transaction] = []
    for i in range(n):
        card = f"card-{rng.randrange(cards):03d}"
        city = rng.choice(list(CITIES))
        lat, lon = CITIES[city]
        rows.append(
            Transaction(
                txn_id=f"t{i:06d}",
                card_id=card,
                ts_ms=base + i * rng.randrange(1_000, 900_000),
                amount=round(rng.lognormvariate(4.0, 1.1), 2),
                mcc=rng.choice(MCCS),
                merchant_id=f"m{rng.randrange(60):03d}",
                lat=lat,
                lon=lon,
                label=1 if rng.random() < 0.03 else 0,
            )
        )
    rows.sort(key=lambda t: (t.ts_ms, t.txn_id))
    return rows


@pytest.fixture(scope="session")
def stream() -> list[Transaction]:
    return synthetic_stream()


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("fraudpipe-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
