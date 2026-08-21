"""FastAPI scoring service.

Exposes two paths onto the *same* code:

* ``POST /score`` -- synchronous scoring of an already-featurized transaction,
  used for latency measurement and for the compose smoke test.
* ``POST /score/raw`` -- takes a raw transaction plus the caller's card state
  and runs the shared feature function itself. This is the escape hatch that
  proves the serving path can reproduce a feature vector on demand, e.g. for
  an analyst asking "why was this declined?".

Prometheus metrics are exported at ``/metrics`` and drive the Grafana latency
panels; the latency histogram buckets are chosen around the 200ms SLO.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from fraudpipe.common.logging import setup_logging
from fraudpipe.config import SETTINGS
from fraudpipe.features.core import process
from fraudpipe.features.state import CardState
from fraudpipe.schemas import FEATURE_ORDER, Decision, FeatureVector, Transaction
from fraudpipe.scorer.model import FraudModel

log = logging.getLogger("scorer.app")

SCORE_LATENCY = Histogram(
    "fraudpipe_score_latency_seconds",
    "Model inference latency inside the scorer",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)
E2E_LATENCY = Histogram(
    "fraudpipe_e2e_latency_seconds",
    "Event time to decision time, end to end",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0),
)
DECISIONS = Counter("fraudpipe_decisions_total", "Decisions emitted", ["decision"])

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(SETTINGS.log_level)
    _state["model"] = FraudModel(SETTINGS.model_path, SETTINGS.model_version)
    _state["threshold"] = SETTINGS.decision_threshold
    log.info("scorer ready threshold=%s", _state["threshold"])
    yield
    _state.clear()


app = FastAPI(title="fraudpipe scorer", version="0.1.0", lifespan=lifespan)


class ScoreRequest(BaseModel):
    txn_id: str
    card_id: str
    ts_ms: int
    label: int | None = None
    features: dict[str, float]


class RawScoreRequest(BaseModel):
    transaction: dict[str, Any]
    card_state: dict[str, Any] | None = Field(
        default=None, description="Serialized CardState; omit for a cold card"
    )


class ScoreResponse(BaseModel):
    txn_id: str
    score: float
    threshold: float
    decision: str
    model_version: str
    latency_ms: float


def _decide(fv: FeatureVector) -> Decision:
    model: FraudModel = _state["model"]
    threshold: float = _state["threshold"]

    t0 = time.perf_counter()
    score = model.score(fv)
    elapsed = time.perf_counter() - t0
    SCORE_LATENCY.observe(elapsed)

    now_ms = int(time.time() * 1000)
    E2E_LATENCY.observe(max(0.0, (now_ms - fv.ts_ms) / 1000.0))
    decision = "REVIEW" if score >= threshold else "APPROVE"
    DECISIONS.labels(decision=decision).inc()

    return Decision(
        txn_id=fv.txn_id,
        card_id=fv.card_id,
        ts_ms=fv.ts_ms,
        score=score,
        threshold=threshold,
        decision=decision,
        label=fv.label,
        model_version=model.version,
        scored_at_ms=now_ms,
        latency_ms=elapsed * 1000.0,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    model: FraudModel | None = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_version": model.version,
        "model_sha256": model.sha256,
        "n_features": len(FEATURE_ORDER),
        "threshold": _state["threshold"],
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    missing = set(FEATURE_ORDER) - set(req.features)
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": sorted(missing)})
    fv = FeatureVector(
        txn_id=req.txn_id,
        card_id=req.card_id,
        ts_ms=req.ts_ms,
        label=req.label,
        features=req.features,
    )
    d = _decide(fv)
    return ScoreResponse(
        txn_id=d.txn_id,
        score=d.score,
        threshold=d.threshold,
        decision=d.decision,
        model_version=d.model_version,
        latency_ms=d.latency_ms,
    )


@app.post("/score/raw")
def score_raw(req: RawScoreRequest) -> JSONResponse:
    """Featurize then score, using the same ``process`` the Spark job calls."""
    txn = Transaction.from_mapping(req.transaction)
    import json as _json

    state = CardState.from_json(
        _json.dumps(req.card_state) if req.card_state else None, txn.card_id
    )
    fv, new_state = process(state, txn)
    d = _decide(fv)
    return JSONResponse(
        {
            "decision": _json.loads(d.to_json()),
            "features": fv.features,
            "card_state": _json.loads(new_state.to_json()),
        }
    )
