"""Environment-driven configuration. Defaults match docker-compose."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = _env("KAFKA_BOOTSTRAP", "localhost:9092")
    topic_raw: str = _env("TOPIC_RAW", "txns.raw")
    topic_featurized: str = _env("TOPIC_FEATURIZED", "txns.featurized")
    topic_scored: str = _env("TOPIC_SCORED", "txns.scored")

    postgres_dsn: str = _env("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud")
    redis_url: str = _env("REDIS_URL", "redis://localhost:6379/0")

    model_path: str = _env("MODEL_PATH", "artifacts/model.onnx")
    model_version: str = _env("MODEL_VERSION", "dev")
    #: Cost-optimal threshold chosen by ``fraudpipe.training.evaluate``.
    decision_threshold: float = float(_env("DECISION_THRESHOLD", "0.5"))

    replay_speedup: float = float(_env("REPLAY_SPEEDUP", "100"))
    log_level: str = _env("LOG_LEVEL", "INFO")


SETTINGS = Settings()
