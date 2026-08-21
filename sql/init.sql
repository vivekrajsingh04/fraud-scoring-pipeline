-- Decisions sink. One row per transaction; txn_id is the natural idempotency
-- key so at-least-once redelivery from Kafka cannot duplicate a decision.
CREATE TABLE IF NOT EXISTS decisions (
    txn_id        TEXT PRIMARY KEY,
    card_id       TEXT        NOT NULL,
    event_ts      TIMESTAMPTZ NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    threshold     DOUBLE PRECISION NOT NULL,
    decision      TEXT        NOT NULL CHECK (decision IN ('APPROVE', 'REVIEW')),
    label         SMALLINT,
    model_version TEXT        NOT NULL,
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    latency_ms    DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_scored_at ON decisions (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_card_ts   ON decisions (card_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_review    ON decisions (scored_at DESC)
    WHERE decision = 'REVIEW';

-- End-to-end latency = decision time - event time. Grafana reads this view.
CREATE OR REPLACE VIEW decision_latency AS
SELECT
    scored_at,
    decision,
    EXTRACT(EPOCH FROM (scored_at - event_ts)) * 1000.0 AS e2e_latency_ms,
    latency_ms AS inference_latency_ms
FROM decisions;

-- Live confusion matrix over labelled traffic, for the Grafana model panel.
CREATE OR REPLACE VIEW decision_quality AS
SELECT
    count(*) FILTER (WHERE decision = 'REVIEW'  AND label = 1) AS true_positives,
    count(*) FILTER (WHERE decision = 'REVIEW'  AND label = 0) AS false_positives,
    count(*) FILTER (WHERE decision = 'APPROVE' AND label = 1) AS false_negatives,
    count(*) FILTER (WHERE decision = 'APPROVE' AND label = 0) AS true_negatives
FROM decisions
WHERE label IS NOT NULL;
