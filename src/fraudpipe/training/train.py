"""Train LightGBM on the offline-replayed features and export to ONNX.

Two things here are deliberate:

1. The split is **temporal**, not random (see ``build_dataset.temporal_split``).
2. The threshold is picked on **validation** by minimising expected cost, then
   applied unchanged to test. The test split is read exactly once, at the end.

ONNX is the serving format so the scorer has no LightGBM (or Python ML stack)
dependency at all -- it loads one file with onnxruntime. The export is verified
before it is written: predictions from the ONNX graph must match the trained
booster within tolerance on a sample, or the run fails.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraudpipe.common.logging import log_with, setup_logging
from fraudpipe.schemas import FEATURE_ORDER
from fraudpipe.training.evaluate import CostModel, cost_curve, full_report, optimal_threshold

log = logging.getLogger("train")

ONNX_TOLERANCE = 1e-5


def train_booster(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 42):
    import lightgbm as lgb

    x_tr = train[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
    x_va = valid[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
    y_tr = train["label"].to_numpy(dtype=int)
    y_va = valid["label"].to_numpy(dtype=int)

    pos = max(int(y_tr.sum()), 1)
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        # Imbalance is handled by reweighting the positive class rather than by
        # resampling, which would distort the score calibration the cost model
        # depends on.
        "scale_pos_weight": (len(y_tr) - pos) / pos,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
    }
    return lgb.train(
        params,
        lgb.Dataset(x_tr, label=y_tr, feature_name=list(FEATURE_ORDER)),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(x_va, label=y_va, feature_name=list(FEATURE_ORDER))],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )


def export_onnx(booster, out_path: str | Path, sample: np.ndarray) -> None:
    # NOTE: this must be onnxmltools' own FloatTensorType, not the identically
    # named class in onnxconverter_common. onnxmltools' LightGBM shape calculator
    # does an exact type check against its own module and rejects the other one
    # with "got an input input with a wrong type" -- which reads like a data
    # problem and is actually an import problem.
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, len(FEATURE_ORDER)]))],
        target_opset=15,
        zipmap=False,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(onnx_model.SerializeToString())

    # Verify the exported graph before trusting it in production.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_scores = _onnx_probabilities(sess, sample.astype(np.float32))
    native = booster.predict(sample, num_iteration=booster.best_iteration)
    max_diff = float(np.max(np.abs(onnx_scores - native)))
    if max_diff > ONNX_TOLERANCE:
        raise RuntimeError(
            f"ONNX export mismatch: max |onnx - lightgbm| = {max_diff:g} "
            f"> {ONNX_TOLERANCE:g}. Refusing to ship a model that does not "
            f"reproduce its own training-time predictions."
        )
    log_with(log, logging.INFO, "onnx export verified", max_abs_diff=max_diff)


def _onnx_probabilities(sess, x: np.ndarray) -> np.ndarray:
    """Extract P(fraud) from a converted LightGBM binary classifier."""
    outputs = sess.run(None, {sess.get_inputs()[0].name: x})
    probs = outputs[1] if len(outputs) > 1 else outputs[0]
    arr = np.asarray(probs)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1].astype(np.float64)
    return arr.ravel().astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    from fraudpipe.training.build_dataset import temporal_split

    p = argparse.ArgumentParser(prog="fraudpipe-train")
    p.add_argument("--features", default="artifacts/features.parquet")
    p.add_argument("--model-out", default="artifacts/model.onnx")
    p.add_argument("--metrics-out", default="artifacts/metrics.json")
    # NOTE: CostModel is a slots=True dataclass, so `CostModel.false_positive_cost`
    # is the slot descriptor, not the default value -- reading a default off the
    # class silently hands argparse a member_descriptor. Instantiate instead.
    defaults = CostModel()
    p.add_argument("--fp-cost", type=float, default=defaults.false_positive_cost)
    p.add_argument("--chargeback-fixed", type=float, default=defaults.chargeback_fixed_cost)
    args = p.parse_args(argv)
    setup_logging()

    df = pd.read_parquet(args.features).dropna(subset=["label"])
    train, valid, test = temporal_split(df)
    log_with(
        log,
        logging.INFO,
        "temporal split",
        train=len(train),
        valid=len(valid),
        test=len(test),
    )

    booster = train_booster(train, valid)
    cost = CostModel(false_positive_cost=args.fp_cost, chargeback_fixed_cost=args.chargeback_fixed)

    def predict(part: pd.DataFrame) -> np.ndarray:
        x = part[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
        return booster.predict(x, num_iteration=booster.best_iteration)

    va_scores, te_scores = predict(valid), predict(test)
    va_amounts = valid["amount"].to_numpy(dtype=float)
    te_amounts = test["amount"].to_numpy(dtype=float)

    threshold, va_cost = optimal_threshold(
        valid["label"].to_numpy(dtype=int), va_scores, va_amounts, cost
    )
    report = full_report(test["label"].to_numpy(dtype=int), te_scores, te_amounts, threshold, cost)
    grid, costs = cost_curve(valid["label"].to_numpy(dtype=int), va_scores, va_amounts, cost)

    export_onnx(booster, args.model_out, test[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)[:512])

    gains = booster.feature_importance("gain")
    metrics = {
        "test": report,
        "validation_cost_at_chosen_threshold": va_cost,
        "chosen_threshold": threshold,
        "cost_model": {
            "false_positive_cost": cost.false_positive_cost,
            "chargeback_fixed_cost": cost.chargeback_fixed_cost,
            "chargeback_loss_fraction": cost.chargeback_loss_fraction,
        },
        "cost_curve": {"threshold": grid.tolist(), "expected_cost": costs.tolist()},
        "feature_importance": dict(
            sorted(
                zip(FEATURE_ORDER, (float(g) for g in gains), strict=True), key=lambda kv: -kv[1]
            )
        ),
        "best_iteration": int(booster.best_iteration or 0),
        "feature_order": list(FEATURE_ORDER),
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    log_with(
        log,
        logging.INFO,
        "training complete",
        pr_auc=report["pr_auc"],
        threshold=threshold,
        model=args.model_out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
