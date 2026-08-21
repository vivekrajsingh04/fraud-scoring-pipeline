"""End-to-end training path: offline replay -> temporal split -> LightGBM -> ONNX.

Runs on the fixture stream so CI needs no Kaggle credentials. What it proves is
the *plumbing*: that the exported ONNX graph reproduces the booster's own
predictions, that the serving wrapper feeds columns in the right order, and that
the temporal split never lets a later row into training.

Model quality on real data is a separate question, answered in the README from
the IEEE-CIS run.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudpipe.schemas import FEATURE_ORDER
from fraudpipe.training.build_dataset import replay_offline, temporal_split, to_frame
from tests.conftest import synthetic_stream

lgb = pytest.importorskip("lightgbm")
pytest.importorskip("onnxmltools")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def frame():
    txns = synthetic_stream(n=6000, cards=150, seed=2024)
    return to_frame(replay_offline(txns))


def test_temporal_split_never_leaks_the_future_into_training(frame):
    train, valid, test = temporal_split(frame)
    assert len(train) and len(valid) and len(test)
    assert train.ts_ms.max() < valid.ts_ms.min()
    assert valid.ts_ms.max() < test.ts_ms.min()
    assert len(train) + len(valid) + len(test) == len(frame)


def test_split_boundary_does_not_straddle_identical_timestamps(frame):
    """Two transactions with the same timestamp must land on the same side."""
    train, valid, test = temporal_split(frame)
    for part_a, part_b in ((train, valid), (valid, test)):
        if len(part_a) and len(part_b):
            assert part_a.ts_ms.iloc[-1] != part_b.ts_ms.iloc[0]


def test_feature_matrix_columns_match_feature_order(frame):
    assert list(frame[list(FEATURE_ORDER)].columns) == list(FEATURE_ORDER)
    assert frame[list(FEATURE_ORDER)].isna().sum().sum() == 0


def test_onnx_export_reproduces_booster_predictions(frame, tmp_path):
    from fraudpipe.training.train import _onnx_probabilities, export_onnx, train_booster

    train, valid, test = temporal_split(frame)
    booster = train_booster(train, valid)

    x_test = test[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
    model_path = tmp_path / "model.onnx"
    # export_onnx raises if the graph and the booster disagree beyond tolerance,
    # so reaching the next line is already most of the assertion.
    export_onnx(booster, model_path, x_test[:256])

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_scores = _onnx_probabilities(sess, x_test)
    native = booster.predict(x_test, num_iteration=booster.best_iteration)
    assert np.max(np.abs(onnx_scores - native)) < 1e-5


def test_serving_wrapper_feeds_columns_in_training_order(frame, tmp_path):
    """A column-order bug is silent and catastrophic. Catch it here."""
    from fraudpipe.scorer.model import FraudModel
    from fraudpipe.training.train import export_onnx, train_booster

    train, valid, test = temporal_split(frame)
    booster = train_booster(train, valid)
    model_path = tmp_path / "model.onnx"
    x_test = test[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)
    export_onnx(booster, model_path, x_test[:256])

    model = FraudModel(model_path, version="test")

    # Build the serving input the way the scorer does: from a FeatureVector's
    # dict, via FEATURE_ORDER -- not from the DataFrame's column order.
    fvs = replay_offline(synthetic_stream(n=300, cards=20, seed=5))
    served = model.score_batch([fv.as_model_input() for fv in fvs[:100]])

    direct = booster.predict(
        np.array([fv.as_model_input() for fv in fvs[:100]], dtype=np.float32),
        num_iteration=booster.best_iteration,
    )
    assert np.max(np.abs(np.array(served) - direct)) < 1e-5
    assert all(0.0 <= s <= 1.0 for s in served)


def test_model_rejects_a_feature_count_mismatch(frame, tmp_path, monkeypatch):
    """If someone edits FEATURE_ORDER without retraining, fail loudly at load."""
    from fraudpipe.scorer import model as model_mod
    from fraudpipe.training.train import export_onnx, train_booster

    train, valid, test = temporal_split(frame)
    booster = train_booster(train, valid)
    model_path = tmp_path / "model.onnx"
    export_onnx(booster, model_path, test[list(FEATURE_ORDER)].to_numpy(dtype=np.float32)[:64])

    monkeypatch.setattr(model_mod, "FEATURE_ORDER", FEATURE_ORDER + ("bogus_feature",))
    with pytest.raises(ValueError, match="out of sync"):
        model_mod.FraudModel(model_path)
