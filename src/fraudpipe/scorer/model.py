"""ONNX model wrapper.

The scorer has no LightGBM dependency: it loads a single ``.onnx`` file with
onnxruntime. The feature vector is assembled from ``FEATURE_ORDER`` -- the same
tuple the trainer used to build its matrix -- so column order cannot drift
between training and serving.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fraudpipe.schemas import FEATURE_ORDER, FeatureVector

log = logging.getLogger("scorer.model")


class FraudModel:
    def __init__(self, path: str | Path, version: str = "dev") -> None:
        self.path = Path(path)
        self.version = version
        opts = ort.SessionOptions()
        # One thread per request-handling worker: p99 latency is dominated by
        # scheduling jitter, not by matmul width, on a tree ensemble this small.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]

        expected = self._sess.get_inputs()[0].shape[-1]
        if isinstance(expected, int) and expected != len(FEATURE_ORDER):
            raise ValueError(
                f"model expects {expected} features but FEATURE_ORDER has "
                f"{len(FEATURE_ORDER)}; the model and the feature module are "
                f"out of sync -- retrain rather than reorder."
            )
        log.info("loaded model %s sha=%s", self.path, self.sha256)

    def score_batch(self, vectors: list[list[float]]) -> list[float]:
        x = np.asarray(vectors, dtype=np.float32).reshape(len(vectors), len(FEATURE_ORDER))
        with self._lock:  # ORT sessions are thread-safe, but pin batching order
            outputs = self._sess.run(None, {self._input_name: x})
        probs = outputs[1] if len(outputs) > 1 else outputs[0]
        arr = np.asarray(probs)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [float(v) for v in arr[:, 1]]
        return [float(v) for v in arr.ravel()]

    def score(self, fv: FeatureVector) -> float:
        return self.score_batch([fv.as_model_input()])[0]
