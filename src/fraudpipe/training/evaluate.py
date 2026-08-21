"""Metrics that matter for fraud, and the cost-optimal threshold.

Accuracy is excluded on purpose: at a 1-3% base rate, "always approve" scores
97-99% and catches nothing. What we report instead:

* **PR-AUC** -- the area under precision/recall, which unlike ROC-AUC does not
  flatter a model on a heavily imbalanced negative class.
* **Precision@k / recall-of-fraud-value@k** -- given that a review team can only
  look at k% of transactions, how much fraud *value* does that budget catch.
* **Cost-optimal threshold** -- a false positive costs customer friction
  (a fixed amount); a false negative costs the chargeback (the transaction
  amount). We sweep the threshold, compute expected cost, and pick the minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Costs in minor currency units (paise). Defaults are illustrative and are
    stated in the README so a reader can substitute their own."""

    #: Cost of wrongly flagging a good transaction: agent time + the customer
    #: friction of a declined or challenged payment.
    false_positive_cost: float = 150.0
    #: Fraction of a missed fraud's value actually lost after recovery.
    chargeback_loss_fraction: float = 1.0
    #: Fixed operational cost per chargeback, on top of the amount.
    chargeback_fixed_cost: float = 500.0


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision -- step-wise PR-AUC, no interpolation."""
    order = np.argsort(-scores, kind="stable")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    total_pos = y.sum()
    if total_pos == 0:
        return 0.0
    recall = tp / total_pos
    d_recall = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * d_recall))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    y = y_true[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.empty(len(y), dtype=float)
    # Average ranks within tied score groups so ties do not bias the statistic.
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_at_budget(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray, budget: float
) -> dict[str, float]:
    """Metrics when only the top ``budget`` fraction of traffic is reviewed."""
    k = max(1, int(round(len(scores) * budget)))
    top = np.argsort(-scores, kind="stable")[:k]
    caught = y_true[top].sum()
    total_fraud = y_true.sum()
    fraud_value = float(amounts[y_true == 1].sum())
    caught_value = float(amounts[top][y_true[top] == 1].sum())
    return {
        "review_rate": budget,
        "reviewed_n": float(k),
        "precision": float(caught / k),
        "recall_count": float(caught / total_fraud) if total_fraud else 0.0,
        "recall_value": (caught_value / fraud_value) if fraud_value else 0.0,
    }


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostModel,
) -> float:
    flagged = scores >= threshold
    fp = int(np.sum(flagged & (y_true == 0)))
    missed = (~flagged) & (y_true == 1)
    fn_value = float(amounts[missed].sum()) * cost.chargeback_loss_fraction
    fn_fixed = int(np.sum(missed)) * cost.chargeback_fixed_cost
    return fp * cost.false_positive_cost + fn_value + fn_fixed


def cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostModel,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost across the threshold range. The README plots this."""
    lo, hi = float(scores.min()), float(scores.max())
    grid = np.linspace(lo, hi, n_points)
    costs = np.array([expected_cost(y_true, scores, amounts, t, cost) for t in grid])
    return grid, costs


def optimal_threshold(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray, cost: CostModel
) -> tuple[float, float]:
    """Threshold minimising expected cost, and that cost.

    Chosen on the *validation* split and then applied unchanged to test --
    tuning it on test would be a subtler form of the same leakage this project
    is about.
    """
    grid, costs = cost_curve(y_true, scores, amounts, cost)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def full_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostModel,
    budgets: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05),
) -> dict[str, object]:
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    baseline = expected_cost(y_true, scores, amounts, threshold=np.inf, cost=cost)
    at_thr = expected_cost(y_true, scores, amounts, threshold, cost)
    return {
        "n": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "threshold": float(threshold),
        "precision_at_threshold": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall_at_threshold": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "alert_rate_at_threshold": float(flagged.mean()),
        "expected_cost_at_threshold": at_thr,
        "expected_cost_approve_all": baseline,
        "cost_saved_vs_approve_all": baseline - at_thr,
        "budgets": [precision_at_budget(y_true, scores, amounts, b) for b in budgets],
    }
