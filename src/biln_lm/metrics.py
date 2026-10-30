"""Evaluation metrics for the downstream benchmarks."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

REGRESSION_METRICS = ("rmse", "mae", "pcc", "spcc")
CLASSIFICATION_METRICS = ("acc", "f1", "mcc", "auroc")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def pcc(y_true, y_pred) -> float:
    return float(pearsonr(y_true, y_pred)[0])


def spcc(y_true, y_pred) -> float:
    return float(spearmanr(y_true, y_pred)[0])


def acc(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))


def mcc(y_true, y_pred) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def auroc(y_true, y_pred) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


METRICS: dict[str, Callable] = {
    "rmse": rmse,
    "mae": mae,
    "pcc": pcc,
    "spcc": spcc,
    "acc": acc,
    "f1": f1,
    "mcc": mcc,
    "auroc": auroc,
}

# Metric each task type is selected on, and whether higher is better.
SELECTION_METRIC = {"regression": ("rmse", False), "classification": ("mcc", True)}


def evaluate(task_type: str, y_true, y_pred, y_score=None) -> dict[str, float]:
    """Compute the metric set for ``regression`` or ``classification``."""
    if task_type == "regression":
        names = REGRESSION_METRICS
    elif task_type == "classification":
        names = CLASSIFICATION_METRICS
    else:
        raise ValueError(f"Unknown task type: {task_type!r}")
    scores = {}
    for name in names:
        values = y_score if name == "auroc" and y_score is not None else y_pred
        try:
            scores[name] = METRICS[name](y_true, values)
        except ValueError:
            scores[name] = float("nan")
    return scores
