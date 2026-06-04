from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


POS_LABEL = "pos"
NEG_LABEL = "neg"
TARGET_COLUMN = "class"


@dataclass(frozen=True)
class MetricResult:
    cost: int
    precision: float
    recall: float
    f1: float
    micro_f1: float
    macro_f1: float
    confusion: tuple[int, int, int, int]
    report: str


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values="na")


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = data.drop(columns=[TARGET_COLUMN])
    y = (data[TARGET_COLUMN] == POS_LABEL).astype(int).to_numpy()
    return x, y


def make_model(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        random_state=random_state,
    )


def aps_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(10 * fp + 500 * fn)


def threshold_grid() -> np.ndarray:
    return np.r_[np.linspace(0.001, 0.1, 100), np.linspace(0.11, 0.9, 80)]


def select_threshold(
    y_true: np.ndarray, scores: np.ndarray, thresholds: Iterable[float] | None = None
) -> tuple[float, int]:
    best_threshold = 0.5
    best_cost: int | None = None
    for threshold in threshold_grid() if thresholds is None else thresholds:
        pred = (scores >= threshold).astype(int)
        cost = aps_cost(y_true, pred)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold, int(best_cost)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return MetricResult(
        cost=aps_cost(y_true, y_pred),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        micro_f1=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion=(int(tn), int(fp), int(fn), int(tp)),
        report=classification_report(
            y_true,
            y_pred,
            target_names=[NEG_LABEL, POS_LABEL],
            digits=4,
            zero_division=0,
        ),
    )


def align_features(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return data.reindex(columns=feature_columns)
