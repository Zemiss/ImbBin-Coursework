from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_aps(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values="na")


def aps_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(10 * fp + 500 * fn)


def main() -> None:
    train = load_aps(DATA_DIR / "ml_train.csv")
    test = load_aps(DATA_DIR / "ml_test.csv")

    x = train.drop(columns=["class"])
    y = (train["class"] == "pos").astype(int)
    x_test = test.drop(columns=["class"])
    y_test = (test["class"] == "pos").astype(int)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_fit, y_fit)
    val_score = model.predict_proba(x_val)[:, 1]
    thresholds = np.linspace(0.02, 0.98, 97)
    costs = [aps_cost(y_val.to_numpy(), (val_score >= t).astype(int)) for t in thresholds]
    threshold = float(thresholds[int(np.argmin(costs))])

    test_score = model.predict_proba(x_test)[:, 1]
    y_pred = (test_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print(f"train shape: {train.shape}, test shape: {test.shape}")
    print(f"positive ratio train/test: {y.mean():.4f} / {y_test.mean():.4f}")
    print(f"selected threshold: {threshold:.2f}")
    print(f"confusion matrix [tn fp fn tp]: {tn} {fp} {fn} {tp}")
    print(f"APS cost: {aps_cost(y_test.to_numpy(), y_pred)}")
    print(f"precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"recall: {recall_score(y_test, y_pred):.4f}")
    print(f"f1: {f1_score(y_test, y_pred):.4f}")
    print(f"micro f1: {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"macro f1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, target_names=["neg", "pos"], digits=4))


if __name__ == "__main__":
    main()
