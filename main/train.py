from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
from sklearn.model_selection import train_test_split

from aps_failure.modeling import (
    TARGET_COLUMN,
    align_features,
    evaluate_predictions,
    load_data,
    make_model,
    select_threshold,
    split_xy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APS failure classifier.")
    parser.add_argument("--train_data", required=True, help="Path to training CSV.")
    parser.add_argument(
        "--model_path",
        default="models/model.joblib",
        help="Output model path.",
    )
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.train_data)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"training data must contain '{TARGET_COLUMN}' column")

    x, y = split_xy(data)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.valid_size,
        stratify=y,
        random_state=args.random_state,
    )

    validation_model = make_model(args.random_state)
    validation_model.fit(x_train, y_train)
    valid_scores = validation_model.predict_proba(x_valid)[:, 1]
    threshold, validation_cost = select_threshold(y_valid, valid_scores)
    valid_pred = (valid_scores >= threshold).astype(int)
    valid_metrics = evaluate_predictions(y_valid, valid_pred)

    final_model = make_model(args.random_state)
    final_model.fit(align_features(data.drop(columns=[TARGET_COLUMN]), list(x.columns)), y)

    package = {
        "model": final_model,
        "threshold": threshold,
        "feature_columns": list(x.columns),
        "validation_metrics": {
            "cost": valid_metrics.cost,
            "precision": valid_metrics.precision,
            "recall": valid_metrics.recall,
            "f1": valid_metrics.f1,
            "micro_f1": valid_metrics.micro_f1,
            "macro_f1": valid_metrics.macro_f1,
            "confusion": valid_metrics.confusion,
        },
        "validation_cost": validation_cost,
        "random_state": args.random_state,
    }

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_path)

    tn, fp, fn, tp = valid_metrics.confusion
    print(f"saved model: {model_path}")
    print(f"selected threshold: {threshold:.4f}")
    print(f"validation confusion matrix [tn fp fn tp]: {tn} {fp} {fn} {tp}")
    print(f"validation APS cost: {valid_metrics.cost}")
    print(f"validation precision: {valid_metrics.precision:.4f}")
    print(f"validation recall: {valid_metrics.recall:.4f}")
    print(f"validation f1: {valid_metrics.f1:.4f}")
    print(f"validation micro f1: {valid_metrics.micro_f1:.4f}")
    print(f"validation macro f1: {valid_metrics.macro_f1:.4f}")


if __name__ == "__main__":
    main()
