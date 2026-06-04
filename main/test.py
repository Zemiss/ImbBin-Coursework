from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import pandas as pd

from modeling import (
    TARGET_COLUMN,
    align_features,
    evaluate_predictions,
    load_data,
    split_xy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate APS failure classifier.")
    parser.add_argument("--test_data", required=True, help="Path to evaluation CSV.")
    parser.add_argument(
        "--model_path",
        default="models/model.joblib",
        help="Path to saved model package.",
    )
    parser.add_argument(
        "--prediction_path",
        default=None,
        help="Optional CSV path for predictions when labels are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = joblib.load(args.model_path)
    model = package["model"]
    threshold = float(package["threshold"])
    feature_columns = package["feature_columns"]

    data = load_data(args.test_data)
    has_label = TARGET_COLUMN in data.columns
    x = data.drop(columns=[TARGET_COLUMN]) if has_label else data
    x = align_features(x, feature_columns)
    scores = model.predict_proba(x)[:, 1]
    y_pred = (scores >= threshold).astype(int)

    if has_label:
        _, y_true = split_xy(data)
        metrics = evaluate_predictions(y_true, y_pred)
        tn, fp, fn, tp = metrics.confusion
        print(f"threshold: {threshold:.4f}")
        print(f"confusion matrix [tn fp fn tp]: {tn} {fp} {fn} {tp}")
        print(f"APS cost: {metrics.cost}")
        print(f"precision: {metrics.precision:.4f}")
        print(f"recall: {metrics.recall:.4f}")
        print(f"f1: {metrics.f1:.4f}")
        print(f"micro f1: {metrics.micro_f1:.4f}")
        print(f"macro f1: {metrics.macro_f1:.4f}")
        print("\nclassification report:")
        print(metrics.report)
    else:
        predictions = pd.DataFrame(
            {
                "score": scores,
                "prediction": ["pos" if label == 1 else "neg" for label in y_pred],
            }
        )
        if args.prediction_path:
            prediction_path = Path(args.prediction_path)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)
            print(f"saved predictions: {prediction_path}")
        else:
            print(predictions.to_csv(index=False))


if __name__ == "__main__":
    main()
