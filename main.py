"""
Main entry point for the Walmart Sales Demand Forecasting project.

Usage examples
--------------
Train XGBoost and save the model:
    python main.py train --model xgboost --data-dir data/ --output-dir models/

Evaluate on a held-out split:
    python main.py train --model random_forest --data-dir data/ --eval-split 0.2

Generate predictions for the Kaggle test set:
    python main.py predict --model-path models/xgboost.pkl --data-dir data/ --output predictions.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from walmart_forecasting.data_loader import load_raw_data, preprocess
from walmart_forecasting.feature_engineering import build_features, ALL_FEATURES, TARGET
from walmart_forecasting.models import get_model
from walmart_forecasting.evaluation import evaluate, print_metrics


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    print(f"[train] Loading data from '{args.data_dir}' …")
    dfs = load_raw_data(args.data_dir)
    train_df, _ = preprocess(dfs)

    print("[train] Building features …")
    train_df = build_features(train_df, is_train=True)

    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]
    X = train_df[feature_cols]
    y = train_df[TARGET]

    if args.eval_split > 0:
        split_idx = int(len(train_df) * (1 - args.eval_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        hol_val = train_df["IsHoliday"].iloc[split_idx:].values
    else:
        X_train, y_train = X, y
        X_val, y_val, hol_val = None, None, None

    print(f"[train] Training '{args.model}' on {len(X_train):,} rows …")
    model = get_model(args.model)
    model.fit(X_train, y_train)

    if X_val is not None:
        preds = model.predict(X_val)
        metrics = evaluate(y_val.values, preds, hol_val)
        print_metrics(metrics, model_name=args.model)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"{args.model}.pkl")
        model.save(save_path)
    else:
        print("[train] --output-dir not specified; model not saved.")


# ---------------------------------------------------------------------------
# Predict command
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    from walmart_forecasting.models import _BaseModel

    print(f"[predict] Loading model from '{args.model_path}' …")
    model = _BaseModel.load(args.model_path)

    print(f"[predict] Loading data from '{args.data_dir}' …")
    dfs = load_raw_data(args.data_dir)
    _, test_df = preprocess(dfs)

    # For lag/rolling features we need the full training history
    train_df, _ = preprocess(dfs)
    train_df = build_features(train_df, is_train=True)

    # Append test rows (no target) and compute lag/rolling on combined df
    combined = pd.concat(
        [train_df, build_features(test_df, is_train=False)], ignore_index=True
    )

    # Re-compute lag features over combined so test rows get look-back context
    from walmart_forecasting.feature_engineering import add_lag_features, add_rolling_features
    combined = combined.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    if TARGET in combined.columns:
        combined = add_lag_features(combined)
        combined = add_rolling_features(combined)

    test_combined = combined[combined["Date"].isin(test_df["Date"].values)].copy()

    feature_cols = [c for c in ALL_FEATURES if c in test_combined.columns]
    preds = model.predict(test_combined[feature_cols])

    submission = test_df[["Store", "Dept", "Date"]].copy()
    submission["Weekly_Sales"] = preds
    submission.to_csv(args.output, index=False)
    print(f"[predict] Predictions saved to '{args.output}'  ({len(submission):,} rows)")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walmart Sales Demand Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = sub.add_parser("train", help="Train a forecasting model")
    train_p.add_argument(
        "--model",
        default="xgboost",
        choices=["linear_regression", "random_forest", "xgboost"],
        help="Model to train",
    )
    train_p.add_argument("--data-dir", default="data/", help="Path to data directory")
    train_p.add_argument("--output-dir", default="models/", help="Directory to save trained model")
    train_p.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Fraction of training data to use as validation (0 = no eval)",
    )

    # ---- predict ----
    predict_p = sub.add_parser("predict", help="Generate predictions for the test set")
    predict_p.add_argument("--model-path", required=True, help="Path to a saved .pkl model file")
    predict_p.add_argument("--data-dir", default="data/", help="Path to data directory")
    predict_p.add_argument("--output", default="submission.csv", help="Output CSV path")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
