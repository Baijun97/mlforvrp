#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_adapter import ModelScorer


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive = y_true == 1
    negative = y_true == 0

    tpr = float((y_pred[positive] == 1).mean()) if positive.any() else 1.0
    tnr = float((y_pred[negative] == 0).mean()) if negative.any() else 1.0
    return 0.5 * (tpr + tnr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the route-end/global-stop threshold from validation.csv. "
            "A non-terminal replay state should continue; a terminal state should stop."
        )
    )
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--feature-columns", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-type",
        choices=["auto", "classifier", "booster", "pickle"],
        default="auto",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["floor", "round", "exact"],
        default="floor",
    )
    args = parser.parse_args()

    features = json.loads(args.feature_columns.read_text(encoding="utf-8"))
    df = pd.read_csv(args.validation)

    required = {
        "instance_name", "route_id", "step_id", "is_terminal_state"
    } | set(features)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Validation CSV is missing columns: {missing}")

    scorer = ModelScorer(args.model, features, args.model_type)

    # Score in manageable batches.
    all_scores = np.empty(len(df), dtype=float)
    batch_size = 100_000
    for start in range(0, len(df), batch_size):
        end = min(len(df), start + batch_size)
        rows = df.iloc[start:end][features].to_dict(orient="records")
        all_scores[start:end] = scorer.predict_scores(rows)

    df["_score"] = all_scores

    state_table = (
        df.groupby(
            ["instance_name", "route_id", "step_id", "is_terminal_state"],
            sort=False,
            as_index=False,
        )["_score"]
        .max()
    )

    # Desired state decision:
    #   non-terminal -> continue (1)
    #   terminal     -> stop (0)
    y_true = (state_table["is_terminal_state"].to_numpy() == 0).astype(int)
    max_scores = state_table["_score"].to_numpy(dtype=float)

    if len(max_scores) == 0:
        raise ValueError("No validation states found.")

    # Search thresholds at unique scores. Downsample when there are too many.
    unique = np.unique(max_scores)
    if len(unique) > 5000:
        quantiles = np.linspace(0.0, 1.0, 5000)
        candidates = np.unique(np.quantile(unique, quantiles))
    else:
        candidates = unique

    eps = 1e-12
    candidates = np.unique(
        np.concatenate(([unique.min() - eps], candidates, [unique.max() + eps]))
    )

    best = None
    for threshold in candidates:
        y_pred = (max_scores >= threshold).astype(int)
        score = balanced_accuracy(y_true, y_pred)
        accuracy = float((y_pred == y_true).mean())
        item = (score, accuracy, -abs(float(threshold) - 0.5), float(threshold))
        if best is None or item > best:
            best = item

    best_balanced, best_accuracy, _, threshold = best

    output = {
        "distance_mode": args.distance_mode,
        "stop_threshold": threshold,
        "max_routes": None,
        "max_steps_per_route": 10000,
        "calibration": {
            "validation_rows": int(len(df)),
            "validation_states": int(len(state_table)),
            "terminal_states": int((y_true == 0).sum()),
            "non_terminal_states": int((y_true == 1).sum()),
            "balanced_accuracy": best_balanced,
            "accuracy": best_accuracy,
        },
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Best stop threshold: {threshold:.8f}")
    print(f"Balanced accuracy: {best_balanced:.4f}")
    print(f"State accuracy: {best_accuracy:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
