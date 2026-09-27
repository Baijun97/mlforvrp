#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_adapter import ModelScorer
from pcvrptw_core import FEATURE_COLUMNS, parse_vrp_file
from routing_pipeline import (
    XGBoostRoutingPipeline,
    load_config,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct PCVRPTW routes using a trained XGBoost candidate-action model."
        )
    )
    parser.add_argument("--instance", required=True, type=Path, help="New .vrp instance.")
    parser.add_argument("--model", required=True, type=Path, help="Trained XGBoost model.")
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=Path(__file__).with_name("feature_columns.json"),
        help="Feature order contract produced by the training-data generator.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional pipeline_config.json, normally produced by calibrate_threshold.py.",
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=None,
        help="Override config threshold. If max candidate score is lower, end route/stop.",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["floor", "round", "exact"],
        default=None,
        help="Must match training-data generation. Uploaded archive uses floor.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["greedy", "topk_uniform", "topk_softmax"],
        default=None,
        help=(
            "Candidate selection after feasibility filtering and XGBoost scoring. "
            "greedy keeps the current Top-1 behaviour; topk_uniform samples uniformly "
            "from Top-k; topk_softmax samples from a softmax over Top-k scores."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k pool size used by topk_uniform/topk_softmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature for topk_softmax; lower is more greedy.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible top-k sampling.",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Optional debugging limit. Default uses instance VEHICLES.",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "classifier", "booster", "pickle"],
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for solution.json, routes.txt, decision_trace.csv.",
    )
    args = parser.parse_args()

    feature_columns = json.loads(
        args.feature_columns.read_text(encoding="utf-8")
    )
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "feature_columns.json does not exactly match the pipeline's "
            "32-feature contract. Training and inference must use identical features."
        )

    instance = parse_vrp_file(args.instance)
    scorer = ModelScorer(
        args.model,
        feature_columns=feature_columns,
        model_type=args.model_type,
    )
    config = load_config(
        args.config,
        stop_threshold=args.stop_threshold,
        distance_mode=args.distance_mode,
        selection_strategy=args.selection_strategy,
        top_k=args.top_k,
        temperature=args.temperature,
        random_seed=args.random_seed,
        max_routes=args.max_routes,
    )

    pipeline = XGBoostRoutingPipeline(instance, scorer, config)
    result = pipeline.solve()
    write_outputs(result, args.output_dir)

    metrics = result["metrics"]
    print(f"Instance: {result['instance_name']}")
    print(f"Routes: {metrics['vehicles_used']}")
    print(f"Visited clients: {metrics['visited_clients']}")
    print(f"Total distance: {metrics['total_distance']}")
    print(f"Collected prize: {metrics['collected_prize']}")
    print(f"Feasible: {metrics['feasible']}")
    print(f"Stop reason: {metrics['stop_reason']}")
    print(f"Runtime: {metrics['runtime_seconds']:.3f} s")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
