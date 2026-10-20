import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import evaluate_xgb_vrptw as base


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--selection-strategy",
        choices=["greedy", "topk_uniform", "topk_softmax"],
        default="greedy",
    )
    parser.add_argument("--selection-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=0)
    options, remaining = parser.parse_known_args()

    if options.selection_k < 1:
        parser.error("--selection-k must be at least 1")
    if not math.isfinite(options.temperature) or options.temperature <= 0:
        parser.error("--temperature must be finite and positive")

    rng = random.Random(options.random_seed)
    trace = []

    def choose_action(model, features, candidates):
        matrix = pd.DataFrame(
            [[row[name] for name in features] for _, row in candidates],
            columns=features,
            dtype="float32",
        )
        scores = model.predict(matrix, validate_features=True)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (
                -float(item[1]),
                float(item[0][1]["travel_time"]),
                item[0][0],
            ),
        )

        k = min(options.selection_k, len(ranked))
        if options.selection_strategy == "greedy":
            k = 1

        pool = ranked[:k]
        if k == 1:
            probabilities = [1.0]
            index = 0
        else:
            if options.selection_strategy == "topk_uniform":
                probabilities = [1.0 / k] * k
            else:
                best = float(pool[0][1])
                weights = [
                    math.exp((float(score) - best) / options.temperature)
                    for _, score in pool
                ]
                total = sum(weights)
                probabilities = [weight / total for weight in weights]
            index = rng.choices(range(k), weights=probabilities, k=1)[0]

        (candidate_id, _), score = pool[index]
        trace.append({
            "decision_index": len(trace) + 1,
            "candidate_id": candidate_id,
            "selected_rank": index + 1,
            "selection_k_used": k,
            "candidate_count": len(ranked),
            "selected_score": float(score),
            "best_score": float(ranked[0][1]),
            "sampling_probability": probabilities[index],
        })
        return candidate_id, float(score)

    base.choose_action = choose_action
    sys.argv = [sys.argv[0]] + remaining
    base.main()

    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output-dir", type=Path, required=True)
    output_args, _ = output_parser.parse_known_args(remaining)
    output = output_args.output_dir.expanduser()

    (output / "selection_config.json").write_text(
        json.dumps(vars(options), indent=2), encoding="utf-8"
    )
    if trace:
        with (output / "selection_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)

    print("\nSelection configuration:", vars(options))
    print("Decisions selecting rank > 1:",
          sum(row["selected_rank"] > 1 for row in trace))


if __name__ == "__main__":
    main()
