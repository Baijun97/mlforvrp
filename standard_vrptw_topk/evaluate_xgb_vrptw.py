#!/usr/bin/env python3
"""Construct complete VRPTW routes with a trained XGBRanker and benchmark them.

Keep this file in the same directory as ``generate_training_dataset.py`` so it
can reuse the *exact* parser, DIMACS distance function, schedule logic, and 41
candidate features used during training.

Example (evaluate the 12 held-out instances with at most 200 customers):

    python3 evaluate_xgb_vrptw.py \
      --model models/xgb_ranker_200/xgboost_ranker.json \
      --prepared-dir processed/vrptw_ranker \
      --instances-dir Instances/VRPTW \
      --output-dir evaluation/xgb_ranker_200 \
      --split test \
      --max-customers 200 \
      --top-k 25

Outputs:
    summary.json             aggregate end-to-end solver metrics
    comparison_summary.csv  OR-Tools-style comparison metrics
    instance_results.csv    one row per test instance
    routes.csv              generated route details
    solutions/*.sol         generated feasible solutions
    partial/*.txt           partial routes for failed constructions

The reported runtime is model inference plus route construction.  It excludes
the one-time model training and loading costs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPS = 1e-9


@dataclass
class ConstructionResult:
    routes: list[list[int]]
    feasible: bool
    failure_reason: str | None
    unserved: list[int]
    inference_seconds: float
    model_calls: int
    decisions: int
    mean_candidates_scored: float


def require(module_name: str, install_name: str | None = None) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        package = install_name or module_name
        raise SystemExit(
            f"Missing dependency: {package}. Install with:\n"
            "  python3 -m pip install -U pandas xgboost"
        ) from exc


def load_training_module(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(
            f"Required file not found: {path}\n"
            "Place evaluate_xgb_vrptw.py beside generate_training_dataset.py."
        )
    spec = importlib.util.spec_from_file_location("vrptw_training_pipeline", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_feature_spec(prepared_dir: Path) -> list[str]:
    path = prepared_dir / "feature_columns.json"
    if not path.is_file():
        raise SystemExit(f"Feature specification not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    features = value.get("features")
    if not isinstance(features, list) or not features:
        raise SystemExit(f"Invalid feature specification: {path}")
    if len(features) != 41:
        raise SystemExit(
            f"Expected the 41-feature standard VRPTW model, found {len(features)}. "
            "Do not use a PCVRPTW/prize-collecting feature specification."
        )
    return [str(feature) for feature in features]


def load_model(model_path: Path, features: list[str]) -> Any:
    xgb = require("xgboost")
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    model = xgb.XGBRanker()
    model.load_model(model_path)
    model_features = model.get_booster().feature_names
    if model_features is not None and list(model_features) != features:
        raise SystemExit(
            "Model feature names/order do not match feature_columns.json. "
            "Use the feature specification saved with this model."
        )
    return model


def read_manifest(
    prepared_dir: Path,
    split: str,
    max_customers: int,
    max_instances: int,
) -> list[dict[str, Any]]:
    path = prepared_dir / "instance_splits.csv"
    if not path.is_file():
        raise SystemExit(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for row in rows:
        if split != "all" and row.get("split") != split:
            continue
        num_customers = int(float(row["num_customers"]))
        if max_customers > 0 and num_customers > max_customers:
            continue
        selected.append({**row, "num_customers": num_customers})
    selected.sort(key=lambda row: row["instance_id"])
    if max_instances > 0:
        selected = selected[:max_instances]
    if not selected:
        raise SystemExit(
            f"No manifest rows selected for split={split!r}, "
            f"max_customers={max_customers}."
        )
    return selected


def unique_file_index(root: Path, suffix: str) -> dict[str, Path]:
    paths = sorted(root.rglob(f"*{suffix}"))
    if not paths:
        raise SystemExit(f"No {suffix} files found under {root}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        if path.stem in index:
            duplicates.setdefault(path.stem, [index[path.stem]]).append(path)
        else:
            index[path.stem] = path
    if duplicates:
        examples = "; ".join(
            f"{stem}: {', '.join(str(p) for p in values)}"
            for stem, values in list(duplicates.items())[:3]
        )
        raise SystemExit(
            "Duplicate file stems prevent unambiguous instance matching. " + examples
        )
    return index


def locate_pairs(
    instances_dir: Path, selected: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], Path, Path]]:
    vrp_index = unique_file_index(instances_dir, ".vrp")
    sol_index = unique_file_index(instances_dir, ".sol")
    pairs = []
    missing = []
    for item in selected:
        instance_id = item["instance_id"]
        vrp_path = vrp_index.get(instance_id)
        sol_path = sol_index.get(instance_id)
        if vrp_path is None or sol_path is None:
            missing.append(instance_id)
            continue
        pairs.append((item, vrp_path, sol_path))
    if missing:
        raise SystemExit(
            "Could not match .vrp and .sol files for manifest IDs, for example: "
            + ", ".join(missing[:10])
        )
    return pairs


def candidate_rows_for_state(
    pipeline: Any,
    instance: Any,
    current_id: int,
    current_time: float,
    route_distance: float,
    route_load: float,
    visited: set[int],
    route_has_customer: bool,
    top_k: int,
    rounding: str,
    candidate_mode: str,
    allow_depot: bool,
) -> list[tuple[int, dict[str, float]]]:
    remaining = sorted(set(instance.customer_ids) - visited)
    pool = remaining.copy()
    if route_has_customer and allow_depot:
        pool.append(0)

    cache: dict[int, dict[str, float]] = {}
    for candidate_id in pool:
        cache[candidate_id] = pipeline.candidate_features(
            instance=instance,
            current_id=current_id,
            candidate_id=candidate_id,
            current_time=current_time,
            route_distance=route_distance,
            route_load=route_load,
            num_visited=len(visited),
            num_remaining=len(remaining),
            rounding=rounding,
        )

    travel_order = sorted(
        pool,
        key=lambda candidate_id: (
            cache[candidate_id]["travel_time"],
            candidate_id,
        ),
    )
    travel_rank = {
        candidate_id: rank
        for rank, candidate_id in enumerate(travel_order, start=1)
    }
    sampling_order = sorted(
        pool,
        key=lambda candidate_id: (
            -cache[candidate_id]["overall_feasible"],
            cache[candidate_id]["travel_time"],
            cache[candidate_id]["candidate_tw_late"],
            candidate_id,
        ),
    )
    if candidate_mode == "topk":
        sampling_order = sampling_order[:top_k]

    rows: list[tuple[int, dict[str, float]]] = []
    for candidate_id in sampling_order:
        row = dict(cache[candidate_id])
        row["candidate_rank_by_travel"] = travel_rank[candidate_id]
        # Hard feasibility mask: infeasible candidates remain useful during
        # training, but they must never be executed by the constructor.
        if int(row["overall_feasible"]) == 1:
            rows.append((candidate_id, row))
    return rows


def choose_action(
    model: Any,
    features: list[str],
    candidates: list[tuple[int, dict[str, float]]],
) -> tuple[int, float]:
    pd = require("pandas")
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
    (candidate_id, _), score = ranked[0]
    return candidate_id, float(score)


def construct_routes(
    pipeline: Any,
    instance: Any,
    model: Any,
    features: list[str],
    top_k: int,
    rounding: str,
    candidate_mode: str,
) -> ConstructionResult:
    visited: set[int] = set()
    all_customers = set(instance.customer_ids)
    routes: list[list[int]] = []
    model_calls = 0
    decisions = 0
    candidate_counts: list[int] = []
    failure_reason: str | None = None
    max_decisions = len(all_customers) + instance.vehicle_limit + 5

    start = time.perf_counter()
    while visited != all_customers:
        if len(routes) >= instance.vehicle_limit:
            failure_reason = (
                f"vehicle_limit_reached_with_{len(all_customers - visited)}_unserved"
            )
            break

        route_id = len(routes) + 1
        current_id = 0
        current_time = instance.depot.tw_early
        route_distance = 0.0
        route_load = 0.0
        sequence: list[int] = []

        while True:
            decisions += 1
            if decisions > max_decisions:
                failure_reason = "decision_safety_limit_reached"
                break

            # On the final available vehicle, returning early would make the
            # remaining customers impossible to serve within the fleet limit.
            last_available_vehicle = route_id >= instance.vehicle_limit
            allow_depot = bool(sequence) and not (
                last_available_vehicle and visited != all_customers
            )
            candidates = candidate_rows_for_state(
                pipeline=pipeline,
                instance=instance,
                current_id=current_id,
                current_time=current_time,
                route_distance=route_distance,
                route_load=route_load,
                visited=visited,
                route_has_customer=bool(sequence),
                top_k=top_k,
                rounding=rounding,
                candidate_mode=candidate_mode,
                allow_depot=allow_depot,
            )
            if not candidates:
                failure_reason = (
                    "no_feasible_customer_at_route_start"
                    if not sequence
                    else "no_feasible_action_in_route"
                )
                break

            candidate_counts.append(len(candidates))
            selected_id, _ = choose_action(model, features, candidates)
            model_calls += 1
            selected_features = next(
                row for candidate_id, row in candidates if candidate_id == selected_id
            )

            if selected_id == 0:
                routes.append(sequence)
                break

            sequence.append(selected_id)
            visited.add(selected_id)
            current_id = selected_id
            current_time = float(selected_features["departure_time"])
            route_distance += float(selected_features["travel_time"])
            route_load = float(selected_features["load_after_candidate"])

            # Force the final depot action through the same model interface so
            # runtime and decision counts include complete route termination.
            if visited == all_customers:
                continue

        if failure_reason is not None:
            if sequence:
                routes.append(sequence)
            break

    elapsed = time.perf_counter() - start
    unserved = sorted(all_customers - visited)
    mean_candidates = statistics.fmean(candidate_counts) if candidate_counts else 0.0
    return ConstructionResult(
        routes=routes,
        feasible=failure_reason is None and not unserved,
        failure_reason=failure_reason,
        unserved=unserved,
        inference_seconds=elapsed,
        model_calls=model_calls,
        decisions=decisions,
        mean_candidates_scored=mean_candidates,
    )


def evaluate_routes(
    pipeline: Any,
    instance: Any,
    construction: ConstructionResult,
    rounding: str,
) -> tuple[list[dict[str, Any]], float, bool, str | None]:
    route_rows = []
    for route_id, sequence in enumerate(construction.routes, start=1):
        route_row, _ = pipeline.schedule_route(
            instance, route_id, sequence, rounding
        )
        route_rows.append(route_row)

    generated_cost = sum(float(row["distance"]) for row in route_rows)
    route_feasible = all(int(row["is_feasible"]) == 1 for row in route_rows)
    all_visits = [customer for route in construction.routes for customer in route]
    unique_visits = len(all_visits) == len(set(all_visits))
    complete = set(all_visits) == set(instance.customer_ids)
    fleet_ok = len(construction.routes) <= instance.vehicle_limit
    feasible = bool(
        construction.feasible
        and route_feasible
        and unique_visits
        and complete
        and fleet_ok
    )
    reason = construction.failure_reason
    if reason is None and not route_feasible:
        reason = "generated_route_constraint_violation"
    elif reason is None and not unique_visits:
        reason = "duplicate_customer_visit"
    elif reason is None and not complete:
        reason = "customer_coverage_failure"
    elif reason is None and not fleet_ok:
        reason = "vehicle_limit_violation"
    return route_rows, generated_cost, feasible, reason


def reference_cost_and_routes(
    pipeline: Any,
    instance: Any,
    solution: Any,
    rounding: str,
) -> tuple[float, int]:
    if solution.claimed_cost is not None:
        return float(solution.claimed_cost), len(solution.routes)
    total = 0.0
    for route_id, sequence in enumerate(solution.routes, start=1):
        row, _ = pipeline.schedule_route(instance, route_id, sequence, rounding)
        total += float(row["distance"])
    return total, len(solution.routes)


def initialise_routes_file(path: Path) -> None:
    fields = [
        "instance_id", "route_id", "sequence", "num_clients", "load",
        "capacity", "capacity_utilization", "distance", "start_time",
        "end_time", "duration", "wait_time", "time_warp",
        "capacity_violation", "is_feasible",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def append_routes(path: Path, instance_id: str, route_rows: list[dict[str, Any]]) -> None:
    fields = [
        "instance_id", "route_id", "sequence", "num_clients", "load",
        "capacity", "capacity_utilization", "distance", "start_time",
        "end_time", "duration", "wait_time", "time_warp",
        "capacity_violation", "is_feasible",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for row in route_rows:
            writer.writerow({name: instance_id if name == "instance_id" else row[name] for name in fields})


def write_solution(path: Path, routes: list[list[int]], cost: float, status: str) -> None:
    lines = [f"# Status: {status}"]
    for route_id, route in enumerate(routes, start=1):
        lines.append(f"Route #{route_id}: {' '.join(map(str, route))}")
    lines.append(f"Cost: {cost:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_summary(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    feasible = [row for row in results if int(row["feasible"]) == 1]
    gaps = [float(row["gap_percent"]) for row in feasible if row["gap_percent"] != ""]
    runtimes = [float(row["inference_seconds"]) for row in results]
    route_differences = [
        int(row["num_routes"]) - int(row["reference_routes"])
        for row in feasible
    ]
    failures = Counter(
        str(row["failure_reason"])
        for row in results
        if int(row["feasible"]) == 0
    )

    def mean_or_none(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    summary = {
        "model": str(args.model.expanduser().resolve()),
        "instances_dir": str(args.instances_dir.expanduser().resolve()),
        "split": args.split,
        "max_customers": args.max_customers if args.max_customers > 0 else None,
        "top_k": args.top_k,
        "candidate_mode": args.candidate_mode,
        "rounding": args.rounding,
        "instances_tested": len(results),
        "feasible_instances": len(feasible),
        "feasible_rate": len(feasible) / len(results) if results else 0.0,
        "feasible_rate_percent": (
            len(feasible) / len(results) * 100.0 if results else 0.0
        ),
        "gap_percent_feasible_only": {
            "count": len(gaps),
            "mean": mean_or_none(gaps),
            "median": statistics.median(gaps) if gaps else None,
            "best": min(gaps) if gaps else None,
            "worst": max(gaps) if gaps else None,
        },
        "routes_feasible_only": {
            "mean_generated": mean_or_none([float(row["num_routes"]) for row in feasible]),
            "mean_reference": mean_or_none([float(row["reference_routes"]) for row in feasible]),
            "mean_difference": mean_or_none([float(value) for value in route_differences]),
        },
        "inference_runtime_seconds_per_instance": {
            "mean": mean_or_none(runtimes),
            "median": statistics.median(runtimes) if runtimes else None,
            "p95": percentile(runtimes, 0.95),
            "total": sum(runtimes),
        },
        "failures": dict(sorted(failures.items())),
        "comparison_note": (
            "Training time is intentionally excluded. Compare feasible rate, "
            "gap, route count, and per-instance inference runtime with OR-Tools "
            "on the same instance IDs and distance convention."
        ),
    }
    return summary


def write_comparison_summary(path: Path, summary: dict[str, Any]) -> None:
    gaps = summary["gap_percent_feasible_only"]
    routes = summary["routes_feasible_only"]
    runtime = summary["inference_runtime_seconds_per_instance"]
    rows = [
        ("Instances tested", summary["instances_tested"]),
        ("Feasible rate", f"{summary['feasible_rate_percent']:.3f}%"),
        ("Mean gap (feasible only)", "" if gaps["mean"] is None else f"{gaps['mean']:.3f}%"),
        ("Median gap (feasible only)", "" if gaps["median"] is None else f"{gaps['median']:.3f}%"),
        ("Best gap (feasible only)", "" if gaps["best"] is None else f"{gaps['best']:.3f}%"),
        ("Worst gap (feasible only)", "" if gaps["worst"] is None else f"{gaps['worst']:.3f}%"),
        ("Mean generated routes", "" if routes["mean_generated"] is None else f"{routes['mean_generated']:.3f}"),
        ("Mean reference routes", "" if routes["mean_reference"] is None else f"{routes['mean_reference']:.3f}"),
        ("Mean inference runtime", "" if runtime["mean"] is None else f"{runtime['mean']:.6f} s"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "XGB route constructor"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of an XGBRanker VRPTW route constructor."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        help="Path to generate_training_dataset.py; defaults to beside this script.",
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="test")
    parser.add_argument("--max-customers", type=int, default=200)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--rounding", choices=("dimacs", "round", "trunc", "none"), default="dimacs")
    parser.add_argument("--candidate-mode", choices=("topk", "all-feasible"), default="topk")
    args = parser.parse_args()

    if args.top_k < 2:
        raise SystemExit("--top-k must be at least 2")

    script_dir = Path(__file__).resolve().parent
    pipeline_path = (
        args.pipeline_script.expanduser().resolve()
        if args.pipeline_script
        else script_dir / "generate_training_dataset.py"
    )
    pipeline = load_training_module(pipeline_path)
    prepared_dir = args.prepared_dir.expanduser().resolve()
    instances_dir = args.instances_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    features = load_feature_spec(prepared_dir)
    model = load_model(args.model.expanduser().resolve(), features)
    selected = read_manifest(
        prepared_dir, args.split, args.max_customers, args.max_instances
    )
    pairs = locate_pairs(instances_dir, selected)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n"
            "Use a new output directory so results from different runs are not mixed."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = output_dir / "solutions"
    partial_dir = output_dir / "partial"
    solutions_dir.mkdir(exist_ok=True)
    partial_dir.mkdir(exist_ok=True)
    routes_path = output_dir / "routes.csv"
    initialise_routes_file(routes_path)
    results_path = output_dir / "instance_results.csv"
    result_fields = [
        "instance_id", "family", "num_customers", "split", "feasible",
        "failure_reason", "num_unserved", "num_routes", "reference_routes",
        "generated_cost", "reference_cost", "gap_percent",
        "inference_seconds", "model_calls", "decisions",
        "mean_candidates_scored", "source_vrp", "source_sol",
    ]

    results: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()

        for current, (manifest_row, vrp_path, sol_path) in enumerate(pairs, start=1):
            instance = pipeline.parse_vrp(vrp_path)
            if instance.name != manifest_row["instance_id"]:
                raise SystemExit(
                    f"NAME mismatch: manifest={manifest_row['instance_id']}, "
                    f"file={vrp_path}, parsed={instance.name}"
                )
            reference_solution = pipeline.parse_solution(sol_path)
            reference_cost, reference_routes = reference_cost_and_routes(
                pipeline, instance, reference_solution, args.rounding
            )

            construction = construct_routes(
                pipeline=pipeline,
                instance=instance,
                model=model,
                features=features,
                top_k=args.top_k,
                rounding=args.rounding,
                candidate_mode=args.candidate_mode,
            )
            route_rows, generated_cost, feasible, reason = evaluate_routes(
                pipeline, instance, construction, args.rounding
            )
            gap = (
                (generated_cost - reference_cost) / reference_cost * 100.0
                if feasible and reference_cost > 0
                else ""
            )
            result = {
                "instance_id": instance.name,
                "family": manifest_row.get("family", ""),
                "num_customers": len(instance.customer_ids),
                "split": manifest_row.get("split", ""),
                "feasible": int(feasible),
                "failure_reason": reason or "",
                "num_unserved": len(construction.unserved),
                "num_routes": len(construction.routes),
                "reference_routes": reference_routes,
                "generated_cost": generated_cost,
                "reference_cost": reference_cost,
                "gap_percent": gap,
                "inference_seconds": construction.inference_seconds,
                "model_calls": construction.model_calls,
                "decisions": construction.decisions,
                "mean_candidates_scored": construction.mean_candidates_scored,
                "source_vrp": str(vrp_path),
                "source_sol": str(sol_path),
            }
            writer.writerow(result)
            handle.flush()
            results.append(result)
            append_routes(routes_path, instance.name, route_rows)

            if feasible:
                destination = solutions_dir / f"{instance.name}.sol"
                status = "feasible"
            else:
                destination = partial_dir / f"{instance.name}.txt"
                status = f"infeasible; {reason}; {len(construction.unserved)} unserved"
            write_solution(destination, construction.routes, generated_cost, status)

            gap_text = f"{float(gap):.3f}%" if gap != "" else "n/a"
            print(
                f"[{current}/{len(pairs)}] {instance.name}: "
                f"feasible={feasible}, routes={len(construction.routes)}/"
                f"{reference_routes}, gap={gap_text}, "
                f"inference={construction.inference_seconds:.3f}s",
                flush=True,
            )

    summary = aggregate_summary(results, args)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_comparison_summary(output_dir / "comparison_summary.csv", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
