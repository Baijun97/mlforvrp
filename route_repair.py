#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pcvrptw_core import (
    Instance,
    close_route,
    execute_action,
    initial_vehicle_state,
    output_customer_id,
    parse_vrp_file,
    verify_routes,
)


EPS = 1e-9


def _extract_routes(solution: Dict[str, object]) -> List[List[int]]:
    """Read routes from the existing pipeline's solution.json."""
    raw_routes = solution.get("routes", [])
    routes: List[List[int]] = []

    for item in raw_routes:
        if isinstance(item, dict):
            nodes = item.get("vrp_node_ids")
            if nodes is None:
                raise ValueError(
                    "A route record is missing 'vrp_node_ids'. "
                    "Expected the solution.json produced by the current ML pipeline."
                )
            routes.append([int(node) for node in nodes])
        elif isinstance(item, list):
            routes.append([int(node) for node in item])
        else:
            raise ValueError(f"Unsupported route entry: {type(item).__name__}")

    return routes


def _route_metrics(
    instance: Instance,
    route: List[int],
    distance_mode: str,
) -> Optional[Dict[str, float]]:
    """Return route metrics, or None if this single route is infeasible."""
    if not route:
        return {
            "distance": 0.0,
            "load": 0.0,
            "return_time": 0.0,
            "depot_return_slack": 0.0,
            "collected_prize": 0.0,
        }

    check = verify_routes(instance, [route], distance_mode)
    if not bool(check["feasible"]):
        return None

    state = initial_vehicle_state(instance)
    for node in route:
        execute_action(instance, state, node, distance_mode)

    closure = close_route(instance, state, distance_mode)

    return {
        "distance": float(closure["total_route_distance"]),
        "load": float(instance.capacity - state.remaining_capacity),
        "return_time": float(closure["return_time"]),
        "depot_return_slack": float(closure["depot_return_slack"]),
        "collected_prize": float(sum(instance.prize[node] for node in route)),
    }


def _solution_distance(
    instance: Instance,
    routes: List[List[int]],
    distance_mode: str,
) -> float:
    check = verify_routes(instance, routes, distance_mode)
    return float(check["total_distance_recomputed"])


def _objective_key(
    vehicle_count: int,
    total_distance: float,
    objective: str,
) -> Tuple[float, ...]:
    if objective == "distance":
        return (float(total_distance),)
    if objective == "vehicles_then_distance":
        return (float(vehicle_count), float(total_distance))
    raise ValueError(
        "objective must be 'distance' or 'vehicles_then_distance'"
    )


def _is_better(
    old_vehicle_count: int,
    old_distance: float,
    new_vehicle_count: int,
    new_distance: float,
    objective: str,
) -> bool:
    old_key = _objective_key(old_vehicle_count, old_distance, objective)
    new_key = _objective_key(new_vehicle_count, new_distance, objective)

    if objective == "distance":
        return new_distance < old_distance - EPS

    if new_vehicle_count < old_vehicle_count:
        return True
    if new_vehicle_count > old_vehicle_count:
        return False
    return new_distance < old_distance - EPS


def _best_insertion(
    instance: Instance,
    routes: List[List[int]],
    node: int,
    distance_mode: str,
) -> Optional[Tuple[int, int, float]]:
    """
    Find the feasible insertion with the smallest resulting route distance.

    Returns:
        (target_route_index, insertion_position, new_target_route_distance)
    """
    best: Optional[Tuple[int, int, float]] = None

    for target_idx, target_route in enumerate(routes):
        for insert_pos in range(len(target_route) + 1):
            candidate = target_route[:insert_pos] + [node] + target_route[insert_pos:]
            metrics = _route_metrics(instance, candidate, distance_mode)
            if metrics is None:
                continue

            candidate_distance = float(metrics["distance"])
            item = (target_idx, insert_pos, candidate_distance)

            if best is None or candidate_distance < best[2] - EPS:
                best = item

    return best


def _try_route_consolidation(
    instance: Instance,
    routes: List[List[int]],
    distance_mode: str,
    objective: str,
    max_source_customers: int,
) -> Optional[Tuple[List[List[int]], Dict[str, object]]]:
    """
    Try to eliminate one short/sparse route by reinserting all of its customers
    into the remaining routes.

    This is a whole-route move: intermediate insertions do not need to improve
    the objective. The final consolidated solution must be feasible and better.
    """
    if len(routes) <= 1:
        return None

    old_distance = _solution_distance(instance, routes, distance_mode)
    old_vehicle_count = len(routes)

    source_order = sorted(
        range(len(routes)),
        key=lambda idx: (len(routes[idx]), idx),
    )

    best_solution: Optional[List[List[int]]] = None
    best_log: Optional[Dict[str, object]] = None
    best_key: Optional[Tuple[float, ...]] = None

    for source_idx in source_order:
        source_route = routes[source_idx]

        if not source_route:
            continue
        if len(source_route) > max_source_customers:
            continue

        trial = [list(route) for idx, route in enumerate(routes) if idx != source_idx]

        success = True
        inserted_nodes: List[int] = []

        # Harder customers first tends to reduce failed consolidation attempts.
        nodes_to_insert = sorted(
            source_route,
            key=lambda node: (
                instance.time_windows[node][1],
                -instance.demand[node],
                node,
            ),
        )

        for node in nodes_to_insert:
            insertion = _best_insertion(instance, trial, node, distance_mode)
            if insertion is None:
                success = False
                break

            target_idx, insert_pos, _ = insertion
            trial[target_idx] = (
                trial[target_idx][:insert_pos]
                + [node]
                + trial[target_idx][insert_pos:]
            )
            inserted_nodes.append(node)

        if not success:
            continue

        verification = verify_routes(instance, trial, distance_mode)
        if not bool(verification["feasible"]):
            continue

        new_distance = float(verification["total_distance_recomputed"])
        new_vehicle_count = len(trial)

        if not _is_better(
            old_vehicle_count,
            old_distance,
            new_vehicle_count,
            new_distance,
            objective,
        ):
            continue

        key = _objective_key(new_vehicle_count, new_distance, objective)
        if best_key is None or key < best_key:
            best_key = key
            best_solution = trial
            best_log = {
                "move_type": "route_consolidation",
                "customer": "",
                "source_route": source_idx + 1,
                "destination_route": "multiple",
                "vehicles_before": old_vehicle_count,
                "vehicles_after": new_vehicle_count,
                "distance_before": old_distance,
                "distance_after": new_distance,
                "distance_change": new_distance - old_distance,
                "details": f"reinserted_nodes={inserted_nodes}",
            }

    if best_solution is None or best_log is None:
        return None

    return best_solution, best_log


def _try_relocate(
    instance: Instance,
    routes: List[List[int]],
    distance_mode: str,
    objective: str,
    max_checks: int,
) -> Optional[Tuple[List[List[int]], Dict[str, object]]]:
    """
    Find one improving single-customer relocate move.

    A customer may be:
      - moved to a different position in the same route, or
      - moved to a different route.

    The move is accepted only if affected routes remain feasible and the global
    objective improves.
    """
    if not routes:
        return None

    route_cache: List[Dict[str, float]] = []
    for route in routes:
        metrics = _route_metrics(instance, route, distance_mode)
        if metrics is None:
            raise ValueError("Input solution contains an infeasible route.")
        route_cache.append(metrics)

    old_distance = sum(float(item["distance"]) for item in route_cache)
    old_vehicle_count = len(routes)

    best_solution: Optional[List[List[int]]] = None
    best_log: Optional[Dict[str, object]] = None
    best_key: Optional[Tuple[float, ...]] = None

    checks = 0

    # Short routes first: this gives consolidation opportunities priority.
    source_order = sorted(range(len(routes)), key=lambda idx: (len(routes[idx]), idx))

    for source_idx in source_order:
        source_route = routes[source_idx]

        for source_pos, node in enumerate(source_route):
            source_without = source_route[:source_pos] + source_route[source_pos:]

            # ---- move to another route ----
            for target_idx, target_route in enumerate(routes):
                if target_idx == source_idx:
                    continue

                for insert_pos in range(len(target_route) + 1):
                    checks += 1
                    if checks > max_checks:
                        break

                    target_with = (
                        target_route[:insert_pos]
                        + [node]
                        + target_route[insert_pos:]
                    )

                    target_metrics = _route_metrics(
                        instance, target_with, distance_mode
                    )
                    if target_metrics is None:
                        continue

                    if source_without:
                        source_metrics = _route_metrics(
                            instance, source_without, distance_mode
                        )
                        if source_metrics is None:
                            continue
                        source_new_distance = float(source_metrics["distance"])
                    else:
                        source_new_distance = 0.0

                    new_distance = (
                        old_distance
                        - float(route_cache[source_idx]["distance"])
                        - float(route_cache[target_idx]["distance"])
                        + source_new_distance
                        + float(target_metrics["distance"])
                    )

                    new_vehicle_count = (
                        old_vehicle_count - 1 if not source_without else old_vehicle_count
                    )

                    if not _is_better(
                        old_vehicle_count,
                        old_distance,
                        new_vehicle_count,
                        new_distance,
                        objective,
                    ):
                        continue

                    trial = [list(route) for route in routes]
                    trial[source_idx] = source_without
                    trial[target_idx] = target_with
                    trial = [route for route in trial if route]

                    verification = verify_routes(instance, trial, distance_mode)
                    if not bool(verification["feasible"]):
                        continue

                    new_distance = float(verification["total_distance_recomputed"])
                    new_vehicle_count = len(trial)

                    if not _is_better(
                        old_vehicle_count,
                        old_distance,
                        new_vehicle_count,
                        new_distance,
                        objective,
                    ):
                        continue

                    key = _objective_key(
                        new_vehicle_count, new_distance, objective
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_solution = trial
                        best_log = {
                            "move_type": "relocate",
                            "customer": node,
                            "source_route": source_idx + 1,
                            "destination_route": target_idx + 1,
                            "vehicles_before": old_vehicle_count,
                            "vehicles_after": new_vehicle_count,
                            "distance_before": old_distance,
                            "distance_after": new_distance,
                            "distance_change": new_distance - old_distance,
                            "details": f"insert_position={insert_pos}",
                        }

                if checks > max_checks:
                    break
            if checks > max_checks:
                break

            # ---- reorder within the same route ----
            if len(source_route) >= 3 and checks <= max_checks:
                base_route = source_without
                for insert_pos in range(len(base_route) + 1):
                    checks += 1
                    if checks > max_checks:
                        break

                    # Re-inserting at the equivalent original place changes nothing.
                    candidate_route = (
                        base_route[:insert_pos]
                        + [node]
                        + base_route[insert_pos:]
                    )
                    if candidate_route == source_route:
                        continue

                    metrics = _route_metrics(instance, candidate_route, distance_mode)
                    if metrics is None:
                        continue

                    new_distance = (
                        old_distance
                        - float(route_cache[source_idx]["distance"])
                        + float(metrics["distance"])
                    )

                    if not _is_better(
                        old_vehicle_count,
                        old_distance,
                        old_vehicle_count,
                        new_distance,
                        objective,
                    ):
                        continue

                    trial = [list(route) for route in routes]
                    trial[source_idx] = candidate_route

                    key = _objective_key(
                        old_vehicle_count, new_distance, objective
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_solution = trial
                        best_log = {
                            "move_type": "relocate_within_route",
                            "customer": node,
                            "source_route": source_idx + 1,
                            "destination_route": source_idx + 1,
                            "vehicles_before": old_vehicle_count,
                            "vehicles_after": old_vehicle_count,
                            "distance_before": old_distance,
                            "distance_after": new_distance,
                            "distance_change": new_distance - old_distance,
                            "details": f"insert_position={insert_pos}",
                        }

        if checks > max_checks:
            break

    if best_solution is None or best_log is None:
        return None

    return best_solution, best_log


def repair_routes(
    instance: Instance,
    routes: List[List[int]],
    distance_mode: str = "floor",
    objective: str = "distance",
    max_iterations: int = 30,
    consolidation_max_customers: int = 5,
    max_relocate_checks: int = 5000,
) -> Tuple[List[List[int]], List[Dict[str, object]], Dict[str, object]]:
    """
    Standalone post-processing repair.

    Important:
      - No model is loaded.
      - No feature schema is used.
      - No training data is touched.
      - The visited-customer set is preserved.
    """
    original_routes = [list(route) for route in routes]

    before = verify_routes(instance, original_routes, distance_mode)
    if not bool(before["feasible"]):
        raise ValueError(
            "Route Repair expects a feasible initial solution. "
            "The supplied routes failed capacity/time-window/depot verification."
        )

    original_visited = sorted(node for route in original_routes for node in route)
    current = [list(route) for route in original_routes]
    trace: List[Dict[str, object]] = []

    for iteration in range(1, max_iterations + 1):
        move = _try_route_consolidation(
            instance,
            current,
            distance_mode,
            objective,
            max_source_customers=consolidation_max_customers,
        )

        if move is None:
            move = _try_relocate(
                instance,
                current,
                distance_mode,
                objective,
                max_checks=max_relocate_checks,
            )

        if move is None:
            break

        current, log = move
        log["iteration"] = iteration
        log["accepted"] = True
        trace.append(log)

        # Safety: repair must never add, remove, or duplicate visited customers.
        repaired_visited = sorted(node for route in current for node in route)
        if repaired_visited != original_visited:
            raise RuntimeError(
                "Route Repair changed the set of visited customers. "
                "This should never happen."
            )

    after = verify_routes(instance, current, distance_mode)

    if not bool(after["feasible"]):
        raise RuntimeError("Internal error: repaired solution is infeasible.")

    summary = {
        "objective": objective,
        "iterations_used": len(trace),
        "vehicles_before": len(original_routes),
        "vehicles_after": len(current),
        "distance_before": float(before["total_distance_recomputed"]),
        "distance_after": float(after["total_distance_recomputed"]),
        "distance_change": (
            float(after["total_distance_recomputed"])
            - float(before["total_distance_recomputed"])
        ),
        "feasible_before": bool(before["feasible"]),
        "feasible_after": bool(after["feasible"]),
    }

    return current, trace, summary


def _build_route_records(
    instance: Instance,
    routes: List[List[int]],
    distance_mode: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    for route_id, route in enumerate(routes, start=1):
        metrics = _route_metrics(instance, route, distance_mode)
        if metrics is None:
            raise RuntimeError("Cannot build output record for an infeasible route.")

        records.append(
            {
                "route_id": route_id,
                "vrp_node_ids": route,
                "customer_ids": [
                    output_customer_id(instance, node) for node in route
                ],
                "customers_served": len(route),
                "load": metrics["load"],
                "distance": metrics["distance"],
                "return_time": metrics["return_time"],
                "depot_return_slack": metrics["depot_return_slack"],
                "collected_prize": metrics["collected_prize"],
                "mean_selected_score": None,
                "min_selected_score": None,
                "route_stop_reason": "route_repair_postprocessing",
            }
        )

    return records


def _write_outputs(
    original_solution: Dict[str, object],
    instance: Instance,
    repaired_routes: List[List[int]],
    trace: List[Dict[str, object]],
    summary: Dict[str, object],
    output_dir: Path,
    distance_mode: str,
    repair_runtime_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    repaired = deepcopy(original_solution)
    repaired["routes"] = _build_route_records(
        instance, repaired_routes, distance_mode
    )

    verification = verify_routes(instance, repaired_routes, distance_mode)
    visited = {node for route in repaired_routes for node in route}
    unvisited = sorted(set(instance.client_ids) - visited)

    metrics = dict(repaired.get("metrics", {}))
    metrics.update(
        {
            "vehicles_used": len(repaired_routes),
            "visited_clients": len(visited),
            "unvisited_clients": len(unvisited),
            "total_distance": float(verification["total_distance_recomputed"]),
            "collected_prize": float(sum(instance.prize[node] for node in visited)),
            "uncollected_prize": float(sum(instance.prize[node] for node in unvisited)),
            "capacity_feasible": bool(verification["capacity_feasible"]),
            "time_window_feasible": bool(verification["time_window_feasible"]),
            "depot_return_feasible": bool(verification["depot_return_feasible"]),
            "duplicate_clients": verification["duplicate_clients"],
            "feasible": bool(verification["feasible"]),
            "repair_runtime_seconds": repair_runtime_seconds,
        }
    )
    repaired["metrics"] = metrics
    repaired["unvisited_vrp_node_ids"] = unvisited
    repaired["unvisited_customer_ids"] = [
        output_customer_id(instance, node) for node in unvisited
    ]
    repaired["route_repair"] = summary

    (output_dir / "solution_repaired.json").write_text(
        json.dumps(repaired, indent=2),
        encoding="utf-8",
    )

    lines: List[str] = []
    for record in repaired["routes"]:
        customer_text = " ".join(map(str, record["customer_ids"]))
        lines.append(f"Route #{record['route_id']}: {customer_text}")

    lines.extend(
        [
            "",
            f"Total distance: {metrics['total_distance']}",
            f"Vehicles used: {metrics['vehicles_used']}",
            f"Visited clients: {metrics['visited_clients']}",
            f"Unvisited clients: {metrics['unvisited_clients']}",
            f"Feasible: {metrics['feasible']}",
            f"Repair runtime seconds: {repair_runtime_seconds:.6f}",
            f"Repair moves accepted: {len(trace)}",
        ]
    )

    (output_dir / "routes_repaired.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    trace_path = output_dir / "repair_trace.csv"
    fields = [
        "iteration",
        "move_type",
        "customer",
        "source_route",
        "destination_route",
        "vehicles_before",
        "vehicles_after",
        "distance_before",
        "distance_after",
        "distance_change",
        "accepted",
        "details",
    ]

    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Route Repair post-processor for an existing ML solution.json. "
            "This script does not load or modify the XGBoost model or feature schema."
        )
    )
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument(
        "--solution",
        required=True,
        type=Path,
        help="Existing solution.json produced by the ML routing pipeline.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--distance-mode",
        choices=["floor", "round", "exact"],
        default="floor",
    )
    parser.add_argument(
        "--objective",
        choices=["distance", "vehicles_then_distance"],
        default="distance",
        help=(
            "distance: never accept a longer total-distance solution; "
            "vehicles_then_distance: prefer fewer routes first, then lower distance."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--consolidation-max-customers",
        type=int,
        default=5,
        help="Only attempt whole-route consolidation on routes up to this size.",
    )
    parser.add_argument(
        "--max-relocate-checks",
        type=int,
        default=5000,
        help="Maximum relocate neighbors evaluated per repair iteration.",
    )
    args = parser.parse_args()

    instance = parse_vrp_file(args.instance)
    solution = json.loads(args.solution.read_text(encoding="utf-8"))
    routes = _extract_routes(solution)

    start = time.perf_counter()
    repaired_routes, trace, summary = repair_routes(
        instance,
        routes,
        distance_mode=args.distance_mode,
        objective=args.objective,
        max_iterations=args.max_iterations,
        consolidation_max_customers=args.consolidation_max_customers,
        max_relocate_checks=args.max_relocate_checks,
    )
    elapsed = time.perf_counter() - start

    _write_outputs(
        solution,
        instance,
        repaired_routes,
        trace,
        summary,
        args.output_dir,
        args.distance_mode,
        elapsed,
    )

    print(f"Vehicles: {summary['vehicles_before']} -> {summary['vehicles_after']}")
    print(
        f"Distance: {summary['distance_before']:.3f} "
        f"-> {summary['distance_after']:.3f}"
    )
    print(f"Feasible: {summary['feasible_after']}")
    print(f"Accepted repair moves: {len(trace)}")
    print(f"Repair runtime: {elapsed:.3f} s")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
