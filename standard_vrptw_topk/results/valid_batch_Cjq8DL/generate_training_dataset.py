#!/usr/bin/env python3
"""Build labelled candidate-ranking data from standard VRPTW .vrp/.sol pairs.

This script is intentionally strict: it accepts standard single-depot
``TYPE: VRPTW`` and ``TYPE: CVRPTW`` instances, rejects prize-collecting files, verifies that the
reference solution visits every customer exactly once, reconstructs capacity
and time-window schedules, and reconciles the DIMACS route cost.

Each decision becomes a ranking group.  The expert next customer has label=1;
at the end of a route the depot is the positive action.  Thus every
``decision_id`` has exactly one positive label, which is suitable for an
XGBoost ranker or another candidate-ranking model.

Examples
--------
Single pair:
    python generate_training_dataset.py --vrp C1_10_1.vrp --sol C1_10_1.sol \
        --output-dir processed/C1_10_1

Batch directory (pairs files by relative path and stem):
    python generate_training_dataset.py --input-dir Instances/VRPTW \
        --output-dir processed/vrptw
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


SECTION_RE = re.compile(r"^[A-Z][A-Z0-9_]*_SECTION$")
ROUTE_RE = re.compile(r"^Route\s+#(\d+)\s*:\s*(.*)$", re.IGNORECASE)
COST_RE = re.compile(r"^Cost\s*:?\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


INSTANCE_FIELDS = [
    "instance_id", "problem_type", "dimension", "num_customers",
    "vehicle_limit", "capacity", "service_time", "edge_weight_type",
    "num_routes", "num_served", "num_unserved", "solution_cost",
    "route_distance", "cost_gap", "num_duplicate_visits", "num_unknown_visits",
    "num_infeasible_routes", "all_required_served", "solution_feasible",
    "schedule_mode", "source_vrp", "source_sol",
]

CUSTOMER_FIELDS = [
    "instance_id", "problem_type", "solution_customer_id", "raw_node_id",
    "is_depot", "x", "y", "demand", "tw_early", "tw_late", "tw_width",
    "service_time", "served_in_solution",
    "route_id", "visit_position",
]

ROUTE_FIELDS = [
    "instance_id", "route_id", "sequence", "num_clients", "load",
    "capacity", "capacity_utilization", "distance", "start_time", "end_time",
    "duration", "travel_time", "service_time_total", "wait_time", "time_warp",
    "capacity_violation", "is_feasible",
]

VISIT_FIELDS = [
    "instance_id", "route_id", "position", "previous_customer_id",
    "customer_id", "is_return_to_depot", "travel_time", "arrival_time",
    "tw_early", "tw_late", "wait_time", "service_start_time",
    "service_time", "departure_time", "time_window_slack", "time_warp",
    "demand", "route_load_after", "remaining_capacity_after",
    "route_distance_after", "capacity_feasible", "time_feasible",
]

CANDIDATE_FIELDS = [
    "instance_id", "problem_type", "decision_id", "route_id",
    "decision_position", "current_customer_id", "candidate_customer_id",
    "candidate_is_depot", "label", "candidate_pool_size",
    "candidate_sample_size", "candidate_rank_by_travel",
    "num_visited_global", "num_remaining_customers", "current_time",
    "route_distance_so_far", "route_load_so_far", "route_load_ratio",
    "remaining_capacity", "remaining_capacity_ratio", "current_x", "current_y",
    "candidate_x", "candidate_y", "delta_x", "delta_y", "euclidean_distance",
    "travel_time", "travel_time_ratio", "arrival_time", "candidate_tw_early",
    "candidate_tw_late", "candidate_tw_width", "candidate_tw_width_ratio",
    "waiting_time", "service_start_time", "candidate_service_time",
    "departure_time", "time_window_slack", "time_window_slack_ratio",
    "lateness", "return_to_depot_time", "return_arrival_time",
    "candidate_demand", "candidate_demand_ratio", "load_after_candidate",
    "remaining_capacity_after", "capacity_feasible", "time_feasible",
    "return_feasible", "overall_feasible",
]

FEATURE_COLUMNS = [
    field for field in CANDIDATE_FIELDS
    if field not in {
        "instance_id", "problem_type", "decision_id", "route_id",
        "decision_position", "current_customer_id", "candidate_customer_id",
        "label", "candidate_pool_size", "candidate_sample_size",
    }
]


@dataclass(frozen=True)
class Node:
    solution_id: int
    raw_id: int
    x: float
    y: float
    demand: float
    tw_early: float
    tw_late: float
    service_time: float
    is_depot: bool


@dataclass
class Instance:
    name: str
    problem_type: str
    dimension: int
    vehicle_limit: int
    capacity: float
    service_time: float
    edge_weight_type: str
    nodes: dict[int, Node]
    source_vrp: Path

    @property
    def customer_ids(self) -> list[int]:
        return sorted(node_id for node_id in self.nodes if node_id != 0)

    @property
    def depot(self) -> Node:
        return self.nodes[0]

    @property
    def horizon(self) -> float:
        return max(1.0, self.depot.tw_late - self.depot.tw_early)


@dataclass
class Solution:
    routes: list[list[int]]
    claimed_cost: float | None
    source_sol: Path


class CSVOutput:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()

    def writerows(self, rows: Iterable[dict]) -> None:
        self.writer.writerows(rows)

    def close(self) -> None:
        self.handle.close()


def parse_vrp(path: Path) -> Instance:
    headers: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if SECTION_RE.match(line):
            section = line
            sections[section] = []
            continue
        if section is not None:
            sections[section].append(line)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    required_sections = {"NODE_COORD_SECTION", "DEMAND_SECTION", "TIME_WINDOW_SECTION", "DEPOT_SECTION"}
    missing = required_sections - sections.keys()
    if missing:
        raise ValueError(f"{path}: missing sections {sorted(missing)}")

    required_headers = {
        "NAME", "TYPE", "DIMENSION", "VEHICLES", "CAPACITY",
        "SERVICE_TIME", "EDGE_WEIGHT_TYPE",
    }
    missing_headers = required_headers - headers.keys()
    if missing_headers:
        raise ValueError(f"{path}: missing headers {sorted(missing_headers)}")

    problem_type = headers["TYPE"].upper()
    accepted_standard_types = {"VRPTW", "CVRPTW"}
    if problem_type not in accepted_standard_types:
        raise ValueError(
            f"{path}: TYPE is {problem_type!r}; this generator accepts only "
            "standard TYPE: VRPTW or TYPE: CVRPTW instances."
        )
    if "PRIZE_SECTION" in sections:
        raise ValueError(
            f"{path}: PRIZE_SECTION found. Use Instances/VRPTW, not PCVRPTW."
        )
    if headers["EDGE_WEIGHT_TYPE"].upper() != "EUC_2D":
        raise ValueError(f"{path}: only EDGE_WEIGHT_TYPE: EUC_2D is supported.")

    depot_ids = [int(line.split()[0]) for line in sections["DEPOT_SECTION"] if int(line.split()[0]) >= 0]
    if len(depot_ids) != 1:
        raise ValueError(f"{path}: this pipeline expects exactly one depot; found {depot_ids}")
    raw_depot = depot_ids[0]

    coords = {
        int(parts[0]): (float(parts[1]), float(parts[2]))
        for line in sections["NODE_COORD_SECTION"]
        if len(parts := line.split()) >= 3
    }
    demands = {
        int(parts[0]): float(parts[1])
        for line in sections["DEMAND_SECTION"]
        if len(parts := line.split()) >= 2
    }
    time_windows = {
        int(parts[0]): (float(parts[1]), float(parts[2]))
        for line in sections["TIME_WINDOW_SECTION"]
        if len(parts := line.split()) >= 3
    }
    coordinate_ids = set(coords)
    if set(demands) != coordinate_ids or set(time_windows) != coordinate_ids:
        raise ValueError(
            f"{path}: node IDs differ across coordinates, demands, and time windows."
        )

    raw_clients = sorted(raw_id for raw_id in coords if raw_id != raw_depot)
    raw_to_solution = {raw_depot: 0}
    raw_to_solution.update({raw_id: idx for idx, raw_id in enumerate(raw_clients, start=1)})

    default_service = float(headers["SERVICE_TIME"])
    nodes: dict[int, Node] = {}
    for raw_id in [raw_depot, *raw_clients]:
        sol_id = raw_to_solution[raw_id]
        x, y = coords[raw_id]
        tw_early, tw_late = time_windows[raw_id]
        is_depot = raw_id == raw_depot
        nodes[sol_id] = Node(
            solution_id=sol_id,
            raw_id=raw_id,
            x=x,
            y=y,
            demand=demands.get(raw_id, 0.0),
            tw_early=tw_early,
            tw_late=tw_late,
            service_time=0.0 if is_depot else default_service,
            is_depot=is_depot,
        )

    dimension = int(headers["DIMENSION"])
    if dimension != len(nodes):
        raise ValueError(f"{path}: DIMENSION={dimension}, parsed nodes={len(nodes)}")

    return Instance(
        name=headers["NAME"],
        problem_type=problem_type,
        dimension=dimension,
        vehicle_limit=int(headers["VEHICLES"]),
        capacity=float(headers["CAPACITY"]),
        service_time=default_service,
        edge_weight_type=headers["EDGE_WEIGHT_TYPE"].upper(),
        nodes=nodes,
        source_vrp=path,
    )


def parse_solution(path: Path) -> Solution:
    routes: list[list[int]] = []
    claimed_cost: float | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        route_match = ROUTE_RE.match(line)
        if route_match:
            routes.append([int(token) for token in route_match.group(2).split()])
            continue
        cost_match = COST_RE.match(line)
        if cost_match:
            claimed_cost = float(cost_match.group(1))
    if not routes:
        raise ValueError(f"{path}: no routes found")
    return Solution(routes=routes, claimed_cost=claimed_cost, source_sol=path)


def distance(instance: Instance, first: int, second: int, rounding: str) -> float:
    a = instance.nodes[first]
    b = instance.nodes[second]
    exact = math.hypot(a.x - b.x, a.y - b.y)
    if rounding == "dimacs":
        return math.floor(exact * 10 + 1e-12) / 10
    if rounding == "round":
        return float(round(exact))
    if rounding == "trunc":
        return float(math.floor(exact))
    return exact


def schedule_route(instance: Instance, route_id: int, sequence: list[int], rounding: str) -> tuple[dict, list[dict]]:
    depot = instance.depot
    current = 0
    current_time = depot.tw_early
    route_distance = 0.0
    route_load = 0.0
    total_wait = 0.0
    total_time_warp = 0.0
    visit_rows: list[dict] = []

    for position, customer_id in enumerate(sequence, start=1):
        node = instance.nodes.get(customer_id)
        if node is None:
            continue
        travel = distance(instance, current, customer_id, rounding)
        arrival = current_time + travel
        wait = max(0.0, node.tw_early - arrival)
        service_start = max(arrival, node.tw_early)
        time_warp = max(0.0, service_start - node.tw_late)
        departure = service_start + node.service_time
        route_distance += travel
        route_load += node.demand
        total_wait += wait
        total_time_warp += time_warp
        visit_rows.append({
            "instance_id": instance.name,
            "route_id": route_id,
            "position": position,
            "previous_customer_id": current,
            "customer_id": customer_id,
            "is_return_to_depot": 0,
            "travel_time": travel,
            "arrival_time": arrival,
            "tw_early": node.tw_early,
            "tw_late": node.tw_late,
            "wait_time": wait,
            "service_start_time": service_start,
            "service_time": node.service_time,
            "departure_time": departure,
            "time_window_slack": node.tw_late - service_start,
            "time_warp": time_warp,
            "demand": node.demand,
            "route_load_after": route_load,
            "remaining_capacity_after": instance.capacity - route_load,
            "route_distance_after": route_distance,
            "capacity_feasible": int(route_load <= instance.capacity + 1e-9),
            "time_feasible": int(time_warp <= 1e-9),
        })
        current = customer_id
        current_time = departure

    travel = distance(instance, current, 0, rounding)
    arrival = current_time + travel
    wait = max(0.0, depot.tw_early - arrival)
    service_start = max(arrival, depot.tw_early)
    time_warp = max(0.0, service_start - depot.tw_late)
    route_distance += travel
    total_wait += wait
    total_time_warp += time_warp
    visit_rows.append({
        "instance_id": instance.name,
        "route_id": route_id,
        "position": len(sequence) + 1,
        "previous_customer_id": current,
        "customer_id": 0,
        "is_return_to_depot": 1,
        "travel_time": travel,
        "arrival_time": arrival,
        "tw_early": depot.tw_early,
        "tw_late": depot.tw_late,
        "wait_time": wait,
        "service_start_time": service_start,
        "service_time": 0.0,
        "departure_time": service_start,
        "time_window_slack": depot.tw_late - service_start,
        "time_warp": time_warp,
        "demand": 0.0,
        "route_load_after": route_load,
        "remaining_capacity_after": instance.capacity - route_load,
        "route_distance_after": route_distance,
        "capacity_feasible": int(route_load <= instance.capacity + 1e-9),
        "time_feasible": int(time_warp <= 1e-9),
    })

    capacity_violation = max(0.0, route_load - instance.capacity)
    route_row = {
        "instance_id": instance.name,
        "route_id": route_id,
        "sequence": " ".join(map(str, sequence)),
        "num_clients": len(sequence),
        "load": route_load,
        "capacity": instance.capacity,
        "capacity_utilization": route_load / instance.capacity if instance.capacity else 0.0,
        "distance": route_distance,
        "start_time": depot.tw_early,
        "end_time": service_start,
        "duration": service_start - depot.tw_early,
        "travel_time": route_distance,
        "service_time_total": sum(instance.nodes[c].service_time for c in sequence if c in instance.nodes),
        "wait_time": total_wait,
        "time_warp": total_time_warp,
        "capacity_violation": capacity_violation,
        "is_feasible": int(capacity_violation <= 1e-9 and total_time_warp <= 1e-9),
    }
    return route_row, visit_rows


def candidate_features(
    instance: Instance,
    current_id: int,
    candidate_id: int,
    current_time: float,
    route_distance: float,
    route_load: float,
    num_visited: int,
    num_remaining: int,
    rounding: str,
) -> dict:
    current = instance.nodes[current_id]
    candidate = instance.nodes[candidate_id]
    travel = distance(instance, current_id, candidate_id, rounding)
    arrival = current_time + travel
    service_start = max(arrival, candidate.tw_early)
    waiting = max(0.0, candidate.tw_early - arrival)
    lateness = max(0.0, service_start - candidate.tw_late)
    departure = service_start + candidate.service_time
    load_after = route_load if candidate.is_depot else route_load + candidate.demand
    return_time = 0.0 if candidate.is_depot else distance(instance, candidate_id, 0, rounding)
    return_arrival = arrival if candidate.is_depot else departure + return_time
    capacity_feasible = load_after <= instance.capacity + 1e-9
    time_feasible = lateness <= 1e-9
    return_feasible = return_arrival <= instance.depot.tw_late + 1e-9
    horizon = instance.horizon
    euclidean = math.hypot(candidate.x - current.x, candidate.y - current.y)

    return {
        "candidate_is_depot": int(candidate.is_depot),
        "num_visited_global": num_visited,
        "num_remaining_customers": num_remaining,
        "current_time": current_time,
        "route_distance_so_far": route_distance,
        "route_load_so_far": route_load,
        "route_load_ratio": route_load / instance.capacity if instance.capacity else 0.0,
        "remaining_capacity": instance.capacity - route_load,
        "remaining_capacity_ratio": (instance.capacity - route_load) / instance.capacity if instance.capacity else 0.0,
        "current_x": current.x,
        "current_y": current.y,
        "candidate_x": candidate.x,
        "candidate_y": candidate.y,
        "delta_x": candidate.x - current.x,
        "delta_y": candidate.y - current.y,
        "euclidean_distance": euclidean,
        "travel_time": travel,
        "travel_time_ratio": travel / horizon,
        "arrival_time": arrival,
        "candidate_tw_early": candidate.tw_early,
        "candidate_tw_late": candidate.tw_late,
        "candidate_tw_width": candidate.tw_late - candidate.tw_early,
        "candidate_tw_width_ratio": (candidate.tw_late - candidate.tw_early) / horizon,
        "waiting_time": waiting,
        "service_start_time": service_start,
        "candidate_service_time": candidate.service_time,
        "departure_time": departure,
        "time_window_slack": candidate.tw_late - service_start,
        "time_window_slack_ratio": (candidate.tw_late - service_start) / horizon,
        "lateness": lateness,
        "return_to_depot_time": return_time,
        "return_arrival_time": return_arrival,
        "candidate_demand": candidate.demand,
        "candidate_demand_ratio": candidate.demand / instance.capacity if instance.capacity else 0.0,
        "load_after_candidate": load_after,
        "remaining_capacity_after": instance.capacity - load_after,
        "capacity_feasible": int(capacity_feasible),
        "time_feasible": int(time_feasible),
        "return_feasible": int(return_feasible),
        "overall_feasible": int(capacity_feasible and time_feasible and return_feasible),
    }


def generate_candidate_rows(
    instance: Instance,
    solution: Solution,
    route_visits: dict[int, list[dict]],
    top_k: int,
    rounding: str,
) -> Iterator[dict]:
    visited_global: set[int] = set()
    all_customers = set(instance.customer_ids)

    for route_id, sequence in enumerate(solution.routes, start=1):
        current_id = 0
        current_time = instance.depot.tw_early
        route_distance = 0.0
        route_load = 0.0
        scheduled_clients = [row for row in route_visits[route_id] if not row["is_return_to_depot"]]

        for decision_position in range(len(sequence) + 1):
            positive_id = sequence[decision_position] if decision_position < len(sequence) else 0
            available_customers = sorted(all_customers - visited_global)
            pool = available_customers.copy()
            if decision_position > 0:
                pool.append(0)  # depot is the route-termination action
            if positive_id not in pool:
                pool.append(positive_id)

            feature_cache = {
                candidate_id: candidate_features(
                    instance=instance,
                    current_id=current_id,
                    candidate_id=candidate_id,
                    current_time=current_time,
                    route_distance=route_distance,
                    route_load=route_load,
                    num_visited=len(visited_global),
                    num_remaining=len(available_customers),
                    rounding=rounding,
                )
                for candidate_id in pool
            }
            ranked_for_sampling = sorted(
                pool,
                key=lambda candidate_id: (
                    -feature_cache[candidate_id]["overall_feasible"],
                    feature_cache[candidate_id]["travel_time"],
                    feature_cache[candidate_id]["candidate_tw_late"],
                    candidate_id,
                ),
            )
            ranked_by_travel = sorted(
                pool,
                key=lambda candidate_id: (
                    feature_cache[candidate_id]["travel_time"],
                    candidate_id,
                ),
            )
            travel_rank = {
                candidate_id: rank
                for rank, candidate_id in enumerate(ranked_by_travel, start=1)
            }
            negatives = [
                candidate_id
                for candidate_id in ranked_for_sampling
                if candidate_id != positive_id
            ]
            sampled = [positive_id, *negatives[: max(0, top_k - 1)]]
            sampled = sorted(set(sampled))
            decision_id = f"{instance.name}:r{route_id:04d}:p{decision_position:04d}"

            for candidate_id in sampled:
                row = {
                    "instance_id": instance.name,
                    "problem_type": instance.problem_type,
                    "decision_id": decision_id,
                    "route_id": route_id,
                    "decision_position": decision_position,
                    "current_customer_id": current_id,
                    "candidate_customer_id": candidate_id,
                    "label": int(candidate_id == positive_id),
                    "candidate_pool_size": len(pool),
                    "candidate_sample_size": len(sampled),
                    "candidate_rank_by_travel": travel_rank[candidate_id],
                }
                row.update(feature_cache[candidate_id])
                yield row

            if positive_id == 0:
                continue
            visit = scheduled_clients[decision_position]
            travel = float(visit["travel_time"])
            route_distance += travel
            route_load = float(visit["route_load_after"])
            current_time = float(visit["departure_time"])
            current_id = positive_id
            visited_global.add(positive_id)


def analyse_pair(instance: Instance, solution: Solution, rounding: str) -> tuple[dict, list[dict], list[dict], dict[int, list[dict]], list[str]]:
    warnings: list[str] = []
    all_visits = [customer_id for route in solution.routes for customer_id in route]
    known_visits = [customer_id for customer_id in all_visits if customer_id in instance.nodes and customer_id != 0]
    unknown_count = len(all_visits) - len(known_visits)
    duplicate_count = len(known_visits) - len(set(known_visits))
    served = set(known_visits)
    unserved = set(instance.customer_ids) - served

    if unserved:
        warnings.append(f"Standard routing instance has {len(unserved)} unserved required customers.")
    if duplicate_count:
        warnings.append(f"Solution contains {duplicate_count} duplicate client visits.")
    if unknown_count:
        warnings.append(f"Solution contains {unknown_count} unknown client IDs.")
    if len(solution.routes) > instance.vehicle_limit:
        warnings.append("Solution uses more routes than the declared vehicle limit.")

    route_rows: list[dict] = []
    visit_rows: list[dict] = []
    visits_by_route: dict[int, list[dict]] = {}
    customer_assignment: dict[int, tuple[int, int]] = {}
    for route_id, sequence in enumerate(solution.routes, start=1):
        for position, customer_id in enumerate(sequence, start=1):
            if customer_id in instance.nodes:
                customer_assignment[customer_id] = (route_id, position)
        route_row, rows = schedule_route(instance, route_id, sequence, rounding)
        route_rows.append(route_row)
        visit_rows.extend(rows)
        visits_by_route[route_id] = rows

    route_distance = sum(float(row["distance"]) for row in route_rows)
    cost_gap = (
        None
        if solution.claimed_cost is None
        else solution.claimed_cost - route_distance
    )
    num_infeasible = sum(1 for row in route_rows if not row["is_feasible"])
    all_required_served = len(unserved) == 0
    solution_feasible = (
        unknown_count == 0
        and duplicate_count == 0
        and len(solution.routes) <= instance.vehicle_limit
        and num_infeasible == 0
        and all_required_served
    )

    instance_row = {
        "instance_id": instance.name,
        "problem_type": instance.problem_type,
        "dimension": instance.dimension,
        "num_customers": len(instance.customer_ids),
        "vehicle_limit": instance.vehicle_limit,
        "capacity": instance.capacity,
        "service_time": instance.service_time,
        "edge_weight_type": instance.edge_weight_type,
        "num_routes": len(solution.routes),
        "num_served": len(served),
        "num_unserved": len(unserved),
        "solution_cost": solution.claimed_cost,
        "route_distance": route_distance,
        "cost_gap": cost_gap,
        "num_duplicate_visits": duplicate_count,
        "num_unknown_visits": unknown_count,
        "num_infeasible_routes": num_infeasible,
        "all_required_served": int(all_required_served),
        "solution_feasible": int(solution_feasible),
        "schedule_mode": "earliest_depot_departure",
        "source_vrp": str(instance.source_vrp),
        "source_sol": str(solution.source_sol),
    }

    customer_rows: list[dict] = []
    for customer_id, node in sorted(instance.nodes.items()):
        assignment = customer_assignment.get(customer_id)
        customer_rows.append({
            "instance_id": instance.name,
            "problem_type": instance.problem_type,
            "solution_customer_id": customer_id,
            "raw_node_id": node.raw_id,
            "is_depot": int(node.is_depot),
            "x": node.x,
            "y": node.y,
            "demand": node.demand,
            "tw_early": node.tw_early,
            "tw_late": node.tw_late,
            "tw_width": node.tw_late - node.tw_early,
            "service_time": node.service_time,
            "served_in_solution": int(customer_id == 0 or customer_id in served),
            "route_id": assignment[0] if assignment else "",
            "visit_position": assignment[1] if assignment else "",
        })

    if solution.claimed_cost is None:
        warnings.append("Solution file has no parseable Cost value.")
    elif abs(cost_gap or 0.0) > 0.11:
        warnings.append(
            f"Claimed cost does not reconcile with DIMACS route distance: gap={cost_gap:.3f}."
        )
    return instance_row, customer_rows, route_rows, visits_by_route, warnings


def find_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.vrp or args.sol:
        if not (args.vrp and args.sol):
            raise ValueError("--vrp and --sol must be provided together")
        return [(args.vrp.resolve(), args.sol.resolve())]

    root = args.input_dir.resolve()
    pairs: list[tuple[Path, Path]] = []
    for vrp_path in sorted(root.rglob("*.vrp")):
        sol_path = vrp_path.with_suffix(".sol")
        if sol_path.exists():
            pairs.append((vrp_path, sol_path))
    if not pairs:
        raise ValueError(f"No same-stem .vrp/.sol pairs found beneath {root}")
    return pairs


def audit_candidate_csv(path: Path) -> dict:
    """Stream-check the ranking contract without loading the CSV into memory."""
    rows = 0
    groups = 0
    positives = 0
    customer_positives = 0
    depot_positives = 0
    groups_without_one_positive = 0
    groups_with_duplicate_candidates = 0
    groups_with_wrong_declared_size = 0
    positive_rows_infeasible = 0
    rows_with_nulls = 0
    min_group_size: int | None = None
    max_group_size = 0

    current_decision: str | None = None
    current_rows = 0
    current_positives = 0
    current_candidates: set[str] = set()
    current_duplicate = False
    current_declared_size: int | None = None

    def finish_group() -> None:
        nonlocal groups, groups_without_one_positive
        nonlocal groups_with_duplicate_candidates, groups_with_wrong_declared_size
        nonlocal min_group_size, max_group_size
        if current_decision is None:
            return
        groups += 1
        if current_positives != 1:
            groups_without_one_positive += 1
        if current_duplicate:
            groups_with_duplicate_candidates += 1
        if current_declared_size != current_rows:
            groups_with_wrong_declared_size += 1
        min_group_size = (
            current_rows
            if min_group_size is None
            else min(min_group_size, current_rows)
        )
        max_group_size = max(max_group_size, current_rows)

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            decision_id = row["decision_id"]
            if decision_id != current_decision:
                finish_group()
                current_decision = decision_id
                current_rows = 0
                current_positives = 0
                current_candidates = set()
                current_duplicate = False
                current_declared_size = int(row["candidate_sample_size"])

            rows += 1
            current_rows += 1
            if any(value == "" for value in row.values()):
                rows_with_nulls += 1

            candidate_id = row["candidate_customer_id"]
            if candidate_id in current_candidates:
                current_duplicate = True
            current_candidates.add(candidate_id)

            label = int(row["label"])
            positives += label
            current_positives += label
            if label:
                is_depot = int(row["candidate_is_depot"])
                depot_positives += is_depot
                customer_positives += 1 - is_depot
                if int(row["overall_feasible"]) != 1:
                    positive_rows_infeasible += 1

    finish_group()
    return {
        "rows": rows,
        "decision_groups": groups,
        "positive_rows": positives,
        "customer_positive_rows": customer_positives,
        "depot_positive_rows": depot_positives,
        "negative_rows": rows - positives,
        "min_group_size": min_group_size or 0,
        "max_group_size": max_group_size,
        "groups_without_exactly_one_positive": groups_without_one_positive,
        "groups_with_duplicate_candidates": groups_with_duplicate_candidates,
        "groups_with_wrong_declared_size": groups_with_wrong_declared_size,
        "positive_rows_infeasible": positive_rows_infeasible,
        "rows_with_nulls": rows_with_nulls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Batch root containing same-path, same-stem .vrp/.sol pairs",
    )
    source.add_argument("--vrp", type=Path, help="Single standard VRPTW .vrp file")
    parser.add_argument("--sol", type=Path, help="Single .sol file (required with --vrp)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=25, help="Candidates retained per decision (default: 25)")
    parser.add_argument(
        "--rounding",
        choices=["dimacs", "round", "trunc", "none"],
        default="dimacs",
        help="Keep 'dimacs' for official PyVRP VRPTW instances",
    )
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("--top-k must be at least 2")

    pairs = find_pairs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "instances": CSVOutput(output_dir / "instances.csv", INSTANCE_FIELDS),
        "customers": CSVOutput(output_dir / "customers.csv", CUSTOMER_FIELDS),
        "routes": CSVOutput(output_dir / "routes.csv", ROUTE_FIELDS),
        "visits": CSVOutput(output_dir / "visits.csv", VISIT_FIELDS),
        "candidates": CSVOutput(output_dir / "candidate_decisions.csv", CANDIDATE_FIELDS),
    }

    reports: list[dict] = []
    seen_instance_names: set[str] = set()
    try:
        for pair_index, (vrp_path, sol_path) in enumerate(pairs, start=1):
            print(
                f"[{pair_index}/{len(pairs)}] Processing {vrp_path.name} ...",
                flush=True,
            )
            instance = parse_vrp(vrp_path)
            if instance.name in seen_instance_names:
                raise ValueError(
                    f"Duplicate internal instance NAME {instance.name!r}. "
                    "Process colliding benchmark families into separate output directories."
                )
            seen_instance_names.add(instance.name)
            solution = parse_solution(sol_path)
            instance_row, customers, routes, visits_by_route, warnings = analyse_pair(instance, solution, args.rounding)
            if not instance_row["solution_feasible"] or warnings:
                details = "; ".join(warnings) or "solution feasibility check failed"
                raise ValueError(f"{instance.name}: refusing invalid training labels: {details}")
            visits = [row for route_id in sorted(visits_by_route) for row in visits_by_route[route_id]]
            candidates = generate_candidate_rows(instance, solution, visits_by_route, args.top_k, args.rounding)

            outputs["instances"].writerows([instance_row])
            outputs["customers"].writerows(customers)
            outputs["routes"].writerows(routes)
            outputs["visits"].writerows(visits)
            outputs["candidates"].writerows(candidates)
            reports.append({
                "instance_id": instance.name,
                "problem_type": instance.problem_type,
                "num_customers": len(instance.customer_ids),
                "num_routes": len(solution.routes),
                "solution_cost": solution.claimed_cost,
                "recomputed_route_distance": instance_row["route_distance"],
                "cost_gap": instance_row["cost_gap"],
                "solution_feasible": bool(instance_row["solution_feasible"]),
                "warnings": warnings,
            })
            print(
                f"[{pair_index}/{len(pairs)}] OK {instance.name}: "
                f"{len(instance.customer_ids)} customers, "
                f"{len(solution.routes)} routes",
                flush=True,
            )
    finally:
        for output in outputs.values():
            output.close()

    (output_dir / "feature_columns.json").write_text(
        json.dumps({"features": FEATURE_COLUMNS, "label": "label", "group": "decision_id"}, indent=2),
        encoding="utf-8",
    )
    candidate_audit = audit_candidate_csv(output_dir / "candidate_decisions.csv")
    quality_ok = all(
        candidate_audit[key] == 0
        for key in [
            "groups_without_exactly_one_positive",
            "groups_with_duplicate_candidates",
            "groups_with_wrong_declared_size",
            "positive_rows_infeasible",
            "rows_with_nulls",
        ]
    )
    quality_report = {
        "quality_ok": quality_ok,
        "pairs_processed": len(pairs),
        "top_k": args.top_k,
        "rounding": args.rounding,
        "candidate_dataset": candidate_audit,
        "instances": reports,
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2),
        encoding="utf-8",
    )
    if not quality_ok:
        raise ValueError(
            "Candidate dataset failed its ranking-contract audit. "
            "See data_quality_report.json."
        )
    print(json.dumps({
        "output_dir": str(output_dir),
        "quality_ok": quality_ok,
        "pairs_processed": len(pairs),
        "top_k": args.top_k,
        "rounding": args.rounding,
        "candidate_dataset": candidate_audit,
    }, indent=2))


if __name__ == "__main__":
    main()
