from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FEATURE_COLUMNS = [
    "vehicle_current_time",
    "vehicle_end_time",
    "vehicle_remaining_capacity",
    "vehicle_capacity_ratio",
    "vehicle_route_distance",
    "vehicle_customers_served",
    "vehicle_trip_number",
    "candidate_remaining_demand",
    "candidate_ready_time",
    "candidate_due_time",
    "candidate_service_time",
    "candidate_time_window_width",
    "candidate_prize",
    "travel_distance",
    "travel_time",
    "arrival_time",
    "waiting_time",
    "service_start_time",
    "service_finish_time",
    "time_window_slack",
    "delivery_amount",
    "capacity_after_delivery",
    "capacity_after_ratio",
    "candidate_demand_after_delivery",
    "candidate_to_depot_distance",
    "candidate_to_depot_time",
    "return_to_depot_time",
    "depot_return_slack",
    "remaining_customer_count",
    "remaining_total_demand",
    "open_vehicle_count",
    "remaining_total_prize",
]

SECTION_NAMES = {
    "NODE_COORD_SECTION",
    "DEMAND_SECTION",
    "TIME_WINDOW_SECTION",
    "PRIZE_SECTION",
    "DEPOT_SECTION",
}


@dataclass(frozen=True)
class Instance:
    name: str
    problem_type: str
    dimension: int
    vehicles: int
    capacity: float
    service_time: float
    edge_weight_type: str
    depot_id: int
    coordinates: Dict[int, Tuple[float, float]]
    demand: Dict[int, float]
    time_windows: Dict[int, Tuple[float, float]]
    prize: Dict[int, float]

    @property
    def client_ids(self) -> List[int]:
        return sorted(node for node in self.coordinates if node != self.depot_id)


@dataclass
class VehicleState:
    current_node: int
    current_time: float
    remaining_capacity: float
    route_distance: float
    customers_served: int
    trip_number: int = 1


def parse_vrp_text(text: str) -> Instance:
    headers: Dict[str, str] = {}
    sections: Dict[str, List[str]] = {}
    current_section: Optional[str] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line in SECTION_NAMES:
            current_section = line
            sections[current_section] = []
            continue

        if line == "EOF":
            break

        if current_section is not None:
            sections[current_section].append(line)
        elif ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    required_headers = [
        "NAME", "TYPE", "DIMENSION", "VEHICLES",
        "CAPACITY", "SERVICE_TIME", "EDGE_WEIGHT_TYPE",
    ]
    missing_headers = [key for key in required_headers if key not in headers]
    if missing_headers:
        raise ValueError(f"Missing VRP headers: {missing_headers}")

    missing_sections = [section for section in SECTION_NAMES if section not in sections]
    if missing_sections:
        raise ValueError(f"Missing VRP sections: {missing_sections}")

    coordinates: Dict[int, Tuple[float, float]] = {}
    for line in sections["NODE_COORD_SECTION"]:
        node, x, y = line.split()
        coordinates[int(node)] = (float(x), float(y))

    demand: Dict[int, float] = {}
    for line in sections["DEMAND_SECTION"]:
        node, value = line.split()
        demand[int(node)] = float(value)

    time_windows: Dict[int, Tuple[float, float]] = {}
    for line in sections["TIME_WINDOW_SECTION"]:
        node, ready, due = line.split()
        time_windows[int(node)] = (float(ready), float(due))

    prize: Dict[int, float] = {}
    for line in sections["PRIZE_SECTION"]:
        node, value = line.split()
        prize[int(node)] = float(value)

    depot_ids = [
        int(line.split()[0])
        for line in sections["DEPOT_SECTION"]
        if line.split() and int(line.split()[0]) != -1
    ]
    if len(depot_ids) != 1:
        raise ValueError(f"Expected exactly one depot, got {depot_ids}")

    dimension = int(headers["DIMENSION"])
    node_sets = [set(coordinates), set(demand), set(time_windows), set(prize)]
    if any(len(node_set) != dimension for node_set in node_sets):
        raise ValueError(
            f"{headers['NAME']}: DIMENSION={dimension}, but one or more sections "
            "have a different number of nodes."
        )
    if len(set(map(frozenset, node_sets))) != 1:
        raise ValueError(f"{headers['NAME']}: node IDs differ between sections.")

    return Instance(
        name=headers["NAME"],
        problem_type=headers["TYPE"],
        dimension=dimension,
        vehicles=int(headers["VEHICLES"]),
        capacity=float(headers["CAPACITY"]),
        service_time=float(headers["SERVICE_TIME"]),
        edge_weight_type=headers["EDGE_WEIGHT_TYPE"],
        depot_id=depot_ids[0],
        coordinates=coordinates,
        demand=demand,
        time_windows=time_windows,
        prize=prize,
    )


def parse_vrp_file(path: Path | str) -> Instance:
    path = Path(path)
    return parse_vrp_text(path.read_text(encoding="utf-8", errors="replace"))


def distance(instance: Instance, i: int, j: int, mode: str = "floor") -> float:
    if instance.edge_weight_type != "EUC_2D":
        raise ValueError(
            f"Only EDGE_WEIGHT_TYPE=EUC_2D is supported, got {instance.edge_weight_type}"
        )

    xi, yi = instance.coordinates[i]
    xj, yj = instance.coordinates[j]
    raw = math.hypot(xi - xj, yi - yj)

    if mode == "floor":
        return float(math.floor(raw))
    if mode == "round":
        return float(math.floor(raw + 0.5))
    if mode == "exact":
        return float(raw)
    raise ValueError(f"Unknown distance mode: {mode}")


def output_customer_id(instance: Instance, vrp_node_id: int) -> int:
    """
    The uploaded PCVRPTW archive uses:
      .vrp depot = node 1
      .vrp customers = nodes 2..N
      .sol-style customer IDs = 1..N-1

    For other depot numbering, retain the original node ID.
    """
    if instance.depot_id == 1 and vrp_node_id > 1:
        return vrp_node_id - 1
    return vrp_node_id


def initial_vehicle_state(instance: Instance) -> VehicleState:
    depot_ready, _ = instance.time_windows[instance.depot_id]
    return VehicleState(
        current_node=instance.depot_id,
        current_time=depot_ready,
        remaining_capacity=instance.capacity,
        route_distance=0.0,
        customers_served=0,
        trip_number=1,
    )


def candidate_features(
    instance: Instance,
    state: VehicleState,
    candidate: int,
    *,
    unvisited_count: int,
    unvisited_total_demand: float,
    unvisited_total_prize: float,
    open_vehicle_count: int,
    distance_mode: str,
) -> Optional[Dict[str, float]]:
    """
    Returns exactly the same 32 feature definitions used by the training-data
    generator. Returns None when the candidate is not feasible.
    """
    candidate_demand = instance.demand[candidate]
    if candidate_demand > state.remaining_capacity + 1e-9:
        return None

    ready, due = instance.time_windows[candidate]
    _, depot_due = instance.time_windows[instance.depot_id]

    travel_distance = distance(instance, state.current_node, candidate, distance_mode)
    # The benchmark has no separate speed or travel-time matrix.
    travel_time = travel_distance

    arrival_time = state.current_time + travel_time
    waiting_time = max(0.0, ready - arrival_time)
    service_start_time = max(arrival_time, ready)

    if service_start_time > due + 1e-9:
        return None

    service_finish_time = service_start_time + instance.service_time
    candidate_to_depot_distance = distance(
        instance, candidate, instance.depot_id, distance_mode
    )
    candidate_to_depot_time = candidate_to_depot_distance
    return_to_depot_time = service_finish_time + candidate_to_depot_time

    # Every accepted action must preserve the ability to return to the depot.
    if return_to_depot_time > depot_due + 1e-9:
        return None

    delivery_amount = candidate_demand
    capacity_after_delivery = state.remaining_capacity - delivery_amount

    features = {
        "vehicle_current_time": state.current_time,
        "vehicle_end_time": depot_due,
        "vehicle_remaining_capacity": state.remaining_capacity,
        "vehicle_capacity_ratio": state.remaining_capacity / instance.capacity,
        "vehicle_route_distance": state.route_distance,
        "vehicle_customers_served": float(state.customers_served),
        "vehicle_trip_number": float(state.trip_number),

        "candidate_remaining_demand": candidate_demand,
        "candidate_ready_time": ready,
        "candidate_due_time": due,
        "candidate_service_time": instance.service_time,
        "candidate_time_window_width": due - ready,
        "candidate_prize": instance.prize[candidate],

        "travel_distance": travel_distance,
        "travel_time": travel_time,
        "arrival_time": arrival_time,
        "waiting_time": waiting_time,
        "service_start_time": service_start_time,
        "service_finish_time": service_finish_time,
        "time_window_slack": due - service_start_time,

        "delivery_amount": delivery_amount,
        "capacity_after_delivery": capacity_after_delivery,
        "capacity_after_ratio": capacity_after_delivery / instance.capacity,
        "candidate_demand_after_delivery": max(
            0.0, candidate_demand - delivery_amount
        ),

        "candidate_to_depot_distance": candidate_to_depot_distance,
        "candidate_to_depot_time": candidate_to_depot_time,
        "return_to_depot_time": return_to_depot_time,
        "depot_return_slack": depot_due - return_to_depot_time,

        "remaining_customer_count": float(unvisited_count),
        "remaining_total_demand": unvisited_total_demand,
        "open_vehicle_count": float(open_vehicle_count),
        "remaining_total_prize": unvisited_total_prize,
    }

    if list(features) != FEATURE_COLUMNS:
        raise RuntimeError("Feature definition/order does not match FEATURE_COLUMNS.")

    return features


def feasible_candidates(
    instance: Instance,
    state: VehicleState,
    unvisited: set[int],
    *,
    open_vehicle_count: int,
    distance_mode: str,
) -> Dict[int, Dict[str, float]]:
    unvisited_total_demand = sum(instance.demand[node] for node in unvisited)
    unvisited_total_prize = sum(instance.prize[node] for node in unvisited)
    unvisited_count = len(unvisited)

    rows: Dict[int, Dict[str, float]] = {}
    for candidate in sorted(unvisited):
        features = candidate_features(
            instance,
            state,
            candidate,
            unvisited_count=unvisited_count,
            unvisited_total_demand=unvisited_total_demand,
            unvisited_total_prize=unvisited_total_prize,
            open_vehicle_count=open_vehicle_count,
            distance_mode=distance_mode,
        )
        if features is not None:
            rows[candidate] = features
    return rows


def execute_action(
    instance: Instance,
    state: VehicleState,
    candidate: int,
    distance_mode: str,
) -> Dict[str, float]:
    ready, due = instance.time_windows[candidate]
    travel_distance = distance(instance, state.current_node, candidate, distance_mode)
    travel_time = travel_distance
    arrival_time = state.current_time + travel_time
    waiting_time = max(0.0, ready - arrival_time)
    service_start_time = max(arrival_time, ready)

    if service_start_time > due + 1e-9:
        raise ValueError(f"Attempted infeasible time-window action to node {candidate}.")

    service_finish_time = service_start_time + instance.service_time
    if instance.demand[candidate] > state.remaining_capacity + 1e-9:
        raise ValueError(f"Attempted infeasible capacity action to node {candidate}.")

    state.current_node = candidate
    state.current_time = service_finish_time
    state.remaining_capacity -= instance.demand[candidate]
    state.route_distance += travel_distance
    state.customers_served += 1

    return {
        "travel_distance": travel_distance,
        "travel_time": travel_time,
        "arrival_time": arrival_time,
        "waiting_time": waiting_time,
        "service_start_time": service_start_time,
        "service_finish_time": service_finish_time,
    }


def close_route(
    instance: Instance,
    state: VehicleState,
    distance_mode: str,
) -> Dict[str, float]:
    return_distance = distance(
        instance, state.current_node, instance.depot_id, distance_mode
    )
    return_time = state.current_time + return_distance
    _, depot_due = instance.time_windows[instance.depot_id]

    if return_time > depot_due + 1e-9:
        raise ValueError(
            f"Route cannot return to depot by due time: {return_time} > {depot_due}"
        )

    return {
        "return_distance": return_distance,
        "return_time": return_time,
        "total_route_distance": state.route_distance + return_distance,
        "depot_return_slack": depot_due - return_time,
    }


def verify_routes(
    instance: Instance,
    routes: List[List[int]],
    distance_mode: str,
) -> Dict[str, object]:
    seen: set[int] = set()
    total_distance = 0.0
    capacity_feasible = True
    time_window_feasible = True
    depot_return_feasible = True
    duplicate_clients: List[int] = []

    for route in routes:
        state = initial_vehicle_state(instance)
        for node in route:
            if node in seen:
                duplicate_clients.append(node)
            seen.add(node)

            if instance.demand[node] > state.remaining_capacity + 1e-9:
                capacity_feasible = False

            ready, due = instance.time_windows[node]
            travel = distance(instance, state.current_node, node, distance_mode)
            arrival = state.current_time + travel
            start = max(arrival, ready)
            if start > due + 1e-9:
                time_window_feasible = False

            # Continue calculation even if invalid so verification remains informative.
            state.current_node = node
            state.current_time = start + instance.service_time
            state.remaining_capacity -= instance.demand[node]
            state.route_distance += travel
            state.customers_served += 1

        return_distance = distance(
            instance, state.current_node, instance.depot_id, distance_mode
        )
        return_time = state.current_time + return_distance
        _, depot_due = instance.time_windows[instance.depot_id]
        if return_time > depot_due + 1e-9:
            depot_return_feasible = False
        total_distance += state.route_distance + return_distance

    return {
        "capacity_feasible": capacity_feasible,
        "time_window_feasible": time_window_feasible,
        "depot_return_feasible": depot_return_feasible,
        "unique_clients": len(seen),
        "duplicate_clients": duplicate_clients,
        "total_distance_recomputed": total_distance,
        "feasible": (
            capacity_feasible
            and time_window_feasible
            and depot_return_feasible
            and not duplicate_clients
        ),
    }
