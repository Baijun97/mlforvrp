from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from model_adapter import ModelScorer
from pcvrptw_core import (
    FEATURE_COLUMNS,
    Instance,
    close_route,
    execute_action,
    feasible_candidates,
    initial_vehicle_state,
    output_customer_id,
    parse_vrp_file,
    verify_routes,
)


@dataclass
class PipelineConfig:
    distance_mode: str = "floor"
    stop_threshold: float = 0.5
    selection_strategy: str = "greedy"
    top_k: int = 10
    temperature: float = 1.0
    random_seed: int = 42
    max_routes: Optional[int] = None
    max_steps_per_route: int = 10000


@dataclass
class Decision:
    route_id: int
    step_id: int
    current_vrp_node_id: int
    current_customer_id: int
    chosen_vrp_node_id: int
    chosen_customer_id: int
    chosen_score: float
    best_score: float
    selected_rank: int
    selected_probability: float
    selection_strategy: str
    top_k_used: int
    feasible_candidate_count: int
    travel_distance: float
    arrival_time: float
    waiting_time: float
    service_start_time: float
    service_finish_time: float
    remaining_capacity_after: float
    route_distance_after: float


class XGBoostRoutingPipeline:
    def __init__(
        self,
        instance: Instance,
        scorer: ModelScorer,
        config: PipelineConfig,
    ):
        self.instance = instance
        self.scorer = scorer
        self.config = config
        self._validate_selection_config()
        self.rng = random.Random(self.config.random_seed)

    def _validate_selection_config(self) -> None:
        allowed = {"greedy", "topk_uniform", "topk_softmax"}
        if self.config.selection_strategy not in allowed:
            raise ValueError(
                f"Unknown selection_strategy={self.config.selection_strategy!r}. "
                f"Choose one of {sorted(allowed)}."
            )
        if self.config.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.config.temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

    def _select_candidate(
        self, ranked: List[Tuple[int, float]]
    ) -> Tuple[int, float, int, float, int]:
        """Select one candidate from an already score-sorted feasible list.

        Returns:
            chosen_node, chosen_score, selected_rank, selected_probability, top_k_used
        """
        if not ranked:
            raise ValueError("Cannot select from an empty candidate list.")

        if self.config.selection_strategy == "greedy":
            node, score = ranked[0]
            return int(node), float(score), 1, 1.0, 1

        k = min(self.config.top_k, len(ranked))
        pool = ranked[:k]

        if self.config.selection_strategy == "topk_uniform":
            probabilities = [1.0 / k] * k
        else:
            # XGBRanker outputs ranking scores, not probabilities. Convert the
            # top-k scores to a stable softmax distribution before sampling.
            scaled_scores = [float(score) / self.config.temperature for _, score in pool]
            max_scaled = max(scaled_scores)
            weights = [math.exp(score - max_scaled) for score in scaled_scores]
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]

        selected_index = self.rng.choices(
            range(k), weights=probabilities, k=1
        )[0]
        node, score = pool[selected_index]
        return (
            int(node),
            float(score),
            selected_index + 1,
            float(probabilities[selected_index]),
            k,
        )

    def solve(self) -> Dict[str, object]:
        start_clock = time.perf_counter()
        unvisited: set[int] = set(self.instance.client_ids)
        routes: List[List[int]] = []
        route_records: List[Dict[str, object]] = []
        decision_trace: List[Decision] = []

        collected_prize = 0.0
        total_distance = 0.0
        stop_reason = "vehicle_limit"

        route_limit = (
            self.instance.vehicles
            if self.config.max_routes is None
            else min(self.instance.vehicles, self.config.max_routes)
        )

        for route_id in range(1, route_limit + 1):
            if not unvisited:
                stop_reason = "all_clients_visited"
                break

            open_vehicle_count = self.instance.vehicles - route_id + 1
            state = initial_vehicle_state(self.instance)
            route: List[int] = []
            route_scores: List[float] = []
            route_stop_reason = "unknown"

            for step_id in range(self.config.max_steps_per_route):
                candidate_map = feasible_candidates(
                    self.instance,
                    state,
                    unvisited,
                    open_vehicle_count=open_vehicle_count,
                    distance_mode=self.config.distance_mode,
                )

                if not candidate_map:
                    route_stop_reason = "no_feasible_candidate"
                    break

                candidate_nodes = list(candidate_map.keys())
                feature_rows = [candidate_map[node] for node in candidate_nodes]
                scores = self.scorer.predict_scores(feature_rows)

                if len(scores) != len(candidate_nodes):
                    raise RuntimeError(
                        "Model returned a different number of scores than candidates."
                    )

                # Highest model score. Ties prefer shorter travel, then lower node ID.
                ranked = sorted(
                    zip(candidate_nodes, scores),
                    key=lambda item: (
                        float(item[1]),
                        -candidate_map[item[0]]["travel_distance"],
                        -item[0],
                    ),
                    reverse=True,
                )
                # The stop decision must use the BEST available score, not a
                # randomly sampled top-k score. Otherwise a low-ranked sample could
                # incorrectly terminate a route even when a strong candidate exists.
                best_score = float(ranked[0][1])

                # Training data contains all-negative route-end/global-stop states.
                # A low maximum score therefore means "do not choose another customer".
                if best_score < self.config.stop_threshold:
                    if route:
                        route_stop_reason = "score_below_threshold_return_to_depot"
                    else:
                        route_stop_reason = "score_below_threshold_global_stop"
                    break

                (
                    chosen_node,
                    chosen_score,
                    selected_rank,
                    selected_probability,
                    top_k_used,
                ) = self._select_candidate(ranked)

                before_node = state.current_node
                action = execute_action(
                    self.instance,
                    state,
                    chosen_node,
                    self.config.distance_mode,
                )
                unvisited.remove(chosen_node)
                route.append(chosen_node)
                route_scores.append(chosen_score)
                collected_prize += self.instance.prize[chosen_node]

                decision_trace.append(
                    Decision(
                        route_id=route_id,
                        step_id=step_id,
                        current_vrp_node_id=before_node,
                        current_customer_id=(
                            0
                            if before_node == self.instance.depot_id
                            else output_customer_id(self.instance, before_node)
                        ),
                        chosen_vrp_node_id=chosen_node,
                        chosen_customer_id=output_customer_id(
                            self.instance, chosen_node
                        ),
                        chosen_score=chosen_score,
                        best_score=best_score,
                        selected_rank=selected_rank,
                        selected_probability=selected_probability,
                        selection_strategy=self.config.selection_strategy,
                        top_k_used=top_k_used,
                        feasible_candidate_count=len(candidate_nodes),
                        travel_distance=action["travel_distance"],
                        arrival_time=action["arrival_time"],
                        waiting_time=action["waiting_time"],
                        service_start_time=action["service_start_time"],
                        service_finish_time=action["service_finish_time"],
                        remaining_capacity_after=state.remaining_capacity,
                        route_distance_after=state.route_distance,
                    )
                )

                if not unvisited:
                    route_stop_reason = "all_clients_visited"
                    break
            else:
                route_stop_reason = "max_steps_per_route"

            # If the model refuses every candidate at the depot, stop opening vehicles.
            if not route and route_stop_reason == "score_below_threshold_global_stop":
                stop_reason = route_stop_reason
                break

            # If no candidate is feasible from a fresh vehicle, no later identical
            # vehicle can do better, so the construction is finished.
            if not route and route_stop_reason == "no_feasible_candidate":
                stop_reason = "no_feasible_candidate_from_depot"
                break

            if route:
                closure = close_route(
                    self.instance, state, self.config.distance_mode
                )
                route_distance = closure["total_route_distance"]
                total_distance += route_distance

                routes.append(route)
                route_records.append(
                    {
                        "route_id": route_id,
                        "vrp_node_ids": route,
                        "customer_ids": [
                            output_customer_id(self.instance, node) for node in route
                        ],
                        "customers_served": len(route),
                        "load": self.instance.capacity - state.remaining_capacity,
                        "distance": route_distance,
                        "return_time": closure["return_time"],
                        "depot_return_slack": closure["depot_return_slack"],
                        "collected_prize": sum(
                            self.instance.prize[node] for node in route
                        ),
                        "mean_selected_score": (
                            sum(route_scores) / len(route_scores)
                            if route_scores else None
                        ),
                        "min_selected_score": (
                            min(route_scores) if route_scores else None
                        ),
                        "route_stop_reason": route_stop_reason,
                    }
                )

            if not unvisited:
                stop_reason = "all_clients_visited"
                break

            if route_id == route_limit:
                stop_reason = "vehicle_limit"

        verification = verify_routes(
            self.instance, routes, self.config.distance_mode
        )

        # Keep one authoritative distance value; assert internal consistency.
        if abs(
            float(verification["total_distance_recomputed"]) - total_distance
        ) > 1e-6:
            raise RuntimeError("Internal route-distance verification failed.")

        elapsed = time.perf_counter() - start_clock
        unvisited_prize = sum(self.instance.prize[node] for node in unvisited)

        return {
            "instance_name": self.instance.name,
            "problem_type": self.instance.problem_type,
            "config": asdict(self.config),
            "routes": route_records,
            "metrics": {
                "vehicles_used": len(routes),
                "visited_clients": sum(len(route) for route in routes),
                "unvisited_clients": len(unvisited),
                "total_distance": total_distance,
                "collected_prize": collected_prize,
                "uncollected_prize": unvisited_prize,
                "runtime_seconds": elapsed,
                "stop_reason": stop_reason,
                "capacity_feasible": verification["capacity_feasible"],
                "time_window_feasible": verification["time_window_feasible"],
                "depot_return_feasible": verification["depot_return_feasible"],
                "duplicate_clients": verification["duplicate_clients"],
                "feasible": verification["feasible"],
            },
            "unvisited_vrp_node_ids": sorted(unvisited),
            "unvisited_customer_ids": [
                output_customer_id(self.instance, node)
                for node in sorted(unvisited)
            ],
            "decision_trace": [asdict(item) for item in decision_trace],
        }


def write_outputs(result: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full machine-readable result.
    (output_dir / "solution.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # Human-readable / .sol-like route list.
    lines = []
    for route in result["routes"]:
        customer_text = " ".join(map(str, route["customer_ids"]))
        lines.append(f"Route #{route['route_id']}: {customer_text}")

    metrics = result["metrics"]
    lines.extend([
        "",
        f"Total distance: {metrics['total_distance']}",
        f"Vehicles used: {metrics['vehicles_used']}",
        f"Visited clients: {metrics['visited_clients']}",
        f"Unvisited clients: {metrics['unvisited_clients']}",
        f"Collected prize: {metrics['collected_prize']}",
        f"Runtime seconds: {metrics['runtime_seconds']:.6f}",
        f"Feasible: {metrics['feasible']}",
        f"Stop reason: {metrics['stop_reason']}",
    ])
    (output_dir / "routes.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    trace = result["decision_trace"]
    trace_path = output_dir / "decision_trace.csv"
    if trace:
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)
    else:
        trace_path.write_text(
            "route_id,step_id,current_vrp_node_id,current_customer_id,"
            "chosen_vrp_node_id,chosen_customer_id,chosen_score\n",
            encoding="utf-8",
        )


def load_config(
    config_path: Optional[Path],
    *,
    stop_threshold: Optional[float],
    distance_mode: Optional[str],
    selection_strategy: Optional[str],
    top_k: Optional[int],
    temperature: Optional[float],
    random_seed: Optional[int],
    max_routes: Optional[int],
) -> PipelineConfig:
    values: Dict[str, object] = {}

    if config_path is not None:
        values.update(json.loads(config_path.read_text(encoding="utf-8")))

    config = PipelineConfig(
        distance_mode=str(values.get("distance_mode", "floor")),
        stop_threshold=float(values.get("stop_threshold", 0.5)),
        selection_strategy=str(values.get("selection_strategy", "greedy")),
        top_k=int(values.get("top_k", 10)),
        temperature=float(values.get("temperature", 1.0)),
        random_seed=int(values.get("random_seed", 42)),
        max_routes=(
            None
            if values.get("max_routes") is None
            else int(values["max_routes"])
        ),
        max_steps_per_route=int(values.get("max_steps_per_route", 10000)),
    )

    if stop_threshold is not None:
        config.stop_threshold = stop_threshold
    if distance_mode is not None:
        config.distance_mode = distance_mode
    if selection_strategy is not None:
        config.selection_strategy = selection_strategy
    if top_k is not None:
        config.top_k = top_k
    if temperature is not None:
        config.temperature = temperature
    if random_seed is not None:
        config.random_seed = random_seed
    if max_routes is not None:
        config.max_routes = max_routes

    return config
